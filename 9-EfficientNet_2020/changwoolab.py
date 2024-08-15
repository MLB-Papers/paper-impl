import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from math import ceil
from functools import partial
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              kernel_size//2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, squeeze_factor=4):
        super(SqueezeExcitation, self).__init__()
        squeeze_channels = in_channels // squeeze_factor
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, 1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        expand_channels = in_channels * expand_ratio
        self.use_residual = (self.stride == 1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, expand_channels, kernel_size=1))

        layers.extend([
            ConvBNReLU(expand_channels, expand_channels, kernel_size=kernel_size, stride=stride, groups=expand_channels),
            SqueezeExcitation(expand_channels),
            nn.Conv2d(expand_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.block(x)
        else:
            return self.block(x)

## Constants determined by small grid search
alpha = 1.2
beta = 1.1
gamma = 1.15

class EfficientNet(nn.Module):
    def __init__(self, phi=0, num_classes=1000, dropout_rate=0.2):
        super(EfficientNet, self).__init__()

        # Calculate scaled width, depth, and resolution
        width_mult = alpha ** phi
        depth_mult = beta ** phi
        resolution_mult = gamma ** phi

        def round_filters(filters, width_mult):
            return int(filters * width_mult)

        def round_repeats(repeats, depth_mult):
            return int(ceil(repeats * depth_mult))

        settings = [
            # expand_ratio, channels, repeats, stride, kernel_size
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]

        out_channels = round_filters(32, width_mult)
        features = [ConvBNReLU(3, out_channels, kernel_size=3, stride=2)]

        in_channels = out_channels
        for expand_ratio, channels, repeats, stride, kernel_size in settings:
            out_channels = round_filters(channels, width_mult)
            repeats = round_repeats(repeats, depth_mult)

            for i in range(repeats):
                stride = stride if i == 0 else 1
                features.append(MBConvBlock(in_channels, out_channels, expand_ratio, stride, kernel_size))
                in_channels = out_channels

        out_channels = round_filters(1280, width_mult)
        features.append(ConvBNReLU(in_channels, out_channels, kernel_size=1))

        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(out_channels, num_classes)

        # Apply resolution scaling
        self.resolution_mult = resolution_mult

    def forward(self, x):
        # Scale the resolution before applying the features
        if self.resolution_mult != 1.0:
            new_size = [int(s * self.resolution_mult) for s in x.shape[2:]]
            x = F.interpolate(x, size=new_size, mode='bilinear', align_corners=False)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

# EfficientNet variants using the phi value to scale width, depth, and resolution
def efficientnet(phi, num_classes=1000):
    return EfficientNet(phi=phi, num_classes=num_classes)

def train_and_evaluate_efficientnet(variant='b0', num_epochs=10, batch_size=64, learning_rate=0.001):
    # Mapping of variant to model function
    variant_map = {
        'b0': efficientnet(phi=0, num_classes=10),
        'b1': efficientnet(phi=1, num_classes=10),
        'b2': efficientnet(phi=2, num_classes=10),
        'b3': efficientnet(phi=3, num_classes=10),
        'b4': efficientnet(phi=4, num_classes=10),
        'b5': efficientnet(phi=5, num_classes=10),
        'b6': efficientnet(phi=6, num_classes=10),
        'b7': efficientnet(phi=7, num_classes=10)
    }

    if variant not in variant_map:
        raise ValueError(f"Variant {variant} is not valid. Choose from 'b0' to 'b7'.")

    # Load CIFAR-10 dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),  # Resize to match EfficientNet input size
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match EfficientNet input size
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize the model, loss function, and optimizer
    model = variant_map[variant]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:    # Print every 100 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    print('Finished Training')

    # Evaluate on the test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')

    return model, accuracy

# Example usage:
for phi in range(8):
    print(f"Testing EfficientNet with phi={phi}")
    model, accuracy = train_and_evaluate_efficientnet(variant=f'b{phi}')
    print(f"Model EfficientNet with phi={phi} achieved an accuracy of {accuracy:.2f}% on CIFAR-10\n")

