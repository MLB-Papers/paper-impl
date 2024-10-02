import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchsummary import summary


class OverFeatAccurate(nn.Module):
    def __init__(self, num_classes=100):
        super(OverFeatAccurate, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(1024 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)
        return x


class OverFeatFast(nn.Module):
    def __init__(self, num_classes=100):
        super(OverFeatFast, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(512 * 7 * 7, 3072)
        self.fc2 = nn.Linear(3072, 3072)
        self.fc3 = nn.Linear(3072, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)
        return x


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

trainset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

accurate_model = OverFeatAccurate(num_classes=100).to(device)
fast_model = OverFeatFast(num_classes=100).to(device)

print("Accurate Model Summary:")
summary(accurate_model, input_size=(3, 224, 224), device=str(device))

print("\nFast Model Summary:")
summary(fast_model, input_size=(3, 224, 224), device=str(device))

criterion = nn.CrossEntropyLoss()


def train(model, optimizer, num_epochs=10):
    model.to(device)
    loss_values = []
    accuracy_values = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        model.train()
        progress_bar = tqdm(
            enumerate(trainloader, 0),
            total=len(trainloader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
        )
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(
                loss=running_loss / (i + 1), accuracy=100 * correct / total
            )

        loss_values.append(running_loss / len(trainloader))
        accuracy_values.append(100 * correct / total)
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {100 * correct / total:.2f}%"
        )

    return loss_values, accuracy_values


epochs = 100

optimizer_accurate = optim.SGD(accurate_model.parameters(), lr=0.01, momentum=0.9)
optimizer_fast = optim.SGD(fast_model.parameters(), lr=0.01, momentum=0.9)

accurate_loss, accurate_acc = train(
    accurate_model, optimizer_accurate, num_epochs=epochs
)
fast_loss, fast_acc = train(fast_model, optimizer_fast, num_epochs=epochs)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(accurate_loss, label="Accurate Model Loss")
plt.plot(fast_loss, label="Fast Model Loss")
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accurate_acc, label="Accurate Model Accuracy")
plt.plot(fast_acc, label="Fast Model Accuracy")
plt.title("Accuracy")
plt.legend()

plt.show()
