import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time

# ==================================================================
# [Track 2 Configuration]
# Optimized Baseline for Fair Comparison (Not PDF standard)
# ==================================================================
BATCH_SIZE = 64         # PDF(4) -> 64 (For stable training)
EPOCHS = 20             # PDF(2) -> 20 (To see convergence)
LEARNING_RATE = 0.001   
MOMENTUM = 0.9          
# ==================================================================

def main():
    # 1. Device Configuration
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Device: CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple M1/M2 MPS")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    # 2. Data Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 3. Define the Network (Same Structure as Baseline)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net().to(device)

    # 4. Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # Lists for visualization
    history_loss = []
    history_acc = []
    
    print(f"\n=== Start Training (Track 2: Batch={BATCH_SIZE}, Epochs={EPOCHS}) ===")
    start_time = time.time()

    # 5. Training Loop
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Track 2 변경점: 로그 출력을 Epoch 단위로 변경 (배치 64라 Step이 적음)
        total_steps = len(trainloader)
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Epoch이 끝날 때마다 로그 출력 및 기록
        avg_loss = running_loss / total_steps
        acc = 100 * correct / total
        print(f'[Epoch {epoch + 1}/{EPOCHS}] loss: {avg_loss:.3f}, acc: {acc:.2f}%')
        
        history_loss.append(avg_loss)
        history_acc.append(acc)

    print('Finished Training')
    print(f"Training time: {time.time() - start_time:.2f} sec")

    # 6. Save Results (Filename changed to baseline_track2...)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_loss, label='Train Loss')
    plt.title(f'Baseline Track 2 (B={BATCH_SIZE}, E={EPOCHS})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_acc, label='Train Acc', color='orange')
    plt.title(f'Baseline Track 2 (B={BATCH_SIZE}, E={EPOCHS})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    plt.savefig(f'baseline_track2_graph_b{BATCH_SIZE}_e{EPOCHS}.png')
    print(f"Graph saved as 'baseline_track2_graph_b{BATCH_SIZE}_e{EPOCHS}.png'")

    # 7. Evaluation
    print("\n=== Evaluation on Test Set ===")
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    final_acc = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {final_acc:.2f}%')

    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix (Track 2, Acc: {final_acc:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'baseline_track2_cm_b{BATCH_SIZE}_e{EPOCHS}.png')
    print(f"Confusion Matrix saved as 'baseline_track2_cm_b{BATCH_SIZE}_e{EPOCHS}.png'")

if __name__ == '__main__':
    main()