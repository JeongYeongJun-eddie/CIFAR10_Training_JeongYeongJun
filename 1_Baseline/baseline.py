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
# [Track 1 Configuration]
# Strictly following the PDF specifications for the Baseline
# ==================================================================
BATCH_SIZE = 4          # PDF Page 6
EPOCHS = 2              # PDF Page 19
LEARNING_RATE = 0.001   # PDF Page 18
MOMENTUM = 0.9          # PDF Page 18
# ==================================================================

def main():
    # 1. Device Configuration
    # Check for CUDA (Colab/PC) or MPS (Mac M1/M2) or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Device: CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple M1/M2 MPS")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    # 2. Data Preprocessing (PDF Page 6)
    # Normalize with mean=(0.5, 0.5, 0.5) and std=(0.5, 0.5, 0.5)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 Dataset
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

    # 3. Define the Network (PDF Page 14-16)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            # Input: 3 channels, Output: 6 channels, Kernel: 5x5
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            # Input: 6 channels, Output: 16 channels, Kernel: 5x5
            self.conv2 = nn.Conv2d(6, 16, 5)
            # Fully Connected Layers
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dims except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net().to(device)

    # 4. Loss Function and Optimizer (PDF Page 18)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # Lists for visualization
    history_loss = []
    history_acc = []
    
    print(f"\n=== Start Training (Track 1: Batch={BATCH_SIZE}, Epochs={EPOCHS}) ===")
    start_time = time.time()

    # 5. Training Loop (PDF Page 19)
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Print every 2000 mini-batches (PDF requirement)
            if i % 2000 == 1999:
                avg_loss = running_loss / 2000
                acc = 100 * correct / total
                print(f'[Epoch {epoch + 1}, Step {i + 1:5d}] loss: {avg_loss:.3f}, acc: {acc:.2f}%')
                
                # Save for plotting
                history_loss.append(avg_loss)
                history_acc.append(acc)
                
                running_loss = 0.0
                correct = 0
                total = 0

    print('Finished Training')
    print(f"Training time: {time.time() - start_time:.2f} sec")

    # 6. Save Results: Loss & Accuracy Graph
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history_loss, label='Train Loss')
    plt.title(f'Baseline Loss (B={BATCH_SIZE}, E={EPOCHS})')
    plt.xlabel('Steps (x2000)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(history_acc, label='Train Acc', color='orange')
    plt.title(f'Baseline Accuracy (B={BATCH_SIZE}, E={EPOCHS})')
    plt.xlabel('Steps (x2000)')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    plt.savefig(f'baseline_track1_graph_b{BATCH_SIZE}_e{EPOCHS}.png')
    print(f"Graph saved as 'baseline_track1_graph_b{BATCH_SIZE}_e{EPOCHS}.png'")

    # 7. Evaluation & Confusion Matrix
    print("\n=== Evaluation on Test Set ===")
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    net.eval() # Set model to evaluation mode
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
    plt.title(f'Confusion Matrix (Baseline Track 1, Acc: {final_acc:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'baseline_track1_cm_b{BATCH_SIZE}_e{EPOCHS}.png')
    print(f"Confusion Matrix saved as 'baseline_track1_cm_b{BATCH_SIZE}_e{EPOCHS}.png'")

if __name__ == '__main__':
    main()