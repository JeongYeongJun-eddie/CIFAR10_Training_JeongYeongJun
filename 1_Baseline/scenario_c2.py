import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import time

# ==================================================================
# [Scenario C-2] Network Capacity x4 (Extreme Width Test)
# Goal: Test the limits of "Diminishing Returns"
# Config: Channels [24, 64] (Baseline was [6, 16])
# ==================================================================
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

def main():
    # 0. Setup Directory
    if not os.path.exists('screenshots'): 
        os.makedirs('screenshots')
    
    # 1. Device Configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Scenario C-2 (x4 Width) on {device}")

    # 2. Data Preprocessing
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 3. Model: VeryWideNetBN (Capacity x4)
    class VeryWideNetBN(nn.Module):
        def __init__(self):
            super().__init__()
            # Quadruple the channels: 6->24, 16->64
            self.conv1 = nn.Conv2d(3, 24, 5)
            self.bn1 = nn.BatchNorm2d(24)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(24, 64, 5)
            self.bn2 = nn.BatchNorm2d(64)
            
            # Scale FC units accordingly
            # Feature map size is same (5x5), but depth is 64
            self.fc1 = nn.Linear(64 * 5 * 5, 480) # Baseline(120) * 4
            self.fc2 = nn.Linear(480, 336)        # Baseline(84) * 4
            self.fc3 = nn.Linear(336, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = VeryWideNetBN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # 4. Training Loop
    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    
    start_time = time.time()
    print(f"\n=== Scenario C-2 Training Started (Total: {EPOCHS} Epochs) ===")

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
        
        model.eval()
        test_loss_sum = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss_sum += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss_sum / len(trainloader)
        avg_test_loss = test_loss_sum / len(testloader)
        acc = 100 * correct / total
        
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(acc)
        
        if (epoch+1) % 5 == 0:
            elapsed = int(time.time() - start_time)
            m, s = divmod(elapsed, 60)
            print(f"[Scenario C-2] Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Acc: {acc:.2f}% ({m}m {s}s)")

    # 5. Save Results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['test_loss'], label='Test Loss', color='red', linestyle='--')
    plt.title('Scenario C-2: Train vs Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['test_acc'], label='Test Accuracy', color='green')
    plt.title('Scenario C-2 Accuracy (x4 Width)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = 'screenshots/scenario_c2_graph.png'
    plt.savefig(save_path)
    
    print("-" * 60)
    print(f"Scenario C-2 Finished.")
    print(f"Final Accuracy: {history['test_acc'][-1]:.2f}%")
    print(f"Total Time: {int(time.time()-start_time)//60}m {int(time.time()-start_time)%60}s")
    print(f"Graph saved to {save_path}")
    print("-" * 60)

if __name__ == '__main__':
    main()