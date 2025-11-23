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
# [Scenario C] Scenario B + Network Capacity x2
# Goal: Demonstrate 'Overfitting' or 'Inefficiency' with large models
# Expected Result: High training time, Potential overfitting
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
    print(f"Running Scenario C (Wide Network) on {device}")

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

    # 3. Model: WideNetBN (Capacity x2)
    class WideNetBN(nn.Module):
        def __init__(self):
            super().__init__()
            # Double the channels: 6->12, 16->32
            self.conv1 = nn.Conv2d(3, 12, 5)
            self.bn1 = nn.BatchNorm2d(12)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(12, 32, 5)
            self.bn2 = nn.BatchNorm2d(32)
            
            # Double the FC units
            self.fc1 = nn.Linear(32 * 5 * 5, 240) # 32 channels
            self.fc2 = nn.Linear(240, 168)
            self.fc3 = nn.Linear(168, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = WideNetBN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # 4. Training Loop
    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    
    start_time = time.time()
    print(f"\n=== Scenario C Training Started (Total: {EPOCHS} Epochs) ===")

    for epoch in range(EPOCHS):
        # [Train Phase]
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
        
        # [Validation Phase]
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
        
        # Statistics
        avg_train_loss = train_loss_sum / len(trainloader)
        avg_test_loss = test_loss_sum / len(testloader)
        acc = 100 * correct / total
        
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(acc)
        
        # Log every 5 epochs
        if (epoch+1) % 5 == 0:
            elapsed = int(time.time() - start_time)
            m, s = divmod(elapsed, 60)
            print(f"[Scenario C] Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Acc: {acc:.2f}% ({m}m {s}s)")

    # 5. Save Results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Losses
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['test_loss'], label='Test Loss', color='red', linestyle='--')
    plt.title('Scenario C: Train vs Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['test_acc'], label='Test Accuracy', color='green')
    plt.title('Scenario C Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = 'screenshots/scenario_c_graph.png'
    plt.savefig(save_path)
    
    print("-" * 60)
    print(f"Scenario C Finished.")
    print(f"Final Accuracy: {history['test_acc'][-1]:.2f}%")
    print(f"Total Time: {int(time.time()-start_time)//60}m {int(time.time()-start_time)%60}s")
    print(f"Graph saved to {save_path}")
    print("-" * 60)

if __name__ == '__main__':
    main()