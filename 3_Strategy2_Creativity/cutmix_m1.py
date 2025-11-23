import torch
import torch.nn as nn
import torch.nn.functional as F  
import torch.optim as optim
import torchvision
from torchvision.transforms import v2
import os
import time
import matplotlib.pyplot as plt
import numpy as np

# ==================================================================
# [Strategy 2] Data-Centric Approach: CutMix
# Model: Baseline NetBN (Fixed, Same as Scenario D)
# Method: CutMix Augmentation (Regularization)
# Fix: num_workers=0 (Prevent OSError on M1)
# ==================================================================
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
MOMENTUM = 0.9
NUM_CLASSES = 10
# ==================================================================

# Global Scope for CutMix (Best Practice)
cutmix = v2.CutMix(num_classes=NUM_CLASSES)

def cutmix_collate_fn(batch):
    return cutmix(*torch.utils.data.default_collate(batch))

def main():
    # 0. Setup Directory
    if not os.path.exists('screenshots'):
        os.makedirs('screenshots')

    # 1. Device Configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Strategy 2 (CutMix) on {device}")

    # 2. Data Preprocessing
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    train_transform = v2.Compose([
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=stats[0], std=stats[1])
    ])
    
    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=stats[0], std=stats[1])
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=test_transform)

    # [Fix] num_workers=0 to prevent 'Too many open files' error
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, 
                                              num_workers=0, collate_fn=cutmix_collate_fn)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ★ Save CutMix Sample Images (Visualization) ★
    print("Generating CutMix sample images...")
    try:
        dataiter = iter(trainloader)
        images, labels = next(dataiter)
        
        # Unnormalize
        mean = torch.tensor(stats[0]).view(3, 1, 1)
        std = torch.tensor(stats[1]).view(3, 1, 1)
        sample_imgs = images[:4] * std + mean
        
        # Save Grid
        grid_img = torchvision.utils.make_grid(sample_imgs, nrow=4)
        plt.figure(figsize=(10, 3))
        plt.imshow(grid_img.permute(1, 2, 0).clip(0, 1))
        plt.axis('off')
        plt.title("CutMix Training Samples")
        plt.savefig(os.path.join('screenshots', 'cutmix_samples.png'))
        plt.close()
        print(f"Saved sample image to 'screenshots/cutmix_samples.png'")
    except Exception as e:
        print(f"Warning: Sample save failed. {e}")

    # 3. Model: NetBN (Same as Scenario D)
    class NetBN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.bn1 = nn.BatchNorm2d(6)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.bn2 = nn.BatchNorm2d(16)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = NetBN().to(device)

    # 4. Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 5. Training Loop
    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    
    start_time = time.time()
    print(f"\n=== Strategy 2 (CutMix) Started (Total: {EPOCHS} Epochs) ===")

    for epoch in range(EPOCHS):
        # Train
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
        
        scheduler.step()

        # Validation
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
            print(f"[CutMix] Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Acc: {acc:.2f}% ({m}m {s}s)")

    print("-" * 60)
    print(f"Strategy 2 Finished.")
    print(f"Final Accuracy: {history['test_acc'][-1]:.2f}%")
    print(f"Total Time: {int(time.time()-start_time)//60}m {int(time.time()-start_time)%60}s")

    # 6. Save Graph
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['test_loss'], label='Test Loss', color='red', linestyle='--')
    plt.title('Strategy 2: Train vs Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['test_acc'], label='Test Accuracy', color='purple')
    plt.title('Strategy 2: Accuracy (CutMix)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join('screenshots', 'cutmix_m1_graph.png')
    plt.savefig(save_path)
    print(f"Graphs saved to '{save_path}'")
    print("-" * 60)

if __name__ == '__main__':
    main()