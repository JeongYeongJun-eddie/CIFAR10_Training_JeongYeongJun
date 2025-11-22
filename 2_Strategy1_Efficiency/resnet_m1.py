import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import time
import matplotlib.pyplot as plt

# ==================================================================
# [Strategy 1] Cost Efficiency Stack
# Architecture: ResNet-9 (Lightweight)
# Optimizer: AdamW + OneCycleLR
# Goal: Accuracy > 85% within 20-30 mins on M1
# ==================================================================
BATCH_SIZE = 128      # Optimized for M1 memory
EPOCHS = 24           # Sufficient for OneCycleLR convergence
MAX_LR = 0.003        # Peak learning rate
WEIGHT_DECAY = 0.05   # Regularization
# ==================================================================

def main():
    # 0. Setup Directory
    if not os.path.exists('screenshots'):
        os.makedirs('screenshots')

    # 1. Device Configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Strategy 1 (ResNet-9) on {device}")

    # 2. Data Preprocessing (Standard Augmentation)
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 3. Model: ResNet-9 Definition
    def conv_bn(in_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool: layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    class ResNet9(nn.Module):
        def __init__(self, in_channels=3, num_classes=10):
            super().__init__()
            self.prep = conv_bn(in_channels, 64)
            
            self.layer1 = conv_bn(64, 128, pool=True)
            self.res1 = nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))
            
            self.layer2 = conv_bn(128, 256, pool=True)
            self.layer3 = conv_bn(256, 512, pool=True)
            self.res3 = nn.Sequential(conv_bn(512, 512), conv_bn(512, 512))
            
            self.classifier = nn.Sequential(
                nn.MaxPool2d(4), 
                nn.Flatten(), 
                nn.Linear(512, num_classes, bias=False)
            )

        def forward(self, x):
            out = self.prep(x)
            out = self.layer1(out)
            out = self.res1(out) + out
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.res3(out) + out
            out = self.classifier(out)
            return out

    model = ResNet9().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, epochs=EPOCHS, 
                                              steps_per_epoch=len(trainloader))

    # 5. Training Loop
    # Tracking both Train and Test losses for comparison
    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    
    start_time = time.time()
    print(f"\n=== ResNet-9 Training Started (Total: {EPOCHS} Epochs) ===")

    for epoch in range(EPOCHS):
        # [Train Phase]
        model.train()
        train_loss_sum = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            
            optimizer.step()
            scheduler.step() # Step per batch
            
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
                loss = nn.CrossEntropyLoss()(outputs, labels)
                test_loss_sum += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate Stats
        avg_train_loss = train_loss_sum / len(trainloader)
        avg_test_loss = test_loss_sum / len(testloader)
        acc = 100 * correct / total
        
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(acc)
        
        # Log Output (Every Epoch)
        elapsed = int(time.time() - start_time)
        m, s = divmod(elapsed, 60)
        print(f"[ResNet] Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Acc: {acc:.2f}% ({m}m {s}s)")

    # 6. Save Results (Graph)
    plt.figure(figsize=(12, 5))
    
    # Left: Loss Comparison (Train vs Test)
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['test_loss'], label='Test Loss', color='red', linestyle='--')
    plt.title('Strategy 1: Train vs Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Right: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['test_acc'], label='Test Accuracy', color='purple')
    plt.title('Strategy 1: Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, 'resnet_m1_graph.png')
    plt.savefig(save_path)
    
    print("-" * 60)
    print(f"Strategy 1 Finished.")
    print(f"Final Accuracy: {history['test_acc'][-1]:.2f}%")
    print(f"Total Time: {int(time.time()-start_time)//60}m {int(time.time()-start_time)%60}s")
    print(f"Graph saved to {save_path}")
    print("-" * 60)

if __name__ == '__main__':
    main()