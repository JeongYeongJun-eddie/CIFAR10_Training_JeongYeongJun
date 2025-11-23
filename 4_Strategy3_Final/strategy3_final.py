import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision
from torchvision.transforms import v2
import os
import time
import matplotlib.pyplot as plt

# ==================================================================
# [Strategy 3: The Final Integration]
# Components: ResNet-9 + AdamW + OneCycleLR + CutMix + TTA
# Narrative: "Merging Architecture Innovation with Data-Centric Methods"
# Safety: num_workers=0 (M1 Stable)
# ==================================================================
BATCH_SIZE = 128
EPOCHS = 50            
MAX_LR = 0.003
WEIGHT_DECAY = 0.05
NUM_CLASSES = 10
# ==================================================================

# Global CutMix Definition
cutmix = v2.CutMix(num_classes=NUM_CLASSES)

def cutmix_collate_fn(batch):
    return cutmix(*torch.utils.data.default_collate(batch))

def main():
    # 0. Setup Directory
    if not os.path.exists('screenshots'): os.makedirs('screenshots')

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running Final Strategy (ResNet + CutMix + TTA) on {device}")

    # 1. Data Prep
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    train_transform = v2.Compose([
        v2.RandomCrop(32, padding=4, padding_mode='reflect'),
        v2.RandomHorizontalFlip(),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=stats[0], std=stats[1])
    ])
    
    test_transform = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=stats[0], std=stats[1])
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=test_transform)

    # [Safety] num_workers=0 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, 
                                              num_workers=0, collate_fn=cutmix_collate_fn)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Model: ResNet-9
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
            self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_classes, bias=False))

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

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, epochs=EPOCHS, 
                                              steps_per_epoch=len(trainloader))
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    start_time = time.time()
    
    print(f"\n=== Final Training Started (ResNet9 + CutMix) for {EPOCHS} Epochs ===")

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
            scheduler.step()
            train_loss_sum += loss.item()
        
        # Validation (Test Loss + Acc)
        model.eval()
        test_loss_sum = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels) # Test Loss 계산
                test_loss_sum += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_train_loss = train_loss_sum / len(trainloader)
        avg_test_loss = test_loss_sum / len(testloader)
        val_acc = 100 * correct / total
        
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(val_acc)
        
        # 5 Epoch마다 출력
        if (epoch+1) % 5 == 0:
            elapsed = int(time.time() - start_time)
            m, s = divmod(elapsed, 60)
            print(f"[Final] Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Acc: {val_acc:.2f}% ({m}m {s}s)")

    elapsed = time.time() - start_time
    print(f"\nTraining Finished in {int(elapsed//60)}m {int(elapsed%60)}s")

    # ★★★ TTA (Test Time Augmentation) ★★★
    print("\n>>> Applying TTA (Test Time Augmentation)...")
    model.eval()
    correct_tta = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 1. Original
            out1 = model(inputs)
            # 2. Flip
            inputs_flipped = torchvision.transforms.functional.hflip(inputs)
            out2 = model(inputs_flipped)
            
            # Ensemble
            final_out = (out1 + out2) / 2
            _, predicted = torch.max(final_out.data, 1)
            total += labels.size(0)
            correct_tta += (predicted == labels).sum().item()

    final_tta_acc = 100 * correct_tta / total
    print("-" * 50)
    print(f"Original Final Acc: {history['test_acc'][-1]:.2f}%")
    print(f"🚀 TTA Final Acc: {final_tta_acc:.2f}%")
    print("-" * 50)

    # Save Graph
    plt.figure(figsize=(12, 5))
    
    # Loss Graph
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['test_loss'], label='Test Loss', color='red', linestyle='--')
    plt.title('Final Strategy: Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy Graph
    plt.subplot(1, 2, 2)
    plt.plot(history['test_acc'], label='Standard Acc')
    plt.axhline(y=final_tta_acc, color='r', linestyle='--', label=f'TTA Acc ({final_tta_acc:.2f}%)')
    plt.title('Final Strategy: Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join('screenshots', 'final_strategy_graph.png')
    plt.savefig(save_path)
    print(f"Graph saved to '{save_path}'")

if __name__ == '__main__':
    main()