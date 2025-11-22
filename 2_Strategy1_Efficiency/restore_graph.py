import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import math

# ==================================================================
# [Visualization Utility]
# Reconstruct training graphs using logged data from the terminal.
# This ensures data preservation without re-running the training.
# ==================================================================
BATCH_SIZE = 64
EPOCHS = 24
MAX_LR = 0.01
# ==================================================================

def main():
    # 0. Setup Directory
    save_dir = 'screenshots'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # -------------------------------------------------------
    # 1. Restored Data (Extracted from Terminal Logs)
    # -------------------------------------------------------
    epochs = range(1, 25)
    
    # Train Loss (Blue Line)
    train_loss = [
        1.2522, 0.8336, 0.6828, 0.5618, 0.4478, 0.3978, 0.3110, 0.2569,
        0.1990, 0.1772, 0.1431, 0.1173, 0.0905, 0.0660, 0.0490, 0.0337,
        0.0242, 0.0199, 0.0149, 0.0111, 0.0082, 0.0066, 0.0054, 0.0188
    ]
    
    # Test Loss (Red Line)
    test_loss = [
        0.9223, 1.1035, 0.8580, 0.7345, 0.5519, 0.5803, 0.5122, 0.4660,
        0.4274, 0.3532, 0.3851, 0.2698, 0.3922, 0.2824, 0.2265, 0.2357,
        0.2318, 0.2221, 0.2340, 0.2266, 0.2212, 0.2248, 0.2251, 0.2240
    ]
    
    # Test Accuracy (Purple Line)
    test_acc = [
        67.82, 64.98, 72.67, 76.18, 80.98, 81.15, 83.63, 84.77,
        86.54, 87.83, 86.25, 91.94, 92.28, 93.08, 93.35, 93.48,
        92.76, 93.66, 93.38, 93.77, 93.84, 93.55, 93.91, 93.66
    ]

    # -------------------------------------------------------
    # 2. Simulate LR Schedule (OneCycleLR)
    # -------------------------------------------------------
    # Calculate total steps based on dataset size (50,000 images)
    total_images = 50000
    steps_per_epoch = math.ceil(total_images / BATCH_SIZE)
    
    # Create dummy model/optimizer to simulate scheduler behavior
    dummy_model = torch.nn.Linear(1, 1)
    optimizer = optim.AdamW(dummy_model.parameters(), lr=MAX_LR)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, epochs=EPOCHS, 
                                              steps_per_epoch=steps_per_epoch)
    
    lr_history = []
    
    # Run scheduler simulation
    for _ in range(EPOCHS):
        for _ in range(steps_per_epoch):
            lr_history.append(scheduler.get_last_lr()[0])
            scheduler.step()

    # -------------------------------------------------------
    # 3. Plot and Save
    # -------------------------------------------------------
    plt.figure(figsize=(18, 5))
    
    # (1) Loss Comparison
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, label='Train Loss', color='blue')
    plt.plot(epochs, test_loss, label='Test Loss', color='red', linestyle='--')
    plt.title('Strategy 1: Train vs Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # (2) Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, test_acc, label='Test Accuracy', color='purple')
    plt.title('Strategy 1: Test Accuracy (Max: 93.66%)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # (3) Learning Rate Schedule
    plt.subplot(1, 3, 3)
    plt.plot(lr_history, color='orange', label='Learning Rate')
    plt.title('OneCycleLR Schedule')
    plt.xlabel('Total Steps')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save to file
    save_path = os.path.join(save_dir, 'resnet_m1_graph_full.png')
    plt.tight_layout()
    plt.savefig(save_path)
    
    print("-" * 50)
    print(f"Visualization Complete.")
    print(f"Graph saved to: {save_path}")
    print("-" * 50)

if __name__ == '__main__':
    main()