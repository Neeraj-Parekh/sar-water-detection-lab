"""
Training Script for India Chip Selector
========================================

Trains ResNet-18 classifier on 120 India SAR chips with:
- SAR-specific augmentations
- Class balancing
- GradCAM visualization
- Early stopping

Usage:
    python train_chip_selector.py --chip-dir ./chips --epochs 30

Author: SAR Water Detection Lab
Date: 2026-01-23
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt

from india_chip_selector import IndiaChipSelector, SARChipDataset, GradCAM, compute_accuracy

# =============================================================================
# SAR-Specific Augmentations
# =============================================================================

class SARRandomFlip:
    """Random horizontal/vertical flip (SAR invariant)."""
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, x):
        if np.random.rand() < self.p:
            x = torch.flip(x, dims=[2])  # Horizontal
        if np.random.rand() < self.p:
            x = torch.flip(x, dims=[1])  # Vertical
        return x

class SARRandomRotate:
    """Random 90-degree rotations."""
    def __init__(self):
        pass
    
    def __call__(self, x):
        k = np.random.randint(0, 4)  # 0, 90, 180, 270 degrees
        return torch.rot90(x, k, dims=[1, 2])

class SARSpeckleNoise:
    """Add multiplicative speckle noise (SAR-specific)."""
    def __init__(self, std=0.05):
        self.std = std
    
    def __call__(self, x):
        noise = torch.randn_like(x) * self.std + 1.0
        return x * noise

def get_train_transforms():
    """SAR-specific training augmentations."""
    return T.Compose([
        SARRandomFlip(p=0.5),
        SARRandomRotate(),
        SARSpeckleNoise(std=0.05),
        T.RandomResizedCrop(size=(512, 512), scale=(0.85, 1.0)),
    ])

def get_val_transforms():
    """No augmentation for validation."""
    return None

# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_acc = 0
    n_batches = 0
    
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        total_acc += compute_accuracy(outputs, labels)
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'accuracy': total_acc / n_batches
    }

def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_acc = 0
    n_batches = 0
    
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            total_acc += compute_accuracy(outputs, labels)
            n_batches += 1
    
    return {
        'val_loss': total_loss / n_batches,
        'val_accuracy': total_acc / n_batches
    }

# =============================================================================
# Main Training
# =============================================================================

def train(chip_dir: Path, output_dir: Path, epochs: int = 30, 
          batch_size: int = 8, lr: float = 1e-4):
    """Main training function."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Starting training at {datetime.now()}")
    
    # Load chips (combine TIF and NPY)
    chip_files = list(chip_dir.glob("*_7band_jrc_f32.tif"))
    chip_files.extend(list(chip_dir.glob("*.npy")))
    
    print(f"Found {len(chip_files)} chips")
    
    if len(chip_files) < 20:
        print(f"⚠️ WARNING: Only {len(chip_files)} chips found. Need at least 20-30 for robust training.")
    
    # Create full dataset (auto-labeled from JRC water coverage)
    full_dataset = SARChipDataset(chip_files)
    
    # Split: 80% train, 20% val
    n_val = int(len(full_dataset) * 0.2)
    n_train = len(full_dataset) - n_val
    
    train_dataset, val_dataset = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply transforms
    train_dataset.dataset.transform = get_train_transforms()
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Model
    model = IndiaChipSelector(pretrained=True).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Training history
    history = {'train': [], 'val': []}
    best_acc = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        # Log
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.3f} | "
              f"Val Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_accuracy']:.3f}")
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        # Save best model
        if val_metrics['val_accuracy'] > best_acc:
            best_acc = val_metrics['val_accuracy']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, output_dir / 'best_model.pt')
            print(f"  → Saved best model (Acc: {best_acc:.3f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (patience={patience})")
            break
    
    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pt')
    
    # Save history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training complete! Best Acc: {best_acc:.3f}")
    
    # Generate GradCAM visualizations
    print("\nGenerating GradCAM visualizations...")
    generate_gradcam_samples(model, val_dataset, output_dir, device)
    
    return model, history

def generate_gradcam_samples(model, val_dataset, output_dir, device, n_samples=5):
    """Generate GradCAM heatmaps for sample chips."""
    gradcam = GradCAM(model)
    gradcam_dir = output_dir / 'gradcam_samples'
    gradcam_dir.mkdir(exist_ok=True)
    
    for i in range(min(n_samples, len(val_dataset))):
        features, label = val_dataset[i]
        features_batch = features.unsqueeze(0).to(device)
        
        # Generate CAM
        cam = gradcam.generate_cam(features_batch, target_class=label)
        
        # Overlay on VV band
        vv_band = features[0].numpy()  # VV
        overlay = gradcam.overlay_heatmap(vv_band, cam, alpha=0.4)
        
        # Save
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(vv_band, cmap='gray')
        plt.title(f"SAR VV Band\nLabel: {'Good' if label == 1 else 'Reject'}")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title("GradCAM Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(gradcam_dir / f'sample_{i:02d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {n_samples} GradCAM visualizations to {gradcam_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train India Chip Selector")
    parser.add_argument("--chip-dir", type=Path, required=True, help="Directory with chip files")
    parser.add_argument("--output-dir", type=Path, default=Path("./chip_selector_models"), help="Output directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    
    args = parser.parse_args()
    
    train(
        chip_dir=args.chip_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
