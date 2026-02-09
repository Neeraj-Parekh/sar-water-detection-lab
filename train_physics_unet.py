"""
Training Script for Physics-Guided U-Net
=========================================

Trains the PhysicsGuidedUNet model on SAR water detection chips.
Can run alongside gpu_equation_search.py on the same GPU.

Usage:
    python train_physics_unet.py --chip-dir ./chips --epochs 50

Author: SAR Water Detection Lab
Date: 2026-01-19
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple

from physics_unet import (
    PhysicsGuidedUNet, PhysicsLoss, 
    compute_iou, compute_f1, device
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Dataset
# =============================================================================

class SARWaterDataset(Dataset):
    """
    Dataset for SAR water detection chips.
    
    Expected chip format: [H, W, 8] numpy array
    Channels: [VV, VH, DEM, Slope, HAND, TWI, ?, Truth]
    """
    
    def __init__(self, chip_files: List[Path], augment: bool = False):
        self.augment = augment
        if self.augment:
            self.chip_files = self._validate_files(chip_files)
        else:
            self.chip_files = chip_files
            
        # Band mapping
        self.band_indices = {
            'vv': 0, 'vh': 1, 'dem': 2, 'slope': 3,
            'hand': 4, 'twi': 5, 'truth': 7
        }
    
    def _validate_files(self, files: List[Path]) -> List[Path]:
        """Check for corrupted files."""
        valid_files = []
        for f in files:
            try:
                # fast check: try to read header/shape without loading full data if possible, 
                # but np.load(mmap_mode='r') is good. Or just try loading.
                # Since we have few chips (118), full load is fine.
                data = np.load(f, mmap_mode='r')
                if data.shape[2] >= 6: # Minimum bands required
                    valid_files.append(f)
            except Exception as e:
                logger.warning(f"Skipping corrupted file {f}: {e}")
        logger.info(f"Kept {len(valid_files)}/{len(files)} valid chips")
        return valid_files
    
    def __len__(self):
        return len(self.chip_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        chip_path = self.chip_files[idx]
        try:
            data = np.load(chip_path).astype(np.float32)
        except Exception as e:
            logger.error(f"Error loading {chip_path}: {e}")
            # Return dummy data to avoid crashing
            return torch.zeros(6, 256, 256), torch.zeros(256, 256), {'chip_path': str(chip_path), 'hand': torch.zeros(256, 256), 'slope': torch.zeros(256, 256)}
        
        # Extract bands
        vv = data[:, :, 0]
        vh = data[:, :, 1]
        dem = data[:, :, 2] if data.shape[2] > 2 else np.zeros_like(vv)
        slope = data[:, :, 3] if data.shape[2] > 3 else np.zeros_like(vv)
        hand = data[:, :, 4] if data.shape[2] > 4 else np.zeros_like(vv)
        twi = data[:, :, 5] if data.shape[2] > 5 else np.zeros_like(vv)
        
        # Truth label
        if data.shape[2] == 8:
            truth = data[:, :, 7]
        elif data.shape[2] == 7:
            truth = data[:, :, 6]
        else:
            truth = np.zeros_like(vv)
        
        # Stack features [6, H, W]
        features = np.stack([vv, vh, dem, hand, slope, twi], axis=0)
        
        # Normalize
        features = self._normalize(features)
        
        # Handle NaN
        features = np.nan_to_num(features, nan=0.0)
        truth = np.nan_to_num(truth, nan=0.0)
        
        # Augmentation
        if self.augment:
            features, truth, hand, slope = self._augment(features, truth, hand, slope)
        
        # To tensors
        features_tensor = torch.from_numpy(features)
        truth_tensor = torch.from_numpy(truth)
        
        metadata = {
            'hand': torch.from_numpy(hand.copy()),
            'slope': torch.from_numpy(slope.copy()),
            'chip_path': str(chip_path)
        }
        
        return features_tensor, truth_tensor, metadata
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize each channel."""
        # VV, VH: typically -30 to 0 dB -> scale to ~0-1
        features[0] = (features[0] + 30) / 30
        features[1] = (features[1] + 30) / 30
        
        # DEM: 0-5000m -> scale
        features[2] = features[2] / 1000
        
        # HAND: 0-50m -> scale
        features[3] = features[3] / 50
        
        # Slope: 0-90 deg -> scale
        features[4] = features[4] / 90
        
        # TWI: 0-20 -> scale
        features[5] = features[5] / 20
        
        return np.clip(features, 0, 1)
    
    def _augment(self, features, truth, hand, slope):
        """Simple augmentations."""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            features = np.flip(features, axis=2).copy()
            truth = np.flip(truth, axis=1).copy()
            hand = np.flip(hand, axis=1).copy()
            slope = np.flip(slope, axis=1).copy()
        
        # Random vertical flip
        if np.random.rand() > 0.5:
            features = np.flip(features, axis=1).copy()
            truth = np.flip(truth, axis=0).copy()
            hand = np.flip(hand, axis=0).copy()
            slope = np.flip(slope, axis=0).copy()
        
        return features, truth, hand, slope


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(model: nn.Module, loader: DataLoader, 
                optimizer: optim.Optimizer, loss_fn: PhysicsLoss) -> Dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_iou = 0
    n_batches = 0
    
    for features, truth, metadata in loader:
        features = features.to(device)
        truth = truth.to(device)
        hand = metadata['hand'].to(device)
        slope = metadata['slope'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(features)
        
        # Loss
        loss, breakdown = loss_fn(logits, truth, hand, slope)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)[:, 1]
            iou = compute_iou(probs, truth)
        
        total_loss += loss.item()
        total_iou += iou
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'iou': total_iou / n_batches
    }


def validate(model: nn.Module, loader: DataLoader, loss_fn: PhysicsLoss) -> Dict:
    """Validate model."""
    model.eval()
    
    total_loss = 0
    total_iou = 0
    total_f1 = 0
    n_batches = 0
    
    with torch.no_grad():
        for features, truth, metadata in loader:
            features = features.to(device)
            truth = truth.to(device)
            hand = metadata['hand'].to(device)
            slope = metadata['slope'].to(device)
            
            logits = model(features)
            loss, _ = loss_fn(logits, truth, hand, slope)
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            iou = compute_iou(probs, truth)
            f1 = compute_f1(probs, truth)
            
            total_loss += loss.item()
            total_iou += iou
            total_f1 += f1
            n_batches += 1
    
    return {
        'val_loss': total_loss / n_batches,
        'val_iou': total_iou / n_batches,
        'val_f1': total_f1 / n_batches
    }


# =============================================================================
# Main Training
# =============================================================================

def train(chip_dir: Path, output_dir: Path, epochs: int = 50,
          batch_size: int = 4, lr: float = 1e-4, val_split: float = 0.2):
    """Main training function."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load chips
    chip_files = list(chip_dir.glob("*.npy"))
    logger.info(f"Found {len(chip_files)} chips")
    
    if len(chip_files) == 0:
        logger.error("No chips found!")
        return
    
    # Split train/val
    np.random.seed(42)
    np.random.shuffle(chip_files)
    
    n_val = int(len(chip_files) * val_split)
    val_files = chip_files[:n_val]
    train_files = chip_files[n_val:]
    
    logger.info(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Create datasets
    train_dataset = SARWaterDataset(train_files, augment=True)
    val_dataset = SARWaterDataset(val_files, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = PhysicsGuidedUNet(in_channels=6, num_classes=2).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    loss_fn = PhysicsLoss(hand_weight=0.3, slope_weight=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Training history
    history = {'train': [], 'val': []}
    best_iou = 0
    
    for epoch in range(epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn)
        
        # Validate
        val_metrics = validate(model, val_loader, loss_fn)
        
        scheduler.step()
        
        # Log
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.3f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f}, IoU: {val_metrics['val_iou']:.3f}, F1: {val_metrics['val_f1']:.3f}"
        )
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        # Save best model
        if val_metrics['val_iou'] > best_iou:
            best_iou = val_metrics['val_iou']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
            }, output_dir / 'best_model.pt')
            logger.info(f"  -> Saved best model (IoU: {best_iou:.3f})")
    
    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pt')
    
    # Save history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Training complete! Best IoU: {best_iou:.3f}")
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Physics-Guided U-Net")
    parser.add_argument("--chip-dir", type=Path, required=True,
                        help="Directory containing NPY chip files")
    parser.add_argument("--output-dir", type=Path, default=Path("./models"),
                        help="Output directory for models")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    
    args = parser.parse_args()
    
    logger.info(f"Device: {device}")
    logger.info(f"Starting training at {datetime.now()}")
    
    train(
        chip_dir=args.chip_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
