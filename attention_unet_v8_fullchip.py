#!/usr/bin/env python3
"""
================================================================================
FULL-CHIP U-NET v8 - River Connectivity Expert
================================================================================

KEY DIFFERENCE FROM v7:
- NO random cropping to 256x256
- Train on FULL 513x513 chips to learn river continuity
- Pad to 512x512 for clean divisibility
- Focus on "long thin connected structures"

Why this matters:
- Rivers are 5-20 pixels wide but 100s of pixels long
- Random crops break river continuity
- U-Net needs to see the full context to learn "connected lines"

Architecture:
- Attention U-Net with residual blocks (same as v7)
- 8 input channels: VV, VH, DEM, SLOPE, HAND, TWI, MNDWI, VH_texture
- Focal + Dice + Boundary Loss
- Mixed precision training for 24GB GPU

Author: SAR Water Detection Project
Date: 2026-01-25
"""

import os
import sys
import json
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.ndimage import uniform_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "random_seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "chip_dir": Path("/home/mit-aoe/sar_water_detection/chips_expanded_npy"),
    "model_dir": Path("/home/mit-aoe/sar_water_detection/models"),
    "results_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    # Model Architecture
    "in_channels": 8,  # VV, VH, DEM, SLOPE, HAND, TWI, MNDWI, VH_texture
    "base_filters": 32,
    "dropout": 0.2,
    # Training - FULL CHIP (no cropping!)
    "batch_size": 2,  # Smaller batch for full 512x512
    "target_size": 512,  # Pad 513 to 512 for clean divisibility
    "num_epochs": 150,
    "learning_rate": 5e-5,  # Lower LR for full chips
    "weight_decay": 1e-4,
    "patience": 25,
    # Loss weights
    "focal_weight": 0.3,
    "dice_weight": 0.4,
    "boundary_weight": 0.2,
    "connectivity_weight": 0.1,  # NEW: Penalize disconnected predictions
    "focal_gamma": 2.0,
    # Augmentation (spatial only, preserve connectivity)
    "use_augmentation": True,
    "aug_flip": True,
    "aug_rotate": True,
    "aug_brightness": 0.1,  # Small SAR intensity variation
    # Mixed precision
    "use_amp": True,
}


# =============================================================================
# ATTENTION GATE
# =============================================================================


class AttentionGate(nn.Module):
    """Attention Gate for skip connections."""

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True), nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True), nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# =============================================================================
# RESIDUAL BLOCK
# =============================================================================


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


# =============================================================================
# ATTENTION U-NET
# =============================================================================


class AttentionUNet(nn.Module):
    """U-Net with Attention Gates and Residual Blocks."""

    def __init__(self, in_channels=8, out_channels=1, base_filters=32, dropout=0.2):
        super().__init__()
        f = base_filters

        # Encoder
        self.enc1 = ResidualBlock(in_channels, f, dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(f, f * 2, dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(f * 2, f * 4, dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResidualBlock(f * 4, f * 8, dropout)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(f * 8, f * 16, dropout)

        # Decoder with Attention
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.att4 = AttentionGate(F_g=f * 8, F_l=f * 8, F_int=f * 4)
        self.dec4 = ResidualBlock(f * 16, f * 8, dropout)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.att3 = AttentionGate(F_g=f * 4, F_l=f * 4, F_int=f * 2)
        self.dec3 = ResidualBlock(f * 8, f * 4, dropout)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.att2 = AttentionGate(F_g=f * 2, F_l=f * 2, F_int=f)
        self.dec2 = ResidualBlock(f * 4, f * 2, dropout)

        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.att1 = AttentionGate(F_g=f, F_l=f, F_int=f // 2)
        self.dec1 = ResidualBlock(f * 2, f, dropout)

        # Output
        self.out = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, self.att4(d4, e4)], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, self.att3(d3, e3)], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self.att2(d2, e2)], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self.att1(d1, e1)], dim=1))

        return self.out(d1)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance."""

    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.clamp(1e-7, 1 - 1e-7)

        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)

        return (alpha_t * focal_weight * bce).mean()


class DiceLoss(nn.Module):
    """Dice Loss for overlap."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )


class BoundaryLoss(nn.Module):
    """Boundary-aware loss for sharp edges."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3),
        )

    def get_boundary(self, mask):
        gx = F.conv2d(mask, self.sobel_x, padding=1)
        gy = F.conv2d(mask, self.sobel_y, padding=1)
        boundary = torch.sqrt(gx**2 + gy**2)
        return (boundary > 0.1).float()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_boundary = self.get_boundary(pred)
        target_boundary = self.get_boundary(target)
        # FIX: Use MSE instead of BCE to avoid AMP autocast issues
        # BCE is unsafe with mixed precision; MSE works well for boundary matching
        return F.mse_loss(pred_boundary, target_boundary)


class ConnectivityLoss(nn.Module):
    """
    NEW: Penalize disconnected water predictions.
    Encourages continuous river structures.
    """

    def __init__(self):
        super().__init__()
        # 3x3 connectivity kernel
        self.register_buffer(
            "conn_kernel", torch.ones(1, 1, 3, 3, dtype=torch.float32) / 9.0
        )

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        # Local connectivity: each water pixel should have water neighbors
        pred_neighbors = F.conv2d(pred, self.conn_kernel, padding=1)
        target_neighbors = F.conv2d(target, self.conn_kernel, padding=1)

        # Water pixels should have similar neighborhood density
        water_mask = target > 0.5
        if water_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        pred_density = pred_neighbors[water_mask]
        target_density = target_neighbors[water_mask]

        # MSE on neighborhood density
        return F.mse_loss(pred_density, target_density)


class CombinedLoss(nn.Module):
    """Combined loss with connectivity awareness."""

    def __init__(
        self,
        focal_weight=0.3,
        dice_weight=0.4,
        boundary_weight=0.2,
        connectivity_weight=0.1,
        gamma=2.0,
    ):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=gamma)
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()
        self.connectivity_loss = ConnectivityLoss()

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.connectivity_weight = connectivity_weight

    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        connectivity = self.connectivity_loss(pred, target)

        total = (
            self.focal_weight * focal
            + self.dice_weight * dice
            + self.boundary_weight * boundary
            + self.connectivity_weight * connectivity
        )

        return total, {
            "focal": focal.item(),
            "dice": dice.item(),
            "boundary": boundary.item(),
            "connectivity": connectivity.item(),
        }


# =============================================================================
# DATASET - FULL CHIP (NO CROPPING!)
# =============================================================================


class FullChipDataset(Dataset):
    """
    Dataset for FULL CHIP training.
    NO random cropping - preserves river connectivity!
    """

    def __init__(
        self, chips: List[np.ndarray], target_size: int = 512, augment: bool = False
    ):
        self.chips = chips
        self.target_size = target_size
        self.augment = augment

        # Normalization stats (same as v7)
        self.norm = {
            "vv": {"mean": -15.0, "std": 5.0},
            "vh": {"mean": -22.0, "std": 5.0},
            "dem": {"mean": 200.0, "std": 200.0},
            "slope": {"mean": 5.0, "std": 8.0},
            "hand": {"mean": 10.0, "std": 15.0},
            "twi": {"mean": 10.0, "std": 5.0},
            "mndwi": {"mean": 0.0, "std": 0.5},
            "vh_texture": {"mean": 0.0, "std": 1.0},
        }

    def __len__(self):
        return len(self.chips)

    def compute_vh_texture(self, vh):
        """Compute VH texture (local variance)."""
        vh_mean = uniform_filter(vh, size=5)
        vh_sq_mean = uniform_filter(vh**2, size=5)
        vh_var = np.maximum(vh_sq_mean - vh_mean**2, 0)
        return np.sqrt(vh_var).astype(np.float32)

    def pad_to_target(
        self, data: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pad chip to target size (512x512)."""
        h, w = data.shape[1], data.shape[2]

        if h >= self.target_size and w >= self.target_size:
            # Center crop if larger
            start_h = (h - self.target_size) // 2
            start_w = (w - self.target_size) // 2
            data = data[
                :,
                start_h : start_h + self.target_size,
                start_w : start_w + self.target_size,
            ]
            label = label[
                start_h : start_h + self.target_size,
                start_w : start_w + self.target_size,
            ]
        else:
            # Pad if smaller
            pad_h = max(0, self.target_size - h)
            pad_w = max(0, self.target_size - w)
            data = np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            label = np.pad(label, ((0, pad_h), (0, pad_w)), mode="reflect")
            data = data[:, : self.target_size, : self.target_size]
            label = label[: self.target_size, : self.target_size]

        return data, label

    def __getitem__(self, idx):
        chip = self.chips[idx]
        n_channels = chip.shape[2]

        # Extract channels
        vv = chip[:, :, 0].astype(np.float32)
        vh = chip[:, :, 1].astype(np.float32)
        dem = chip[:, :, 2].astype(np.float32)
        slope = np.clip(chip[:, :, 3].astype(np.float32), 0, 90)
        hand = np.clip(chip[:, :, 4].astype(np.float32), 0, 500)
        twi = np.clip(chip[:, :, 5].astype(np.float32), 0, 30)
        label = (chip[:, :, 6] > 0).astype(np.float32)
        mndwi = (
            np.clip(chip[:, :, 7].astype(np.float32), -1, 1)
            if n_channels > 7
            else np.zeros_like(vv)
        )

        # Compute VH texture
        vh_texture = self.compute_vh_texture(vh)

        # Fix NaN
        vv = np.nan_to_num(vv, nan=-20.0)
        vh = np.nan_to_num(vh, nan=-25.0)
        dem = np.nan_to_num(dem, nan=0.0)
        hand = np.nan_to_num(hand, nan=100.0)
        twi = np.nan_to_num(twi, nan=5.0)
        mndwi = np.nan_to_num(mndwi, nan=0.0)

        # Stack channels: (C, H, W)
        data = np.stack([vv, vh, dem, slope, hand, twi, mndwi, vh_texture], axis=0)

        # Normalize
        keys = ["vv", "vh", "dem", "slope", "hand", "twi", "mndwi", "vh_texture"]
        for i, key in enumerate(keys):
            data[i] = (data[i] - self.norm[key]["mean"]) / self.norm[key]["std"]
        data = np.clip(data, -5, 5)

        # Augmentation (preserves connectivity - only spatial transforms)
        if self.augment:
            # Horizontal flip
            if np.random.random() > 0.5:
                data = np.flip(data, axis=2).copy()
                label = np.flip(label, axis=1).copy()
            # Vertical flip
            if np.random.random() > 0.5:
                data = np.flip(data, axis=1).copy()
                label = np.flip(label, axis=0).copy()
            # 90-degree rotations
            k = np.random.randint(4)
            data = np.rot90(data, k, axes=(1, 2)).copy()
            label = np.rot90(label, k).copy()
            # Small intensity variation (SAR noise simulation)
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.05, size=(2, 1, 1))  # Only VV, VH
                data[:2] = data[:2] + noise.astype(np.float32)

        # Pad to target size (512x512)
        data, label = self.pad_to_target(data, label)

        return torch.from_numpy(data), torch.from_numpy(label[np.newaxis])


# =============================================================================
# TRAINING
# =============================================================================


def compute_iou(pred, target):
    """Compute IoU."""
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    target_bin = (target > 0.5).float()

    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return (intersection / union).item()


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0
    total_iou = 0
    loss_components = {"focal": 0, "dice": 0, "boundary": 0, "connectivity": 0}

    for data, target in loader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                output = model(data)
                loss, components = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss, components = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_iou += compute_iou(output, target)

        for k, v in components.items():
            loss_components[k] += v

    n = len(loader)
    return total_loss / n, total_iou / n, {k: v / n for k, v in loss_components.items()}


def validate(model, loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_iou = 0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss, _ = criterion(output, target)

            total_loss += loss.item()
            total_iou += compute_iou(output, target)

    n = len(loader)
    return total_loss / n, total_iou / n


# =============================================================================
# MAIN
# =============================================================================


def main():
    logger.info("=" * 70)
    logger.info("FULL-CHIP ATTENTION U-NET v8 TRAINING")
    logger.info("Key: NO cropping - full 512x512 for river connectivity!")
    logger.info("=" * 70)

    # Set seeds
    torch.manual_seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])

    device = torch.device(CONFIG["device"])
    logger.info(f"Device: {device}")

    # Create directories
    CONFIG["model_dir"].mkdir(exist_ok=True)
    CONFIG["results_dir"].mkdir(exist_ok=True)

    # Load chips
    logger.info("Loading chips...")
    chip_files = sorted(CONFIG["chip_dir"].glob("*_with_truth.npy"))
    chips = [np.load(f) for f in chip_files]
    logger.info(f"Loaded {len(chips)} chips")

    # Split (80/20)
    n_test = max(1, int(len(chips) * 0.2))
    indices = np.random.permutation(len(chips))
    test_chips = [chips[i] for i in indices[:n_test]]
    train_chips = [chips[i] for i in indices[n_test:]]

    logger.info(f"Train: {len(train_chips)}, Test: {len(test_chips)}")
    logger.info(
        f"Target size: {CONFIG['target_size']}x{CONFIG['target_size']} (FULL CHIP)"
    )

    # Datasets - NO CROPPING!
    train_dataset = FullChipDataset(
        train_chips,
        target_size=CONFIG["target_size"],
        augment=CONFIG["use_augmentation"],
    )
    test_dataset = FullChipDataset(
        test_chips, target_size=CONFIG["target_size"], augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    model = AttentionUNet(
        in_channels=CONFIG["in_channels"],
        base_filters=CONFIG["base_filters"],
        dropout=CONFIG["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # Loss and optimizer
    criterion = CombinedLoss(
        focal_weight=CONFIG["focal_weight"],
        dice_weight=CONFIG["dice_weight"],
        boundary_weight=CONFIG["boundary_weight"],
        connectivity_weight=CONFIG["connectivity_weight"],
        gamma=CONFIG["focal_gamma"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # Mixed precision scaler
    scaler = GradScaler() if CONFIG["use_amp"] else None

    # Training loop
    best_iou = 0
    patience_counter = 0
    history = []

    logger.info("Starting training...")
    start_time = time.time()

    for epoch in range(CONFIG["num_epochs"]):
        train_loss, train_iou, loss_components = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_iou = validate(model, test_loader, criterion, device)

        scheduler.step()

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_iou": train_iou,
                "val_loss": val_loss,
                "val_iou": val_iou,
                **{f"train_{k}": v for k, v in loss_components.items()},
            }
        )

        logger.info(
            f"Epoch {epoch + 1}/{CONFIG['num_epochs']}: "
            f"Train Loss={train_loss:.4f}, Train IoU={train_iou:.4f}, "
            f"Val Loss={val_loss:.4f}, Val IoU={val_iou:.4f}"
        )

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0

            model_path = CONFIG["model_dir"] / "attention_unet_v8_fullchip_best.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_iou": best_iou,
                    "config": {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in CONFIG.items()
                    },
                },
                model_path,
            )
            logger.info(f"  Saved best model: IoU={best_iou:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    train_time = time.time() - start_time

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "train_time_seconds": train_time,
        "best_iou": best_iou,
        "final_epoch": epoch + 1,
        "key_difference": "FULL CHIP training (512x512) - no random cropping!",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()},
        "history": history,
    }

    results_path = CONFIG["results_dir"] / "attention_unet_v8_fullchip_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best IoU: {best_iou:.4f}")
    logger.info(f"Training time: {train_time / 60:.1f} minutes")
    logger.info(
        f"Model saved to: {CONFIG['model_dir'] / 'attention_unet_v8_fullchip_best.pth'}"
    )
    logger.info(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
