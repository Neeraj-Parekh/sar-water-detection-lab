#!/usr/bin/env python3
"""
================================================================================
ATTENTION U-NET v7 - Modern Architecture with Advanced Loss Functions
================================================================================

Upgrades from previous versions:
1. Attention Gates in skip connections (focus on water boundaries)
2. Residual blocks in encoder/decoder
3. Focal Loss + Dice Loss + Boundary Loss
4. MNDWI + Slope as additional input channels (8 channels total)
5. Deep supervision

Input Channels:
- VV, VH (SAR)
- DEM, SLOPE, HAND, TWI (Terrain)
- MNDWI (Optical, if available)
- VH_texture (computed)

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
from scipy.ndimage import uniform_filter, gaussian_filter, sobel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

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
    # Model
    "in_channels": 8,  # VV, VH, DEM, SLOPE, HAND, TWI, MNDWI, VH_texture
    "base_filters": 32,
    "dropout": 0.3,
    # Training
    "batch_size": 8,
    "image_size": 256,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "patience": 15,
    # Loss weights
    "focal_weight": 0.4,
    "dice_weight": 0.4,
    "boundary_weight": 0.2,
    "focal_gamma": 2.0,  # Focal loss gamma
}


# =============================================================================
# ATTENTION GATE
# =============================================================================


class AttentionGate(nn.Module):
    """
    Attention Gate for skip connections.
    Helps the model focus on relevant regions (water boundaries).
    """

    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Number of feature-maps from gating signal (from decoder)
            F_l: Number of feature-maps from skip connection (from encoder)
            F_int: Number of intermediate feature-maps
        """
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
        """
        Args:
            g: Gating signal from decoder (upsampled)
            x: Skip connection from encoder
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# =============================================================================
# RESIDUAL BLOCK
# =============================================================================


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    Helps with gradient flow in deep networks.
    """

    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)

        # Skip connection
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.relu(out)

        return out


# =============================================================================
# ATTENTION U-NET
# =============================================================================


class AttentionUNet(nn.Module):
    """
    U-Net with Attention Gates and Residual Blocks.
    """

    def __init__(self, in_channels=8, out_channels=1, base_filters=32, dropout=0.3):
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

        # Decoder with Attention Gates
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

        # Deep supervision heads
        self.ds4 = nn.Conv2d(f * 8, out_channels, 1)
        self.ds3 = nn.Conv2d(f * 4, out_channels, 1)
        self.ds2 = nn.Conv2d(f * 2, out_channels, 1)

    def forward(self, x, deep_supervision=False):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with attention
        d4 = self.up4(b)
        e4_att = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4_att], dim=1))

        d3 = self.up3(d4)
        e3_att = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3_att], dim=1))

        d2 = self.up2(d3)
        e2_att = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2_att], dim=1))

        d1 = self.up1(d2)
        e1_att = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1_att], dim=1))

        out = self.out(d1)

        if deep_supervision:
            # Return intermediate outputs for deep supervision
            ds4 = F.interpolate(
                self.ds4(d4), size=x.shape[2:], mode="bilinear", align_corners=False
            )
            ds3 = F.interpolate(
                self.ds3(d3), size=x.shape[2:], mode="bilinear", align_corners=False
            )
            ds2 = F.interpolate(
                self.ds2(d2), size=x.shape[2:], mode="bilinear", align_corners=False
            )
            return out, ds4, ds3, ds2

        return out


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses training on hard examples.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.clamp(1e-7, 1 - 1e-7)

        # Binary cross entropy
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)

        # Focal weight
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma

        # Alpha weighting
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)

        loss = alpha_t * focal_weight * bce
        return loss.mean()


class DiceLoss(nn.Module):
    """
    Dice Loss for better overlap optimization.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        return 1 - dice


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss for sharp edges.
    Penalizes incorrect boundary predictions more heavily.
    """

    def __init__(self):
        super().__init__()
        # Sobel filters for edge detection
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
        """Extract boundary from mask using Sobel filters."""
        gx = F.conv2d(mask, self.sobel_x, padding=1)
        gy = F.conv2d(mask, self.sobel_y, padding=1)
        boundary = torch.sqrt(gx**2 + gy**2)
        boundary = (boundary > 0.1).float()
        return boundary

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        # Get boundaries
        pred_boundary = self.get_boundary(pred)
        target_boundary = self.get_boundary(target)

        # Boundary BCE loss
        bce = F.binary_cross_entropy(
            pred_boundary.clamp(1e-7, 1 - 1e-7), target_boundary, reduction="mean"
        )

        return bce


class CombinedLoss(nn.Module):
    """
    Combined loss: Focal + Dice + Boundary
    """

    def __init__(
        self, focal_weight=0.4, dice_weight=0.4, boundary_weight=0.2, gamma=2.0
    ):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=gamma)
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        boundary = self.boundary_loss(pred, target)

        total = (
            self.focal_weight * focal
            + self.dice_weight * dice
            + self.boundary_weight * boundary
        )

        return total, {
            "focal": focal.item(),
            "dice": dice.item(),
            "boundary": boundary.item(),
        }


# =============================================================================
# DATASET
# =============================================================================


class SARDatasetV7(Dataset):
    """Dataset with 8 channels including MNDWI and VH texture."""

    def __init__(
        self, chips: List[np.ndarray], image_size: int = 256, augment: bool = False
    ):
        self.chips = chips
        self.image_size = image_size
        self.augment = augment

        # Normalization stats
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
        vh_np = vh if isinstance(vh, np.ndarray) else vh.numpy()
        vh_mean = uniform_filter(vh_np, size=5)
        vh_sq_mean = uniform_filter(vh_np**2, size=5)
        vh_var = np.maximum(vh_sq_mean - vh_mean**2, 0)
        return np.sqrt(vh_var).astype(np.float32)

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

        # Stack channels
        data = np.stack([vv, vh, dem, slope, hand, twi, mndwi, vh_texture], axis=0)

        # Normalize
        keys = ["vv", "vh", "dem", "slope", "hand", "twi", "mndwi", "vh_texture"]
        for i, key in enumerate(keys):
            data[i] = (data[i] - self.norm[key]["mean"]) / self.norm[key]["std"]
        data = np.clip(data, -5, 5)

        # Augmentation
        if self.augment:
            if np.random.random() > 0.5:
                data = np.flip(data, axis=2).copy()
                label = np.flip(label, axis=1).copy()
            if np.random.random() > 0.5:
                data = np.flip(data, axis=1).copy()
                label = np.flip(label, axis=0).copy()
            k = np.random.randint(4)
            data = np.rot90(data, k, axes=(1, 2)).copy()
            label = np.rot90(label, k).copy()

        # Random crop to image_size
        h, w = data.shape[1], data.shape[2]
        if h > self.image_size and w > self.image_size:
            i = np.random.randint(0, h - self.image_size)
            j = np.random.randint(0, w - self.image_size)
            data = data[:, i : i + self.image_size, j : j + self.image_size]
            label = label[i : i + self.image_size, j : j + self.image_size]
        else:
            # Pad if needed
            pad_h = max(0, self.image_size - h)
            pad_w = max(0, self.image_size - w)
            if pad_h > 0 or pad_w > 0:
                data = np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
                label = np.pad(label, ((0, pad_h), (0, pad_w)), mode="reflect")
                data = data[:, : self.image_size, : self.image_size]
                label = label[: self.image_size, : self.image_size]

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


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_iou = 0
    loss_components = {"focal": 0, "dice": 0, "boundary": 0}

    for data, target in loader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss, components = criterion(output, target)
        loss.backward()

        # Gradient clipping
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


def main():
    """Main training function."""
    logger.info("=" * 70)
    logger.info("ATTENTION U-NET v7 TRAINING")
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

    # Split
    n_test = max(1, int(len(chips) * 0.2))
    indices = np.random.permutation(len(chips))
    test_chips = [chips[i] for i in indices[:n_test]]
    train_chips = [chips[i] for i in indices[n_test:]]

    logger.info(f"Train: {len(train_chips)}, Test: {len(test_chips)}")

    # Datasets
    train_dataset = SARDatasetV7(train_chips, CONFIG["image_size"], augment=True)
    test_dataset = SARDatasetV7(test_chips, CONFIG["image_size"], augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4
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
        gamma=CONFIG["focal_gamma"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Training loop
    best_iou = 0
    patience_counter = 0
    history = []

    logger.info("Starting training...")
    start_time = time.time()

    for epoch in range(CONFIG["num_epochs"]):
        train_loss, train_iou, loss_components = train_epoch(
            model, train_loader, criterion, optimizer, device
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

            model_path = CONFIG["model_dir"] / "attention_unet_v7_best.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_iou": best_iou,
                    "config": CONFIG,
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
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()},
        "history": history,
    }

    results_path = CONFIG["results_dir"] / "attention_unet_v7_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best IoU: {best_iou:.4f}")
    logger.info(f"Training time: {train_time / 60:.1f} minutes")
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
