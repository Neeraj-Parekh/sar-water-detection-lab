#!/usr/bin/env python3
"""
================================================================================
IMPROVED U-NET v5 WITH EDGE-AWARE LOSS AND PHYSICS GUIDANCE
================================================================================
Based on research findings, this version includes:
1. Edge-aware loss for boundary refinement
2. Dice + Focal + Edge combined loss
3. Physics-guided constraints (HAND, slope)
4. Better data augmentation

Author: SAR Water Detection Project
Date: January 2026
================================================================================
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
import rasterio
from scipy.ndimage import uniform_filter, sobel, zoom
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "random_seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "chip_dirs": [
        Path("/home/mit-aoe/sar_water_detection/chips"),
        Path("/home/mit-aoe/sar_water_detection/chips_expanded"),
    ],
    "model_dir": Path("/home/mit-aoe/sar_water_detection/models"),
    "results_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    "batch_size": 4,
    "image_size": 256,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "physics_weight": 0.1,
    "edge_weight": 0.3,
}

CONFIG["model_dir"].mkdir(exist_ok=True)
CONFIG["results_dir"].mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(CONFIG["results_dir"] / "unet_v5_training.log"),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATASET
# =============================================================================


class SARDataset(Dataset):
    def __init__(
        self, chips: List[np.ndarray], image_size: int = 256, augment: bool = False
    ):
        self.chips = chips
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, idx):
        chip = self.chips[idx]

        # Extract bands: VV, VH, DEM, HAND, SLOPE, TWI
        vv = chip[0]
        vh = chip[1]
        dem = chip[3]
        hand = chip[4]
        slope = chip[5]
        twi = chip[6]
        mask = chip[7]

        # Stack input
        data = np.stack([vv, vh, dem, hand, slope, twi], axis=0)

        # Resize
        h, w = mask.shape
        if h != self.image_size or w != self.image_size:
            data = zoom(data, (1, self.image_size / h, self.image_size / w), order=1)
            mask = zoom(mask, (self.image_size / h, self.image_size / w), order=0)

        # Normalize
        for c in range(data.shape[0]):
            mean = np.nanmean(data[c])
            std = np.nanstd(data[c]) + 1e-8
            data[c] = np.clip((data[c] - mean) / std, -10, 10)

        # Handle NaN
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.nan_to_num(mask, nan=0.0)

        # Augmentation
        if self.augment:
            # Random flip
            if np.random.random() > 0.5:
                data = np.flip(data, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.random() > 0.5:
                data = np.flip(data, axis=1).copy()
                mask = np.flip(mask, axis=0).copy()
            # Random rotation (90 degrees)
            k = np.random.randint(4)
            data = np.rot90(data, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k).copy()

        # Compute edge map for edge-aware loss
        edge_x = sobel(mask, axis=0)
        edge_y = sobel(mask, axis=1)
        edge_map = np.sqrt(edge_x**2 + edge_y**2)
        edge_map = (edge_map > 0.1).astype(np.float32)

        # Get HAND and slope for physics loss
        hand_norm = (
            hand
            if h == self.image_size
            else zoom(hand, (self.image_size / h, self.image_size / w), order=1)
        )
        slope_norm = (
            slope
            if h == self.image_size
            else zoom(slope, (self.image_size / h, self.image_size / w), order=1)
        )

        return {
            "input": torch.from_numpy(data).float(),
            "mask": torch.from_numpy(mask).float().unsqueeze(0),
            "edge": torch.from_numpy(edge_map).float().unsqueeze(0),
            "hand": torch.from_numpy(hand_norm).float().unsqueeze(0),
            "slope": torch.from_numpy(slope_norm).float().unsqueeze(0),
        }


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        return x * self.sigmoid(avg_out + max_out).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channel_attn = ChannelAttention(channels)
        self.spatial_attn = SpatialAttention()

    def forward(self, x):
        return self.spatial_attn(self.channel_attn(x))


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_cbam: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_ch) if use_cbam else nn.Identity()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return self.cbam(x)


class UNetV5(nn.Module):
    """
    Improved U-Net with:
    - CBAM attention
    - Edge detection head
    - Deeper architecture
    """

    def __init__(self, in_channels: int = 6):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(0.2)

        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        # Output heads
        self.seg_head = nn.Conv2d(32, 1, 1)  # Segmentation
        self.edge_head = nn.Conv2d(32, 1, 1)  # Edge detection

    def forward(self, x):
        # Handle NaN/Inf
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.drop(self.pool(e2)))
        e4 = self.enc4(self.drop(self.pool(e3)))

        # Bottleneck
        b = self.bottleneck(self.drop(self.pool(e4)))

        # Decoder
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Outputs
        seg_logits = self.seg_head(d1)
        edge_logits = self.edge_head(d1)

        return seg_logits, edge_logits


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


class EdgeAwareLoss(nn.Module):
    """Edge-aware loss that penalizes errors near boundaries more."""

    def __init__(self, boundary_weight: float = 2.0):
        super().__init__()
        self.boundary_weight = boundary_weight

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, edge_map: torch.Tensor
    ) -> torch.Tensor:
        # Standard BCE
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        # Weight by edge proximity
        weights = 1 + self.boundary_weight * edge_map
        weighted_bce = (bce * weights).mean()

        return weighted_bce


class PhysicsGuidedLoss(nn.Module):
    """Physics-guided loss using HAND and slope constraints."""

    def __init__(self, hand_threshold: float = 10.0, slope_threshold: float = 15.0):
        super().__init__()
        self.hand_threshold = hand_threshold
        self.slope_threshold = slope_threshold

    def forward(
        self, pred: torch.Tensor, hand: torch.Tensor, slope: torch.Tensor
    ) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)

        # Penalize water predictions at high HAND values
        high_hand_mask = (hand > self.hand_threshold).float()
        hand_loss = (pred_prob * high_hand_mask).mean()

        # Penalize water predictions on steep slopes
        steep_mask = (slope > self.slope_threshold).float()
        slope_loss = (pred_prob * steep_mask).mean()

        return hand_loss + slope_loss


class CombinedLoss(nn.Module):
    """Combined loss for water segmentation."""

    def __init__(self, edge_weight: float = 0.3, physics_weight: float = 0.1):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.edge_aware_loss = EdgeAwareLoss()
        self.physics_loss = PhysicsGuidedLoss()
        self.edge_weight = edge_weight
        self.physics_weight = physics_weight

    def forward(self, seg_pred, edge_pred, target, edge_map, hand, slope):
        # Segmentation losses
        dice = self.dice_loss(seg_pred, target)
        focal = self.focal_loss(seg_pred, target)
        edge_aware = self.edge_aware_loss(seg_pred, target, edge_map)

        # Edge detection loss
        edge_loss = F.binary_cross_entropy_with_logits(edge_pred, edge_map)

        # Physics loss
        physics = self.physics_loss(seg_pred, hand, slope)

        # Combined
        total = (
            0.5 * dice
            + 0.3 * focal
            + 0.2 * edge_aware
            + self.edge_weight * edge_loss
            + self.physics_weight * physics
        )

        return total, {
            "dice": dice.item(),
            "focal": focal.item(),
            "edge_aware": edge_aware.item(),
            "edge": edge_loss.item(),
            "physics": physics.item(),
        }


# =============================================================================
# TRAINING
# =============================================================================


def compute_iou(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> float:
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = (target > threshold).float()

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection

    return (intersection / (union + 1e-8)).item()


def load_chips() -> List[np.ndarray]:
    chips = []
    for chip_dir in CONFIG["chip_dirs"]:
        if not chip_dir.exists():
            continue
        for f in chip_dir.glob("*.npy"):
            try:
                chip = np.load(f).astype(np.float32)
                if chip.ndim == 3 and chip.shape[0] != 8:
                    chip = chip.transpose(2, 0, 1)
                if chip.shape[0] >= 8 and np.nansum(chip[7]) > 0:
                    chips.append(chip)
            except:
                pass
        for f in chip_dir.glob("*.tif"):
            try:
                with rasterio.open(f) as src:
                    chip = src.read().astype(np.float32)
                if chip.shape[0] >= 8 and np.nansum(chip[7]) > 0:
                    chips.append(chip)
            except:
                pass
    return chips


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_iou = 0

    for batch in loader:
        inputs = batch["input"].to(device)
        masks = batch["mask"].to(device)
        edges = batch["edge"].to(device)
        hand = batch["hand"].to(device)
        slope = batch["slope"].to(device)

        optimizer.zero_grad()

        seg_pred, edge_pred = model(inputs)
        loss, loss_dict = criterion(seg_pred, edge_pred, masks, edges, hand, slope)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_iou += compute_iou(seg_pred, masks)

    return total_loss / len(loader), total_iou / len(loader)


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_iou = 0

    with torch.no_grad():
        for batch in loader:
            inputs = batch["input"].to(device)
            masks = batch["mask"].to(device)
            edges = batch["edge"].to(device)
            hand = batch["hand"].to(device)
            slope = batch["slope"].to(device)

            seg_pred, edge_pred = model(inputs)
            loss, _ = criterion(seg_pred, edge_pred, masks, edges, hand, slope)

            total_loss += loss.item()
            total_iou += compute_iou(seg_pred, masks)

    return total_loss / len(loader), total_iou / len(loader)


def main():
    logger.info("=" * 80)
    logger.info("U-NET V5 TRAINING WITH EDGE-AWARE LOSS")
    logger.info("=" * 80)
    logger.info(f"Device: {CONFIG['device']}")
    logger.info(f"Started: {datetime.now().isoformat()}")

    torch.manual_seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])

    device = torch.device(CONFIG["device"])

    # Load data
    logger.info("\nLoading chips...")
    chips = load_chips()
    logger.info(f"Loaded {len(chips)} chips")

    # Split
    train_chips, temp_chips = train_test_split(
        chips, test_size=0.30, random_state=CONFIG["random_seed"]
    )
    val_chips, test_chips = train_test_split(
        temp_chips, test_size=0.50, random_state=CONFIG["random_seed"]
    )

    logger.info(
        f"Train: {len(train_chips)}, Val: {len(val_chips)}, Test: {len(test_chips)}"
    )

    # Datasets
    train_dataset = SARDataset(train_chips, CONFIG["image_size"], augment=True)
    val_dataset = SARDataset(val_chips, CONFIG["image_size"], augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4
    )

    # Model
    model = UNetV5(in_channels=6).to(device)
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {params:,}")

    # Loss and optimizer
    criterion = CombinedLoss(
        edge_weight=CONFIG["edge_weight"], physics_weight=CONFIG["physics_weight"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Training
    best_val_iou = 0
    results = {"train_loss": [], "train_iou": [], "val_loss": [], "val_iou": []}

    logger.info("\nStarting training...")

    for epoch in range(CONFIG["num_epochs"]):
        train_loss, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_iou = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        results["train_loss"].append(train_loss)
        results["train_iou"].append(train_iou)
        results["val_loss"].append(val_loss)
        results["val_iou"].append(val_iou)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_iou": val_iou,
                },
                CONFIG["model_dir"] / "unet_v5_best.pth",
            )
            logger.info(
                f"Epoch {epoch + 1:3d}: Train Loss={train_loss:.4f}, IoU={train_iou:.4f} | "
                f"Val Loss={val_loss:.4f}, IoU={val_iou:.4f} [BEST]"
            )
        elif (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1:3d}: Train Loss={train_loss:.4f}, IoU={train_iou:.4f} | "
                f"Val Loss={val_loss:.4f}, IoU={val_iou:.4f}"
            )

    # Save results
    results["best_val_iou"] = best_val_iou
    results["parameters"] = params

    with open(CONFIG["results_dir"] / "unet_v5_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nBest Validation IoU: {best_val_iou:.4f}")
    logger.info(f"Model saved to: {CONFIG['model_dir'] / 'unet_v5_best.pth'}")
    logger.info(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
