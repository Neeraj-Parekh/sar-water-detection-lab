#!/usr/bin/env python3
"""
================================================================================
U-NET v6 - FIXED VERSION WITH BETTER GENERALIZATION
================================================================================
Fixes for overfitting:
1. Consistent data split using same method as evaluation
2. Stronger regularization (higher dropout, weight decay)
3. More aggressive data augmentation
4. Global normalization instead of per-image
5. Smaller model capacity
6. Early stopping with patience

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
from scipy.ndimage import uniform_filter, sobel, zoom, gaussian_filter
from scipy.ndimage import rotate as nd_rotate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    "batch_size": 8,  # Larger batch for stability
    "image_size": 256,
    "num_epochs": 150,
    "learning_rate": 3e-4,  # Higher initial LR
    "weight_decay": 1e-3,  # MUCH stronger weight decay
    "dropout_rate": 0.4,  # Higher dropout
    "patience": 20,  # Early stopping patience
    # Global normalization stats (computed from training data)
    "norm_stats": {
        "vv": {"mean": -15.0, "std": 5.0},
        "vh": {"mean": -22.0, "std": 5.0},
        "dem": {"mean": 200.0, "std": 200.0},
        "hand": {"mean": 10.0, "std": 15.0},
        "slope": {"mean": 5.0, "std": 8.0},
        "twi": {"mean": 10.0, "std": 5.0},
    },
}

CONFIG["model_dir"].mkdir(exist_ok=True)
CONFIG["results_dir"].mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(CONFIG["results_dir"] / "unet_v6_training.log"),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATASET WITH STRONGER AUGMENTATION
# =============================================================================


class SARDatasetV6(Dataset):
    def __init__(
        self, chips: List[np.ndarray], image_size: int = 256, augment: bool = False
    ):
        self.chips = chips
        self.image_size = image_size
        self.augment = augment
        self.norm = CONFIG["norm_stats"]

    def __len__(self):
        return len(self.chips)

    def _normalize(self, data):
        """Global normalization using fixed stats."""
        keys = ["vv", "vh", "dem", "hand", "slope", "twi"]
        for i, key in enumerate(keys):
            data[i] = (data[i] - self.norm[key]["mean"]) / self.norm[key]["std"]
        return np.clip(data, -5, 5)

    def _augment(self, data, mask):
        """Strong augmentation."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            data = np.flip(data, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()

        # Random vertical flip
        if np.random.random() > 0.5:
            data = np.flip(data, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()

        # Random 90-degree rotation
        k = np.random.randint(4)
        data = np.rot90(data, k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k).copy()

        # Random noise on SAR channels
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.1, data[:2].shape).astype(np.float32)
            data[:2] = data[:2] + noise

        # Random brightness/contrast on SAR
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-0.2, 0.2)
            data[:2] = alpha * data[:2] + beta

        # Gaussian blur occasionally
        if np.random.random() > 0.7:
            sigma = np.random.uniform(0.5, 1.5)
            for c in range(data.shape[0]):
                data[c] = gaussian_filter(data[c], sigma=sigma)

        return data, mask

    def __getitem__(self, idx):
        chip = self.chips[idx]

        # Extract bands
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
            hand = zoom(hand, (self.image_size / h, self.image_size / w), order=1)
            slope = zoom(slope, (self.image_size / h, self.image_size / w), order=1)

        # Handle NaN before normalization
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.nan_to_num(mask, nan=0.0)
        hand = np.nan_to_num(hand, nan=0.0)
        slope = np.nan_to_num(slope, nan=0.0)

        # Global normalization
        data = self._normalize(data)

        # Augmentation
        if self.augment:
            data, mask = self._augment(data, mask)

        # Compute edge map
        edge_x = sobel(mask, axis=0)
        edge_y = sobel(mask, axis=1)
        edge_map = np.sqrt(edge_x**2 + edge_y**2)
        edge_map = (edge_map > 0.1).astype(np.float32)

        return {
            "input": torch.from_numpy(data.copy()).float(),
            "mask": torch.from_numpy(mask.copy()).float().unsqueeze(0),
            "edge": torch.from_numpy(edge_map.copy()).float().unsqueeze(0),
            "hand": torch.from_numpy(hand.copy()).float().unsqueeze(0),
            "slope": torch.from_numpy(slope.copy()).float().unsqueeze(0),
        }


# =============================================================================
# SMALLER MODEL WITH STRONGER REGULARIZATION
# =============================================================================


class ConvBlockV6(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNetV6(nn.Module):
    """
    Smaller U-Net with stronger regularization.
    Reduced channel counts to prevent overfitting.
    """

    def __init__(self, in_channels: int = 6, dropout: float = 0.4):
        super().__init__()

        # Smaller channel counts
        ch = [24, 48, 96, 192, 384]  # Reduced from [32, 64, 128, 256, 512]

        # Encoder
        self.enc1 = ConvBlockV6(in_channels, ch[0], dropout=0)
        self.enc2 = ConvBlockV6(ch[0], ch[1], dropout=dropout * 0.5)
        self.enc3 = ConvBlockV6(ch[1], ch[2], dropout=dropout * 0.75)
        self.enc4 = ConvBlockV6(ch[2], ch[3], dropout=dropout)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlockV6(ch[3], ch[4], dropout=dropout)

        # Decoder
        self.up4 = nn.ConvTranspose2d(ch[4], ch[3], 2, stride=2)
        self.dec4 = ConvBlockV6(ch[3] * 2, ch[3], dropout=dropout)
        self.up3 = nn.ConvTranspose2d(ch[3], ch[2], 2, stride=2)
        self.dec3 = ConvBlockV6(ch[2] * 2, ch[2], dropout=dropout * 0.75)
        self.up2 = nn.ConvTranspose2d(ch[2], ch[1], 2, stride=2)
        self.dec2 = ConvBlockV6(ch[1] * 2, ch[1], dropout=dropout * 0.5)
        self.up1 = nn.ConvTranspose2d(ch[1], ch[0], 2, stride=2)
        self.dec1 = ConvBlockV6(ch[0] * 2, ch[0], dropout=0)

        # Output heads
        self.seg_head = nn.Conv2d(ch[0], 1, 1)
        self.edge_head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        seg_logits = self.seg_head(d1)
        edge_logits = self.edge_head(d1)

        return seg_logits, edge_logits


# =============================================================================
# SIMPLIFIED LOSS (removing physics loss that may hurt generalization)
# =============================================================================


class DiceBCELoss(nn.Module):
    """Simple Dice + BCE loss - often works better than complex losses."""

    def __init__(self, dice_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = 1 - dice_weight

    def forward(self, pred, target):
        # BCE
        bce = F.binary_cross_entropy_with_logits(pred, target)

        # Dice
        pred_prob = torch.sigmoid(pred)
        intersection = (pred_prob * target).sum()
        dice = 1 - (2 * intersection + 1) / (pred_prob.sum() + target.sum() + 1)

        return self.dice_weight * dice + self.bce_weight * bce


class CombinedLossV6(nn.Module):
    """Simplified combined loss."""

    def __init__(self, edge_weight: float = 0.2):
        super().__init__()
        self.seg_loss = DiceBCELoss(dice_weight=0.5)
        self.edge_weight = edge_weight

    def forward(self, seg_pred, edge_pred, target, edge_map, hand, slope):
        seg = self.seg_loss(seg_pred, target)
        edge = F.binary_cross_entropy_with_logits(edge_pred, edge_map)
        total = seg + self.edge_weight * edge
        return total, {"seg": seg.item(), "edge": edge.item()}


# =============================================================================
# TRAINING WITH EARLY STOPPING
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
        loss, _ = criterion(seg_pred, edge_pred, masks, edges, hand, slope)

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
    logger.info("U-NET V6 TRAINING - FIXED FOR GENERALIZATION")
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

    # CONSISTENT SPLIT - using same method as evaluation
    np.random.seed(CONFIG["random_seed"])
    n = len(chips)
    idx = np.random.permutation(n)
    train_idx = idx[: int(0.70 * n)]
    val_idx = idx[int(0.70 * n) : int(0.85 * n)]
    test_idx = idx[int(0.85 * n) :]

    train_chips = [chips[i] for i in train_idx]
    val_chips = [chips[i] for i in val_idx]
    test_chips = [chips[i] for i in test_idx]

    logger.info(
        f"Train: {len(train_chips)}, Val: {len(val_chips)}, Test: {len(test_chips)}"
    )

    # Datasets
    train_dataset = SARDatasetV6(train_chips, CONFIG["image_size"], augment=True)
    val_dataset = SARDatasetV6(val_chips, CONFIG["image_size"], augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4
    )

    # Model - smaller with more dropout
    model = UNetV6(in_channels=6, dropout=CONFIG["dropout_rate"]).to(device)
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {params:,}")

    # Loss and optimizer
    criterion = CombinedLossV6(edge_weight=0.2)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)

    # Training with early stopping
    best_val_iou = 0
    patience_counter = 0
    results = {"train_loss": [], "train_iou": [], "val_loss": [], "val_iou": []}

    logger.info("\nStarting training...")

    for epoch in range(CONFIG["num_epochs"]):
        train_loss, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_iou = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_iou)

        results["train_loss"].append(train_loss)
        results["train_iou"].append(train_iou)
        results["val_loss"].append(val_loss)
        results["val_iou"].append(val_iou)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_iou": val_iou,
                },
                CONFIG["model_dir"] / "unet_v6_best.pth",
            )
            logger.info(
                f"Epoch {epoch + 1:3d}: Train Loss={train_loss:.4f}, IoU={train_iou:.4f} | "
                f"Val Loss={val_loss:.4f}, IoU={val_iou:.4f} [BEST]"
            )
        else:
            patience_counter += 1
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1:3d}: Train Loss={train_loss:.4f}, IoU={train_iou:.4f} | "
                    f"Val Loss={val_loss:.4f}, IoU={val_iou:.4f}"
                )

        # Early stopping
        if patience_counter >= CONFIG["patience"]:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    checkpoint = torch.load(CONFIG["model_dir"] / "unet_v6_best.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dataset = SARDatasetV6(test_chips, CONFIG["image_size"], augment=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    test_ious = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input"].to(device)
            masks = batch["mask"].to(device)
            seg_pred, _ = model(inputs)
            iou = compute_iou(seg_pred, masks)
            test_ious.append(iou)

    test_mean_iou = np.mean(test_ious)
    test_std_iou = np.std(test_ious)

    logger.info(f"Test IoU: {test_mean_iou:.4f} +/- {test_std_iou:.4f}")

    # Save results
    results["best_val_iou"] = best_val_iou
    results["test_mean_iou"] = test_mean_iou
    results["test_std_iou"] = test_std_iou
    results["parameters"] = params

    with open(CONFIG["results_dir"] / "unet_v6_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nBest Validation IoU: {best_val_iou:.4f}")
    logger.info(f"Test IoU: {test_mean_iou:.4f}")
    logger.info(f"Model saved to: {CONFIG['model_dir'] / 'unet_v6_best.pth'}")
    logger.info(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
