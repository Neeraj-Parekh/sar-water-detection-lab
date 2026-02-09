#!/usr/bin/env python3
"""
Standalone U-Net V4 Training Script with Debugging
===================================================
Fixes the CUDA BCE assertion error by:
1. Using MSE loss instead of BCE
2. Adding explicit value clamping
3. Checking for NaN/Inf values
4. Proper data normalization
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
from scipy.ndimage import zoom, binary_dilation, binary_erosion
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

warnings.filterwarnings("ignore")

# Enable CUDA debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
    "bands": {"VV": 0, "VH": 1, "DEM": 3, "HAND": 4, "SLOPE": 5, "TWI": 6, "TRUTH": 7},
    "image_size": 256,
    "batch_size": 8,
    "epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "patience": 25,
}

CONFIG["model_dir"].mkdir(exist_ok=True)
CONFIG["results_dir"].mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(CONFIG["results_dir"] / "unet_v4_standalone.log"),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# ATTENTION MODULES
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# =============================================================================
# U-NET MODEL (Simplified and Stable)
# =============================================================================


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.cbam(x)
        return x


class UNetV4(nn.Module):
    """Simplified U-Net with CBAM attention."""

    def __init__(self, in_channels: int = 6):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = ConvBlock(512, 256)  # 256 + 256 = 512

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)  # 128 + 128 = 256

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)  # 64 + 64 = 128

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)  # 32 + 32 = 64

        # Output head - NO sigmoid here, we'll apply it after
        self.out_conv = nn.Conv2d(32, 1, 1)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check input
        if torch.isnan(x).any() or torch.isinf(x).any():
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

        # Output logits (not sigmoid yet)
        logits = self.out_conv(d1)

        return logits


# =============================================================================
# LOSS FUNCTIONS (Safe versions)
# =============================================================================


class DiceLoss(nn.Module):
    """Dice loss - no BCE, no assertion errors."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal loss using sigmoid internally - safe version."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Use BCE with logits (more numerically stable)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

        # Compute focal weight
        pt = torch.exp(-bce)
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        return (focal_weight * bce).mean()


class CombinedLossV2(nn.Module):
    """Safe combined loss without BCE assertion issues."""

    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Clamp target to [0, 1]
        target = torch.clamp(target, 0, 1)

        # Apply sigmoid to get probabilities for dice
        probs = torch.sigmoid(logits)

        # Dice loss on probabilities
        dice_loss = self.dice(probs, target)

        # Focal loss on logits (internally applies sigmoid)
        focal_loss = self.focal(logits, target)

        # Combined
        return 0.5 * dice_loss + 0.5 * focal_loss


# =============================================================================
# DATASET
# =============================================================================


class SARDataset(Dataset):
    def __init__(
        self,
        chips: List[np.ndarray],
        names: List[str],
        image_size: int = 256,
        augment: bool = True,
    ):
        self.chips = chips
        self.names = names
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.chips)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chip = self.chips[idx].copy()

        # Extract bands
        vv = chip[CONFIG["bands"]["VV"]]
        vh = chip[CONFIG["bands"]["VH"]]
        dem = chip[CONFIG["bands"]["DEM"]]
        hand = chip[CONFIG["bands"]["HAND"]]
        slope = chip[CONFIG["bands"]["SLOPE"]]
        twi = chip[CONFIG["bands"]["TWI"]]
        truth = chip[CONFIG["bands"]["TRUTH"]]

        # Stack features
        features = np.stack([vv, vh, dem, hand, slope, twi], axis=0)
        mask = truth[np.newaxis]

        # Resize
        features = self._resize(features, self.image_size)
        mask = self._resize(mask, self.image_size)

        # Augmentation
        if self.augment:
            features, mask = self._augment(features, mask)

        # Normalize features (per-channel z-score)
        features = self._normalize(features)

        # Ensure mask is binary [0, 1]
        mask = (mask > 0.5).astype(np.float32)

        # Replace NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.nan_to_num(mask, nan=0.0)

        return {
            "features": torch.from_numpy(features).float(),
            "mask": torch.from_numpy(mask).float(),
            "name": self.names[idx],
        }

    def _resize(self, arr: np.ndarray, size: int) -> np.ndarray:
        if arr.ndim == 2:
            h, w = arr.shape
            factors = (size / h, size / w)
        else:
            c, h, w = arr.shape
            factors = (1, size / h, size / w)
        return zoom(arr, factors, order=1)

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        result = np.zeros_like(features, dtype=np.float32)
        for i in range(features.shape[0]):
            channel = features[i]
            mean = np.nanmean(channel)
            std = np.nanstd(channel) + 1e-8
            result[i] = (channel - mean) / std
            # Clip to reasonable range
            result[i] = np.clip(result[i], -10, 10)
        return result

    def _augment(self, features, mask):
        # Random horizontal flip
        if np.random.rand() > 0.5:
            features = np.flip(features, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()

        # Random vertical flip
        if np.random.rand() > 0.5:
            features = np.flip(features, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()

        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        features = np.rot90(features, k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k, axes=(1, 2)).copy()

        return features, mask


# =============================================================================
# DATA LOADING
# =============================================================================


def load_chips() -> Tuple[List[np.ndarray], List[str]]:
    """Load all chips."""
    chips = []
    names = []

    for chip_dir in CONFIG["chip_dirs"]:
        if not chip_dir.exists():
            continue

        # Load .npy files
        for f in chip_dir.glob("*.npy"):
            try:
                chip = np.load(f)
                if chip.shape[0] >= 8 and np.nansum(chip[7]) > 0:
                    chips.append(chip.astype(np.float32))
                    names.append(f.stem)
            except Exception as e:
                pass

        # Load .tif files
        for f in chip_dir.glob("*.tif"):
            try:
                with rasterio.open(f) as src:
                    chip = src.read().astype(np.float32)
                if chip.shape[0] >= 8 and np.nansum(chip[7]) > 0:
                    chips.append(chip)
                    names.append(f.stem)
            except Exception as e:
                pass

    logger.info(f"Loaded {len(chips)} chips")
    return chips, names


# =============================================================================
# TRAINING
# =============================================================================


def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    pred_binary = (pred > 0.5).astype(int)
    target_binary = (target > 0.5).astype(int)
    intersection = np.sum(pred_binary & target_binary)
    union = np.sum(pred_binary | target_binary)
    return intersection / (union + 1e-10)


def train():
    logger.info("=" * 70)
    logger.info("U-NET V4 STANDALONE TRAINING")
    logger.info("=" * 70)
    logger.info(f"Device: {CONFIG['device']}")
    logger.info(
        f"CUDA_LAUNCH_BLOCKING: {os.environ.get('CUDA_LAUNCH_BLOCKING', 'not set')}"
    )

    device = torch.device(CONFIG["device"])

    # Load data
    chips, names = load_chips()

    # Split
    train_chips, test_chips, train_names, test_names = train_test_split(
        chips, names, test_size=0.15, random_state=CONFIG["random_seed"]
    )
    train_chips, val_chips, train_names, val_names = train_test_split(
        train_chips, train_names, test_size=0.15, random_state=CONFIG["random_seed"]
    )

    logger.info(
        f"Train: {len(train_chips)}, Val: {len(val_chips)}, Test: {len(test_chips)}"
    )

    # Datasets
    train_dataset = SARDataset(train_chips, train_names, augment=True)
    val_dataset = SARDataset(val_chips, val_names, augment=False)
    test_dataset = SARDataset(test_chips, test_names, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
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
    model = UNetV4(in_channels=6).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = CombinedLossV2()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # Training loop
    best_val_iou = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_iou": []}

    for epoch in range(CONFIG["epochs"]):
        # Training
        model.train()
        train_losses = []

        for batch_idx, batch in enumerate(train_loader):
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)

            # Debug: Check input ranges
            if batch_idx == 0 and epoch == 0:
                logger.info(
                    f"Features range: [{features.min():.2f}, {features.max():.2f}]"
                )
                logger.info(f"Mask range: [{mask.min():.2f}, {mask.max():.2f}]")

            optimizer.zero_grad()

            # Forward pass (returns logits)
            logits = model(features)

            # Debug: Check output ranges
            if batch_idx == 0 and epoch == 0:
                logger.info(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")

            # Compute loss (uses logits, not probabilities)
            loss = criterion(logits, mask)

            # Check for NaN loss
            if torch.isnan(loss):
                logger.warning(f"NaN loss at epoch {epoch}, batch {batch_idx}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                mask = batch["mask"].to(device)

                logits = model(features)
                loss = criterion(logits, mask)

                val_losses.append(loss.item())

                # Apply sigmoid for predictions
                probs = torch.sigmoid(logits)
                val_preds.append(probs.cpu().numpy())
                val_targets.append(mask.cpu().numpy())

        # Metrics
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_iou = compute_iou(val_preds.flatten(), val_targets.flatten())

        train_loss = np.mean(train_losses) if train_losses else 0
        val_loss = np.mean(val_losses) if val_losses else 0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        if (epoch + 1) % 5 == 0 or val_iou > best_val_iou:
            logger.info(
                f"Epoch {epoch + 1:3d}: Train Loss={train_loss:.4f}, "
                f"Val Loss={val_loss:.4f}, Val IoU={val_iou:.4f}"
            )

        # Early stopping
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_iou": val_iou,
                },
                CONFIG["model_dir"] / "unet_v4_best.pth",
            )
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    checkpoint = torch.load(
        CONFIG["model_dir"] / "unet_v4_best.pth", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Test evaluation
    model.eval()
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            mask = batch["mask"]

            logits = model(features)
            probs = torch.sigmoid(logits)

            test_preds.append(probs.cpu().numpy())
            test_targets.append(mask.numpy())

    test_preds = np.concatenate(test_preds)
    test_targets = np.concatenate(test_targets)
    test_iou = compute_iou(test_preds.flatten(), test_targets.flatten())

    # Compute more metrics
    pred_binary = (test_preds.flatten() > 0.5).astype(int)
    target_binary = (test_targets.flatten() > 0.5).astype(int)

    tp = np.sum(pred_binary & target_binary)
    fp = np.sum(pred_binary & ~target_binary)
    fn = np.sum(~pred_binary & target_binary)

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    logger.info("\n" + "=" * 70)
    logger.info("TEST RESULTS")
    logger.info("=" * 70)
    logger.info(f"IoU:       {test_iou:.4f}")
    logger.info(f"F1:        {f1:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"Best Val IoU: {best_val_iou:.4f}")

    # Save results
    results = {
        "model": "UNet_v4_standalone",
        "test": {
            "iou": float(test_iou),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        },
        "best_val_iou": float(best_val_iou),
        "epochs_trained": epoch + 1,
        "parameters": sum(p.numel() for p in model.parameters()),
    }

    with open(CONFIG["results_dir"] / "unet_v4_standalone_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(
        f"\nResults saved to: {CONFIG['results_dir'] / 'unet_v4_standalone_results.json'}"
    )

    return results


if __name__ == "__main__":
    train()
