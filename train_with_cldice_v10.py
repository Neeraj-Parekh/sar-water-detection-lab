#!/usr/bin/env python3
"""
================================================================================
TRAIN WITH CLDICE v10 - Topology-Preserving Training
================================================================================

This script trains the Attention U-Net with:
1. Proper clDice loss (CVPR 2021) - Differentiable soft morphology
2. FocalTversky loss - High recall for thin structures
3. Hausdorff distance loss - Boundary accuracy
4. Physics constraints - HAND/Slope penalties

Key Changes from v9:
- Replaces broken SkeletonLoss with proper differentiable clDice
- Adds deep supervision option
- Better learning rate scheduling
- Gradient clipping for stability

Author: SAR Water Detection Project
Date: 2026-01-26
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
from skimage.filters import frangi, hessian

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import our new losses
try:
    from cldice_loss import clDiceLoss, clDiceBCELoss
    from comprehensive_loss import (
        ComprehensiveLoss,
        get_river_detection_loss,
        get_lake_detection_loss,
        FocalTverskyLoss,
        BoundaryLoss,
    )

    CUSTOM_LOSSES_AVAILABLE = True
    logger.info("Custom losses loaded successfully")
except ImportError as e:
    logger.warning(f"Custom losses not available: {e}")
    CUSTOM_LOSSES_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "version": "10.0-clDice",
    "random_seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "chip_dir": Path("/home/mit-aoe/sar_water_detection/chips_expanded_npy"),
    "model_dir": Path("/home/mit-aoe/sar_water_detection/models"),
    "results_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    # Model Architecture
    "in_channels": 9,  # VV, VH, DEM, SLOPE, HAND, TWI, MNDWI, VH_texture, FRANGI
    "base_filters": 32,
    "dropout": 0.15,
    # Training
    "batch_size": 2,
    "target_size": 512,
    "num_epochs": 150,
    "learning_rate": 5e-5,  # Lower LR for clDice stability
    "weight_decay": 1e-4,
    "patience": 25,
    "gradient_clip": 1.0,  # Gradient clipping for stability
    # Loss Configuration (clDice focused)
    "loss_type": "river",  # "river", "lake", or "combined"
    "use_deep_supervision": False,  # Add auxiliary heads
    # clDice specific
    "cldice_iter": 10,  # Soft skeleton iterations
    "cldice_weight": 0.4,  # High weight for topology
    "bce_weight": 0.2,
    "dice_weight": 0.2,
    "focal_tversky_weight": 0.2,
    # Augmentation
    "use_augmentation": True,
    "aug_flip_prob": 0.5,
    "aug_rotate_prob": 0.5,
    # Scheduler
    "scheduler": "onecycle",  # "cosine" or "onecycle"
    "warmup_epochs": 5,
    # AMP (mixed precision) - disabled for stability
    "use_amp": False,
}


# =============================================================================
# FRANGI VESSELNESS FILTER
# =============================================================================


def compute_frangi_vesselness(
    vh: np.ndarray, scales: List[int] = [1, 2, 3]
) -> np.ndarray:
    """Compute Frangi Vesselness filter for river detection."""
    vh_norm = (vh - vh.min()) / (vh.max() - vh.min() + 1e-8)
    vh_inv = 1.0 - vh_norm

    try:
        vesselness = frangi(
            vh_inv.astype(np.float64),
            sigmas=scales,
            black_ridges=False,
            mode="reflect",
        )
    except Exception:
        vesselness = hessian(vh_inv.astype(np.float64), sigmas=scales, mode="reflect")

    vesselness = np.nan_to_num(vesselness, nan=0.0)
    if vesselness.max() > 0:
        vesselness = vesselness / vesselness.max()

    return vesselness.astype(np.float32)


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

    def __init__(self, in_channels, out_channels, dropout=0.15):
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
# ATTENTION U-NET WITH DEEP SUPERVISION
# =============================================================================


class AttentionUNetV10(nn.Module):
    """
    Attention U-Net with optional Deep Supervision.

    Deep supervision adds auxiliary output heads at each decoder level,
    providing gradient flow to earlier layers and improving training.
    """

    def __init__(
        self,
        in_channels: int = 9,
        out_channels: int = 1,
        base_filters: int = 32,
        dropout: float = 0.15,
        deep_supervision: bool = False,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
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

        # Main output
        self.out = nn.Conv2d(f, out_channels, 1)

        # Deep supervision outputs (at 1/2, 1/4, 1/8 resolution)
        if deep_supervision:
            self.ds_out4 = nn.Conv2d(f * 8, out_channels, 1)
            self.ds_out3 = nn.Conv2d(f * 4, out_channels, 1)
            self.ds_out2 = nn.Conv2d(f * 2, out_channels, 1)

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

        out = self.out(d1)

        if self.deep_supervision and self.training:
            # Return auxiliary outputs for deep supervision
            ds4 = F.interpolate(
                self.ds_out4(d4),
                size=out.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            ds3 = F.interpolate(
                self.ds_out3(d3),
                size=out.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            ds2 = F.interpolate(
                self.ds_out2(d2),
                size=out.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            return out, [ds4, ds3, ds2]

        return out


# =============================================================================
# DATASET
# =============================================================================


class WaterChipDataset(Dataset):
    """Dataset for water detection chips with Frangi preprocessing."""

    def __init__(
        self,
        chip_files: List[Path],
        target_size: int = 512,
        augment: bool = False,
        aug_flip_prob: float = 0.5,
        aug_rotate_prob: float = 0.5,
    ):
        self.chip_files = chip_files
        self.target_size = target_size
        self.augment = augment
        self.aug_flip_prob = aug_flip_prob
        self.aug_rotate_prob = aug_rotate_prob

    def __len__(self):
        return len(self.chip_files)

    def preprocess_chip(self, chip: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess chip to 9 channels + label."""
        # Extract channels
        vv = chip[:, :, 0]
        vh = chip[:, :, 1]
        dem = chip[:, :, 2]
        slope = chip[:, :, 3]
        hand = chip[:, :, 4]
        twi = chip[:, :, 5]
        label = chip[:, :, 6]
        mndwi = chip[:, :, 7] if chip.shape[2] > 7 else np.zeros_like(vv)

        # Compute derived features
        vh_texture = uniform_filter(vh**2, size=5) - uniform_filter(vh, size=5) ** 2
        vh_texture = np.sqrt(np.maximum(vh_texture, 0))

        frangi_feat = compute_frangi_vesselness(vh)

        # Normalize features
        def normalize(x, vmin=None, vmax=None):
            if vmin is None:
                vmin = np.nanpercentile(x, 1)
            if vmax is None:
                vmax = np.nanpercentile(x, 99)
            return np.clip((x - vmin) / (vmax - vmin + 1e-8), 0, 1)

        features = np.stack(
            [
                normalize(vv, -30, 0),
                normalize(vh, -35, -5),
                normalize(dem, 0, 2000),
                normalize(slope, 0, 45),
                normalize(hand, 0, 100),
                normalize(twi, 0, 20),
                normalize(mndwi, -1, 1),
                normalize(vh_texture),
                frangi_feat,  # Already 0-1
            ],
            axis=-1,
        )

        # Handle NaN
        features = np.nan_to_num(features, nan=0.0)
        label = np.nan_to_num(label, nan=0.0)

        return features.astype(np.float32), label.astype(np.float32)

    def augment_sample(
        self, features: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentations."""
        # Random horizontal flip
        if np.random.random() < self.aug_flip_prob:
            features = np.flip(features, axis=1).copy()
            label = np.flip(label, axis=1).copy()

        # Random vertical flip
        if np.random.random() < self.aug_flip_prob:
            features = np.flip(features, axis=0).copy()
            label = np.flip(label, axis=0).copy()

        # Random 90-degree rotations
        if np.random.random() < self.aug_rotate_prob:
            k = np.random.randint(1, 4)
            features = np.rot90(features, k, axes=(0, 1)).copy()
            label = np.rot90(label, k, axes=(0, 1)).copy()

        return features, label

    def __getitem__(self, idx):
        chip_path = self.chip_files[idx]
        chip = np.load(chip_path)

        features, label = self.preprocess_chip(chip)

        # Resize if needed
        if features.shape[0] != self.target_size:
            # Simple center crop or pad
            h, w = features.shape[:2]
            if h > self.target_size:
                start = (h - self.target_size) // 2
                features = features[
                    start : start + self.target_size, start : start + self.target_size
                ]
                label = label[
                    start : start + self.target_size, start : start + self.target_size
                ]
            elif h < self.target_size:
                pad = (self.target_size - h) // 2
                features = np.pad(
                    features, ((pad, pad), (pad, pad), (0, 0)), mode="reflect"
                )
                label = np.pad(label, ((pad, pad), (pad, pad)), mode="reflect")
                features = features[: self.target_size, : self.target_size]
                label = label[: self.target_size, : self.target_size]

        if self.augment:
            features, label = self.augment_sample(features, label)

        # Convert to torch tensors (CHW format)
        features = torch.from_numpy(features.transpose(2, 0, 1))
        label = torch.from_numpy(label).unsqueeze(0)

        return features, label


# =============================================================================
# COMBINED LOSS WITH CLDICE
# =============================================================================


class CombinedClDiceLoss(nn.Module):
    """
    Combined loss with proper clDice for topology preservation.

    Components:
    - clDice: Preserves river centerlines (CVPR 2021)
    - BCE: Pixel-wise accuracy
    - Dice: Region overlap
    - FocalTversky: High recall for thin structures
    """

    def __init__(
        self,
        cldice_weight: float = 0.4,
        bce_weight: float = 0.2,
        dice_weight: float = 0.2,
        focal_tversky_weight: float = 0.2,
        cldice_iter: int = 10,
    ):
        super().__init__()
        self.cldice_weight = cldice_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_tversky_weight = focal_tversky_weight

        if CUSTOM_LOSSES_AVAILABLE:
            self.cldice = clDiceLoss(iter_=cldice_iter, smooth=1.0)
            self.focal_tversky = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)
        else:
            self.cldice = None
            self.focal_tversky = None

        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)
        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + 1.0) / (pred_flat.sum() + target_flat.sum() + 1.0)
        return 1.0 - dice

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        aux_preds: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred_prob = torch.sigmoid(pred)

        # BCE loss
        bce = self.bce(pred, target)

        # Dice loss
        dice = self.dice_loss(pred, target)

        # clDice loss (if available)
        # Note: clDiceLoss returns (loss, metrics) tuple
        if self.cldice is not None:
            cldice_result = self.cldice(pred_prob, target)
            # Handle both tuple return and direct tensor return
            if isinstance(cldice_result, tuple):
                cldice = cldice_result[0]  # First element is the loss tensor
            else:
                cldice = cldice_result
        else:
            cldice = torch.tensor(0.0, device=pred.device)

        # Focal Tversky loss (if available)
        # Note: FocalTverskyLoss also returns (loss, metrics) tuple
        if self.focal_tversky is not None:
            ft_result = self.focal_tversky(pred, target)
            if isinstance(ft_result, tuple):
                ft = ft_result[0]
            else:
                ft = ft_result
        else:
            ft = torch.tensor(0.0, device=pred.device)

        # Combine
        total = (
            self.cldice_weight * cldice
            + self.bce_weight * bce
            + self.dice_weight * dice
            + self.focal_tversky_weight * ft
        )

        # Deep supervision losses (auxiliary heads)
        if aux_preds is not None:
            ds_weights = [0.4, 0.2, 0.1]  # Decreasing weights for deeper levels
            for i, aux in enumerate(aux_preds):
                if i < len(ds_weights):
                    aux_loss = self.bce(aux, target) + self.dice_loss(aux, target)
                    total = total + ds_weights[i] * aux_loss

        # Numerical stability
        if torch.isnan(total) or torch.isinf(total):
            logger.warning("NaN/Inf in loss, using BCE fallback")
            total = bce

        metrics = {
            "cldice": cldice.item() if torch.is_tensor(cldice) else 0.0,
            "bce": bce.item(),
            "dice": dice.item(),
            "focal_tversky": ft.item() if torch.is_tensor(ft) else 0.0,
            "total": total.item(),
        }

        return total, metrics


# =============================================================================
# TRAINING LOOP
# =============================================================================


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
    gradient_clip: float = 1.0,
    deep_supervision: bool = False,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_metrics = {}

    for batch_idx, (features, labels) in enumerate(loader):
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        if deep_supervision:
            outputs, aux_outputs = model(features)
            loss, metrics = loss_fn(outputs, labels, aux_outputs)
        else:
            outputs = model(features)
            loss, metrics = loss_fn(outputs, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v

    n_batches = len(loader)
    avg_loss = total_loss / n_batches
    avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}

    return {"loss": avg_loss, **avg_metrics}


def validate_epoch(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: str
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_metrics = {}

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss, metrics = loss_fn(outputs, labels)

            # Calculate IoU
            pred_binary = (torch.sigmoid(outputs) > 0.5).float()
            intersection = (pred_binary * labels).sum()
            union = pred_binary.sum() + labels.sum() - intersection
            iou = (intersection + 1e-8) / (union + 1e-8)

            total_loss += loss.item()
            total_iou += iou.item()
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v

    n_batches = len(loader)
    avg_loss = total_loss / n_batches
    avg_iou = total_iou / n_batches
    avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}

    return {"loss": avg_loss, "iou": avg_iou, **avg_metrics}


# =============================================================================
# MAIN TRAINING
# =============================================================================


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("ATTENTION U-NET v10 WITH CLDICE TRAINING")
    logger.info("=" * 60)

    # Set random seed
    np.random.seed(CONFIG["random_seed"])
    torch.manual_seed(CONFIG["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG["random_seed"])

    device = CONFIG["device"]
    logger.info(f"Using device: {device}")

    # Create directories
    CONFIG["model_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["results_dir"].mkdir(parents=True, exist_ok=True)

    # Load chip files
    chip_dir = CONFIG["chip_dir"]
    if not chip_dir.exists():
        logger.error(f"Chip directory not found: {chip_dir}")
        return

    chip_files = sorted(chip_dir.glob("*.npy"))
    logger.info(f"Found {len(chip_files)} chips")

    if len(chip_files) == 0:
        logger.error("No chips found!")
        return

    # Split into train/val
    n_val = max(1, int(len(chip_files) * 0.2))
    np.random.shuffle(chip_files)
    val_files = chip_files[:n_val]
    train_files = chip_files[n_val:]

    logger.info(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # Create datasets
    train_dataset = WaterChipDataset(
        train_files,
        target_size=CONFIG["target_size"],
        augment=CONFIG["use_augmentation"],
        aug_flip_prob=CONFIG["aug_flip_prob"],
        aug_rotate_prob=CONFIG["aug_rotate_prob"],
    )

    val_dataset = WaterChipDataset(
        val_files, target_size=CONFIG["target_size"], augment=False
    )

    # Create dataloaders
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

    # Create model
    model = AttentionUNetV10(
        in_channels=CONFIG["in_channels"],
        out_channels=1,
        base_filters=CONFIG["base_filters"],
        dropout=CONFIG["dropout"],
        deep_supervision=CONFIG["use_deep_supervision"],
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # Create loss function
    if CUSTOM_LOSSES_AVAILABLE and CONFIG["loss_type"] == "river":
        logger.info("Using river-optimized loss (clDice + FocalTversky)")
        loss_fn = CombinedClDiceLoss(
            cldice_weight=CONFIG["cldice_weight"],
            bce_weight=CONFIG["bce_weight"],
            dice_weight=CONFIG["dice_weight"],
            focal_tversky_weight=CONFIG["focal_tversky_weight"],
            cldice_iter=CONFIG["cldice_iter"],
        )
    else:
        logger.info("Using standard combined loss")
        loss_fn = CombinedClDiceLoss()

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    # Create scheduler
    if CONFIG["scheduler"] == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=CONFIG["learning_rate"] * 10,
            epochs=CONFIG["num_epochs"],
            steps_per_epoch=len(train_loader),
            pct_start=CONFIG["warmup_epochs"] / CONFIG["num_epochs"],
        )
    else:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Training loop
    best_iou = 0.0
    patience_counter = 0
    history = {"train": [], "val": []}

    for epoch in range(CONFIG["num_epochs"]):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            gradient_clip=CONFIG["gradient_clip"],
            deep_supervision=CONFIG["use_deep_supervision"],
        )

        # Validate
        val_metrics = validate_epoch(model, val_loader, loss_fn, device)

        # Update scheduler
        if CONFIG["scheduler"] != "onecycle":
            scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log progress
        logger.info(
            f"Epoch {epoch + 1}/{CONFIG['num_epochs']} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"clDice: {val_metrics.get('cldice', 0):.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Save best model
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            patience_counter = 0

            model_path = CONFIG["model_dir"] / "attention_unet_v10_cldice_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_iou": best_iou,
                    "config": CONFIG,
                },
                model_path,
            )
            logger.info(f"Saved best model with IoU: {best_iou:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    # Save final results
    results = {
        "version": CONFIG["version"],
        "best_iou": best_iou,
        "final_epoch": epoch + 1,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()},
        "history": history,
    }

    results_path = (
        CONFIG["results_dir"]
        / f"training_v10_cldice_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info(f"Training complete! Best IoU: {best_iou:.4f}")
    logger.info(f"Results saved to: {results_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
