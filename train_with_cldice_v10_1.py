#!/usr/bin/env python3
"""
================================================================================
TRAIN WITH CLDICE v10.1 - Curriculum Learning + Better Hyperparameters
================================================================================

Improvements over v10:
1. Curriculum learning: Start with Dice+BCE, gradually add clDice
2. Higher learning rate with proper warm-up
3. Longer patience for early stopping
4. Cosine annealing with warm restarts
5. Better clDice weight scheduling

Author: SAR Water Detection Project
Date: 2026-01-26
"""

import os
import sys
import json
import time
import logging
import warnings
import math
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import our new losses
try:
    from cldice_loss import clDiceLoss, clDiceBCELoss
    from comprehensive_loss import FocalTverskyLoss

    CUSTOM_LOSSES_AVAILABLE = True
    logger.info("Custom losses loaded successfully")
except ImportError as e:
    logger.warning(f"Custom losses not available: {e}")
    CUSTOM_LOSSES_AVAILABLE = False


# =============================================================================
# CONFIGURATION - IMPROVED
# =============================================================================

CONFIG = {
    "version": "10.1-curriculum",
    "random_seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "chip_dir": Path("/home/mit-aoe/sar_water_detection/chips_expanded_npy"),
    "model_dir": Path("/home/mit-aoe/sar_water_detection/models"),
    "results_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    # Model Architecture
    "in_channels": 9,
    "base_filters": 32,
    "dropout": 0.2,  # Slightly higher dropout
    # Training - IMPROVED
    "batch_size": 2,
    "target_size": 512,
    "num_epochs": 200,  # More epochs
    "learning_rate": 2e-4,  # Higher LR
    "min_lr": 1e-6,
    "weight_decay": 1e-4,
    "patience": 40,  # More patience
    "gradient_clip": 1.0,
    # Curriculum Learning for clDice
    "warmup_epochs": 10,  # BCE+Dice only for first 10 epochs
    "cldice_rampup_epochs": 30,  # Gradually increase clDice weight over 30 epochs
    "max_cldice_weight": 0.5,  # Maximum clDice weight
    # Final loss weights (after ramp-up)
    "bce_weight": 0.2,
    "dice_weight": 0.2,
    "focal_tversky_weight": 0.1,
    # clDice specific
    "cldice_iter": 15,  # More iterations for better skeleton
    # Augmentation - More aggressive
    "use_augmentation": True,
    "aug_flip_prob": 0.5,
    "aug_rotate_prob": 0.5,
    "aug_noise_prob": 0.3,  # Add noise augmentation
    # Scheduler
    "scheduler": "cosine_warm",
    "T_0": 20,  # Restart every 20 epochs
    "T_mult": 2,  # Double period after each restart
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


class ResidualBlock(nn.Module):
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


class AttentionUNetV10(nn.Module):
    def __init__(self, in_channels=9, out_channels=1, base_filters=32, dropout=0.15):
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

        self.out = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

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
# DATASET WITH MORE AUGMENTATION
# =============================================================================


class WaterChipDataset(Dataset):
    def __init__(
        self,
        chip_files: List[Path],
        target_size: int = 512,
        augment: bool = False,
        aug_flip_prob: float = 0.5,
        aug_rotate_prob: float = 0.5,
        aug_noise_prob: float = 0.3,
    ):
        self.chip_files = chip_files
        self.target_size = target_size
        self.augment = augment
        self.aug_flip_prob = aug_flip_prob
        self.aug_rotate_prob = aug_rotate_prob
        self.aug_noise_prob = aug_noise_prob

    def __len__(self):
        return len(self.chip_files)

    def preprocess_chip(self, chip: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vv = chip[:, :, 0]
        vh = chip[:, :, 1]
        dem = chip[:, :, 2]
        slope = chip[:, :, 3]
        hand = chip[:, :, 4]
        twi = chip[:, :, 5]
        label = chip[:, :, 6]
        mndwi = chip[:, :, 7] if chip.shape[2] > 7 else np.zeros_like(vv)

        vh_texture = uniform_filter(vh**2, size=5) - uniform_filter(vh, size=5) ** 2
        vh_texture = np.sqrt(np.maximum(vh_texture, 0))
        frangi_feat = compute_frangi_vesselness(vh)

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
                frangi_feat,
            ],
            axis=-1,
        )

        features = np.nan_to_num(features, nan=0.0)
        label = np.nan_to_num(label, nan=0.0)

        return features.astype(np.float32), label.astype(np.float32)

    def augment_sample(
        self, features: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        # Add Gaussian noise to SAR channels (VV, VH)
        if np.random.random() < self.aug_noise_prob:
            noise_std = np.random.uniform(0.01, 0.05)
            features[:, :, 0] += np.random.normal(0, noise_std, features[:, :, 0].shape)
            features[:, :, 1] += np.random.normal(0, noise_std, features[:, :, 1].shape)
            features = np.clip(features, 0, 1)

        return features, label

    def __getitem__(self, idx):
        chip_path = self.chip_files[idx]
        chip = np.load(chip_path)

        features, label = self.preprocess_chip(chip)

        if features.shape[0] != self.target_size:
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

        features = torch.from_numpy(features.transpose(2, 0, 1))
        label = torch.from_numpy(label).unsqueeze(0)

        return features, label


# =============================================================================
# CURRICULUM LEARNING LOSS
# =============================================================================


class CurriculumClDiceLoss(nn.Module):
    """
    Loss with curriculum learning: gradually increase clDice weight.

    Phase 1 (warmup): BCE + Dice only
    Phase 2 (ramp-up): Gradually add clDice
    Phase 3 (full): Full clDice weight
    """

    def __init__(
        self,
        warmup_epochs: int = 10,
        rampup_epochs: int = 30,
        max_cldice_weight: float = 0.5,
        bce_weight: float = 0.2,
        dice_weight: float = 0.2,
        focal_tversky_weight: float = 0.1,
        cldice_iter: int = 15,
    ):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.rampup_epochs = rampup_epochs
        self.max_cldice_weight = max_cldice_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_tversky_weight = focal_tversky_weight

        self.current_epoch = 0

        if CUSTOM_LOSSES_AVAILABLE:
            self.cldice = clDiceLoss(iter_=cldice_iter, smooth=1.0)
            self.focal_tversky = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)
        else:
            self.cldice = None
            self.focal_tversky = None

        self.bce = nn.BCEWithLogitsLoss()

    def set_epoch(self, epoch: int):
        """Update current epoch for curriculum scheduling."""
        self.current_epoch = epoch

    def get_cldice_weight(self) -> float:
        """Get current clDice weight based on curriculum."""
        if self.current_epoch < self.warmup_epochs:
            # Phase 1: No clDice
            return 0.0
        elif self.current_epoch < self.warmup_epochs + self.rampup_epochs:
            # Phase 2: Linear ramp-up
            progress = (self.current_epoch - self.warmup_epochs) / self.rampup_epochs
            return self.max_cldice_weight * progress
        else:
            # Phase 3: Full weight
            return self.max_cldice_weight

    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)
        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + 1.0) / (pred_flat.sum() + target_flat.sum() + 1.0)
        return 1.0 - dice

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred_prob = torch.sigmoid(pred)

        # Get current clDice weight
        cldice_weight = self.get_cldice_weight()

        # BCE loss
        bce = self.bce(pred, target)

        # Dice loss
        dice = self.dice_loss(pred, target)

        # clDice loss
        if self.cldice is not None and cldice_weight > 0:
            cldice_result = self.cldice(pred_prob, target)
            if isinstance(cldice_result, tuple):
                cldice = cldice_result[0]
            else:
                cldice = cldice_result
        else:
            cldice = torch.tensor(0.0, device=pred.device)

        # Focal Tversky loss
        if self.focal_tversky is not None:
            ft_result = self.focal_tversky(pred, target)
            if isinstance(ft_result, tuple):
                ft = ft_result[0]
            else:
                ft = ft_result
        else:
            ft = torch.tensor(0.0, device=pred.device)

        # Combine with curriculum weight
        total = (
            cldice_weight * cldice
            + self.bce_weight * bce
            + self.dice_weight * dice
            + self.focal_tversky_weight * ft
        )

        # Normalize by total weight
        total_weight = (
            cldice_weight
            + self.bce_weight
            + self.dice_weight
            + self.focal_tversky_weight
        )
        total = total / total_weight

        if torch.isnan(total) or torch.isinf(total):
            logger.warning("NaN/Inf in loss, using BCE fallback")
            total = bce

        metrics = {
            "cldice": cldice.item() if torch.is_tensor(cldice) else 0.0,
            "cldice_weight": cldice_weight,
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
    loss_fn: CurriculumClDiceLoss,
    device: str,
    gradient_clip: float = 1.0,
    epoch: int = 0,
) -> Dict[str, float]:
    model.train()
    loss_fn.set_epoch(epoch)

    total_loss = 0.0
    total_metrics = {}

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss, metrics = loss_fn(outputs, labels)

        loss.backward()

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
    model: nn.Module, loader: DataLoader, loss_fn: CurriculumClDiceLoss, device: str
) -> Dict[str, float]:
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

            pred_binary = (torch.sigmoid(outputs) > 0.5).float()
            intersection = (pred_binary * labels).sum()
            union = pred_binary.sum() + labels.sum() - intersection
            iou = (intersection + 1e-8) / (union + 1e-8)

            total_loss += loss.item()
            total_iou += iou.item()
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v

    n_batches = len(loader)
    return {
        "loss": total_loss / n_batches,
        "iou": total_iou / n_batches,
        **{k: v / n_batches for k, v in total_metrics.items()},
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    logger.info("=" * 60)
    logger.info("ATTENTION U-NET v10.1 WITH CURRICULUM CLDICE")
    logger.info("=" * 60)

    np.random.seed(CONFIG["random_seed"])
    torch.manual_seed(CONFIG["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG["random_seed"])

    device = CONFIG["device"]
    logger.info(f"Using device: {device}")

    CONFIG["model_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["results_dir"].mkdir(parents=True, exist_ok=True)

    chip_dir = CONFIG["chip_dir"]
    chip_files = sorted(chip_dir.glob("*.npy"))
    logger.info(f"Found {len(chip_files)} chips")

    if len(chip_files) == 0:
        return

    n_val = max(1, int(len(chip_files) * 0.2))
    np.random.shuffle(chip_files)
    val_files = chip_files[:n_val]
    train_files = chip_files[n_val:]

    logger.info(f"Train: {len(train_files)}, Val: {len(val_files)}")

    train_dataset = WaterChipDataset(
        train_files,
        target_size=CONFIG["target_size"],
        augment=CONFIG["use_augmentation"],
        aug_flip_prob=CONFIG["aug_flip_prob"],
        aug_rotate_prob=CONFIG["aug_rotate_prob"],
        aug_noise_prob=CONFIG["aug_noise_prob"],
    )

    val_dataset = WaterChipDataset(
        val_files, target_size=CONFIG["target_size"], augment=False
    )

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

    model = AttentionUNetV10(
        in_channels=CONFIG["in_channels"],
        out_channels=1,
        base_filters=CONFIG["base_filters"],
        dropout=CONFIG["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    loss_fn = CurriculumClDiceLoss(
        warmup_epochs=CONFIG["warmup_epochs"],
        rampup_epochs=CONFIG["cldice_rampup_epochs"],
        max_cldice_weight=CONFIG["max_cldice_weight"],
        bce_weight=CONFIG["bce_weight"],
        dice_weight=CONFIG["dice_weight"],
        focal_tversky_weight=CONFIG["focal_tversky_weight"],
        cldice_iter=CONFIG["cldice_iter"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=CONFIG["T_0"], T_mult=CONFIG["T_mult"], eta_min=CONFIG["min_lr"]
    )

    best_iou = 0.0
    patience_counter = 0
    history = {"train": [], "val": []}

    logger.info(
        f"Curriculum: warmup={CONFIG['warmup_epochs']}, rampup={CONFIG['cldice_rampup_epochs']}, max_cldice={CONFIG['max_cldice_weight']}"
    )

    for epoch in range(CONFIG["num_epochs"]):
        epoch_start = time.time()

        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            gradient_clip=CONFIG["gradient_clip"],
            epoch=epoch,
        )

        val_metrics = validate_epoch(model, val_loader, loss_fn, device)

        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log with curriculum info
        cldice_w = loss_fn.get_cldice_weight()
        phase = (
            "warmup"
            if epoch < CONFIG["warmup_epochs"]
            else (
                "rampup"
                if epoch < CONFIG["warmup_epochs"] + CONFIG["cldice_rampup_epochs"]
                else "full"
            )
        )

        logger.info(
            f"Epoch {epoch + 1}/{CONFIG['num_epochs']} [{phase}] | "
            f"Train: {train_metrics['loss']:.4f} | "
            f"Val: {val_metrics['loss']:.4f} | "
            f"IoU: {val_metrics['iou']:.4f} | "
            f"clDice_w: {cldice_w:.2f} | "
            f"Time: {epoch_time:.1f}s"
        )

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            patience_counter = 0

            model_path = (
                CONFIG["model_dir"] / "attention_unet_v10.1_curriculum_best.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_iou": best_iou,
                    "config": {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in CONFIG.items()
                    },
                },
                model_path,
            )
            logger.info(f"  -> New best IoU: {best_iou:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    results = {
        "version": CONFIG["version"],
        "best_iou": best_iou,
        "final_epoch": epoch + 1,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()},
    }

    results_path = (
        CONFIG["results_dir"]
        / f"training_v10.1_curriculum_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info(f"Training complete! Best IoU: {best_iou:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
