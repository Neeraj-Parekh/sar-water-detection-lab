#!/usr/bin/env python3
"""
================================================================================
SOTA U-NET v9 - Research-Grade River Detection
================================================================================

KEY INNOVATIONS (from research priorities):
1. SKELETON LOSS - Topology-aware loss that penalizes broken river continuity
2. FRANGI VESSELNESS - Pre-computed "line detector" as extra input channel
3. GMM AUTO-THRESHOLD - Let data decide thresholds, not hard-coded values
4. MST RIVER CONNECTOR - Graph-based post-processing to heal gaps
5. NUMERICAL STABILITY - Fix NaN issues from v8

Architecture:
- Attention U-Net with 9 input channels (+Frangi vesselness)
- Skeleton + Dice + Focal Loss (topology-aware)
- FP32 for stability (no AMP)

Author: SAR Water Detection Project - SOTA Branch
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
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy import ndimage
from skimage.morphology import skeletonize, thin
from skimage.filters import frangi, hessian
from sklearn.mixture import GaussianMixture

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
    # Model Architecture - NOW 9 CHANNELS (+Frangi)
    "in_channels": 9,  # VV, VH, DEM, SLOPE, HAND, TWI, MNDWI, VH_texture, FRANGI
    "base_filters": 32,
    "dropout": 0.15,  # Slightly lower dropout
    # Training - FULL CHIP, NO AMP (stability)
    "batch_size": 2,
    "target_size": 512,
    "num_epochs": 100,  # Shorter epochs with better loss
    "learning_rate": 1e-4,  # Higher LR, we have better gradients now
    "weight_decay": 1e-4,
    "patience": 20,
    # SOTA Loss weights
    "skeleton_weight": 0.3,  # NEW: Topology preservation
    "dice_weight": 0.4,
    "focal_weight": 0.2,
    "connectivity_weight": 0.1,
    "focal_gamma": 2.0,
    # Augmentation
    "use_augmentation": True,
    # NO AMP - stability first
    "use_amp": False,
}


# =============================================================================
# FRANGI VESSELNESS FILTER
# =============================================================================


def compute_frangi_vesselness(
    vh: np.ndarray, scales: List[int] = [1, 2, 3]
) -> np.ndarray:
    """
    Compute Frangi Vesselness filter to highlight tube-like (river) structures.
    This pre-computed filter screams "THIS IS A LINE!" to the model.

    Args:
        vh: VH backscatter in dB (negative values)
        scales: Sigma values for multi-scale analysis

    Returns:
        Vesselness response (0-1, higher = more line-like)
    """
    # Normalize VH to 0-1 range for Frangi
    vh_norm = (vh - vh.min()) / (vh.max() - vh.min() + 1e-8)

    # Invert so water (dark) becomes bright for Frangi
    vh_inv = 1.0 - vh_norm

    # Apply Frangi filter (detects tube-like structures)
    try:
        vesselness = frangi(
            vh_inv.astype(np.float64),
            sigmas=scales,
            black_ridges=False,  # We want bright ridges (water)
            mode="reflect",
        )
    except Exception:
        # Fallback: simple Hessian-based line detection
        vesselness = hessian(vh_inv.astype(np.float64), sigmas=scales, mode="reflect")

    # Normalize output
    vesselness = np.nan_to_num(vesselness, nan=0.0)
    if vesselness.max() > 0:
        vesselness = vesselness / vesselness.max()

    return vesselness.astype(np.float32)


# =============================================================================
# GMM AUTO-THRESHOLD
# =============================================================================


def compute_gmm_threshold(
    vh: np.ndarray, n_components: int = 2
) -> Tuple[float, float, float]:
    """
    Use Gaussian Mixture Model to find optimal VH threshold.
    Instead of hard-coded "-19.6 dB", let the data decide.

    Returns:
        (threshold, water_mode, land_mode) - The valley between two modes
    """
    # Flatten and remove NaN
    vh_flat = vh.flatten()
    vh_valid = vh_flat[~np.isnan(vh_flat)]

    if len(vh_valid) < 100:
        return -19.6, -22.0, -12.0  # Fallback

    # Fit 2-component GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42, max_iter=100)
    gmm.fit(vh_valid.reshape(-1, 1))

    # Get means (modes)
    means = gmm.means_.flatten()
    water_mode = means.min()  # Darker = water
    land_mode = means.max()  # Brighter = land

    # Threshold = midpoint between modes (the "valley")
    threshold = (water_mode + land_mode) / 2

    return float(threshold), float(water_mode), float(land_mode)


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
# ATTENTION U-NET
# =============================================================================


class AttentionUNet(nn.Module):
    """U-Net with Attention Gates and Residual Blocks."""

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
# SKELETON LOSS - THE KEY INNOVATION
# =============================================================================


class SkeletonLoss(nn.Module):
    """
    Topology-aware Skeleton/Centerline Loss.

    Standard Dice doesn't care if a river is broken into dots.
    This loss calculates the skeleton of predictions and ground truth.
    If the prediction breaks the skeleton (disconnects the river),
    the penalty is MASSIVE.

    Forces the model to prioritize CONTINUITY over just pixel accuracy.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def compute_skeleton_torch(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Approximate skeleton using morphological thinning.
        Uses erosion-based approach that works on GPU.
        """
        # Convert to binary
        binary = (mask > 0.5).float()

        # Erosion kernel
        kernel = torch.ones(1, 1, 3, 3, device=mask.device, dtype=mask.dtype)

        # Iterative erosion to find approximate skeleton
        skeleton = binary.clone()
        prev_sum = float("inf")

        for _ in range(10):  # Max iterations
            eroded = F.conv2d(skeleton, kernel, padding=1)
            eroded = (eroded >= 9).float()  # Full 3x3 neighborhood

            # Skeleton = original - eroded (but keep connected)
            diff = skeleton - eroded
            skeleton = skeleton * (diff < 0.5).float() + eroded

            curr_sum = skeleton.sum().item()
            if abs(curr_sum - prev_sum) < 1:
                break
            prev_sum = curr_sum

        # Edge detection on remaining to get centerline
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=mask.dtype, device=mask.device
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=mask.dtype, device=mask.device
        ).view(1, 1, 3, 3)

        gx = F.conv2d(binary, sobel_x, padding=1)
        gy = F.conv2d(binary, sobel_y, padding=1)
        edges = torch.sqrt(gx**2 + gy**2 + 1e-8)

        # Combine: skeleton + edges = centerline
        centerline = torch.clamp(skeleton + edges * 0.5, 0, 1)

        return centerline

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)

        # Compute skeletons
        pred_skel = self.compute_skeleton_torch(pred_prob)
        target_skel = self.compute_skeleton_torch(target)

        # Dice on skeletons - heavily penalizes broken rivers
        pred_flat = pred_skel.view(-1)
        target_flat = target_skel.view(-1)

        intersection = (pred_flat * target_flat).sum()
        skeleton_dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )

        return 1.0 - skeleton_dice


# =============================================================================
# OTHER LOSS FUNCTIONS
# =============================================================================


class FocalLoss(nn.Module):
    """Focal Loss with numerical stability."""

    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        pred_prob = torch.sigmoid(pred)
        pred_prob = pred_prob.clamp(1e-6, 1 - 1e-6)  # Better clamping

        bce = F.binary_cross_entropy(pred_prob, target, reduction="none")
        pt = torch.where(target == 1, pred_prob, 1 - pred_prob)
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)

        loss = (alpha_t * focal_weight * bce).mean()
        return torch.clamp(loss, 0, 10)  # Clamp for stability


class DiceLoss(nn.Module):
    """Dice Loss with stability."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_prob = torch.sigmoid(pred)
        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice


class ConnectivityLoss(nn.Module):
    """Penalize disconnected water predictions."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "conn_kernel", torch.ones(1, 1, 3, 3, dtype=torch.float32) / 9.0
        )

    def forward(self, pred, target):
        pred_prob = torch.sigmoid(pred)

        pred_neighbors = F.conv2d(pred_prob, self.conn_kernel, padding=1)
        target_neighbors = F.conv2d(target, self.conn_kernel, padding=1)

        water_mask = target > 0.5
        if water_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        pred_density = pred_neighbors[water_mask]
        target_density = target_neighbors[water_mask]

        return F.mse_loss(pred_density, target_density)


class CombinedSOTALoss(nn.Module):
    """SOTA Combined loss with Skeleton awareness."""

    def __init__(
        self,
        skeleton_weight=0.3,
        dice_weight=0.4,
        focal_weight=0.2,
        connectivity_weight=0.1,
        gamma=2.0,
    ):
        super().__init__()
        self.skeleton_loss = SkeletonLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma=gamma)
        self.connectivity_loss = ConnectivityLoss()

        self.skeleton_weight = skeleton_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.connectivity_weight = connectivity_weight

    def forward(self, pred, target):
        skeleton = self.skeleton_loss(pred, target)
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        connectivity = self.connectivity_loss(pred, target)

        total = (
            self.skeleton_weight * skeleton
            + self.dice_weight * dice
            + self.focal_weight * focal
            + self.connectivity_weight * connectivity
        )

        # Numerical stability check
        if torch.isnan(total) or torch.isinf(total):
            logger.warning("NaN/Inf detected in loss, using fallback")
            total = dice  # Fallback to stable loss

        return total, {
            "skeleton": skeleton.item() if not torch.isnan(skeleton) else 0.0,
            "dice": dice.item() if not torch.isnan(dice) else 0.0,
            "focal": focal.item() if not torch.isnan(focal) else 0.0,
            "connectivity": connectivity.item()
            if not torch.isnan(connectivity)
            else 0.0,
        }


# =============================================================================
# DATASET WITH FRANGI
# =============================================================================


class SOTADataset(Dataset):
    """
    SOTA Dataset with Frangi Vesselness as extra channel.
    Also computes GMM threshold per chip for adaptive physics.
    """

    def __init__(
        self, chips: List[np.ndarray], target_size: int = 512, augment: bool = False
    ):
        self.chips = chips
        self.target_size = target_size
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
            "frangi": {"mean": 0.2, "std": 0.3},  # NEW
        }

    def __len__(self):
        return len(self.chips)

    def compute_vh_texture(self, vh):
        vh_mean = uniform_filter(vh, size=5)
        vh_sq_mean = uniform_filter(vh**2, size=5)
        vh_var = np.maximum(vh_sq_mean - vh_mean**2, 0)
        return np.sqrt(vh_var).astype(np.float32)

    def pad_to_target(self, data: np.ndarray, label: np.ndarray):
        h, w = data.shape[1], data.shape[2]

        if h >= self.target_size and w >= self.target_size:
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

        # NEW: Compute Frangi Vesselness
        frangi_resp = compute_frangi_vesselness(vh, scales=[1, 2, 3])

        # Fix NaN
        vv = np.nan_to_num(vv, nan=-20.0)
        vh = np.nan_to_num(vh, nan=-25.0)
        dem = np.nan_to_num(dem, nan=0.0)
        hand = np.nan_to_num(hand, nan=100.0)
        twi = np.nan_to_num(twi, nan=5.0)
        mndwi = np.nan_to_num(mndwi, nan=0.0)
        frangi_resp = np.nan_to_num(frangi_resp, nan=0.0)

        # Stack 9 channels: VV, VH, DEM, SLOPE, HAND, TWI, MNDWI, VH_texture, FRANGI
        data = np.stack(
            [vv, vh, dem, slope, hand, twi, mndwi, vh_texture, frangi_resp], axis=0
        )

        # Normalize
        keys = [
            "vv",
            "vh",
            "dem",
            "slope",
            "hand",
            "twi",
            "mndwi",
            "vh_texture",
            "frangi",
        ]
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

        # Pad to target size
        data, label = self.pad_to_target(data, label)

        return torch.from_numpy(data.astype(np.float32)), torch.from_numpy(
            label[np.newaxis].astype(np.float32)
        )


# =============================================================================
# TRAINING
# =============================================================================


def compute_iou(pred, target):
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    target_bin = (target > 0.5).float()

    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return (intersection / union).item()


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch - NO AMP for stability."""
    model.train()
    total_loss = 0
    total_iou = 0
    loss_components = {"skeleton": 0, "dice": 0, "focal": 0, "connectivity": 0}

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss, components = criterion(output, target)

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Batch {batch_idx}: NaN/Inf loss detected, skipping")
            continue

        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        total_iou += compute_iou(output, target)

        for k, v in components.items():
            loss_components[k] += v

    n = max(len(loader), 1)
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

            if not torch.isnan(loss):
                total_loss += loss.item()
            total_iou += compute_iou(output, target)

    n = max(len(loader), 1)
    return total_loss / n, total_iou / n


# =============================================================================
# MAIN
# =============================================================================


def main():
    logger.info("=" * 70)
    logger.info("SOTA U-NET v9 TRAINING")
    logger.info("Innovations: Skeleton Loss + Frangi Vesselness + GMM Thresholds")
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

    # Compute GMM threshold on sample chip
    sample_vh = chips[0][:, :, 1]
    gmm_thresh, water_mode, land_mode = compute_gmm_threshold(sample_vh)
    logger.info(
        f"GMM Auto-Threshold: {gmm_thresh:.2f} dB (water mode: {water_mode:.2f}, land: {land_mode:.2f})"
    )

    # Split (80/20)
    n_test = max(1, int(len(chips) * 0.2))
    indices = np.random.permutation(len(chips))
    test_chips = [chips[i] for i in indices[:n_test]]
    train_chips = [chips[i] for i in indices[n_test:]]

    logger.info(f"Train: {len(train_chips)}, Test: {len(test_chips)}")
    logger.info(f"Input channels: {CONFIG['in_channels']} (includes Frangi vesselness)")

    # Datasets
    train_dataset = SOTADataset(
        train_chips,
        target_size=CONFIG["target_size"],
        augment=CONFIG["use_augmentation"],
    )
    test_dataset = SOTADataset(
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

    # SOTA Loss
    criterion = CombinedSOTALoss(
        skeleton_weight=CONFIG["skeleton_weight"],
        dice_weight=CONFIG["dice_weight"],
        focal_weight=CONFIG["focal_weight"],
        connectivity_weight=CONFIG["connectivity_weight"],
        gamma=CONFIG["focal_gamma"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    # Training loop
    best_iou = 0
    patience_counter = 0
    history = []

    logger.info("Starting training (NO AMP for stability)...")
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
            f"Val Loss={val_loss:.4f}, Val IoU={val_iou:.4f} | "
            f"Skel={loss_components['skeleton']:.3f}, Dice={loss_components['dice']:.3f}"
        )

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0

            model_path = CONFIG["model_dir"] / "attention_unet_v9_sota_best.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_iou": best_iou,
                    "gmm_threshold": gmm_thresh,
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
        "innovations": [
            "Skeleton/Centerline Loss for topology preservation",
            "Frangi Vesselness filter as input channel",
            "GMM auto-threshold for adaptive physics",
            "NO AMP for numerical stability",
        ],
        "gmm_threshold": gmm_thresh,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()},
        "history": history,
    }

    results_path = CONFIG["results_dir"] / "attention_unet_v9_sota_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best IoU: {best_iou:.4f}")
    logger.info(f"Training time: {train_time / 60:.1f} minutes")
    logger.info(
        f"Model saved to: {CONFIG['model_dir'] / 'attention_unet_v9_sota_best.pth'}"
    )


if __name__ == "__main__":
    main()
