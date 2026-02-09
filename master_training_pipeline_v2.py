#!/usr/bin/env python3
"""
Master SAR Water Detection Training Pipeline v2
================================================
Fixed version with proper data handling.

Author: AI Assistant
Date: 2026-01-24
GPU: NVIDIA RTX A5000 (24GB)
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    jaccard_score,
)
import lightgbm as lgb
from scipy import ndimage

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("master_training_v2.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "chip_dir": Path("/home/mit-aoe/sar_water_detection/chips"),
    "output_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    "model_dir": Path("/home/mit-aoe/sar_water_detection/models"),
    "random_seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 4,  # Reduced for memory
    "num_workers": 0,  # Avoid multiprocessing issues
    "image_size": 256,  # Resize to consistent size
    # Band indices (chips are H x W x 8, with 8th being truth)
    "feature_bands": 7,  # VV, VH, MNDWI, DEM, HAND, SLOPE, TWI
    "has_truth": True,
    # Training splits
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
}

# Create output directories
CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
CONFIG["model_dir"].mkdir(parents=True, exist_ok=True)

np.random.seed(CONFIG["random_seed"])
torch.manual_seed(CONFIG["random_seed"])


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================


def load_chip(filepath: Path) -> Optional[np.ndarray]:
    """Load a single chip file and standardize format."""
    try:
        data = np.load(filepath, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            data = data["arr_0"]

        # Check format: (H, W, C) or (C, H, W)
        if data.ndim == 3:
            if data.shape[2] <= 10:  # Likely (H, W, C)
                data = np.transpose(data, (2, 0, 1))  # Convert to (C, H, W)

        return data
    except Exception as e:
        logger.warning(f"Failed to load {filepath}: {e}")
        return None


def load_all_chips(chip_dir: Path) -> Tuple[List[np.ndarray], List[str]]:
    """Load all chip files from directory."""
    chip_files = sorted(chip_dir.glob("*.npy"))
    chips = []
    names = []

    for f in chip_files:
        chip = load_chip(f)
        if chip is not None:
            chips.append(chip)
            names.append(f.stem)

    logger.info(f"Loaded {len(chips)} chips from {chip_dir}")
    if chips:
        logger.info(f"Chip shape example: {chips[0].shape}")
    return chips, names


def resize_chip(chip: np.ndarray, size: int) -> np.ndarray:
    """Resize chip to consistent size using bilinear interpolation."""
    c, h, w = chip.shape
    if h == size and w == size:
        return chip

    # Use scipy zoom for numpy array
    zoom_factors = (1, size / h, size / w)
    from scipy.ndimage import zoom

    resized = zoom(chip.astype(np.float32), zoom_factors, order=1)
    return resized


def extract_features_lightgbm(chip: np.ndarray) -> Dict[str, float]:
    """Extract physics-meaningful features for LightGBM."""
    # Chip is (C, H, W) with C >= 7
    if chip.shape[0] < 7:
        return {}

    vv = chip[0]
    vh = chip[1]
    mndwi = chip[2]
    dem = chip[3]
    hand = chip[4]
    slope = chip[5]
    twi = chip[6]

    features = {}

    # Backscatter statistics
    for name, arr in [("vv", vv), ("vh", vh)]:
        features[f"{name}_mean"] = np.nanmean(arr)
        features[f"{name}_std"] = np.nanstd(arr)
        features[f"{name}_min"] = np.nanmin(arr)
        features[f"{name}_max"] = np.nanmax(arr)
        features[f"{name}_p5"] = np.nanpercentile(arr, 5)
        features[f"{name}_p25"] = np.nanpercentile(arr, 25)
        features[f"{name}_p50"] = np.nanpercentile(arr, 50)
        features[f"{name}_p75"] = np.nanpercentile(arr, 75)
        features[f"{name}_p95"] = np.nanpercentile(arr, 95)

    # Polarization features
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(vv, vh + 1e-6)
        diff = vv - vh
    features["ratio_mean"] = np.nanmean(ratio)
    features["ratio_std"] = np.nanstd(ratio)
    features["diff_mean"] = np.nanmean(diff)
    features["diff_std"] = np.nanstd(diff)

    # Terrain features
    features["hand_mean"] = np.nanmean(hand)
    features["hand_std"] = np.nanstd(hand)
    features["hand_low_pct"] = np.nanmean(hand < 5) * 100
    features["slope_mean"] = np.nanmean(slope)
    features["slope_flat_pct"] = np.nanmean(slope < 5) * 100
    features["twi_mean"] = np.nanmean(twi)
    features["twi_high_pct"] = np.nanmean(twi > 10) * 100

    # MNDWI features
    features["mndwi_mean"] = np.nanmean(mndwi)
    features["mndwi_std"] = np.nanstd(mndwi)
    features["mndwi_pos_pct"] = np.nanmean(mndwi > 0) * 100

    return features


def get_water_percentage(chip: np.ndarray) -> float:
    """Get water percentage from chip (using truth mask or MNDWI)."""
    if chip.shape[0] >= 8:
        truth = chip[7]
        return np.nanmean(truth > 0.5) * 100
    else:
        mndwi = chip[2]
        return np.nanmean(mndwi > 0) * 100


# =============================================================================
# MODEL 1: LIGHTGBM FOR PIXEL-WISE CLASSIFICATION
# =============================================================================


def train_lightgbm_pixel(chips: List[np.ndarray], names: List[str]) -> Dict:
    """
    Train LightGBM for pixel-wise water detection.
    This is the approach that beat deep learning on Sen1Floods11.
    """
    logger.info("=" * 60)
    logger.info("TRAINING MODEL 1: LightGBM Pixel-wise Classifier")
    logger.info("=" * 60)

    start_time = time.time()

    # Extract pixel-level features and labels
    X_all = []
    y_all = []

    for chip_idx, chip in enumerate(chips):
        if chip.shape[0] < 8:
            continue

        vv = chip[0]
        vh = chip[1]
        mndwi = chip[2]
        hand = chip[4]
        slope = chip[5]
        twi = chip[6]
        truth = chip[7]

        h, w = vv.shape

        # Sample pixels (stratified)
        water_idx = np.where((truth > 0.5).flatten())[0]
        nonwater_idx = np.where((truth <= 0.5).flatten())[0]

        n_samples = 500  # Per class per chip

        if len(water_idx) > 0:
            water_samples = np.random.choice(
                water_idx, size=min(n_samples, len(water_idx)), replace=False
            )
        else:
            water_samples = np.array([], dtype=int)

        if len(nonwater_idx) > 0:
            nonwater_samples = np.random.choice(
                nonwater_idx, size=min(n_samples, len(nonwater_idx)), replace=False
            )
        else:
            nonwater_samples = np.array([], dtype=int)

        for idx in np.concatenate([water_samples, nonwater_samples]):
            i, j = idx // w, idx % w

            # Extract features for this pixel
            feat = [
                vv[i, j],
                vh[i, j],
                vv[i, j] - vh[i, j],
                vv[i, j] / (vh[i, j] + 1e-6),
                hand[i, j],
                slope[i, j],
                twi[i, j],
                mndwi[i, j],
            ]

            # Add local statistics (3x3 window)
            for arr in [vv, vh]:
                i_min, i_max = max(0, i - 1), min(h, i + 2)
                j_min, j_max = max(0, j - 1), min(w, j + 2)
                patch = arr[i_min:i_max, j_min:j_max]
                feat.extend([np.mean(patch), np.std(patch)])

            X_all.append(feat)
            y_all.append(1 if truth[i, j] > 0.5 else 0)

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)

    # Clean data
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]

    logger.info(f"Total samples: {len(X)}")
    logger.info(f"Class balance: {np.mean(y):.2%} water")

    if len(X) == 0:
        return {"status": "failed", "error": "No valid samples"}

    # Train/val/test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=CONFIG["random_seed"], stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.2,
        random_state=CONFIG["random_seed"],
        stratify=y_trainval,
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Feature names
    feature_names = [
        "VV",
        "VH",
        "VV-VH",
        "VV/VH",
        "HAND",
        "SLOPE",
        "TWI",
        "MNDWI",
        "VV_local_mean",
        "VV_local_std",
        "VH_local_mean",
        "VH_local_std",
    ]

    # Train LightGBM
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        num_leaves=63,
        learning_rate=0.05,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=CONFIG["random_seed"],
        verbose=-1,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    # Evaluate
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]

    results = {
        "model": "LightGBM_PixelWise",
        "params": {
            "n_estimators": model.n_estimators_,
            "max_depth": 8,
            "features": feature_names,
        },
        "test": {
            "accuracy": accuracy_score(y_test, y_pred_test),
            "f1": f1_score(y_test, y_pred_test),
            "precision": precision_score(y_test, y_pred_test),
            "recall": recall_score(y_test, y_pred_test),
            "iou": jaccard_score(y_test, y_pred_test),
        },
        "feature_importance": dict(
            zip(feature_names, model.feature_importances_.tolist())
        ),
        "training_time_seconds": time.time() - start_time,
        "n_samples": len(X),
    }

    # Save model
    model_path = CONFIG["model_dir"] / "lightgbm_pixel.txt"
    model.booster_.save_model(str(model_path))
    results["model_path"] = str(model_path)

    logger.info(f"LightGBM Pixel-wise Results:")
    logger.info(f"  Test Accuracy: {results['test']['accuracy']:.4f}")
    logger.info(f"  Test F1:       {results['test']['f1']:.4f}")
    logger.info(f"  Test IoU:      {results['test']['iou']:.4f}")

    # Top features
    sorted_feat = sorted(
        results["feature_importance"].items(), key=lambda x: x[1], reverse=True
    )
    logger.info("Top 5 Features:")
    for name, imp in sorted_feat[:5]:
        logger.info(f"  {name}: {imp:.4f}")

    return results


# =============================================================================
# PYTORCH DATASET FOR SEGMENTATION
# =============================================================================


class SARSegmentationDataset(Dataset):
    """PyTorch Dataset for SAR water segmentation."""

    def __init__(
        self, chips: List[np.ndarray], names: List[str], image_size: int = 256
    ):
        self.chips = chips
        self.names = names
        self.image_size = image_size

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, idx):
        chip = self.chips[idx].copy().astype(np.float32)

        # Chip is (C, H, W)
        features = chip[:7]  # First 7 bands

        if chip.shape[0] >= 8:
            mask = chip[7:8]  # 8th band is truth, keep as (1, H, W)
        else:
            mask = (chip[2:3] > 0).astype(np.float32)  # MNDWI as proxy

        # Resize to consistent size
        features = self._resize(features, self.image_size)
        mask = self._resize(mask, self.image_size)

        # Normalize features
        features = self._normalize(features)

        # Binarize mask
        mask = (mask > 0.5).astype(np.float32)

        return torch.from_numpy(features), torch.from_numpy(mask)

    def _resize(self, arr: np.ndarray, size: int) -> np.ndarray:
        """Resize array to (C, size, size)."""
        from scipy.ndimage import zoom

        c, h, w = arr.shape
        if h == size and w == size:
            return arr
        zoom_factors = (1, size / h, size / w)
        return zoom(arr, zoom_factors, order=1)

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize each band."""
        normalized = np.zeros_like(features)

        # VV, VH: -30 to 0 dB
        normalized[0] = np.clip((features[0] + 30) / 30, 0, 1)
        normalized[1] = np.clip((features[1] + 30) / 30, 0, 1)

        # MNDWI: -1 to 1
        normalized[2] = np.clip((features[2] + 1) / 2, 0, 1)

        # DEM: 0 to 5000m
        normalized[3] = np.clip(features[3] / 1000, 0, 5)

        # HAND: 0 to 50m
        normalized[4] = np.clip(features[4] / 50, 0, 1)

        # Slope: 0 to 90
        normalized[5] = np.clip(features[5] / 90, 0, 1)

        # TWI: 0 to 20
        normalized[6] = np.clip(features[6] / 20, 0, 1)

        return normalized


# =============================================================================
# MODEL 2: PHYSICS-GUIDED LIGHTWEIGHT UNET
# =============================================================================


class ConvBlock(nn.Module):
    """Double convolution block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class LightweightUNet(nn.Module):
    """
    Lightweight U-Net with physics attention.
    ~500K parameters (vs 31M for full U-Net).
    """

    def __init__(self, in_channels=7, num_classes=1):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        # Output
        self.out = nn.Conv2d(32, num_classes, 1)

        # Physics attention (HAND-based)
        self.hand_attention = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, apply_physics=True):
        # Store HAND for attention (normalized, channel 4)
        hand = x[:, 4:5, :, :]

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = self.out(d1)

        # Apply physics attention (water at low HAND)
        if apply_physics:
            # Resize HAND to match output
            hand_resized = F.interpolate(
                hand, size=out.shape[-2:], mode="bilinear", align_corners=False
            )
            physics_attn = 1 - self.hand_attention(
                hand_resized
            )  # High attention where HAND is low
            out = out * physics_attn

        return out


class PhysicsLoss(nn.Module):
    """Combined BCE + physics constraints."""

    def __init__(self, hand_weight=0.2, slope_weight=0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.hand_weight = hand_weight
        self.slope_weight = slope_weight

    def forward(self, pred, target, features):
        # BCE loss
        bce = self.bce(pred, target)

        probs = torch.sigmoid(pred)

        # HAND constraint: penalize water at high elevation
        hand = features[:, 4:5, :, :]
        hand_resized = F.interpolate(
            hand, size=pred.shape[-2:], mode="bilinear", align_corners=False
        )
        hand_loss = (probs * hand_resized).mean()

        # Slope constraint: penalize water on steep slopes
        slope = features[:, 5:6, :, :]
        slope_resized = F.interpolate(
            slope, size=pred.shape[-2:], mode="bilinear", align_corners=False
        )
        steep_mask = (slope_resized > 0.2).float()
        slope_loss = (probs * steep_mask).mean()

        total = bce + self.hand_weight * hand_loss + self.slope_weight * slope_loss

        return total, {
            "bce": bce.item(),
            "hand": hand_loss.item(),
            "slope": slope_loss.item(),
        }


def compute_iou(pred, target, threshold=0.5):
    """Compute IoU metric."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_binary * target).sum()
    union = ((pred_binary + target) > 0).float().sum()
    return (intersection / (union + 1e-6)).item()


def train_physics_unet(chips: List[np.ndarray], names: List[str]) -> Dict:
    """Train physics-guided lightweight U-Net."""
    logger.info("=" * 60)
    logger.info("TRAINING MODEL 2: Physics-Guided Lightweight U-Net")
    logger.info("=" * 60)

    start_time = time.time()
    device = CONFIG["device"]

    # Filter chips with 8 bands
    valid_chips = [c for c in chips if c.shape[0] >= 8]
    valid_names = [n for c, n in zip(chips, names) if c.shape[0] >= 8]

    logger.info(f"Valid chips with truth mask: {len(valid_chips)}")

    if len(valid_chips) < 10:
        return {"status": "failed", "error": "Not enough chips with truth masks"}

    # Split data
    train_chips, test_chips = train_test_split(
        valid_chips, test_size=0.2, random_state=CONFIG["random_seed"]
    )
    train_chips, val_chips = train_test_split(
        train_chips, test_size=0.2, random_state=CONFIG["random_seed"]
    )

    logger.info(
        f"Train: {len(train_chips)}, Val: {len(val_chips)}, Test: {len(test_chips)}"
    )

    # Create datasets
    train_dataset = SARSegmentationDataset(train_chips, [], CONFIG["image_size"])
    val_dataset = SARSegmentationDataset(val_chips, [], CONFIG["image_size"])
    test_dataset = SARSegmentationDataset(test_chips, [], CONFIG["image_size"])

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0
    )

    # Create model
    model = LightweightUNet(in_channels=7, num_classes=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    criterion = PhysicsLoss(hand_weight=0.2, slope_weight=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # Training
    best_val_iou = 0
    history = {"train_loss": [], "val_loss": [], "val_iou": []}
    num_epochs = 100

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        for features, masks in train_loader:
            features = features.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss, _ = criterion(outputs, masks, features)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        val_iou = 0
        with torch.no_grad():
            for features, masks in val_loader:
                features = features.to(device)
                masks = masks.to(device)

                outputs = model(features)
                loss, _ = criterion(outputs, masks, features)
                val_loss += loss.item()
                val_iou += compute_iou(outputs, masks)

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        if epoch % 20 == 0:
            logger.info(
                f"Epoch {epoch}/{num_epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val IoU={val_iou:.4f}"
            )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(
                model.state_dict(), CONFIG["model_dir"] / "physics_unet_best.pth"
            )

    # Test evaluation
    model.load_state_dict(torch.load(CONFIG["model_dir"] / "physics_unet_best.pth"))
    model.eval()

    test_iou = 0
    with torch.no_grad():
        for features, masks in test_loader:
            features = features.to(device)
            masks = masks.to(device)
            outputs = model(features)
            test_iou += compute_iou(outputs, masks)

    test_iou /= len(test_loader)

    results = {
        "model": "PhysicsGuidedLightweightUNet",
        "params": {
            "num_parameters": n_params,
            "epochs": num_epochs,
            "image_size": CONFIG["image_size"],
        },
        "val": {
            "best_iou": best_val_iou,
        },
        "test": {
            "iou": test_iou,
        },
        "model_path": str(CONFIG["model_dir"] / "physics_unet_best.pth"),
        "training_time_seconds": time.time() - start_time,
    }

    logger.info(f"Physics U-Net Results:")
    logger.info(f"  Val IoU:  {best_val_iou:.4f}")
    logger.info(f"  Test IoU: {test_iou:.4f}")
    logger.info(f"  Training Time: {results['training_time_seconds']:.2f}s")

    return results


# =============================================================================
# MODEL 3: MULTI-TASK MODEL
# =============================================================================


class MultiTaskUNet(nn.Module):
    """Multi-task U-Net for water segmentation + edge detection + confidence."""

    def __init__(self, in_channels=7):
        super().__init__()

        # Shared encoder
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)

        # Shared decoder
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        # Task heads
        self.mask_head = nn.Conv2d(32, 1, 1)
        self.edge_head = nn.Conv2d(32, 1, 1)
        self.conf_head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Decoder
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return {
            "mask": self.mask_head(d1),
            "edge": self.edge_head(d1),
            "confidence": torch.sigmoid(self.conf_head(d1)),
        }


def compute_edges(masks: torch.Tensor) -> torch.Tensor:
    """Compute edge maps from masks using Sobel-like gradient."""
    # Pad
    padded = F.pad(masks, (1, 1, 1, 1), mode="replicate")

    # Gradient
    grad_x = padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]
    grad_y = padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]

    edges = (grad_x.abs() + grad_y.abs() > 0).float()
    return edges


def train_multitask(chips: List[np.ndarray], names: List[str]) -> Dict:
    """Train multi-task model."""
    logger.info("=" * 60)
    logger.info("TRAINING MODEL 3: Multi-Task U-Net")
    logger.info("=" * 60)

    start_time = time.time()
    device = CONFIG["device"]

    # Filter valid chips
    valid_chips = [c for c in chips if c.shape[0] >= 8]

    if len(valid_chips) < 10:
        return {"status": "failed", "error": "Not enough chips"}

    # Split
    train_chips, test_chips = train_test_split(
        valid_chips, test_size=0.2, random_state=CONFIG["random_seed"]
    )
    train_chips, val_chips = train_test_split(
        train_chips, test_size=0.2, random_state=CONFIG["random_seed"]
    )

    logger.info(
        f"Train: {len(train_chips)}, Val: {len(val_chips)}, Test: {len(test_chips)}"
    )

    # Datasets
    train_dataset = SARSegmentationDataset(train_chips, [], CONFIG["image_size"])
    val_dataset = SARSegmentationDataset(val_chips, [], CONFIG["image_size"])
    test_dataset = SARSegmentationDataset(test_chips, [], CONFIG["image_size"])

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0
    )

    # Model
    model = MultiTaskUNet(in_channels=7).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_iou = 0
    num_epochs = 100

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        for features, masks in train_loader:
            features = features.to(device)
            masks = masks.to(device)
            edges = compute_edges(masks)

            optimizer.zero_grad()
            outputs = model(features)

            mask_loss = bce(outputs["mask"], masks)
            edge_loss = bce(outputs["edge"], edges)

            loss = mask_loss + 0.5 * edge_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_iou = 0
        with torch.no_grad():
            for features, masks in val_loader:
                features = features.to(device)
                masks = masks.to(device)
                outputs = model(features)
                val_iou += compute_iou(outputs["mask"], masks)

        val_iou /= len(val_loader)

        if epoch % 20 == 0:
            logger.info(
                f"Epoch {epoch}/{num_epochs}: Train Loss={train_loss:.4f}, Val IoU={val_iou:.4f}"
            )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(
                model.state_dict(), CONFIG["model_dir"] / "multitask_unet_best.pth"
            )

    # Test
    model.load_state_dict(torch.load(CONFIG["model_dir"] / "multitask_unet_best.pth"))
    model.eval()

    test_iou = 0
    with torch.no_grad():
        for features, masks in test_loader:
            features = features.to(device)
            masks = masks.to(device)
            outputs = model(features)
            test_iou += compute_iou(outputs["mask"], masks)

    test_iou /= len(test_loader)

    results = {
        "model": "MultiTaskUNet",
        "params": {"num_parameters": n_params},
        "test": {"iou": test_iou},
        "val": {"best_iou": best_val_iou},
        "training_time_seconds": time.time() - start_time,
    }

    logger.info(f"Multi-Task Results:")
    logger.info(f"  Test IoU: {test_iou:.4f}")

    return results


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Run all training pipelines."""
    logger.info("=" * 80)
    logger.info("MASTER SAR WATER DETECTION TRAINING PIPELINE v2")
    logger.info("=" * 80)
    logger.info(f"Device: {CONFIG['device']}")
    logger.info(f"Started: {datetime.now().isoformat()}")

    # Load chips
    chips, names = load_all_chips(CONFIG["chip_dir"])

    if not chips:
        logger.error("No chips found!")
        return

    all_results = {}

    # Model 1: LightGBM Pixel-wise
    try:
        all_results["lightgbm_pixel"] = train_lightgbm_pixel(chips, names)
    except Exception as e:
        logger.error(f"LightGBM failed: {e}")
        import traceback

        traceback.print_exc()
        all_results["lightgbm_pixel"] = {"status": "failed", "error": str(e)}

    # Model 2: Physics U-Net
    try:
        all_results["physics_unet"] = train_physics_unet(chips, names)
    except Exception as e:
        logger.error(f"Physics U-Net failed: {e}")
        import traceback

        traceback.print_exc()
        all_results["physics_unet"] = {"status": "failed", "error": str(e)}

    # Model 3: Multi-Task
    try:
        all_results["multitask"] = train_multitask(chips, names)
    except Exception as e:
        logger.error(f"Multi-Task failed: {e}")
        import traceback

        traceback.print_exc()
        all_results["multitask"] = {"status": "failed", "error": str(e)}

    # Save results
    results_path = CONFIG["output_dir"] / "training_results_v2.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)

    # Summary
    logger.info("\nRESULTS SUMMARY:")
    for model_name, results in all_results.items():
        if "test" in results:
            logger.info(f"\n{model_name}:")
            for k, v in results["test"].items():
                if isinstance(v, float):
                    logger.info(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
