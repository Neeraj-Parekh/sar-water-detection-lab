#!/usr/bin/env python3
"""
Master SAR Water Detection Training Pipeline
=============================================
Trains multiple models on SAR chip data:
1. LightGBM baseline (fast, interpretable)
2. PySR Equation Search (symbolic regression)
3. Physics-Guided SegFormer
4. Multi-Task Model

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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import lightgbm as lgb
from scipy import ndimage
from skimage.feature import local_binary_pattern
import albumentations as A

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("master_training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "chip_dir": Path("/home/mit-aoe/sar_water_detection/chips"),
    "output_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    "model_dir": Path("/home/mit-aoe/sar_water_detection/models"),
    "random_seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 8,
    "num_workers": 4,
    # Band indices in 7-band chip
    "bands": {
        "VV": 0,
        "VH": 1,
        "MNDWI": 2,
        "DEM": 3,
        "HAND": 4,
        "SLOPE": 5,
        "TWI": 6,
    },
    # Training splits
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
}

# Create output directories
CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
CONFIG["model_dir"].mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================


def load_chip(filepath: Path) -> Optional[np.ndarray]:
    """Load a single chip file."""
    try:
        data = np.load(filepath, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            data = data["arr_0"]
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
    return chips, names


def extract_features_lightgbm(chip: np.ndarray) -> Dict[str, float]:
    """
    Extract physics-meaningful features from a chip for LightGBM.
    Returns a flat dictionary of features.
    """
    # Get individual bands (assuming 7-band format with truth as 8th)
    if chip.shape[0] >= 7:
        vv = chip[0]
        vh = chip[1]
        mndwi = chip[2]
        dem = chip[3]
        hand = chip[4]
        slope = chip[5]
        twi = chip[6]
    else:
        logger.warning(f"Chip has unexpected shape: {chip.shape}")
        return {}

    features = {}

    # Backscatter statistics
    features["vv_mean"] = np.nanmean(vv)
    features["vv_std"] = np.nanstd(vv)
    features["vv_min"] = np.nanmin(vv)
    features["vv_max"] = np.nanmax(vv)
    features["vv_p5"] = np.nanpercentile(vv, 5)
    features["vv_p25"] = np.nanpercentile(vv, 25)
    features["vv_p50"] = np.nanpercentile(vv, 50)
    features["vv_p75"] = np.nanpercentile(vv, 75)
    features["vv_p95"] = np.nanpercentile(vv, 95)

    features["vh_mean"] = np.nanmean(vh)
    features["vh_std"] = np.nanstd(vh)
    features["vh_min"] = np.nanmin(vh)
    features["vh_max"] = np.nanmax(vh)
    features["vh_p5"] = np.nanpercentile(vh, 5)
    features["vh_p25"] = np.nanpercentile(vh, 25)
    features["vh_p50"] = np.nanpercentile(vh, 50)
    features["vh_p75"] = np.nanpercentile(vh, 75)
    features["vh_p95"] = np.nanpercentile(vh, 95)

    # Polarization ratios (critical for water detection)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = vv / (vh + 1e-6)
        diff = vv - vh
    features["ratio_mean"] = np.nanmean(ratio)
    features["ratio_std"] = np.nanstd(ratio)
    features["diff_mean"] = np.nanmean(diff)
    features["diff_std"] = np.nanstd(diff)

    # Terrain features
    features["hand_mean"] = np.nanmean(hand)
    features["hand_std"] = np.nanstd(hand)
    features["hand_p25"] = np.nanpercentile(hand, 25)
    features["hand_p75"] = np.nanpercentile(hand, 75)
    features["hand_low_pct"] = np.nanmean(hand < 5) * 100  # % of pixels with HAND < 5m

    features["slope_mean"] = np.nanmean(slope)
    features["slope_std"] = np.nanstd(slope)
    features["slope_flat_pct"] = np.nanmean(slope < 5) * 100  # % of flat pixels

    features["twi_mean"] = np.nanmean(twi)
    features["twi_std"] = np.nanstd(twi)
    features["twi_high_pct"] = np.nanmean(twi > 10) * 100  # % of high TWI

    # MNDWI (optical reference)
    features["mndwi_mean"] = np.nanmean(mndwi)
    features["mndwi_std"] = np.nanstd(mndwi)
    features["mndwi_pos_pct"] = np.nanmean(mndwi > 0) * 100  # % positive MNDWI

    # Texture features (local variance)
    try:
        vv_local_var = ndimage.generic_filter(vv.astype(float), np.var, size=3)
        vh_local_var = ndimage.generic_filter(vh.astype(float), np.var, size=3)
        features["vv_local_var_mean"] = np.nanmean(vv_local_var)
        features["vh_local_var_mean"] = np.nanmean(vh_local_var)
    except:
        features["vv_local_var_mean"] = 0
        features["vh_local_var_mean"] = 0

    # Histogram shape (kurtosis indicator for water presence)
    try:
        from scipy.stats import kurtosis, skew

        features["vv_kurtosis"] = kurtosis(vv.flatten(), nan_policy="omit")
        features["vh_kurtosis"] = kurtosis(vh.flatten(), nan_policy="omit")
        features["vv_skew"] = skew(vv.flatten(), nan_policy="omit")
        features["vh_skew"] = skew(vh.flatten(), nan_policy="omit")
    except:
        features["vv_kurtosis"] = 0
        features["vh_kurtosis"] = 0
        features["vv_skew"] = 0
        features["vh_skew"] = 0

    return features


def get_chip_label(chip: np.ndarray) -> int:
    """
    Get label for chip (binary: has significant water or not).
    Uses MNDWI as reference for now.
    """
    if chip.shape[0] >= 8:
        # If 8th band exists, use it as truth
        truth = chip[7]
        water_pct = np.nanmean(truth > 0.5) * 100
    else:
        # Use MNDWI as proxy
        mndwi = chip[2]
        water_pct = np.nanmean(mndwi > 0) * 100

    return 1 if water_pct > 5 else 0  # >5% water = positive class


# =============================================================================
# MODEL 1: LIGHTGBM BASELINE
# =============================================================================


def train_lightgbm(chips: List[np.ndarray], names: List[str]) -> Dict:
    """
    Train LightGBM baseline model for chip classification.
    Returns training results dictionary.
    """
    logger.info("=" * 60)
    logger.info("TRAINING MODEL 1: LightGBM Baseline")
    logger.info("=" * 60)

    start_time = time.time()

    # Extract features and labels
    X_data = []
    y_data = []
    valid_names = []

    for chip, name in zip(chips, names):
        features = extract_features_lightgbm(chip)
        if features:
            X_data.append(features)
            y_data.append(get_chip_label(chip))
            valid_names.append(name)

    # Convert to arrays
    feature_names = list(X_data[0].keys())
    X = np.array([[f[k] for k in feature_names] for f in X_data])
    y = np.array(y_data)

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Class distribution: {np.bincount(y)}")

    # Train/val/test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=CONFIG["test_ratio"],
        random_state=CONFIG["random_seed"],
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=CONFIG["val_ratio"] / (1 - CONFIG["test_ratio"]),
        random_state=CONFIG["random_seed"],
        stratify=y_trainval,
    )

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train LightGBM
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        num_leaves=31,
        learning_rate=0.1,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=CONFIG["random_seed"],
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )

    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    results = {
        "model": "LightGBM",
        "params": {
            "n_estimators": model.n_estimators_,
            "max_depth": 6,
            "num_leaves": 31,
            "features": feature_names,
        },
        "train": {
            "accuracy": accuracy_score(y_train, y_pred_train),
            "f1": f1_score(y_train, y_pred_train, zero_division=0),
            "precision": precision_score(y_train, y_pred_train, zero_division=0),
            "recall": recall_score(y_train, y_pred_train, zero_division=0),
        },
        "val": {
            "accuracy": accuracy_score(y_val, y_pred_val),
            "f1": f1_score(y_val, y_pred_val, zero_division=0),
            "precision": precision_score(y_val, y_pred_val, zero_division=0),
            "recall": recall_score(y_val, y_pred_val, zero_division=0),
        },
        "test": {
            "accuracy": accuracy_score(y_test, y_pred_test),
            "f1": f1_score(y_test, y_pred_test, zero_division=0),
            "precision": precision_score(y_test, y_pred_test, zero_division=0),
            "recall": recall_score(y_test, y_pred_test, zero_division=0),
        },
        "feature_importance": dict(
            zip(feature_names, model.feature_importances_.tolist())
        ),
        "training_time_seconds": time.time() - start_time,
    }

    # Save model
    model_path = CONFIG["model_dir"] / "lightgbm_baseline.txt"
    model.booster_.save_model(str(model_path))
    results["model_path"] = str(model_path)

    # Log results
    logger.info(f"LightGBM Results:")
    logger.info(f"  Train Accuracy: {results['train']['accuracy']:.4f}")
    logger.info(f"  Val Accuracy:   {results['val']['accuracy']:.4f}")
    logger.info(f"  Test Accuracy:  {results['test']['accuracy']:.4f}")
    logger.info(f"  Test F1:        {results['test']['f1']:.4f}")
    logger.info(f"  Training Time:  {results['training_time_seconds']:.2f}s")

    # Log top features
    sorted_features = sorted(
        results["feature_importance"].items(), key=lambda x: x[1], reverse=True
    )
    logger.info("Top 10 Features:")
    for feat, imp in sorted_features[:10]:
        logger.info(f"  {feat}: {imp:.4f}")

    return results


# =============================================================================
# MODEL 2: PYSR EQUATION SEARCH
# =============================================================================


def prepare_pysr_data(chips: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for symbolic regression.
    Returns X (features) and y (water percentage).
    """
    X_data = []
    y_data = []

    for chip in chips:
        if chip.shape[0] < 7:
            continue

        vv = chip[0]
        vh = chip[1]
        mndwi = chip[2]
        hand = chip[4]
        slope = chip[5]
        twi = chip[6]

        # Sample pixels (too many for full symbolic regression)
        h, w = vv.shape
        sample_idx = np.random.choice(h * w, size=min(1000, h * w), replace=False)

        for idx in sample_idx:
            i, j = idx // w, idx % w

            X_data.append(
                [
                    vv[i, j],
                    vh[i, j],
                    vv[i, j] - vh[i, j],
                    vv[i, j] / (vh[i, j] + 1e-6),
                    hand[i, j],
                    slope[i, j],
                    twi[i, j],
                ]
            )

            # Target: is this pixel likely water?
            y_data.append(1 if mndwi[i, j] > 0.1 else 0)

    return np.array(X_data), np.array(y_data)


def run_equation_search(chips: List[np.ndarray]) -> Dict:
    """
    Run PySR symbolic regression to discover water detection equations.
    """
    logger.info("=" * 60)
    logger.info("TRAINING MODEL 2: PySR Equation Search")
    logger.info("=" * 60)

    start_time = time.time()

    try:
        from pysr import PySRRegressor
    except ImportError:
        logger.warning("PySR not installed. Skipping equation search.")
        return {"status": "skipped", "reason": "PySR not installed"}

    # Prepare data
    logger.info("Preparing data for symbolic regression...")
    X, y = prepare_pysr_data(chips)

    # Handle NaN/Inf
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]

    logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")
    logger.info(f"Class balance: {np.mean(y):.2%} positive")

    # Subsample for speed (if too many samples)
    if len(X) > 50000:
        idx = np.random.choice(len(X), 50000, replace=False)
        X = X[idx]
        y = y[idx]
        logger.info(f"Subsampled to {len(X)} samples")

    # Configure PySR
    feature_names = ["VV", "VH", "VV_minus_VH", "VV_div_VH", "HAND", "SLOPE", "TWI"]

    model = PySRRegressor(
        niterations=200,  # Reduced for faster testing
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["abs", "sqrt"],
        constraints={
            "/": (-1, 1),
        },
        maxsize=20,
        parsimony=0.01,
        populations=20,
        population_size=50,
        progress=True,
        verbosity=1,
        timeout_in_seconds=3600,  # 1 hour timeout for this run
        temp_equation_file=True,
        random_state=CONFIG["random_seed"],
    )

    logger.info("Starting equation search (this may take a while)...")
    model.fit(X, y, variable_names=feature_names)

    # Extract results
    equations = []
    for i, eq in enumerate(model.equations_):
        equations.append(
            {
                "equation": str(eq.equation),
                "complexity": int(eq.complexity),
                "loss": float(eq.loss),
                "score": float(eq.score) if hasattr(eq, "score") else 0,
            }
        )

    results = {
        "model": "PySR_EquationSearch",
        "equations": equations,
        "best_equation": str(model.sympy())
        if hasattr(model, "sympy")
        else equations[-1]["equation"]
        if equations
        else "None",
        "feature_names": feature_names,
        "n_samples": len(X),
        "training_time_seconds": time.time() - start_time,
    }

    logger.info(f"Found {len(equations)} equations")
    if equations:
        logger.info(f"Best equation: {results['best_equation']}")

    return results


# =============================================================================
# MODEL 3: PHYSICS-GUIDED SEGFORMER
# =============================================================================


class SARDataset(Dataset):
    """PyTorch Dataset for SAR chips."""

    def __init__(self, chips: List[np.ndarray], names: List[str], transform=None):
        self.chips = chips
        self.names = names
        self.transform = transform

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, idx):
        chip = self.chips[idx].copy()

        # Separate features and mask
        if chip.shape[0] >= 8:
            features = chip[:7]  # First 7 bands
            mask = chip[7]  # 8th band is truth
        else:
            features = chip[:7]
            # Use MNDWI as proxy mask
            mask = (chip[2] > 0).astype(np.float32)

        # Normalize features
        features = self._normalize(features)

        # Convert to tensors
        features = torch.from_numpy(features).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return features, mask

    def _normalize(self, features):
        """Normalize each band to [0, 1] range."""
        normalized = np.zeros_like(features, dtype=np.float32)

        # VV, VH: typically -30 to 0 dB
        normalized[0] = np.clip((features[0] + 30) / 30, 0, 1)
        normalized[1] = np.clip((features[1] + 30) / 30, 0, 1)

        # MNDWI: -1 to 1
        normalized[2] = np.clip((features[2] + 1) / 2, 0, 1)

        # DEM: 0 to 5000m
        normalized[3] = np.clip(features[3] / 1000, 0, 5)

        # HAND: 0 to 50m
        normalized[4] = np.clip(features[4] / 50, 0, 1)

        # Slope: 0 to 90 degrees
        normalized[5] = np.clip(features[5] / 90, 0, 1)

        # TWI: 0 to 20
        normalized[6] = np.clip(features[6] / 20, 0, 1)

        return normalized


class PhysicsGuidedLoss(nn.Module):
    """
    Physics-guided loss function for water segmentation.
    Combines BCE with physical constraints.
    """

    def __init__(self, alpha=0.3, beta=0.2, gamma=0.1):
        super().__init__()
        self.alpha = alpha  # HAND constraint weight
        self.beta = beta  # Slope constraint weight
        self.gamma = gamma  # Backscatter constraint weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target, features):
        """
        pred: (B, 1, H, W) logits
        target: (B, 1, H, W) binary mask
        features: (B, 7, H, W) normalized input features
        """
        # Standard BCE loss
        bce_loss = self.bce(pred, target)

        # Get probabilities
        probs = torch.sigmoid(pred)

        # Physics constraint 1: HAND correlation
        # Water should be at low elevation (low HAND)
        hand = features[:, 4:5, :, :]  # Normalized HAND
        hand_loss = (probs * hand).mean()  # Penalize water at high HAND

        # Physics constraint 2: Slope exclusion
        # No water on steep slopes (> ~17 degrees = 0.19 normalized)
        slope = features[:, 5:6, :, :]
        steep_mask = (slope > 0.17).float()
        slope_loss = (probs * steep_mask).mean()

        # Physics constraint 3: Backscatter
        # Water has low VH backscatter (< -18 dB = 0.4 normalized)
        vh = features[:, 1:2, :, :]
        bright_mask = (vh > 0.5).float()  # > -15 dB
        backscatter_loss = (probs * bright_mask).mean()

        # Combined loss
        total_loss = (
            bce_loss
            + self.alpha * hand_loss
            + self.beta * slope_loss
            + self.gamma * backscatter_loss
        )

        return total_loss, {
            "bce": bce_loss.item(),
            "hand": hand_loss.item(),
            "slope": slope_loss.item(),
            "backscatter": backscatter_loss.item(),
        }


class LightweightSegformer(nn.Module):
    """
    Lightweight segmentation model inspired by SegFormer-B0.
    Uses MobileNetV3 backbone + simple decoder.
    """

    def __init__(self, in_channels=7, num_classes=1):
        super().__init__()

        # Encoder (simple ConvNet for speed)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Final classifier
        self.classifier = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        # Store input size
        input_size = x.shape[-2:]

        # Encode
        features = self.encoder(x)

        # Decode
        decoded = self.decoder(features)

        # Classify
        out = self.classifier(decoded)

        # Resize to input size if needed
        if out.shape[-2:] != input_size:
            out = F.interpolate(
                out, size=input_size, mode="bilinear", align_corners=False
            )

        return out


def train_physics_segformer(chips: List[np.ndarray], names: List[str]) -> Dict:
    """
    Train physics-guided lightweight SegFormer.
    """
    logger.info("=" * 60)
    logger.info("TRAINING MODEL 3: Physics-Guided SegFormer")
    logger.info("=" * 60)

    start_time = time.time()
    device = CONFIG["device"]

    # Split data
    train_chips, test_chips, train_names, test_names = train_test_split(
        chips, names, test_size=CONFIG["test_ratio"], random_state=CONFIG["random_seed"]
    )
    train_chips, val_chips, train_names, val_names = train_test_split(
        train_chips,
        train_names,
        test_size=CONFIG["val_ratio"] / (1 - CONFIG["test_ratio"]),
        random_state=CONFIG["random_seed"],
    )

    logger.info(
        f"Train: {len(train_chips)}, Val: {len(val_chips)}, Test: {len(test_chips)}"
    )

    # Create datasets
    train_dataset = SARDataset(train_chips, train_names)
    val_dataset = SARDataset(val_chips, val_names)
    test_dataset = SARDataset(test_chips, test_names)

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
    model = LightweightSegformer(in_channels=7, num_classes=1).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = PhysicsGuidedLoss(alpha=0.3, beta=0.2, gamma=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_iou": []}

    num_epochs = 50

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

                # Compute IoU
                preds = (torch.sigmoid(outputs) > 0.5).float()
                intersection = (preds * masks).sum()
                union = (preds + masks).clamp(0, 1).sum()
                val_iou += (intersection / (union + 1e-6)).item()

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        if epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch}/{num_epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val IoU={val_iou:.4f}"
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(), CONFIG["model_dir"] / "physics_segformer_best.pth"
            )

    # Test evaluation
    model.load_state_dict(
        torch.load(CONFIG["model_dir"] / "physics_segformer_best.pth")
    )
    model.eval()

    test_iou = 0
    test_f1 = 0
    with torch.no_grad():
        for features, masks in test_loader:
            features = features.to(device)
            masks = masks.to(device)

            outputs = model(features)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # IoU
            intersection = (preds * masks).sum()
            union = (preds + masks).clamp(0, 1).sum()
            test_iou += (intersection / (union + 1e-6)).item()

            # F1
            tp = (preds * masks).sum()
            fp = (preds * (1 - masks)).sum()
            fn = ((1 - preds) * masks).sum()
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            test_f1 += (2 * precision * recall / (precision + recall + 1e-6)).item()

    test_iou /= len(test_loader)
    test_f1 /= len(test_loader)

    results = {
        "model": "PhysicsGuidedSegFormer",
        "params": {
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "epochs": num_epochs,
            "batch_size": CONFIG["batch_size"],
            "physics_weights": {"alpha": 0.3, "beta": 0.2, "gamma": 0.1},
        },
        "val": {
            "best_loss": best_val_loss,
            "best_iou": max(history["val_iou"]),
        },
        "test": {
            "iou": test_iou,
            "f1": test_f1,
        },
        "history": history,
        "model_path": str(CONFIG["model_dir"] / "physics_segformer_best.pth"),
        "training_time_seconds": time.time() - start_time,
    }

    logger.info(f"Physics-Guided SegFormer Results:")
    logger.info(f"  Test IoU: {test_iou:.4f}")
    logger.info(f"  Test F1:  {test_f1:.4f}")
    logger.info(f"  Training Time: {results['training_time_seconds']:.2f}s")

    return results


# =============================================================================
# MODEL 4: MULTI-TASK MODEL
# =============================================================================


class MultiTaskModel(nn.Module):
    """
    Multi-task model for:
    1. Water segmentation mask
    2. Edge detection
    3. Water type classification
    4. Confidence estimation
    """

    def __init__(self, in_channels=7, num_water_types=5):
        super().__init__()

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Shared decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Task-specific heads
        self.mask_head = nn.Conv2d(16, 1, 1)  # Water mask
        self.edge_head = nn.Conv2d(16, 1, 1)  # Edge map
        self.confidence_head = nn.Conv2d(16, 1, 1)  # Confidence

        # Global pooling + classification head for water type
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.type_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_water_types),
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        # Encode
        features = self.encoder(x)

        # Decode
        decoded = self.decoder(features)

        # Resize to input if needed
        if decoded.shape[-2:] != input_size:
            decoded = F.interpolate(
                decoded, size=input_size, mode="bilinear", align_corners=False
            )

        # Task outputs
        mask = self.mask_head(decoded)
        edge = self.edge_head(decoded)
        confidence = self.confidence_head(decoded)

        # Water type (global classification)
        pooled = self.global_pool(decoded)
        water_type = self.type_head(pooled)

        return {
            "mask": mask,
            "edge": edge,
            "confidence": confidence,
            "water_type": water_type,
        }


def train_multitask_model(chips: List[np.ndarray], names: List[str]) -> Dict:
    """
    Train multi-task model.
    """
    logger.info("=" * 60)
    logger.info("TRAINING MODEL 4: Multi-Task Model")
    logger.info("=" * 60)

    start_time = time.time()
    device = CONFIG["device"]

    # Split data (same as segformer)
    train_chips, test_chips, train_names, test_names = train_test_split(
        chips, names, test_size=CONFIG["test_ratio"], random_state=CONFIG["random_seed"]
    )
    train_chips, val_chips, train_names, val_names = train_test_split(
        train_chips,
        train_names,
        test_size=CONFIG["val_ratio"] / (1 - CONFIG["test_ratio"]),
        random_state=CONFIG["random_seed"],
    )

    logger.info(
        f"Train: {len(train_chips)}, Val: {len(val_chips)}, Test: {len(test_chips)}"
    )

    # Create datasets
    train_dataset = SARDataset(train_chips, train_names)
    val_dataset = SARDataset(val_chips, val_names)
    test_dataset = SARDataset(test_chips, test_names)

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
    model = MultiTaskModel(in_channels=7, num_water_types=5).to(device)
    logger.info(
        f"Multi-Task Model parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    # Loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_iou": []}
    num_epochs = 50

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        for features, masks in train_loader:
            features = features.to(device)
            masks = masks.to(device)

            # Create edge target using Sobel-like gradient
            with torch.no_grad():
                # Simple edge detection
                masks_padded = F.pad(masks, (1, 1, 1, 1), mode="replicate")
                grad_x = masks_padded[:, :, 1:-1, 2:] - masks_padded[:, :, 1:-1, :-2]
                grad_y = masks_padded[:, :, 2:, 1:-1] - masks_padded[:, :, :-2, 1:-1]
                edges = (grad_x.abs() + grad_y.abs() > 0).float()

            # Dummy water type labels (based on water percentage)
            water_pct = masks.mean(dim=(1, 2, 3))
            water_types = torch.zeros(len(water_pct), dtype=torch.long, device=device)
            water_types[water_pct > 0.3] = 1  # Large lake
            water_types[(water_pct > 0.1) & (water_pct <= 0.3)] = 2  # River
            water_types[(water_pct > 0.05) & (water_pct <= 0.1)] = 3  # Wetland
            water_types[(water_pct > 0) & (water_pct <= 0.05)] = 4  # Stream

            optimizer.zero_grad()
            outputs = model(features)

            # Multi-task loss
            mask_loss = bce_loss(outputs["mask"], masks)
            edge_loss = bce_loss(outputs["edge"], edges)
            conf_loss = bce_loss(outputs["confidence"], masks)  # Confidence = certainty
            type_loss = ce_loss(outputs["water_type"], water_types)

            total_loss = mask_loss + 0.5 * edge_loss + 0.3 * conf_loss + 0.2 * type_loss
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

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
                loss = bce_loss(outputs["mask"], masks)
                val_loss += loss.item()

                # IoU
                preds = (torch.sigmoid(outputs["mask"]) > 0.5).float()
                intersection = (preds * masks).sum()
                union = (preds + masks).clamp(0, 1).sum()
                val_iou += (intersection / (union + 1e-6)).item()

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        if epoch % 10 == 0:
            logger.info(
                f"Epoch {epoch}/{num_epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val IoU={val_iou:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(), CONFIG["model_dir"] / "multitask_model_best.pth"
            )

    # Test evaluation
    model.load_state_dict(torch.load(CONFIG["model_dir"] / "multitask_model_best.pth"))
    model.eval()

    test_iou = 0
    test_edge_acc = 0
    with torch.no_grad():
        for features, masks in test_loader:
            features = features.to(device)
            masks = masks.to(device)

            outputs = model(features)
            preds = (torch.sigmoid(outputs["mask"]) > 0.5).float()

            # IoU
            intersection = (preds * masks).sum()
            union = (preds + masks).clamp(0, 1).sum()
            test_iou += (intersection / (union + 1e-6)).item()

    test_iou /= len(test_loader)

    results = {
        "model": "MultiTaskModel",
        "params": {
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "epochs": num_epochs,
            "batch_size": CONFIG["batch_size"],
            "tasks": ["mask", "edge", "confidence", "water_type"],
        },
        "val": {
            "best_loss": best_val_loss,
            "best_iou": max(history["val_iou"]),
        },
        "test": {
            "iou": test_iou,
        },
        "history": history,
        "model_path": str(CONFIG["model_dir"] / "multitask_model_best.pth"),
        "training_time_seconds": time.time() - start_time,
    }

    logger.info(f"Multi-Task Model Results:")
    logger.info(f"  Test IoU: {test_iou:.4f}")
    logger.info(f"  Training Time: {results['training_time_seconds']:.2f}s")

    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Run all training pipelines."""
    logger.info("=" * 80)
    logger.info("MASTER SAR WATER DETECTION TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Device: {CONFIG['device']}")
    logger.info(f"Chip directory: {CONFIG['chip_dir']}")
    logger.info(f"Output directory: {CONFIG['output_dir']}")
    logger.info("=" * 80)

    # Load all chips
    chips, names = load_all_chips(CONFIG["chip_dir"])

    if len(chips) == 0:
        logger.error("No chips found! Exiting.")
        return

    all_results = {}

    # Model 1: LightGBM Baseline
    try:
        lightgbm_results = train_lightgbm(chips, names)
        all_results["lightgbm"] = lightgbm_results
    except Exception as e:
        logger.error(f"LightGBM training failed: {e}")
        all_results["lightgbm"] = {"status": "failed", "error": str(e)}

    # Model 2: PySR Equation Search (optional, can be slow)
    # Uncomment to run:
    # try:
    #     pysr_results = run_equation_search(chips)
    #     all_results['pysr'] = pysr_results
    # except Exception as e:
    #     logger.error(f"PySR search failed: {e}")
    #     all_results['pysr'] = {'status': 'failed', 'error': str(e)}

    # Model 3: Physics-Guided SegFormer
    try:
        segformer_results = train_physics_segformer(chips, names)
        all_results["physics_segformer"] = segformer_results
    except Exception as e:
        logger.error(f"Physics-Guided SegFormer training failed: {e}")
        all_results["physics_segformer"] = {"status": "failed", "error": str(e)}

    # Model 4: Multi-Task Model
    try:
        multitask_results = train_multitask_model(chips, names)
        all_results["multitask"] = multitask_results
    except Exception as e:
        logger.error(f"Multi-Task training failed: {e}")
        all_results["multitask"] = {"status": "failed", "error": str(e)}

    # Save all results
    results_path = CONFIG["output_dir"] / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {results_path}")

    # Print summary
    logger.info("\n" + "=" * 40)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 40)

    for model_name, results in all_results.items():
        if isinstance(results, dict) and "test" in results:
            logger.info(f"\n{model_name}:")
            for metric, value in results.get("test", {}).items():
                if isinstance(value, float):
                    logger.info(f"  {metric}: {value:.4f}")
                else:
                    logger.info(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
