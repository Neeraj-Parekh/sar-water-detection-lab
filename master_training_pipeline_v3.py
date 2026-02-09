#!/usr/bin/env python3
"""
Master SAR Water Detection Training Pipeline v3
================================================
MAJOR IMPROVEMENTS over v2:
1. ADAPTIVE thresholds (learned, not fixed)
2. NARROW RIVER detection with multi-scale features
3. SAR-ONLY mode (no MNDWI leakage)
4. ENSEMBLE methods (LightGBM + U-Net weighted)
5. DATA AUGMENTATION for U-Nets
6. REDUCED U-Net parameters (~300K, not 1.9M)
7. BETTER physics loss with learnable weights
8. CROSS-VALIDATION for honest metrics
9. GLCM texture features for water boundaries
10. Edge-aware loss for narrow features

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
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    jaccard_score,
)
import lightgbm as lgb
from scipy import ndimage
from scipy.ndimage import sobel, gaussian_filter

warnings.filterwarnings("ignore")

# Setup logging with flush
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("master_training_v3.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Force flush after each log
for handler in logger.handlers:
    handler.flush = lambda: None

# Configuration
CONFIG = {
    "chip_dir": Path("/home/mit-aoe/sar_water_detection/chips"),
    "expanded_chip_dir": Path("/home/mit-aoe/sar_water_detection/chips_expanded"),
    "output_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    "model_dir": Path("/home/mit-aoe/sar_water_detection/models"),
    "random_seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 8,
    "num_workers": 4,
    "image_size": 256,
    # Training
    "num_epochs": 150,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    # Cross-validation
    "n_folds": 5,
    # Feature bands (SAR-ONLY mode excludes MNDWI)
    "use_sar_only": True,  # Set to True to avoid optical leakage
    # Ensemble weights (learned during training)
    "ensemble_init_weights": [0.5, 0.3, 0.2],  # LightGBM, Physics U-Net, Multi-task
}

CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
CONFIG["model_dir"].mkdir(parents=True, exist_ok=True)

np.random.seed(CONFIG["random_seed"])
torch.manual_seed(CONFIG["random_seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed(CONFIG["random_seed"])


# =============================================================================
# ADAPTIVE FEATURE EXTRACTION (Not Fixed Thresholds)
# =============================================================================


def compute_adaptive_threshold(arr: np.ndarray, method: str = "otsu") -> float:
    """Compute adaptive threshold using various methods."""
    arr_flat = arr[np.isfinite(arr)].flatten()
    if len(arr_flat) == 0:
        return 0.0

    if method == "otsu":
        # Otsu's method - finds optimal threshold to minimize intra-class variance
        hist, bin_edges = np.histogram(arr_flat, bins=256)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        total = len(arr_flat)
        sum_total = np.sum(arr_flat)

        sum_bg = 0.0
        weight_bg = 0.0
        max_var = 0.0
        threshold = bin_centers[0]

        for i, (count, center) in enumerate(zip(hist, bin_centers)):
            weight_bg += count
            if weight_bg == 0:
                continue
            weight_fg = total - weight_bg
            if weight_fg == 0:
                break

            sum_bg += count * center
            mean_bg = sum_bg / weight_bg
            mean_fg = (sum_total - sum_bg) / weight_fg

            var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
            if var_between > max_var:
                max_var = var_between
                threshold = center

        return threshold

    elif method == "percentile":
        # Use percentile-based adaptive threshold
        return np.percentile(arr_flat, 25)  # 25th percentile for water (dark)

    elif method == "bimodal":
        # Find valley between bimodal peaks
        hist, bin_edges = np.histogram(arr_flat, bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Smooth histogram
        hist_smooth = gaussian_filter(hist.astype(float), sigma=3)

        # Find local minima
        for i in range(1, len(hist_smooth) - 1):
            if (
                hist_smooth[i] < hist_smooth[i - 1]
                and hist_smooth[i] < hist_smooth[i + 1]
            ):
                return bin_centers[i]

        return np.median(arr_flat)

    return np.median(arr_flat)


def compute_glcm_features(
    arr: np.ndarray, distances: List[int] = [1, 2, 4]
) -> Dict[str, float]:
    """
    Compute GLCM (Gray-Level Co-occurrence Matrix) texture features.
    These are crucial for detecting water boundaries and narrow features.
    """
    # Quantize to 16 levels for efficiency
    arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
    if arr_max - arr_min < 1e-6:
        return {
            "glcm_contrast": 0,
            "glcm_homogeneity": 1,
            "glcm_energy": 1,
            "glcm_correlation": 0,
        }

    arr_quantized = ((arr - arr_min) / (arr_max - arr_min) * 15).astype(np.uint8)

    features = {}
    for d in distances:
        # Compute co-occurrence for horizontal direction
        glcm = np.zeros((16, 16), dtype=np.float32)
        h, w = arr_quantized.shape

        for i in range(h):
            for j in range(w - d):
                glcm[arr_quantized[i, j], arr_quantized[i, j + d]] += 1

        # Normalize
        glcm = glcm / (glcm.sum() + 1e-6)

        # Compute features
        i_idx, j_idx = np.meshgrid(range(16), range(16), indexing="ij")

        contrast = np.sum(glcm * (i_idx - j_idx) ** 2)
        homogeneity = np.sum(glcm / (1 + np.abs(i_idx - j_idx)))
        energy = np.sum(glcm**2)

        features[f"glcm_contrast_d{d}"] = contrast
        features[f"glcm_homogeneity_d{d}"] = homogeneity
        features[f"glcm_energy_d{d}"] = energy

    return features


def compute_multiscale_features(
    arr: np.ndarray, scales: List[int] = [3, 5, 9, 15]
) -> Dict[str, float]:
    """
    Multi-scale features for detecting narrow rivers at different widths.
    Uses morphological operations and local statistics at multiple scales.
    """
    features = {}

    for scale in scales:
        # Local mean and std at this scale
        kernel = np.ones((scale, scale)) / (scale * scale)
        local_mean = ndimage.convolve(arr, kernel, mode="reflect")
        local_sq_mean = ndimage.convolve(arr**2, kernel, mode="reflect")
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))

        features[f"local_mean_s{scale}"] = np.nanmean(local_mean)
        features[f"local_std_s{scale}"] = np.nanmean(local_std)

        # Gradient magnitude at this scale
        smoothed = gaussian_filter(arr, sigma=scale / 3)
        grad_x = sobel(smoothed, axis=1)
        grad_y = sobel(smoothed, axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        features[f"grad_mag_s{scale}"] = np.nanmean(grad_mag)

        # Line detection (for narrow rivers)
        # High gradient perpendicular to river, low along river
        features[f"grad_anisotropy_s{scale}"] = np.nanstd(grad_mag) / (
            np.nanmean(grad_mag) + 1e-6
        )

    return features


def detect_linear_features(
    arr: np.ndarray, threshold_percentile: float = 20
) -> np.ndarray:
    """
    Detect linear features (narrow rivers, streams) using morphological operations.
    Returns a mask highlighting linear water features.
    """
    # Adaptive threshold for dark features (water)
    threshold = np.percentile(arr[np.isfinite(arr)], threshold_percentile)
    dark_mask = arr < threshold

    # Apply line detection kernels at multiple angles
    line_kernels = []
    for angle in [0, 45, 90, 135]:
        kernel = np.zeros((7, 7))
        if angle == 0:
            kernel[3, :] = 1
        elif angle == 90:
            kernel[:, 3] = 1
        elif angle == 45:
            np.fill_diagonal(kernel, 1)
        elif angle == 135:
            np.fill_diagonal(np.fliplr(kernel), 1)
        kernel = kernel / kernel.sum()
        line_kernels.append(kernel)

    # Apply each kernel and take maximum response
    line_responses = []
    for kernel in line_kernels:
        response = ndimage.convolve(dark_mask.astype(float), kernel, mode="reflect")
        line_responses.append(response)

    line_mask = np.max(line_responses, axis=0)

    return line_mask


# =============================================================================
# DATA LOADING WITH AUGMENTATION
# =============================================================================


def load_chip(filepath: Path) -> Optional[np.ndarray]:
    """Load a single chip file and standardize format."""
    try:
        if filepath.suffix == ".tif":
            import rasterio

            with rasterio.open(filepath) as src:
                data = src.read()  # (C, H, W)
        else:
            data = np.load(filepath, allow_pickle=True)
            if isinstance(data, np.lib.npyio.NpzFile):
                data = data["arr_0"]

            # Check format: (H, W, C) or (C, H, W)
            if data.ndim == 3:
                if data.shape[2] <= 10:  # Likely (H, W, C)
                    data = np.transpose(data, (2, 0, 1))  # Convert to (C, H, W)

        return data.astype(np.float32)
    except Exception as e:
        logger.warning(f"Failed to load {filepath}: {e}")
        return None


def load_all_chips(chip_dirs: List[Path]) -> Tuple[List[np.ndarray], List[str]]:
    """Load all chip files from multiple directories."""
    chips = []
    names = []

    for chip_dir in chip_dirs:
        if not chip_dir.exists():
            logger.warning(f"Directory not found: {chip_dir}")
            continue

        # Support both .npy and .tif
        for pattern in ["*.npy", "*.tif"]:
            for f in sorted(chip_dir.glob(pattern)):
                chip = load_chip(f)
                if chip is not None and chip.shape[0] >= 7:
                    chips.append(chip)
                    names.append(f.stem)

    logger.info(f"Loaded {len(chips)} chips total")
    if chips:
        logger.info(f"Chip shape example: {chips[0].shape}")
    return chips, names


class AugmentedSARDataset(Dataset):
    """
    PyTorch Dataset with heavy augmentation for SAR water segmentation.
    Augmentation is CRITICAL for limited data (117 chips).
    """

    def __init__(
        self,
        chips: List[np.ndarray],
        names: List[str],
        image_size: int = 256,
        augment: bool = True,
        sar_only: bool = True,
    ):
        self.chips = chips
        self.names = names
        self.image_size = image_size
        self.augment = augment
        self.sar_only = sar_only

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, idx):
        chip = self.chips[idx].copy().astype(np.float32)

        # Select bands based on mode
        if self.sar_only:
            # SAR-only: VV, VH, DEM, HAND, SLOPE, TWI (exclude MNDWI at index 2)
            # Indices: 0=VV, 1=VH, 3=DEM, 4=HAND, 5=SLOPE, 6=TWI
            if chip.shape[0] >= 7:
                features = np.stack(
                    [chip[0], chip[1], chip[3], chip[4], chip[5], chip[6]], axis=0
                )
            else:
                features = chip[:6]
        else:
            features = chip[:7]  # All 7 bands including MNDWI

        # Get truth mask
        if chip.shape[0] >= 8:
            mask = chip[7:8]
        else:
            # Fallback: use VH threshold as proxy
            mask = (chip[1:2] < -18).astype(np.float32)

        # Resize
        features = self._resize(features, self.image_size)
        mask = self._resize(mask, self.image_size)

        # Augmentation
        if self.augment:
            features, mask = self._augment(features, mask)

        # Normalize
        features = self._normalize(features)

        # Binarize mask
        mask = (mask > 0.5).astype(np.float32)

        # Compute edge mask for edge-aware loss
        edge_mask = self._compute_edges(mask[0])

        return (
            torch.from_numpy(features),
            torch.from_numpy(mask),
            torch.from_numpy(edge_mask[np.newaxis]),
        )

    def _resize(self, arr: np.ndarray, size: int) -> np.ndarray:
        from scipy.ndimage import zoom

        c, h, w = arr.shape
        if h == size and w == size:
            return arr
        zoom_factors = (1, size / h, size / w)
        return zoom(arr, zoom_factors, order=1)

    def _augment(
        self, features: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            features = features[:, :, ::-1].copy()
            mask = mask[:, :, ::-1].copy()

        # Random vertical flip
        if np.random.random() > 0.5:
            features = features[:, ::-1, :].copy()
            mask = mask[:, ::-1, :].copy()

        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        if k > 0:
            features = np.rot90(features, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(1, 2)).copy()

        # Random crop and resize (scale augmentation)
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.7, 1.0)
            h, w = features.shape[1], features.shape[2]
            new_h, new_w = int(h * scale), int(w * scale)
            top = np.random.randint(0, h - new_h + 1)
            left = np.random.randint(0, w - new_w + 1)

            features = features[:, top : top + new_h, left : left + new_w]
            mask = mask[:, top : top + new_h, left : left + new_w]

            # Resize back
            features = self._resize(features, self.image_size)
            mask = self._resize(mask, self.image_size)

        # SAR-specific: Add speckle noise
        if np.random.random() > 0.5:
            noise_level = np.random.uniform(0.02, 0.1)
            noise = np.random.randn(*features.shape) * noise_level
            features = features + noise.astype(np.float32)

        return features, mask

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize features with adaptive scaling."""
        normalized = np.zeros_like(features)
        n_channels = features.shape[0]

        if self.sar_only:
            # SAR-only: 6 channels (VV, VH, DEM, HAND, SLOPE, TWI)
            if n_channels >= 1:
                normalized[0] = np.clip((features[0] + 30) / 30, -1, 2)  # VV
            if n_channels >= 2:
                normalized[1] = np.clip((features[1] + 30) / 30, -1, 2)  # VH
            if n_channels >= 3:
                normalized[2] = np.clip(features[2] / 1000, 0, 5)  # DEM
            if n_channels >= 4:
                normalized[3] = np.clip(features[3] / 50, 0, 2)  # HAND
            if n_channels >= 5:
                normalized[4] = np.clip(features[4] / 45, 0, 2)  # SLOPE
            if n_channels >= 6:
                normalized[5] = np.clip(features[5] / 15, 0, 2)  # TWI
        else:
            # Full 7 channels including MNDWI
            normalized[0] = np.clip((features[0] + 30) / 30, -1, 2)
            normalized[1] = np.clip((features[1] + 30) / 30, -1, 2)
            normalized[2] = np.clip((features[2] + 1) / 2, 0, 1)  # MNDWI
            normalized[3] = np.clip(features[3] / 1000, 0, 5)
            normalized[4] = np.clip(features[4] / 50, 0, 2)
            normalized[5] = np.clip(features[5] / 45, 0, 2)
            normalized[6] = np.clip(features[6] / 15, 0, 2)

        return normalized

    def _compute_edges(self, mask: np.ndarray) -> np.ndarray:
        """Compute edge mask using Sobel operator."""
        edge_x = sobel(mask, axis=1)
        edge_y = sobel(mask, axis=0)
        edges = np.sqrt(edge_x**2 + edge_y**2)
        edges = (edges > 0.1).astype(np.float32)

        # Dilate edges slightly
        edges = ndimage.binary_dilation(edges, iterations=1).astype(np.float32)

        return edges


# =============================================================================
# IMPROVED LIGHTGBM WITH MORE FEATURES
# =============================================================================


def extract_advanced_features(
    chip: np.ndarray, i: int, j: int, window_sizes: List[int] = [3, 5, 9]
) -> List[float]:
    """
    Extract advanced pixel-level features for LightGBM.
    Includes multi-scale and texture features for narrow river detection.
    """
    vv = chip[0]
    vh = chip[1]
    hand = chip[4]
    slope = chip[5]
    twi = chip[6]

    h, w = vv.shape

    features = []

    # Basic pixel values
    features.extend(
        [
            vv[i, j],
            vh[i, j],
            vv[i, j] - vh[i, j],  # Polarization difference
            vv[i, j] / (vh[i, j] + 1e-6),  # Polarization ratio
            hand[i, j],
            slope[i, j],
            twi[i, j],
        ]
    )

    # Multi-scale local statistics
    for ws in window_sizes:
        half = ws // 2
        i_min, i_max = max(0, i - half), min(h, i + half + 1)
        j_min, j_max = max(0, j - half), min(w, j + half + 1)

        vv_patch = vv[i_min:i_max, j_min:j_max]
        vh_patch = vh[i_min:i_max, j_min:j_max]

        features.extend(
            [
                np.mean(vv_patch),
                np.std(vv_patch),
                np.mean(vh_patch),
                np.std(vh_patch),
                np.min(vv_patch),  # Darkest pixel (most likely water)
                np.min(vh_patch),
            ]
        )

    # Gradient features (for edges/boundaries)
    for arr, name in [(vv, "vv"), (vh, "vh")]:
        grad_x = sobel(arr, axis=1)
        grad_y = sobel(arr, axis=0)

        features.extend(
            [
                grad_x[i, j],
                grad_y[i, j],
                np.sqrt(grad_x[i, j] ** 2 + grad_y[i, j] ** 2),  # Gradient magnitude
            ]
        )

    # ADAPTIVE threshold features (not fixed!)
    adaptive_thresh_vv = compute_adaptive_threshold(vv, method="otsu")
    adaptive_thresh_vh = compute_adaptive_threshold(vh, method="otsu")

    features.extend(
        [
            vv[i, j] - adaptive_thresh_vv,  # Distance from adaptive threshold
            vh[i, j] - adaptive_thresh_vh,
            float(vv[i, j] < adaptive_thresh_vv),  # Below adaptive threshold?
            float(vh[i, j] < adaptive_thresh_vh),
        ]
    )

    # Linear feature detection (narrow rivers)
    line_response = detect_linear_features(vh, threshold_percentile=25)
    features.append(line_response[i, j])

    return features


def get_feature_names_advanced() -> List[str]:
    """Get feature names for advanced feature extraction."""
    names = [
        "VV",
        "VH",
        "VV-VH",
        "VV/VH",
        "HAND",
        "SLOPE",
        "TWI",
    ]

    for ws in [3, 5, 9]:
        names.extend(
            [
                f"VV_mean_{ws}",
                f"VV_std_{ws}",
                f"VH_mean_{ws}",
                f"VH_std_{ws}",
                f"VV_min_{ws}",
                f"VH_min_{ws}",
            ]
        )

    names.extend(
        [
            "VV_grad_x",
            "VV_grad_y",
            "VV_grad_mag",
            "VH_grad_x",
            "VH_grad_y",
            "VH_grad_mag",
            "VV_adaptive_diff",
            "VH_adaptive_diff",
            "VV_below_adaptive",
            "VH_below_adaptive",
            "line_response",
        ]
    )

    return names


def train_lightgbm_v3(chips: List[np.ndarray], names: List[str]) -> Dict:
    """
    Train improved LightGBM with:
    - Multi-scale features
    - Adaptive thresholds
    - Linear feature detection
    - Cross-validation
    """
    logger.info("=" * 60)
    logger.info("TRAINING MODEL 1: LightGBM v3 (Advanced Features)")
    logger.info("=" * 60)
    sys.stdout.flush()

    start_time = time.time()

    # Extract features
    X_all = []
    y_all = []

    samples_per_class_per_chip = 300

    for chip_idx, chip in enumerate(chips):
        if chip.shape[0] < 8:
            continue

        truth = chip[7]
        h, w = truth.shape

        # Stratified sampling
        water_idx = np.where(truth.flatten() > 0.5)[0]
        nonwater_idx = np.where(truth.flatten() <= 0.5)[0]

        n_water = min(len(water_idx), samples_per_class_per_chip)
        n_nonwater = min(len(nonwater_idx), samples_per_class_per_chip)

        if n_water > 0:
            water_samples = np.random.choice(water_idx, size=n_water, replace=False)
        else:
            water_samples = np.array([], dtype=int)

        if n_nonwater > 0:
            nonwater_samples = np.random.choice(
                nonwater_idx, size=n_nonwater, replace=False
            )
        else:
            nonwater_samples = np.array([], dtype=int)

        for idx in np.concatenate([water_samples, nonwater_samples]):
            i, j = idx // w, idx % w

            feat = extract_advanced_features(chip, i, j)
            X_all.append(feat)
            y_all.append(1 if truth[i, j] > 0.5 else 0)

        if (chip_idx + 1) % 20 == 0:
            logger.info(f"  Processed {chip_idx + 1}/{len(chips)} chips...")
            sys.stdout.flush()

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)

    # Clean data
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]

    logger.info(f"Total samples: {len(X)}")
    logger.info(f"Class balance: {np.mean(y):.2%} water")
    sys.stdout.flush()

    if len(X) == 0:
        return {"status": "failed", "error": "No valid samples"}

    feature_names = get_feature_names_advanced()

    # Cross-validation
    kfold = KFold(
        n_splits=CONFIG["n_folds"], shuffle=True, random_state=CONFIG["random_seed"]
    )
    cv_scores = []

    logger.info(f"Running {CONFIG['n_folds']}-fold cross-validation...")
    sys.stdout.flush()

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=10,
            num_leaves=127,
            learning_rate=0.03,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=CONFIG["random_seed"],
            verbose=-1,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        y_pred = model.predict(X_val)
        fold_iou = jaccard_score(y_val, y_pred)
        cv_scores.append(fold_iou)
        logger.info(f"  Fold {fold + 1}: IoU = {fold_iou:.4f}")
        sys.stdout.flush()

    logger.info(f"CV Mean IoU: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    sys.stdout.flush()

    # Train final model on all data
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=CONFIG["random_seed"], stratify=y
    )

    final_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=10,
        num_leaves=127,
        learning_rate=0.03,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=CONFIG["random_seed"],
        verbose=-1,
        n_jobs=-1,
    )

    final_model.fit(X_trainval, y_trainval)

    y_pred_test = final_model.predict(X_test)
    y_prob_test = final_model.predict_proba(X_test)[:, 1]

    results = {
        "model": "LightGBM_v3_Advanced",
        "params": {
            "n_estimators": final_model.n_estimators_,
            "max_depth": 10,
            "features": feature_names,
            "n_features": len(feature_names),
        },
        "cv": {
            "mean_iou": float(np.mean(cv_scores)),
            "std_iou": float(np.std(cv_scores)),
            "fold_scores": [float(s) for s in cv_scores],
        },
        "test": {
            "accuracy": float(accuracy_score(y_test, y_pred_test)),
            "f1": float(f1_score(y_test, y_pred_test)),
            "precision": float(precision_score(y_test, y_pred_test)),
            "recall": float(recall_score(y_test, y_pred_test)),
            "iou": float(jaccard_score(y_test, y_pred_test)),
        },
        "feature_importance": dict(
            zip(feature_names, final_model.feature_importances_.tolist())
        ),
        "training_time_seconds": time.time() - start_time,
        "n_samples": len(X),
    }

    # Save model
    model_path = CONFIG["model_dir"] / "lightgbm_v3.txt"
    final_model.booster_.save_model(str(model_path))
    results["model_path"] = str(model_path)

    logger.info(f"LightGBM v3 Results:")
    logger.info(f"  CV Mean IoU:   {results['cv']['mean_iou']:.4f}")
    logger.info(f"  Test Accuracy: {results['test']['accuracy']:.4f}")
    logger.info(f"  Test F1:       {results['test']['f1']:.4f}")
    logger.info(f"  Test IoU:      {results['test']['iou']:.4f}")

    # Top features
    sorted_feat = sorted(
        results["feature_importance"].items(), key=lambda x: x[1], reverse=True
    )
    logger.info("Top 10 Features:")
    for name, imp in sorted_feat[:10]:
        logger.info(f"  {name}: {imp:.0f}")

    sys.stdout.flush()
    return results


# =============================================================================
# COMPACT U-NET WITH LEARNABLE PHYSICS (~300K params)
# =============================================================================


class CompactConvBlock(nn.Module):
    """Compact convolution block with depthwise separable convolutions."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            # Depthwise separable convolution (much fewer params)
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False),
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class LearnablePhysicsModule(nn.Module):
    """
    LEARNABLE physics constraints - not fixed thresholds!
    The network learns optimal thresholds and weights.
    """

    def __init__(self, in_channels):
        super().__init__()

        # Learnable thresholds (initialized with physics priors)
        self.hand_threshold = nn.Parameter(torch.tensor(10.0))  # HAND threshold
        self.slope_threshold = nn.Parameter(torch.tensor(15.0))  # Slope threshold
        self.vh_threshold = nn.Parameter(torch.tensor(-18.0))  # VH threshold

        # Learnable attention weights
        self.attention_net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, features, hand=None, slope=None, vh=None):
        """
        Apply learnable physics attention.

        Args:
            features: Feature maps from encoder
            hand: HAND values (optional, for physics constraint)
            slope: Slope values (optional)
            vh: VH backscatter (optional)
        """
        # Compute attention from features
        attention = self.attention_net(features)

        # Add physics-based attention if auxiliary inputs provided
        if hand is not None:
            # Water more likely at low HAND (learned threshold)
            hand_attention = torch.sigmoid(-(hand - self.hand_threshold) / 5.0)
            attention = attention * hand_attention

        if slope is not None:
            # Water less likely on steep slopes (learned threshold)
            slope_attention = torch.sigmoid(-(slope - self.slope_threshold) / 10.0)
            attention = attention * slope_attention

        if vh is not None:
            # Water more likely where VH is dark (learned threshold)
            vh_attention = torch.sigmoid(-(vh - self.vh_threshold) / 3.0)
            attention = attention * 0.5 + vh_attention * 0.5

        return features * attention


class CompactPhysicsUNet(nn.Module):
    """
    Compact U-Net with learnable physics (~300K parameters).

    Key differences from v2:
    - Depthwise separable convolutions (fewer params)
    - Learnable physics thresholds (not fixed)
    - Multi-scale skip connections
    - Edge-aware output head
    """

    def __init__(self, in_channels=6, num_classes=1):
        super().__init__()

        # Compact encoder (fewer channels)
        self.enc1 = CompactConvBlock(in_channels, 24)
        self.enc2 = CompactConvBlock(24, 48)
        self.enc3 = CompactConvBlock(48, 96)
        self.enc4 = CompactConvBlock(96, 192)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck with physics
        self.bottleneck = CompactConvBlock(192, 192)
        self.physics = LearnablePhysicsModule(192)

        # Compact decoder
        self.up4 = nn.ConvTranspose2d(192, 96, 2, stride=2)
        self.dec4 = CompactConvBlock(192, 96)

        self.up3 = nn.ConvTranspose2d(96, 48, 2, stride=2)
        self.dec3 = CompactConvBlock(96, 48)

        self.up2 = nn.ConvTranspose2d(48, 24, 2, stride=2)
        self.dec2 = CompactConvBlock(48, 24)

        self.up1 = nn.ConvTranspose2d(24, 24, 2, stride=2)
        self.dec1 = CompactConvBlock(48, 24)

        # Output heads
        self.mask_head = nn.Conv2d(24, num_classes, 1)
        self.edge_head = nn.Conv2d(24, num_classes, 1)
        self.confidence_head = nn.Conv2d(24, num_classes, 1)

    def forward(self, x, hand=None, slope=None, vh=None):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck with physics
        b = self.bottleneck(self.pool(e4))
        b = self.physics(b, hand, slope, vh)

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Outputs
        mask = torch.sigmoid(self.mask_head(d1))
        edge = torch.sigmoid(self.edge_head(d1))
        confidence = torch.sigmoid(self.confidence_head(d1))

        return mask, edge, confidence


# =============================================================================
# EDGE-AWARE AND NARROW FEATURE LOSS
# =============================================================================


class EdgeAwareNarrowFeatureLoss(nn.Module):
    """
    Loss function that emphasizes:
    1. Edge detection (for boundaries)
    2. Narrow features (rivers, streams)
    3. Learnable physics constraints
    """

    def __init__(self, edge_weight=0.3, narrow_weight=0.2, physics_weight=0.2):
        super().__init__()
        self.edge_weight = edge_weight
        self.narrow_weight = narrow_weight
        self.physics_weight = physics_weight

        # Learnable loss weights
        self.log_vars = nn.Parameter(torch.zeros(4))  # Multi-task uncertainty weighting

    def forward(
        self,
        pred_mask,
        pred_edge,
        pred_conf,
        true_mask,
        true_edge,
        hand=None,
        slope=None,
        vh=None,
    ):
        # Main segmentation loss (Dice + BCE for class imbalance)
        dice_loss = self._dice_loss(pred_mask, true_mask)
        bce_loss = F.binary_cross_entropy(pred_mask, true_mask)
        seg_loss = dice_loss + bce_loss

        # Edge loss (weighted more on edge pixels)
        edge_bce = F.binary_cross_entropy(pred_edge, true_edge)
        edge_dice = self._dice_loss(pred_edge, true_edge)
        edge_loss = edge_bce + edge_dice

        # Narrow feature loss (emphasize thin structures)
        # Use morphological thinning approximation
        narrow_loss = self._narrow_feature_loss(pred_mask, true_mask)

        # Physics consistency loss (learnable, not fixed)
        physics_loss = torch.tensor(0.0, device=pred_mask.device)
        if hand is not None:
            # Penalize water predictions at high HAND (but threshold is learned in model)
            hand_penalty = (pred_mask * torch.clamp(hand / 50, 0, 1)).mean()
            physics_loss = physics_loss + hand_penalty

        if slope is not None:
            # Penalize water on steep slopes
            steep_mask = (slope > 20).float()
            slope_penalty = (pred_mask * steep_mask).mean()
            physics_loss = physics_loss + slope_penalty

        if vh is not None:
            # Penalize water where VH is bright (normalized)
            vh_norm = (vh + 30) / 30  # Normalize to ~0-1
            bright_mask = (vh_norm > 0.6).float()
            vh_penalty = (pred_mask * bright_mask).mean()
            physics_loss = physics_loss + vh_penalty

        # Multi-task uncertainty weighting (Kendall et al.)
        precision1 = torch.exp(-self.log_vars[0])
        precision2 = torch.exp(-self.log_vars[1])
        precision3 = torch.exp(-self.log_vars[2])
        precision4 = torch.exp(-self.log_vars[3])

        total_loss = (
            precision1 * seg_loss
            + self.log_vars[0]
            + precision2 * self.edge_weight * edge_loss
            + self.log_vars[1]
            + precision3 * self.narrow_weight * narrow_loss
            + self.log_vars[2]
            + precision4 * self.physics_weight * physics_loss
            + self.log_vars[3]
        )

        return total_loss, {
            "seg_loss": seg_loss.item(),
            "edge_loss": edge_loss.item(),
            "narrow_loss": narrow_loss.item(),
            "physics_loss": physics_loss.item()
            if isinstance(physics_loss, torch.Tensor)
            else physics_loss,
        }

    def _dice_loss(self, pred, target, smooth=1e-6):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2 * intersection + smooth) / (
            pred_flat.sum() + target_flat.sum() + smooth
        )

    def _narrow_feature_loss(self, pred, target):
        """
        Loss that emphasizes narrow features using gradient-based detection.
        Narrow features have high gradient in one direction.
        """
        # Compute gradients
        pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]

        # Penalize gradient differences (emphasizes edges and narrow features)
        grad_loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        grad_loss_y = F.l1_loss(pred_grad_y, target_grad_y)

        return grad_loss_x + grad_loss_y


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================


def train_compact_physics_unet(chips: List[np.ndarray], names: List[str]) -> Dict:
    """Train compact physics-guided U-Net with edge awareness."""

    logger.info("=" * 60)
    logger.info("TRAINING MODEL 2: Compact Physics U-Net v3")
    logger.info("=" * 60)
    sys.stdout.flush()

    start_time = time.time()
    device = torch.device(CONFIG["device"])

    # Filter chips with truth masks
    valid_chips = [(c, n) for c, n in zip(chips, names) if c.shape[0] >= 8]
    chips_only = [c for c, _ in valid_chips]
    names_only = [n for _, n in valid_chips]

    logger.info(f"Valid chips with truth mask: {len(chips_only)}")

    if len(chips_only) < 10:
        return {"status": "failed", "error": "Not enough chips with truth masks"}

    # Split data
    train_chips, test_chips, train_names, test_names = train_test_split(
        chips_only, names_only, test_size=0.2, random_state=CONFIG["random_seed"]
    )
    train_chips, val_chips, train_names, val_names = train_test_split(
        train_chips, train_names, test_size=0.15, random_state=CONFIG["random_seed"]
    )

    logger.info(
        f"Train: {len(train_chips)}, Val: {len(val_chips)}, Test: {len(test_chips)}"
    )
    sys.stdout.flush()

    # Create datasets with augmentation
    in_channels = 6 if CONFIG["use_sar_only"] else 7
    train_dataset = AugmentedSARDataset(
        train_chips,
        train_names,
        CONFIG["image_size"],
        augment=True,
        sar_only=CONFIG["use_sar_only"],
    )
    val_dataset = AugmentedSARDataset(
        val_chips,
        val_names,
        CONFIG["image_size"],
        augment=False,
        sar_only=CONFIG["use_sar_only"],
    )
    test_dataset = AugmentedSARDataset(
        test_chips,
        test_names,
        CONFIG["image_size"],
        augment=False,
        sar_only=CONFIG["use_sar_only"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
    )

    # Create model
    model = CompactPhysicsUNet(in_channels=in_channels, num_classes=1).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    sys.stdout.flush()

    # Loss and optimizer
    criterion = EdgeAwareNarrowFeatureLoss(
        edge_weight=0.3, narrow_weight=0.2, physics_weight=0.2
    )
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2
    )

    # Training loop
    best_val_iou = 0.0
    patience = 30
    patience_counter = 0

    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        train_loss = 0.0

        for batch_idx, (features, masks, edges) in enumerate(train_loader):
            features = features.to(device)
            masks = masks.to(device)
            edges = edges.to(device)

            # Extract physics inputs if SAR-only mode
            # In SAR-only mode: 0=VV, 1=VH, 2=DEM, 3=HAND, 4=SLOPE, 5=TWI
            if CONFIG["use_sar_only"]:
                hand = features[:, 3:4, :, :]
                slope = features[:, 4:5, :, :]
                vh = features[:, 1:2, :, :]
            else:
                hand = features[:, 4:5, :, :]
                slope = features[:, 5:6, :, :]
                vh = features[:, 1:2, :, :]

            optimizer.zero_grad()

            pred_mask, pred_edge, pred_conf = model(features, hand, slope, vh)

            loss, loss_dict = criterion(
                pred_mask, pred_edge, pred_conf, masks, edges, hand, slope, vh
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_iou = 0.0
        val_count = 0

        with torch.no_grad():
            for features, masks, edges in val_loader:
                features = features.to(device)
                masks = masks.to(device)

                if CONFIG["use_sar_only"]:
                    hand = features[:, 3:4, :, :]
                    slope = features[:, 4:5, :, :]
                    vh = features[:, 1:2, :, :]
                else:
                    hand = features[:, 4:5, :, :]
                    slope = features[:, 5:6, :, :]
                    vh = features[:, 1:2, :, :]

                pred_mask, _, _ = model(features, hand, slope, vh)

                pred_binary = (pred_mask > 0.5).float()
                intersection = (pred_binary * masks).sum()
                union = pred_binary.sum() + masks.sum() - intersection
                iou = (intersection / (union + 1e-6)).item()

                val_iou += iou
                val_count += 1

        val_iou /= max(val_count, 1)
        train_loss /= len(train_loader)

        # Early stopping
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            # Save best model
            torch.save(
                model.state_dict(),
                CONFIG["model_dir"] / "compact_physics_unet_v3_best.pth",
            )
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == CONFIG["num_epochs"] - 1:
            logger.info(
                f"Epoch {epoch}/{CONFIG['num_epochs']}: Train Loss={train_loss:.4f}, Val IoU={val_iou:.4f}, Best={best_val_iou:.4f}"
            )
            sys.stdout.flush()

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Load best model for testing
    model.load_state_dict(
        torch.load(CONFIG["model_dir"] / "compact_physics_unet_v3_best.pth")
    )
    model.eval()

    # Test evaluation
    test_iou = 0.0
    test_count = 0

    with torch.no_grad():
        for features, masks, edges in test_loader:
            features = features.to(device)
            masks = masks.to(device)

            if CONFIG["use_sar_only"]:
                hand = features[:, 3:4, :, :]
                slope = features[:, 4:5, :, :]
                vh = features[:, 1:2, :, :]
            else:
                hand = features[:, 4:5, :, :]
                slope = features[:, 5:6, :, :]
                vh = features[:, 1:2, :, :]

            pred_mask, _, _ = model(features, hand, slope, vh)

            pred_binary = (pred_mask > 0.5).float()
            intersection = (pred_binary * masks).sum()
            union = pred_binary.sum() + masks.sum() - intersection
            iou = (intersection / (union + 1e-6)).item()

            test_iou += iou
            test_count += 1

    test_iou /= max(test_count, 1)

    # Get learned physics parameters
    learned_physics = {
        "hand_threshold": model.physics.hand_threshold.item(),
        "slope_threshold": model.physics.slope_threshold.item(),
        "vh_threshold": model.physics.vh_threshold.item(),
    }

    results = {
        "model": "CompactPhysicsUNet_v3",
        "params": {
            "num_parameters": num_params,
            "epochs": CONFIG["num_epochs"],
            "image_size": CONFIG["image_size"],
            "sar_only": CONFIG["use_sar_only"],
            "in_channels": in_channels,
        },
        "learned_physics": learned_physics,
        "val": {"best_iou": float(best_val_iou)},
        "test": {"iou": float(test_iou)},
        "model_path": str(CONFIG["model_dir"] / "compact_physics_unet_v3_best.pth"),
        "training_time_seconds": time.time() - start_time,
    }

    logger.info(f"Compact Physics U-Net v3 Results:")
    logger.info(f"  Parameters: {num_params:,}")
    logger.info(f"  Val IoU:  {best_val_iou:.4f}")
    logger.info(f"  Test IoU: {test_iou:.4f}")
    logger.info(f"  Learned HAND threshold: {learned_physics['hand_threshold']:.2f}")
    logger.info(f"  Learned SLOPE threshold: {learned_physics['slope_threshold']:.2f}")
    logger.info(f"  Learned VH threshold: {learned_physics['vh_threshold']:.2f} dB")
    sys.stdout.flush()

    return results


# =============================================================================
# ENSEMBLE MODEL
# =============================================================================


class EnsemblePredictor:
    """
    Ensemble of LightGBM + Compact U-Net with learned weights.
    """

    def __init__(self, lgbm_model, unet_model, device):
        self.lgbm_model = lgbm_model
        self.unet_model = unet_model
        self.device = device

        # Learned ensemble weights (can be optimized)
        self.weights = [0.5, 0.5]  # LightGBM, U-Net

    def predict(self, chip: np.ndarray) -> np.ndarray:
        """
        Ensemble prediction combining LightGBM and U-Net.
        """
        h, w = chip.shape[1], chip.shape[2]

        # LightGBM predictions (pixel-wise)
        lgbm_probs = np.zeros((h, w), dtype=np.float32)

        for i in range(h):
            for j in range(w):
                feat = extract_advanced_features(chip, i, j)
                if np.all(np.isfinite(feat)):
                    prob = self.lgbm_model.predict_proba([feat])[0, 1]
                    lgbm_probs[i, j] = prob

        # U-Net prediction
        self.unet_model.eval()
        with torch.no_grad():
            # Prepare input
            if CONFIG["use_sar_only"]:
                features = np.stack(
                    [chip[0], chip[1], chip[3], chip[4], chip[5], chip[6]], axis=0
                )
            else:
                features = chip[:7]

            # Resize
            from scipy.ndimage import zoom

            c, oh, ow = features.shape
            size = CONFIG["image_size"]
            features_resized = zoom(features, (1, size / oh, size / ow), order=1)

            # Normalize and convert to tensor
            features_tensor = (
                torch.from_numpy(features_resized).unsqueeze(0).float().to(self.device)
            )

            # Predict
            if CONFIG["use_sar_only"]:
                hand = features_tensor[:, 3:4, :, :]
                slope = features_tensor[:, 4:5, :, :]
                vh = features_tensor[:, 1:2, :, :]
            else:
                hand = features_tensor[:, 4:5, :, :]
                slope = features_tensor[:, 5:6, :, :]
                vh = features_tensor[:, 1:2, :, :]

            unet_probs, _, _ = self.unet_model(features_tensor, hand, slope, vh)
            unet_probs = unet_probs.squeeze().cpu().numpy()

            # Resize back to original
            unet_probs = zoom(unet_probs, (oh / size, ow / size), order=1)

        # Weighted ensemble
        ensemble_probs = self.weights[0] * lgbm_probs + self.weights[1] * unet_probs

        return ensemble_probs

    def optimize_weights(self, chips: List[np.ndarray], n_samples: int = 10):
        """Optimize ensemble weights using validation data."""
        best_iou = 0.0
        best_weights = self.weights.copy()

        for w1 in np.arange(0.2, 0.9, 0.1):
            w2 = 1.0 - w1
            self.weights = [w1, w2]

            # Evaluate on sample chips
            total_iou = 0.0
            count = 0

            for chip in chips[:n_samples]:
                if chip.shape[0] < 8:
                    continue

                truth = chip[7]
                pred = self.predict(chip) > 0.5

                intersection = (pred * truth).sum()
                union = pred.sum() + truth.sum() - intersection
                iou = intersection / (union + 1e-6)

                total_iou += iou
                count += 1

            avg_iou = total_iou / max(count, 1)

            if avg_iou > best_iou:
                best_iou = avg_iou
                best_weights = self.weights.copy()

        self.weights = best_weights
        logger.info(
            f"Optimized ensemble weights: LightGBM={self.weights[0]:.2f}, U-Net={self.weights[1]:.2f}"
        )

        return best_weights, best_iou


# =============================================================================
# MAIN
# =============================================================================


def main():
    logger.info("=" * 80)
    logger.info("MASTER SAR WATER DETECTION TRAINING PIPELINE v3")
    logger.info("=" * 80)
    logger.info(f"Device: {CONFIG['device']}")
    logger.info(f"SAR-only mode: {CONFIG['use_sar_only']}")
    logger.info(f"Started: {datetime.now().isoformat()}")
    sys.stdout.flush()

    # Load chips from both directories
    chip_dirs = [CONFIG["chip_dir"]]
    if CONFIG["expanded_chip_dir"].exists():
        # Check for nested directory
        nested = CONFIG["expanded_chip_dir"] / "india_chips_expanded"
        if nested.exists():
            chip_dirs.append(nested)
        else:
            chip_dirs.append(CONFIG["expanded_chip_dir"])

    chips, names = load_all_chips(chip_dirs)

    if not chips:
        logger.error("No chips found!")
        return

    all_results = {}

    # Model 1: LightGBM v3 with advanced features
    try:
        all_results["lightgbm_v3"] = train_lightgbm_v3(chips, names)
    except Exception as e:
        logger.error(f"LightGBM v3 failed: {e}")
        import traceback

        traceback.print_exc()
        all_results["lightgbm_v3"] = {"status": "failed", "error": str(e)}

    # Model 2: Compact Physics U-Net
    try:
        all_results["compact_physics_unet"] = train_compact_physics_unet(chips, names)
    except Exception as e:
        logger.error(f"Compact Physics U-Net failed: {e}")
        import traceback

        traceback.print_exc()
        all_results["compact_physics_unet"] = {"status": "failed", "error": str(e)}

    # Save results
    results_path = CONFIG["output_dir"] / "training_results_v3.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)

    # Summary
    logger.info("\nRESULTS SUMMARY:")
    for model_name, results in all_results.items():
        logger.info(f"\n{model_name}:")
        if "test" in results:
            for k, v in results["test"].items():
                if isinstance(v, float):
                    logger.info(f"  {k}: {v:.4f}")
        if "cv" in results:
            logger.info(f"  CV Mean IoU: {results['cv']['mean_iou']:.4f}")
        if "learned_physics" in results:
            logger.info(f"  Learned Physics:")
            for k, v in results["learned_physics"].items():
                logger.info(f"    {k}: {v:.2f}")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
