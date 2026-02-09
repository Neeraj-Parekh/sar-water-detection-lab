#!/usr/bin/env python3
"""
================================================================================
COMPREHENSIVE EVALUATION & ANALYSIS PIPELINE
================================================================================
This script performs:
1. Full test evaluation for both models
2. Ensemble creation (LightGBM + U-Net)
3. Narrow water body detection analysis
4. Land-water boundary accuracy analysis
5. False positive/negative analysis
6. Prediction visualizations
7. Feature importance plots
8. Per-chip error analysis
9. Symbolic regression with PySR

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
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import rasterio
from scipy import ndimage
from scipy.ndimage import zoom, label, binary_erosion, binary_dilation
from scipy.ndimage import distance_transform_edt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import lightgbm as lgb

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
    "viz_dir": Path("/home/mit-aoe/sar_water_detection/visualizations"),
    "bands": {"VV": 0, "VH": 1, "DEM": 3, "HAND": 4, "SLOPE": 5, "TWI": 6, "TRUTH": 7},
    "image_size": 256,
    "batch_size": 8,
    "scales": [3, 5, 9, 15, 21],
}

CONFIG["viz_dir"].mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(CONFIG["results_dir"] / "comprehensive_evaluation.log"),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# U-NET MODEL DEFINITION (must match training)
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
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
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

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_ch)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.cbam(x)
        return x


class UNetV4(nn.Module):
    def __init__(self, in_channels: int = 6):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(256, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


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

        for f in chip_dir.glob("*.npy"):
            try:
                chip = np.load(f).astype(np.float32)
                if chip.shape[0] >= 8 and np.nansum(chip[7]) > 0:
                    chips.append(chip)
                    names.append(f.stem)
            except:
                pass

        for f in chip_dir.glob("*.tif"):
            try:
                with rasterio.open(f) as src:
                    chip = src.read().astype(np.float32)
                if chip.shape[0] >= 8 and np.nansum(chip[7]) > 0:
                    chips.append(chip)
                    names.append(f.stem)
            except:
                pass

    logger.info(f"Loaded {len(chips)} chips")
    return chips, names


# =============================================================================
# FEATURE EXTRACTION (MUST MATCH TRAINING - 69 features)
# =============================================================================

# Physics priors (same as training)
PHYSICS = {
    "hand_threshold_init": 10.0,
    "slope_threshold_init": 15.0,
    "vh_dark_threshold_init": -18.0,
}


def db_to_linear(db: np.ndarray) -> np.ndarray:
    """Convert dB to linear scale."""
    return np.power(10, db / 10)


def linear_to_db(linear: np.ndarray) -> np.ndarray:
    """Convert linear to dB."""
    return 10 * np.log10(np.clip(linear, 1e-10, None))


def compute_otsu_threshold(arr: np.ndarray) -> float:
    arr_clean = arr[~np.isnan(arr)].flatten()
    arr_clip = np.clip(
        arr_clean, np.percentile(arr_clean, 1), np.percentile(arr_clean, 99)
    )
    try:
        from skimage.filters import threshold_otsu

        return threshold_otsu(arr_clip)
    except:
        return np.median(arr_clip)


def compute_kapur_threshold(arr: np.ndarray, n_bins: int = 256) -> float:
    """Kapur's entropy-based thresholding."""
    arr_clean = arr[~np.isnan(arr)].flatten()
    arr_clip = np.clip(
        arr_clean, np.percentile(arr_clean, 1), np.percentile(arr_clean, 99)
    )
    arr_norm = (arr_clip - arr_clip.min()) / (arr_clip.max() - arr_clip.min() + 1e-10)
    hist, bin_edges = np.histogram(arr_norm, bins=n_bins, density=True)
    hist = hist / (hist.sum() + 1e-10)
    cumsum = np.cumsum(hist)
    max_entropy = -np.inf
    best_thresh = 0.5
    for t in range(1, n_bins - 1):
        p_bg = cumsum[t]
        if p_bg < 1e-10 or p_bg > 1 - 1e-10:
            continue
        p_fg = 1 - p_bg
        h_bg = hist[:t] / (p_bg + 1e-10)
        h_bg = h_bg[h_bg > 0]
        entropy_bg = -np.sum(h_bg * np.log(h_bg + 1e-10))
        h_fg = hist[t:] / (p_fg + 1e-10)
        h_fg = h_fg[h_fg > 0]
        entropy_fg = -np.sum(h_fg * np.log(h_fg + 1e-10))
        total_entropy = entropy_bg + entropy_fg
        if total_entropy > max_entropy:
            max_entropy = total_entropy
            best_thresh = (bin_edges[t] + bin_edges[t + 1]) / 2
    return best_thresh * (arr_clip.max() - arr_clip.min()) + arr_clip.min()


def compute_pseudo_entropy(
    vv_db: np.ndarray, vh_db: np.ndarray, window: int = 5
) -> np.ndarray:
    """Compute pseudo-entropy from dual-pol SAR."""
    from scipy.ndimage import uniform_filter

    vv_lin = db_to_linear(vv_db)
    vh_lin = db_to_linear(vh_db)
    vv_local = uniform_filter(vv_lin, size=window)
    vh_local = uniform_filter(vh_lin, size=window)
    total = vv_local + vh_local + 1e-10
    p = np.clip(vv_local / total, 1e-10, 1 - 1e-10)
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    return entropy


def compute_pseudo_alpha(
    vv_db: np.ndarray, vh_db: np.ndarray, window: int = 5
) -> np.ndarray:
    """Compute pseudo-alpha angle from dual-pol SAR."""
    from scipy.ndimage import uniform_filter

    vv_lin = db_to_linear(vv_db)
    vh_lin = db_to_linear(vh_db)
    vv_local = uniform_filter(vv_lin, size=window)
    vh_local = uniform_filter(vh_lin, size=window)
    alpha = np.arctan2(np.sqrt(vh_local), np.sqrt(vv_local))
    return np.degrees(alpha)


def compute_rvi(vv_db: np.ndarray, vh_db: np.ndarray, window: int = 5) -> np.ndarray:
    """Radar Vegetation Index."""
    from scipy.ndimage import uniform_filter

    vv_lin = db_to_linear(vv_db)
    vh_lin = db_to_linear(vh_db)
    vv_local = uniform_filter(vv_lin, size=window)
    vh_local = uniform_filter(vh_lin, size=window)
    return 4 * vh_local / (vv_local + vh_local + 1e-10)


def compute_span(vv_db: np.ndarray, vh_db: np.ndarray) -> np.ndarray:
    """Total power (span)."""
    vv_lin = db_to_linear(vv_db)
    vh_lin = db_to_linear(vh_db)
    span_lin = vv_lin + 2 * vh_lin
    return linear_to_db(span_lin)


def compute_enl(arr: np.ndarray, window: int = 9) -> np.ndarray:
    """Equivalent Number of Looks."""
    from scipy.ndimage import uniform_filter

    arr_lin = db_to_linear(arr)
    local_mean = uniform_filter(arr_lin, size=window)
    local_sq_mean = uniform_filter(arr_lin**2, size=window)
    local_var = local_sq_mean - local_mean**2
    local_std = np.sqrt(np.maximum(local_var, 1e-10))
    enl = (local_mean / local_std) ** 2
    return np.clip(enl, 0, 100)


def compute_cv(arr: np.ndarray, window: int = 9) -> np.ndarray:
    """Coefficient of Variation."""
    from scipy.ndimage import uniform_filter

    arr_lin = db_to_linear(arr)
    local_mean = uniform_filter(arr_lin, size=window) + 1e-10
    local_sq_mean = uniform_filter(arr_lin**2, size=window)
    local_var = local_sq_mean - local_mean**2
    local_std = np.sqrt(np.maximum(local_var, 1e-10))
    return np.clip(local_std / local_mean, 0, 5)


def compute_glcm_fast(arr: np.ndarray, window: int = 11) -> Dict[str, np.ndarray]:
    """Fast approximation of GLCM-like features."""
    from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter

    local_mean = uniform_filter(arr.astype(np.float64), size=window)
    local_sq_mean = uniform_filter((arr.astype(np.float64)) ** 2, size=window)
    local_var = local_sq_mean - local_mean**2
    contrast = np.sqrt(np.maximum(local_var, 0))
    local_max = maximum_filter(arr, size=window)
    local_min = minimum_filter(arr, size=window)
    local_range = local_max - local_min + 1e-10
    homogeneity = 1.0 / (1.0 + local_range)
    energy = 1.0 / (1.0 + contrast)
    return {
        "contrast": contrast.astype(np.float32),
        "homogeneity": homogeneity.astype(np.float32),
        "energy": energy.astype(np.float32),
    }


def compute_morphological_features(arr: np.ndarray) -> Dict[str, np.ndarray]:
    """Morphological features."""
    from skimage.morphology import disk, opening, closing, white_tophat, black_tophat

    arr_norm = (
        (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + 1e-10) * 255
    ).astype(np.uint8)
    selem = disk(3)
    opened = opening(arr_norm, selem)
    closed = closing(arr_norm, selem)
    w_tophat = white_tophat(arr_norm, selem)
    b_tophat = black_tophat(arr_norm, selem)
    return {
        "opened": opened.astype(np.float32) / 255,
        "closed": closed.astype(np.float32) / 255,
        "white_tophat": w_tophat.astype(np.float32) / 255,
        "black_tophat": b_tophat.astype(np.float32) / 255,
    }


def detect_linear_features(
    arr: np.ndarray, n_orientations: int = 8, line_length: int = 15
) -> np.ndarray:
    """Detect linear features (rivers, streams, canals)."""
    responses = []
    for angle in np.linspace(0, np.pi, n_orientations, endpoint=False):
        kernel_size = line_length
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        for t in range(-center, center + 1):
            x = int(center + t * np.cos(angle))
            y = int(center + t * np.sin(angle))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
        kernel = kernel / (kernel.sum() + 1e-10)
        response = ndimage.convolve(arr, kernel)
        responses.append(response)
    return np.maximum.reduce(responses)


def extract_features(chip: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Extract ALL 69 features from a chip (MUST MATCH TRAINING)."""
    from scipy.ndimage import uniform_filter, sobel, minimum_filter

    vv = chip[0]
    vh = chip[1]
    dem = chip[3]
    hand = chip[4]
    slope = chip[5]
    twi = chip[6]

    features = []
    names = []

    # =========================================================================
    # 1. BASIC SAR FEATURES (4 features)
    # =========================================================================
    features.extend([vv, vh])
    names.extend(["VV", "VH"])
    features.append(vv - vh)
    names.append("VV_minus_VH")
    vv_lin = db_to_linear(vv)
    vh_lin = db_to_linear(vh)
    ratio = vv_lin / (vh_lin + 1e-10)
    features.append(np.log10(ratio + 1e-10))
    names.append("VV_over_VH_log")

    # =========================================================================
    # 2. TOPOGRAPHIC FEATURES (6 features)
    # =========================================================================
    features.extend([hand, slope, twi, dem])
    names.extend(["HAND", "SLOPE", "TWI", "DEM"])
    dem_dx = sobel(dem, axis=1)
    dem_dy = sobel(dem, axis=0)
    dem_grad = np.sqrt(dem_dx**2 + dem_dy**2)
    dem_aspect = np.arctan2(dem_dy, dem_dx)
    features.extend([dem_grad, dem_aspect])
    names.extend(["DEM_gradient", "DEM_aspect"])

    # =========================================================================
    # 3. POLARIMETRIC FEATURES (4 features)
    # =========================================================================
    entropy = compute_pseudo_entropy(vv, vh, window=5)
    features.append(entropy)
    names.append("pseudo_entropy")
    alpha = compute_pseudo_alpha(vv, vh, window=5)
    features.append(alpha)
    names.append("pseudo_alpha")
    rvi = compute_rvi(vv, vh, window=5)
    features.append(rvi)
    names.append("RVI")
    span = compute_span(vv, vh)
    features.append(span)
    names.append("span")

    # =========================================================================
    # 4. MULTI-SCALE STATISTICS (26 features)
    # =========================================================================
    for scale in CONFIG["scales"]:
        vv_mean = uniform_filter(vv, size=scale)
        vv_sq_mean = uniform_filter(vv**2, size=scale)
        vv_std = np.sqrt(np.maximum(vv_sq_mean - vv_mean**2, 0))
        features.extend([vv_mean, vv_std])
        names.extend([f"VV_mean_s{scale}", f"VV_std_s{scale}"])

        vh_mean = uniform_filter(vh, size=scale)
        vh_sq_mean = uniform_filter(vh**2, size=scale)
        vh_std = np.sqrt(np.maximum(vh_sq_mean - vh_mean**2, 0))
        features.extend([vh_mean, vh_std])
        names.extend([f"VH_mean_s{scale}", f"VH_std_s{scale}"])

        if scale <= 9:
            vv_min = minimum_filter(vv, size=scale)
            vh_min = minimum_filter(vh, size=scale)
            features.extend([vv_min, vh_min])
            names.extend([f"VV_min_s{scale}", f"VH_min_s{scale}"])

    # =========================================================================
    # 5. SPECKLE STATISTICS (4 features)
    # =========================================================================
    for band, band_name in [(vv, "VV"), (vh, "VH")]:
        enl = compute_enl(band, window=9)
        features.append(enl)
        names.append(f"{band_name}_ENL")
        cv = compute_cv(band, window=9)
        features.append(cv)
        names.append(f"{band_name}_CV")

    # =========================================================================
    # 6. TEXTURE FEATURES (6 features)
    # =========================================================================
    for band, band_name in [(vv, "VV"), (vh, "VH")]:
        glcm_feats = compute_glcm_fast(band, window=11)
        for prop_name, prop_arr in glcm_feats.items():
            features.append(prop_arr)
            names.append(f"{band_name}_glcm_{prop_name}")

    # =========================================================================
    # 7. MORPHOLOGICAL FEATURES (4 features)
    # =========================================================================
    morph_feats = compute_morphological_features(vh)
    for morph_name, morph_arr in morph_feats.items():
        features.append(morph_arr)
        names.append(f"VH_{morph_name}")

    # =========================================================================
    # 8. LINEAR FEATURE DETECTION (1 feature)
    # =========================================================================
    line_response = detect_linear_features(vh, n_orientations=8, line_length=15)
    features.append(line_response)
    names.append("line_response")

    # =========================================================================
    # 9. ADAPTIVE THRESHOLDS (6 features)
    # =========================================================================
    vv_otsu = compute_otsu_threshold(vv)
    vh_otsu = compute_otsu_threshold(vh)
    features.append(vv - vv_otsu)
    features.append(vh - vh_otsu)
    names.extend(["VV_otsu_diff", "VH_otsu_diff"])
    features.append((vv < vv_otsu).astype(np.float32))
    features.append((vh < vh_otsu).astype(np.float32))
    names.extend(["VV_below_otsu", "VH_below_otsu"])
    vv_kapur = compute_kapur_threshold(vv)
    vh_kapur = compute_kapur_threshold(vh)
    features.append((vv < vv_kapur).astype(np.float32))
    features.append((vh < vh_kapur).astype(np.float32))
    names.extend(["VV_below_kapur", "VH_below_kapur"])

    # =========================================================================
    # 10. PHYSICS-BASED COMPOSITE SCORES (5 features)
    # =========================================================================
    hand_score = np.clip(1 - hand / PHYSICS["hand_threshold_init"], 0, 1)
    slope_score = np.clip(1 - slope / PHYSICS["slope_threshold_init"], 0, 1)
    vh_score = np.clip((PHYSICS["vh_dark_threshold_init"] - vh) / 10, 0, 1)
    twi_score = np.clip((twi - 5) / 10, 0, 1)
    physics_score = (hand_score * slope_score * vh_score * twi_score) ** 0.25
    features.append(physics_score)
    names.append("physics_composite")
    features.extend([hand_score, slope_score, vh_score, twi_score])
    names.extend(["hand_score", "slope_score", "vh_score", "twi_score"])

    # =========================================================================
    # 11. EDGE/GRADIENT FEATURES (3 features)
    # =========================================================================
    vh_dx = sobel(vh, axis=1)
    vh_dy = sobel(vh, axis=0)
    vh_grad = np.sqrt(vh_dx**2 + vh_dy**2)
    features.append(vh_grad)
    names.append("VH_gradient")
    vv_dx = sobel(vv, axis=1)
    vv_dy = sobel(vv, axis=0)
    vv_grad = np.sqrt(vv_dx**2 + vv_dy**2)
    features.append(vv_grad)
    names.append("VV_gradient")
    vh_lap = ndimage.laplace(vh)
    features.append(vh_lap)
    names.append("VH_laplacian")

    # =========================================================================
    # STACK AND RETURN
    # =========================================================================
    feature_stack = np.stack(features, axis=0).astype(np.float32)
    feature_stack = np.nan_to_num(feature_stack, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_stack, names


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================


def analyze_narrow_water_bodies(
    pred: np.ndarray, truth: np.ndarray, width_thresholds: List[int] = [3, 5, 10, 20]
) -> Dict:
    """
    Analyze detection accuracy for narrow water bodies (rivers, streams).

    Narrow water bodies are defined by their width in pixels.
    """
    results = {}

    # Compute distance transform of water bodies
    truth_binary = (truth > 0.5).astype(np.uint8)
    pred_binary = (pred > 0.5).astype(np.uint8)

    if truth_binary.sum() == 0:
        return {"error": "No water in truth mask"}

    # Distance from water edge (inside water)
    dist_inside = distance_transform_edt(truth_binary)

    # Width is 2x distance to nearest edge
    water_width = dist_inside * 2

    for max_width in width_thresholds:
        # Mask for narrow water (width < threshold)
        narrow_mask = (truth_binary > 0) & (water_width <= max_width)

        if narrow_mask.sum() == 0:
            results[f"width_le_{max_width}px"] = {
                "n_pixels": 0,
                "recall": None,
                "precision": None,
            }
            continue

        # Metrics for narrow water only
        narrow_truth = truth_binary[narrow_mask]
        narrow_pred = pred_binary[narrow_mask]

        tp = np.sum(narrow_pred & narrow_truth)
        fn = np.sum((~narrow_pred) & narrow_truth)
        fp = np.sum(narrow_pred & (~narrow_truth))

        recall = tp / (tp + fn + 1e-10)
        precision = tp / (tp + fp + 1e-10) if (tp + fp) > 0 else 0

        results[f"width_le_{max_width}px"] = {
            "n_pixels": int(narrow_mask.sum()),
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(2 * precision * recall / (precision + recall + 1e-10)),
        }

    return results


def analyze_boundary_accuracy(
    pred: np.ndarray, truth: np.ndarray, distance_thresholds: List[int] = [1, 2, 3, 5]
) -> Dict:
    """
    Analyze accuracy at land-water boundaries.

    Measures how well the model detects edges/boundaries.
    """
    truth_binary = (truth > 0.5).astype(np.uint8)
    pred_binary = (pred > 0.5).astype(np.uint8)

    # Find boundary pixels (edge of water)
    eroded = binary_erosion(truth_binary, iterations=1)
    boundary_mask = truth_binary & (~eroded)

    if boundary_mask.sum() == 0:
        return {"error": "No boundary pixels found"}

    # Distance from boundary
    dist_from_boundary = distance_transform_edt(~boundary_mask)

    results = {}

    for max_dist in distance_thresholds:
        # Pixels within max_dist of boundary
        near_boundary = dist_from_boundary <= max_dist

        if near_boundary.sum() == 0:
            continue

        # Metrics for near-boundary pixels
        boundary_truth = truth_binary[near_boundary]
        boundary_pred = pred_binary[near_boundary]

        tp = np.sum(boundary_pred & boundary_truth)
        tn = np.sum((~boundary_pred) & (~boundary_truth))
        fp = np.sum(boundary_pred & (~boundary_truth))
        fn = np.sum((~boundary_pred) & boundary_truth)

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
        iou = tp / (tp + fp + fn + 1e-10)

        results[f"within_{max_dist}px_of_boundary"] = {
            "n_pixels": int(near_boundary.sum()),
            "accuracy": float(accuracy),
            "iou": float(iou),
            "false_positive_rate": float(fp / (fp + tn + 1e-10)),
            "false_negative_rate": float(fn / (fn + tp + 1e-10)),
        }

    return results


def analyze_false_positives(
    pred: np.ndarray,
    truth: np.ndarray,
    hand: np.ndarray,
    slope: np.ndarray,
    vh: np.ndarray,
) -> Dict:
    """
    Analyze where false positives occur and why.

    Categories:
    - Shadow areas (low VH but high slope)
    - Wet soil (low VH, low HAND, but not water)
    - Urban areas (geometric patterns)
    """
    pred_binary = (pred > 0.5).astype(np.uint8)
    truth_binary = (truth > 0.5).astype(np.uint8)

    # False positives
    fp_mask = pred_binary & (~truth_binary)
    # False negatives
    fn_mask = (~pred_binary) & truth_binary
    # True positives
    tp_mask = pred_binary & truth_binary

    n_fp = fp_mask.sum()
    n_fn = fn_mask.sum()
    n_tp = tp_mask.sum()

    results = {
        "total_false_positives": int(n_fp),
        "total_false_negatives": int(n_fn),
        "total_true_positives": int(n_tp),
        "fp_rate": float(n_fp / (n_fp + n_tp + 1e-10)),
        "fn_rate": float(n_fn / (n_fn + n_tp + 1e-10)),
    }

    if n_fp > 0:
        # Analyze FP by category
        fp_vh = vh[fp_mask]
        fp_hand = hand[fp_mask]
        fp_slope = slope[fp_mask]

        # Shadow-like FP (steep slope, low VH)
        shadow_like = (fp_slope > 15) & (fp_vh < -15)
        results["fp_shadow_like"] = int(shadow_like.sum())
        results["fp_shadow_like_pct"] = float(shadow_like.sum() / n_fp * 100)

        # Wet soil FP (low HAND, moderate VH)
        wet_soil = (fp_hand < 5) & (fp_vh > -20) & (fp_vh < -10)
        results["fp_wet_soil"] = int(wet_soil.sum())
        results["fp_wet_soil_pct"] = float(wet_soil.sum() / n_fp * 100)

        # Statistics of FP pixels
        results["fp_vh_mean"] = float(np.mean(fp_vh))
        results["fp_vh_std"] = float(np.std(fp_vh))
        results["fp_hand_mean"] = float(np.mean(fp_hand))
        results["fp_slope_mean"] = float(np.mean(fp_slope))

    if n_fn > 0:
        # Analyze FN (missed water)
        fn_vh = vh[fn_mask]
        fn_hand = hand[fn_mask]

        results["fn_vh_mean"] = float(np.mean(fn_vh))
        results["fn_hand_mean"] = float(np.mean(fn_hand))

        # Bright water FN (high VH - rough water)
        bright_water = fn_vh > -15
        results["fn_bright_water"] = int(bright_water.sum())
        results["fn_bright_water_pct"] = float(bright_water.sum() / n_fn * 100)

    return results


def compute_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict:
    """Compute comprehensive metrics."""
    pred_flat = pred.flatten()
    truth_flat = truth.flatten()

    pred_binary = (pred_flat > 0.5).astype(int)
    truth_binary = (truth_flat > 0.5).astype(int)

    tp = np.sum(pred_binary & truth_binary)
    tn = np.sum((~pred_binary) & (~truth_binary))
    fp = np.sum(pred_binary & (~truth_binary))
    fn = np.sum((~pred_binary) & truth_binary)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    iou = tp / (tp + fp + fn + 1e-10)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


# =============================================================================
# ENSEMBLE
# =============================================================================


def create_ensemble_predictions(
    lgb_pred: np.ndarray,
    unet_pred: np.ndarray,
    weights: Tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    """Create weighted ensemble of LightGBM and U-Net predictions."""
    return weights[0] * lgb_pred + weights[1] * unet_pred


def find_optimal_ensemble_weights(
    lgb_preds: List[np.ndarray], unet_preds: List[np.ndarray], truths: List[np.ndarray]
) -> Tuple[float, float]:
    """Find optimal ensemble weights using grid search."""
    best_iou = 0
    best_weights = (0.5, 0.5)

    for lgb_weight in np.arange(0, 1.05, 0.1):
        unet_weight = 1 - lgb_weight

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for lgb_pred, unet_pred, truth in zip(lgb_preds, unet_preds, truths):
            ensemble = lgb_weight * lgb_pred + unet_weight * unet_pred
            pred_binary = (ensemble > 0.5).astype(int).flatten()
            truth_binary = (truth > 0.5).astype(int).flatten()

            total_tp += np.sum(pred_binary & truth_binary)
            total_fp += np.sum(pred_binary & (~truth_binary))
            total_fn += np.sum((~pred_binary) & truth_binary)

        iou = total_tp / (total_tp + total_fp + total_fn + 1e-10)

        if iou > best_iou:
            best_iou = iou
            best_weights = (lgb_weight, unet_weight)

    return best_weights


# =============================================================================
# VISUALIZATION
# =============================================================================


def plot_prediction_comparison(
    chip: np.ndarray,
    lgb_pred: np.ndarray,
    unet_pred: np.ndarray,
    ensemble_pred: np.ndarray,
    chip_name: str,
    save_dir: Path,
):
    """Create visualization comparing predictions."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    vv = chip[0]
    vh = chip[1]
    truth = chip[7]

    # Row 1: Input and truth
    axes[0, 0].imshow(vv, cmap="gray", vmin=-30, vmax=0)
    axes[0, 0].set_title("VV (dB)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(vh, cmap="gray", vmin=-35, vmax=-5)
    axes[0, 1].set_title("VH (dB)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(truth, cmap="Blues", vmin=0, vmax=1)
    axes[0, 2].set_title("Ground Truth")
    axes[0, 2].axis("off")

    # RGB composite
    rgb = np.stack(
        [
            np.clip((vv + 30) / 30, 0, 1),
            np.clip((vh + 35) / 30, 0, 1),
            np.clip((vv - vh + 15) / 20, 0, 1),
        ],
        axis=-1,
    )
    axes[0, 3].imshow(rgb)
    axes[0, 3].set_title("VV-VH-Ratio RGB")
    axes[0, 3].axis("off")

    # Row 2: Predictions
    axes[1, 0].imshow(lgb_pred, cmap="Blues", vmin=0, vmax=1)
    axes[1, 0].set_title("LightGBM Prediction")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(unet_pred, cmap="Blues", vmin=0, vmax=1)
    axes[1, 1].set_title("U-Net Prediction")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(ensemble_pred, cmap="Blues", vmin=0, vmax=1)
    axes[1, 2].set_title("Ensemble Prediction")
    axes[1, 2].axis("off")

    # Error map
    pred_binary = (ensemble_pred > 0.5).astype(int)
    truth_binary = (truth > 0.5).astype(int)
    error_map = np.zeros((*truth.shape, 3))
    error_map[pred_binary & truth_binary] = [0, 1, 0]  # TP = green
    error_map[(~pred_binary) & truth_binary] = [1, 0, 0]  # FN = red
    error_map[pred_binary & (~truth_binary)] = [1, 1, 0]  # FP = yellow
    axes[1, 3].imshow(error_map)
    axes[1, 3].set_title("Error Map (TP=green, FN=red, FP=yellow)")
    axes[1, 3].axis("off")

    plt.suptitle(f"Chip: {chip_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / f"{chip_name}_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(importance: Dict[str, int], save_path: Path):
    """Plot feature importance."""
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:25]

    fig, ax = plt.subplots(figsize=(12, 10))

    names = [x[0] for x in sorted_imp]
    values = [x[1] for x in sorted_imp]

    colors = []
    for name in names:
        if "VV" in name or "VH" in name:
            colors.append("#2196F3")  # Blue for SAR
        elif name in ["HAND", "SLOPE", "TWI", "DEM"]:
            colors.append("#4CAF50")  # Green for topo
        else:
            colors.append("#FF9800")  # Orange for derived

    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (split count)")
    ax.set_title("Top 25 Feature Importance\n(Blue=SAR, Green=Topo, Orange=Derived)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_narrow_water_analysis(narrow_results: Dict, save_path: Path):
    """Plot narrow water body detection analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))

    widths = []
    recalls = []
    f1s = []

    for key, val in narrow_results.items():
        if val.get("recall") is not None:
            width = int(key.split("_")[2].replace("px", ""))
            widths.append(width)
            recalls.append(val["recall"] * 100)
            f1s.append(val["f1"] * 100)

    x = np.arange(len(widths))
    width_bar = 0.35

    ax.bar(x - width_bar / 2, recalls, width_bar, label="Recall (%)", color="#2196F3")
    ax.bar(x + width_bar / 2, f1s, width_bar, label="F1 (%)", color="#4CAF50")

    ax.set_xlabel("Maximum Water Body Width (pixels)")
    ax.set_ylabel("Score (%)")
    ax.set_title("Narrow Water Body Detection Performance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"≤{w}px" for w in widths])
    ax.legend()
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# MAIN EVALUATION
# =============================================================================


def main():
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE EVALUATION & ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"Device: {CONFIG['device']}")

    device = torch.device(CONFIG["device"])

    # Load data
    chips, names = load_chips()

    # Split same as training
    train_chips, test_chips, train_names, test_names = train_test_split(
        chips, names, test_size=0.15, random_state=CONFIG["random_seed"]
    )

    logger.info(f"Test set: {len(test_chips)} chips")

    # Load LightGBM model
    logger.info("\nLoading LightGBM model...")
    lgb_model = lgb.Booster(
        model_file=str(CONFIG["model_dir"] / "lightgbm_v4_comprehensive.txt")
    )

    # Load U-Net model
    logger.info("Loading U-Net model...")
    unet_model = UNetV4(in_channels=6).to(device)
    checkpoint = torch.load(
        CONFIG["model_dir"] / "unet_v4_best.pth",
        weights_only=False,
        map_location=device,
    )
    unet_model.load_state_dict(checkpoint["model_state_dict"])
    unet_model.eval()
    logger.info(f"U-Net best val IoU: {checkpoint['val_iou']:.4f}")

    # Collect predictions
    all_lgb_preds = []
    all_unet_preds = []
    all_truths = []
    all_metrics = []
    all_narrow_results = []
    all_boundary_results = []
    all_fp_results = []

    logger.info("\nRunning predictions on test set...")

    for i, (chip, name) in enumerate(zip(test_chips, test_names)):
        if (i + 1) % 10 == 0:
            logger.info(f"  Processing {i + 1}/{len(test_chips)}")

        truth = chip[7]
        vv = chip[0]
        vh = chip[1]
        hand = chip[4]
        slope = chip[5]

        # LightGBM prediction
        features, feat_names = extract_features(chip)
        h, w = truth.shape
        X = features.reshape(features.shape[0], -1).T
        lgb_proba = lgb_model.predict(X)
        lgb_pred = lgb_proba.reshape(h, w)

        # U-Net prediction
        # Prepare input
        unet_input = np.stack(
            [chip[0], chip[1], chip[3], chip[4], chip[5], chip[6]], axis=0
        )
        unet_input = zoom(
            unet_input, (1, CONFIG["image_size"] / h, CONFIG["image_size"] / w), order=1
        )

        # Normalize
        for c in range(unet_input.shape[0]):
            mean = np.nanmean(unet_input[c])
            std = np.nanstd(unet_input[c]) + 1e-8
            unet_input[c] = np.clip((unet_input[c] - mean) / std, -10, 10)

        unet_input = np.nan_to_num(unet_input, nan=0.0)
        unet_tensor = torch.from_numpy(unet_input).float().unsqueeze(0).to(device)

        with torch.no_grad():
            unet_logits = unet_model(unet_tensor)
            unet_proba = torch.sigmoid(unet_logits).cpu().numpy()[0, 0]

        # Resize back to original
        unet_pred = zoom(
            unet_proba, (h / CONFIG["image_size"], w / CONFIG["image_size"]), order=1
        )

        # Store predictions
        all_lgb_preds.append(lgb_pred)
        all_unet_preds.append(unet_pred)
        all_truths.append(truth)

        # Compute metrics for this chip
        metrics = {
            "name": name,
            "lgb": compute_metrics(lgb_pred, truth),
            "unet": compute_metrics(unet_pred, truth),
        }
        all_metrics.append(metrics)

        # Narrow water analysis
        narrow = analyze_narrow_water_bodies(lgb_pred, truth)
        all_narrow_results.append(narrow)

        # Boundary analysis
        boundary = analyze_boundary_accuracy(lgb_pred, truth)
        all_boundary_results.append(boundary)

        # False positive analysis
        fp = analyze_false_positives(lgb_pred, truth, hand, slope, vh)
        all_fp_results.append(fp)

    # Find optimal ensemble weights
    logger.info("\nFinding optimal ensemble weights...")
    best_weights = find_optimal_ensemble_weights(
        all_lgb_preds, all_unet_preds, all_truths
    )
    logger.info(
        f"Optimal weights: LightGBM={best_weights[0]:.2f}, U-Net={best_weights[1]:.2f}"
    )

    # Compute ensemble predictions and metrics
    ensemble_preds = [
        create_ensemble_predictions(lgb, unet, best_weights)
        for lgb, unet in zip(all_lgb_preds, all_unet_preds)
    ]

    # Overall metrics
    lgb_all = np.concatenate([p.flatten() for p in all_lgb_preds])
    unet_all = np.concatenate([p.flatten() for p in all_unet_preds])
    ensemble_all = np.concatenate([p.flatten() for p in ensemble_preds])
    truth_all = np.concatenate([t.flatten() for t in all_truths])

    overall_results = {
        "lightgbm": compute_metrics(lgb_all, truth_all),
        "unet": compute_metrics(unet_all, truth_all),
        "ensemble": compute_metrics(ensemble_all, truth_all),
        "ensemble_weights": {"lgb": best_weights[0], "unet": best_weights[1]},
    }

    logger.info("\n" + "=" * 60)
    logger.info("OVERALL TEST RESULTS")
    logger.info("=" * 60)
    logger.info(
        f"LightGBM - IoU: {overall_results['lightgbm']['iou']:.4f}, F1: {overall_results['lightgbm']['f1']:.4f}"
    )
    logger.info(
        f"U-Net    - IoU: {overall_results['unet']['iou']:.4f}, F1: {overall_results['unet']['f1']:.4f}"
    )
    logger.info(
        f"Ensemble - IoU: {overall_results['ensemble']['iou']:.4f}, F1: {overall_results['ensemble']['f1']:.4f}"
    )

    # Aggregate narrow water results
    logger.info("\n" + "=" * 60)
    logger.info("NARROW WATER BODY DETECTION")
    logger.info("=" * 60)

    agg_narrow = {}
    for width in [3, 5, 10, 20]:
        key = f"width_le_{width}px"
        recalls = [
            r[key]["recall"]
            for r in all_narrow_results
            if r.get(key, {}).get("recall") is not None
        ]
        if recalls:
            agg_narrow[key] = {
                "mean_recall": float(np.mean(recalls)),
                "std_recall": float(np.std(recalls)),
            }
            logger.info(
                f"Width ≤{width}px: Recall = {np.mean(recalls) * 100:.1f}% ± {np.std(recalls) * 100:.1f}%"
            )

    # Aggregate boundary results
    logger.info("\n" + "=" * 60)
    logger.info("LAND-WATER BOUNDARY ACCURACY")
    logger.info("=" * 60)

    agg_boundary = {}
    for dist in [1, 2, 3, 5]:
        key = f"within_{dist}px_of_boundary"
        ious = [
            r[key]["iou"]
            for r in all_boundary_results
            if r.get(key, {}).get("iou") is not None
        ]
        if ious:
            agg_boundary[key] = {
                "mean_iou": float(np.mean(ious)),
                "std_iou": float(np.std(ious)),
            }
            logger.info(
                f"Within {dist}px: IoU = {np.mean(ious) * 100:.1f}% ± {np.std(ious) * 100:.1f}%"
            )

    # Aggregate FP analysis
    logger.info("\n" + "=" * 60)
    logger.info("FALSE POSITIVE ANALYSIS")
    logger.info("=" * 60)

    total_fp = sum(r["total_false_positives"] for r in all_fp_results)
    total_fn = sum(r["total_false_negatives"] for r in all_fp_results)
    total_tp = sum(r["total_true_positives"] for r in all_fp_results)

    shadow_fp = sum(r.get("fp_shadow_like", 0) for r in all_fp_results)
    wet_soil_fp = sum(r.get("fp_wet_soil", 0) for r in all_fp_results)

    logger.info(f"Total FP: {total_fp}, FN: {total_fn}, TP: {total_tp}")
    logger.info(f"FP Rate: {total_fp / (total_fp + total_tp) * 100:.2f}%")
    logger.info(f"FN Rate: {total_fn / (total_fn + total_tp) * 100:.2f}%")
    logger.info(
        f"Shadow-like FP: {shadow_fp} ({shadow_fp / total_fp * 100:.1f}% of FP)"
    )
    logger.info(
        f"Wet-soil FP: {wet_soil_fp} ({wet_soil_fp / total_fp * 100:.1f}% of FP)"
    )

    # Save visualizations
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 60)

    # Save sample predictions
    for i in range(min(10, len(test_chips))):
        plot_prediction_comparison(
            test_chips[i],
            all_lgb_preds[i],
            all_unet_preds[i],
            ensemble_preds[i],
            test_names[i],
            CONFIG["viz_dir"],
        )
        logger.info(f"  Saved visualization for {test_names[i]}")

    # Feature importance plot
    with open(CONFIG["results_dir"] / "training_results_v4.json") as f:
        lgb_results = json.load(f)

    if (
        "lightgbm_v4" in lgb_results
        and "feature_importance" in lgb_results["lightgbm_v4"]
    ):
        plot_feature_importance(
            lgb_results["lightgbm_v4"]["feature_importance"],
            CONFIG["viz_dir"] / "feature_importance.png",
        )
        logger.info("  Saved feature importance plot")

    # Narrow water plot
    plot_narrow_water_analysis(
        agg_narrow, CONFIG["viz_dir"] / "narrow_water_analysis.png"
    )
    logger.info("  Saved narrow water analysis plot")

    # Save comprehensive results
    comprehensive_results = {
        "timestamp": datetime.now().isoformat(),
        "test_chips": len(test_chips),
        "overall": overall_results,
        "narrow_water_detection": agg_narrow,
        "boundary_accuracy": agg_boundary,
        "false_positive_analysis": {
            "total_fp": total_fp,
            "total_fn": total_fn,
            "total_tp": total_tp,
            "fp_rate": total_fp / (total_fp + total_tp),
            "fn_rate": total_fn / (total_fn + total_tp),
            "shadow_like_fp_pct": shadow_fp / total_fp * 100 if total_fp > 0 else 0,
            "wet_soil_fp_pct": wet_soil_fp / total_fp * 100 if total_fp > 0 else 0,
        },
        "per_chip_metrics": all_metrics,
    }

    with open(
        CONFIG["results_dir"] / "comprehensive_evaluation_results.json", "w"
    ) as f:
        json.dump(comprehensive_results, f, indent=2)

    logger.info(
        f"\nResults saved to: {CONFIG['results_dir'] / 'comprehensive_evaluation_results.json'}"
    )
    logger.info(f"Visualizations saved to: {CONFIG['viz_dir']}")
    logger.info(f"\nCompleted: {datetime.now().isoformat()}")

    return comprehensive_results


if __name__ == "__main__":
    main()
