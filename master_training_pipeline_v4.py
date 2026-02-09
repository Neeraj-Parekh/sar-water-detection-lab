#!/usr/bin/env python3
"""
================================================================================
MASTER SAR WATER DETECTION TRAINING PIPELINE v4
================================================================================
Comprehensive physics-guided water detection with advanced mathematics.

Key Features:
1. POLARIMETRIC SAR DECOMPOSITION
   - Dual-pol pseudo-entropy, pseudo-alpha, pseudo-anisotropy
   - VV/VH ratio (water indicator)
   - Radar Vegetation Index (RVI)

2. TEXTURE ANALYSIS (GLCM)
   - Contrast, dissimilarity, homogeneity, energy, correlation, ASM
   - Multi-scale texture at 3 window sizes

3. SPECKLE STATISTICS
   - Equivalent Number of Looks (ENL)
   - Coefficient of Variation (CV)
   - Local heterogeneity metrics

4. MORPHOLOGICAL FEATURES
   - Opening/closing for noise removal
   - Top-hat transforms for bright/dark features
   - Linear structure detection (oriented filters)

5. PHYSICS-BASED FEATURES
   - HAND (Height Above Nearest Drainage)
   - TWI (Topographic Wetness Index)
   - Slope-based water probability
   - Adaptive thresholds (Otsu, Kapur entropy)

6. ADVANCED LOSS FUNCTIONS
   - Lovasz-Softmax (IoU-optimized)
   - Focal Tversky (handles class imbalance)
   - Boundary-aware loss (edges)
   - Combined physics-aware loss

7. ATTENTION MECHANISMS
   - CBAM (Channel + Spatial attention)
   - Physics attention (learned thresholds)
   - Multi-scale attention fusion

8. ENSEMBLE FUSION
   - LightGBM pixel-wise
   - U-Net semantic
   - Weighted average with uncertainty

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
from scipy import ndimage, stats
from scipy.ndimage import uniform_filter, gaussian_filter
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import disk, opening, closing, white_tophat, black_tophat
from skimage.filters import threshold_otsu, gabor
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
import lightgbm as lgb

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
    "version": "4.0",
    "random_seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # Data paths (will be set based on environment)
    "chip_dirs": [],
    "model_dir": None,
    "results_dir": None,
    # SAR-only mode (no MNDWI optical leakage)
    "sar_only": True,
    # Band indices
    "bands": {
        "VV": 0,
        "VH": 1,
        "MNDWI": 2,
        "DEM": 3,
        "HAND": 4,
        "SLOPE": 5,
        "TWI": 6,
        "TRUTH": 7,
    },
    # Physics priors (learnable but initialized with domain knowledge)
    "physics": {
        "vh_dark_threshold_init": -18.0,  # dB, water is dark
        "vv_dark_threshold_init": -15.0,  # dB
        "hand_threshold_init": 10.0,  # meters, water at low HAND
        "slope_threshold_init": 15.0,  # degrees, water on flat areas
        "twi_threshold_init": 12.0,  # high TWI = wet
    },
    # Multi-scale windows for features
    "scales": [3, 5, 9, 15, 21],
    # GLCM parameters
    "glcm": {
        "distances": [1, 2, 4],
        "angles": [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        "quantize_levels": 64,
    },
    # LightGBM hyperparameters
    "lgb_params": {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "max_depth": 8,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
    },
    # U-Net training
    "unet": {
        "image_size": 256,
        "batch_size": 8,
        "epochs": 150,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "patience": 30,
    },
    # Loss weights
    "loss_weights": {
        "bce": 0.3,
        "lovasz": 0.3,
        "focal_tversky": 0.2,
        "boundary": 0.1,
        "physics": 0.1,
    },
    # Cross-validation
    "n_folds": 5,
    "sample_rate": 0.1,  # Fraction of pixels to sample
}

# Setup paths based on environment
if os.path.exists("/home/mit-aoe/sar_water_detection"):
    # GPU server
    CONFIG["chip_dirs"] = [
        Path("/home/mit-aoe/sar_water_detection/chips"),
        Path("/home/mit-aoe/sar_water_detection/chips_expanded"),
    ]
    CONFIG["model_dir"] = Path("/home/mit-aoe/sar_water_detection/models")
    CONFIG["results_dir"] = Path("/home/mit-aoe/sar_water_detection/results")
else:
    # Local machine
    CONFIG["chip_dirs"] = [
        Path("/media/neeraj-parekh/Data1/sar soil system/chips/gui/chips"),
        Path("/media/neeraj-parekh/Data1/sar soil system/chips/gui/chips_expanded"),
    ]
    CONFIG["model_dir"] = Path(
        "/media/neeraj-parekh/Data1/sar soil system/chips/gui/models"
    )
    CONFIG["results_dir"] = Path(
        "/media/neeraj-parekh/Data1/sar soil system/chips/gui/results"
    )

CONFIG["model_dir"].mkdir(exist_ok=True)
CONFIG["results_dir"].mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(CONFIG["results_dir"] / "training_v4.log"),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# MATHEMATICAL FOUNDATIONS
# =============================================================================

"""
POLARIMETRIC SAR THEORY
=======================
For dual-pol (VV, VH) data, we can compute pseudo-polarimetric parameters:

1. PSEUDO-ENTROPY (H_pseudo):
   Measures randomness of scattering. Water has LOW entropy (specular reflection).
   
   For dual-pol: H = -sum(p_i * log2(p_i)) where p_i = lambda_i / sum(lambda)
   With only VV and VH, we use eigenvalues of 2x2 covariance matrix:
   C2 = [[<|S_VV|^2>, <S_VV*S_VH>], [<S_VH*S_VV>, <|S_VH|^2>]]
   
   Simplified: H_pseudo = -p*log2(p) - (1-p)*log2(1-p) where p = VV/(VV+VH)

2. PSEUDO-ALPHA:
   Scattering mechanism indicator. Low alpha = surface scattering (water).
   alpha_pseudo = arctan(|VH|/|VV|) in radians

3. VV/VH RATIO (Depolarization Ratio):
   High ratio = low depolarization = smooth surface = water
   DR = VV_linear / VH_linear

4. RADAR VEGETATION INDEX (RVI):
   RVI = 4 * VH / (VV + VH)
   Low RVI indicates water (low volume scattering)


TEXTURE ANALYSIS (GLCM)
=======================
Gray-Level Co-occurrence Matrix captures spatial relationships:

P(i,j|d,θ) = probability of finding gray levels i and j separated by distance d at angle θ

Properties computed:
- Contrast: sum((i-j)^2 * P(i,j)) - high for heterogeneous regions
- Dissimilarity: sum(|i-j| * P(i,j)) - similar to contrast
- Homogeneity: sum(P(i,j) / (1 + |i-j|)) - high for water (uniform)
- Energy (ASM): sum(P(i,j)^2) - high for uniform regions
- Correlation: sum((i-μ_i)(j-μ_j)P(i,j) / (σ_i * σ_j))
- Entropy: -sum(P(i,j) * log(P(i,j))) - measure of randomness


SPECKLE STATISTICS
==================
SAR images have multiplicative speckle noise. Statistics help characterize it:

1. Equivalent Number of Looks (ENL):
   ENL = (μ/σ)^2 where μ=mean, σ=std
   High ENL = homogeneous region = water
   Typically ENL > 10 for water bodies

2. Coefficient of Variation (CV):
   CV = σ/μ
   Low CV = homogeneous = water
   For water: CV ≈ 0.2-0.3 (in linear scale)


TOPOGRAPHIC INDICES
===================
1. HAND (Height Above Nearest Drainage):
   HAND = DEM - elevation_of_nearest_stream
   Water bodies: HAND ≈ 0-5 meters

2. TWI (Topographic Wetness Index):
   TWI = ln(a / tan(β))
   where a = upstream contributing area, β = slope
   High TWI = likely to accumulate water

3. Slope:
   Water is flat, typically < 5 degrees


LOSS FUNCTIONS
==============
1. Lovasz-Softmax:
   Directly optimizes Jaccard index (IoU)
   L_lovasz = lovasz_extension(errors, labels)
   
2. Focal Tversky Loss:
   Handles class imbalance by focusing on hard examples
   TI = (TP) / (TP + α*FN + β*FP)
   L_ft = (1 - TI)^γ where γ focuses on hard examples

3. Boundary Loss:
   Penalizes errors near edges more heavily
   L_boundary = weighted_BCE where weight = exp(distance_to_edge)
"""


# =============================================================================
# FEATURE EXTRACTION FUNCTIONS
# =============================================================================


def db_to_linear(db: np.ndarray) -> np.ndarray:
    """Convert dB to linear scale: linear = 10^(dB/10)"""
    return np.power(10, db / 10)


def linear_to_db(linear: np.ndarray) -> np.ndarray:
    """Convert linear to dB: dB = 10 * log10(linear)"""
    return 10 * np.log10(np.clip(linear, 1e-10, None))


def compute_pseudo_entropy(
    vv_db: np.ndarray, vh_db: np.ndarray, window: int = 5
) -> np.ndarray:
    """
    Compute pseudo-entropy from dual-pol SAR.

    H_pseudo = -p*log2(p) - (1-p)*log2(1-p)
    where p = VV_linear / (VV_linear + VH_linear)

    Water has LOW entropy (specular reflection, dominated by VV).
    """
    vv_lin = db_to_linear(vv_db)
    vh_lin = db_to_linear(vh_db)

    # Local means for stable estimation
    vv_local = uniform_filter(vv_lin, size=window)
    vh_local = uniform_filter(vh_lin, size=window)

    total = vv_local + vh_local + 1e-10
    p = vv_local / total

    # Clip to avoid log(0)
    p = np.clip(p, 1e-10, 1 - 1e-10)

    # Shannon entropy (base 2)
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    return entropy


def compute_pseudo_alpha(
    vv_db: np.ndarray, vh_db: np.ndarray, window: int = 5
) -> np.ndarray:
    """
    Compute pseudo-alpha angle from dual-pol SAR.

    alpha = arctan(|VH| / |VV|) in radians

    Low alpha (< 30°) = surface scattering = water
    High alpha (> 45°) = volume scattering = vegetation
    """
    vv_lin = db_to_linear(vv_db)
    vh_lin = db_to_linear(vh_db)

    vv_local = uniform_filter(vv_lin, size=window)
    vh_local = uniform_filter(vh_lin, size=window)

    # Alpha in radians
    alpha = np.arctan2(np.sqrt(vh_local), np.sqrt(vv_local))

    # Convert to degrees for interpretability
    return np.degrees(alpha)


def compute_depolarization_ratio(vv_db: np.ndarray, vh_db: np.ndarray) -> np.ndarray:
    """
    Compute VV/VH ratio (depolarization ratio).

    High ratio = low depolarization = smooth surface = WATER
    Low ratio = high depolarization = rough/vegetated
    """
    return vv_db - vh_db  # In dB, ratio becomes difference


def compute_rvi(vv_db: np.ndarray, vh_db: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Radar Vegetation Index.

    RVI = 4 * VH / (VV + VH)

    Range: 0-1
    Low RVI = water (low volume scattering)
    High RVI = vegetation (high volume scattering)
    """
    vv_lin = db_to_linear(vv_db)
    vh_lin = db_to_linear(vh_db)

    vv_local = uniform_filter(vv_lin, size=window)
    vh_local = uniform_filter(vh_lin, size=window)

    rvi = 4 * vh_local / (vv_local + vh_local + 1e-10)

    return rvi


def compute_span(vv_db: np.ndarray, vh_db: np.ndarray) -> np.ndarray:
    """
    Total power (span) = VV + 2*VH (for reciprocal case).
    """
    vv_lin = db_to_linear(vv_db)
    vh_lin = db_to_linear(vh_db)

    span_lin = vv_lin + 2 * vh_lin
    return linear_to_db(span_lin)


def compute_glcm_features(
    arr: np.ndarray,
    window: int = 21,
    distances: List[int] = [1, 2],
    angles: List[float] = [0, np.pi / 2],
    levels: int = 32,
) -> Dict[str, np.ndarray]:
    """
    Compute GLCM texture features using sliding windows.

    Features: contrast, dissimilarity, homogeneity, energy, correlation, ASM
    """
    # Quantize to levels
    arr_min, arr_max = np.nanpercentile(arr, [1, 99])
    arr_norm = np.clip((arr - arr_min) / (arr_max - arr_min + 1e-10), 0, 1)
    arr_quant = (arr_norm * (levels - 1)).astype(np.uint8)

    h, w = arr.shape
    pad = window // 2

    # Initialize output arrays
    features = {
        "contrast": np.zeros_like(arr, dtype=np.float32),
        "dissimilarity": np.zeros_like(arr, dtype=np.float32),
        "homogeneity": np.zeros_like(arr, dtype=np.float32),
        "energy": np.zeros_like(arr, dtype=np.float32),
        "correlation": np.zeros_like(arr, dtype=np.float32),
    }

    # Compute GLCM for each window (stride for speed)
    stride = window // 2

    for i in range(pad, h - pad, stride):
        for j in range(pad, w - pad, stride):
            patch = arr_quant[i - pad : i + pad + 1, j - pad : j + pad + 1]

            try:
                glcm = graycomatrix(
                    patch,
                    distances=distances,
                    angles=angles,
                    levels=levels,
                    symmetric=True,
                    normed=True,
                )

                for prop in features.keys():
                    val = graycoprops(glcm, prop).mean()
                    # Fill the stride block
                    i_end = min(i + stride, h - pad)
                    j_end = min(j + stride, w - pad)
                    features[prop][i:i_end, j:j_end] = val
            except:
                pass

    # Fill edges
    for prop in features:
        features[prop][:pad, :] = features[prop][pad, :]
        features[prop][-pad:, :] = features[prop][-pad - 1, :]
        features[prop][:, :pad] = features[prop][:, pad : pad + 1]
        features[prop][:, -pad:] = features[prop][:, -pad - 1 : -pad]

    return features


def compute_glcm_fast(arr: np.ndarray, window: int = 11) -> Dict[str, np.ndarray]:
    """
    Fast approximation of GLCM-like features using local statistics.
    Much faster than full GLCM computation.
    """
    # Local variance (approximates contrast)
    local_mean = uniform_filter(arr.astype(np.float64), size=window)
    local_sq_mean = uniform_filter((arr.astype(np.float64)) ** 2, size=window)
    local_var = local_sq_mean - local_mean**2
    contrast = np.sqrt(np.maximum(local_var, 0))

    # Homogeneity approximation (inverse of local range)
    from scipy.ndimage import maximum_filter, minimum_filter

    local_max = maximum_filter(arr, size=window)
    local_min = minimum_filter(arr, size=window)
    local_range = local_max - local_min + 1e-10
    homogeneity = 1.0 / (1.0 + local_range)

    # Energy approximation (uniformity)
    energy = 1.0 / (1.0 + contrast)

    return {
        "contrast": contrast.astype(np.float32),
        "homogeneity": homogeneity.astype(np.float32),
        "energy": energy.astype(np.float32),
    }


def compute_enl(arr: np.ndarray, window: int = 9) -> np.ndarray:
    """
    Equivalent Number of Looks.

    ENL = (μ/σ)^2

    High ENL = homogeneous region = likely water
    Water typically has ENL > 10
    """
    arr_lin = db_to_linear(arr)

    local_mean = uniform_filter(arr_lin, size=window)
    local_sq_mean = uniform_filter(arr_lin**2, size=window)
    local_var = local_sq_mean - local_mean**2
    local_std = np.sqrt(np.maximum(local_var, 1e-10))

    enl = (local_mean / local_std) ** 2

    # Clip to reasonable range
    return np.clip(enl, 0, 100)


def compute_coefficient_of_variation(arr: np.ndarray, window: int = 9) -> np.ndarray:
    """
    Coefficient of Variation.

    CV = σ/μ

    Low CV = homogeneous = likely water
    Water typically has CV < 0.3 (in linear scale)
    """
    arr_lin = db_to_linear(arr)

    local_mean = uniform_filter(arr_lin, size=window) + 1e-10
    local_sq_mean = uniform_filter(arr_lin**2, size=window)
    local_var = local_sq_mean - local_mean**2
    local_std = np.sqrt(np.maximum(local_var, 1e-10))

    cv = local_std / local_mean

    return np.clip(cv, 0, 5)


def compute_local_heterogeneity(arr: np.ndarray, window: int = 9) -> np.ndarray:
    """
    Local heterogeneity using entropy of local histogram.

    Low heterogeneity = uniform = water
    """
    from scipy.ndimage import generic_filter

    def local_entropy(values):
        """Compute entropy of local window."""
        values = values[~np.isnan(values)]
        if len(values) < 2:
            return 0
        # Simple histogram-based entropy
        hist, _ = np.histogram(values, bins=8, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist + 1e-10))

    # Use strided computation for speed
    h, w = arr.shape
    result = np.zeros_like(arr)
    stride = window // 2
    pad = window // 2

    for i in range(0, h - window, stride):
        for j in range(0, w - window, stride):
            patch = arr[i : i + window, j : j + window]
            ent = local_entropy(patch.flatten())
            i_end = min(i + stride, h)
            j_end = min(j + stride, w)
            result[i:i_end, j:j_end] = ent

    return result


def compute_morphological_features(arr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Morphological operations for noise removal and feature extraction.
    """
    # Normalize to 0-255 for morphological operations
    arr_norm = (
        (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr) + 1e-10) * 255
    ).astype(np.uint8)

    selem = disk(3)

    # Opening: removes bright noise, preserves dark features (water)
    opened = opening(arr_norm, selem)

    # Closing: removes dark noise, preserves bright features
    closed = closing(arr_norm, selem)

    # White top-hat: bright features smaller than structuring element
    w_tophat = white_tophat(arr_norm, selem)

    # Black top-hat: dark features smaller than structuring element
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
    """
    Detect linear features (rivers, streams, canals) using oriented filters.

    Uses Gabor-like filters at multiple orientations.
    """
    responses = []

    for angle in np.linspace(0, np.pi, n_orientations, endpoint=False):
        # Create line kernel
        kernel_size = line_length
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2

        # Draw line at angle
        for t in range(-center, center + 1):
            x = int(center + t * np.cos(angle))
            y = int(center + t * np.sin(angle))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1

        kernel = kernel / (kernel.sum() + 1e-10)

        # Apply filter
        response = ndimage.convolve(arr, kernel)
        responses.append(response)

    # Max response across all orientations
    max_response = np.maximum.reduce(responses)

    return max_response


def compute_otsu_threshold(arr: np.ndarray) -> float:
    """Compute Otsu's threshold for adaptive thresholding."""
    arr_clean = arr[~np.isnan(arr)].flatten()
    arr_clip = np.clip(
        arr_clean, np.percentile(arr_clean, 1), np.percentile(arr_clean, 99)
    )

    try:
        threshold = threshold_otsu(arr_clip)
    except:
        threshold = np.median(arr_clip)

    return threshold


def compute_kapur_threshold(arr: np.ndarray, n_bins: int = 256) -> float:
    """
    Kapur's entropy-based thresholding.

    Maximizes the sum of entropies of foreground and background.
    Often better than Otsu for bimodal distributions with unequal variances.
    """
    arr_clean = arr[~np.isnan(arr)].flatten()
    arr_clip = np.clip(
        arr_clean, np.percentile(arr_clean, 1), np.percentile(arr_clean, 99)
    )

    # Normalize to 0-1
    arr_norm = (arr_clip - arr_clip.min()) / (arr_clip.max() - arr_clip.min() + 1e-10)

    # Histogram
    hist, bin_edges = np.histogram(arr_norm, bins=n_bins, density=True)
    hist = hist / (hist.sum() + 1e-10)

    # Cumulative sums
    cumsum = np.cumsum(hist)

    # Find threshold that maximizes sum of entropies
    max_entropy = -np.inf
    best_thresh = 0.5

    for t in range(1, n_bins - 1):
        # Background probability
        p_bg = cumsum[t]
        if p_bg < 1e-10 or p_bg > 1 - 1e-10:
            continue

        # Foreground probability
        p_fg = 1 - p_bg

        # Background entropy
        h_bg = hist[:t] / (p_bg + 1e-10)
        h_bg = h_bg[h_bg > 0]
        entropy_bg = -np.sum(h_bg * np.log(h_bg + 1e-10))

        # Foreground entropy
        h_fg = hist[t:] / (p_fg + 1e-10)
        h_fg = h_fg[h_fg > 0]
        entropy_fg = -np.sum(h_fg * np.log(h_fg + 1e-10))

        total_entropy = entropy_bg + entropy_fg

        if total_entropy > max_entropy:
            max_entropy = total_entropy
            best_thresh = (bin_edges[t] + bin_edges[t + 1]) / 2

    # Convert back to original scale
    threshold = best_thresh * (arr_clip.max() - arr_clip.min()) + arr_clip.min()

    return threshold


def extract_comprehensive_features(
    chip: np.ndarray, chip_name: str = ""
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract ALL features from a chip.

    Returns:
        features: (N_features, H, W) array
        feature_names: list of feature names
    """
    vv = chip[CONFIG["bands"]["VV"]]
    vh = chip[CONFIG["bands"]["VH"]]
    dem = chip[CONFIG["bands"]["DEM"]]
    hand = chip[CONFIG["bands"]["HAND"]]
    slope = chip[CONFIG["bands"]["SLOPE"]]
    twi = chip[CONFIG["bands"]["TWI"]]

    features = []
    names = []

    # =========================================================================
    # 1. BASIC SAR FEATURES
    # =========================================================================
    features.extend([vv, vh])
    names.extend(["VV", "VH"])

    # VV-VH difference (in dB)
    features.append(vv - vh)
    names.append("VV_minus_VH")

    # VV/VH ratio (depolarization)
    vv_lin = db_to_linear(vv)
    vh_lin = db_to_linear(vh)
    ratio = vv_lin / (vh_lin + 1e-10)
    features.append(np.log10(ratio + 1e-10))  # Log scale
    names.append("VV_over_VH_log")

    # =========================================================================
    # 2. TOPOGRAPHIC FEATURES
    # =========================================================================
    features.extend([hand, slope, twi, dem])
    names.extend(["HAND", "SLOPE", "TWI", "DEM"])

    # DEM derivatives
    dem_dx = ndimage.sobel(dem, axis=1)
    dem_dy = ndimage.sobel(dem, axis=0)
    dem_grad = np.sqrt(dem_dx**2 + dem_dy**2)
    dem_aspect = np.arctan2(dem_dy, dem_dx)
    features.extend([dem_grad, dem_aspect])
    names.extend(["DEM_gradient", "DEM_aspect"])

    # =========================================================================
    # 3. POLARIMETRIC FEATURES
    # =========================================================================
    # Pseudo-entropy
    entropy = compute_pseudo_entropy(vv, vh, window=5)
    features.append(entropy)
    names.append("pseudo_entropy")

    # Pseudo-alpha
    alpha = compute_pseudo_alpha(vv, vh, window=5)
    features.append(alpha)
    names.append("pseudo_alpha")

    # RVI
    rvi = compute_rvi(vv, vh, window=5)
    features.append(rvi)
    names.append("RVI")

    # Span (total power)
    span = compute_span(vv, vh)
    features.append(span)
    names.append("span")

    # =========================================================================
    # 4. MULTI-SCALE STATISTICS
    # =========================================================================
    for scale in CONFIG["scales"]:
        # VV statistics
        vv_mean = uniform_filter(vv, size=scale)
        vv_sq_mean = uniform_filter(vv**2, size=scale)
        vv_std = np.sqrt(np.maximum(vv_sq_mean - vv_mean**2, 0))

        features.extend([vv_mean, vv_std])
        names.extend([f"VV_mean_s{scale}", f"VV_std_s{scale}"])

        # VH statistics
        vh_mean = uniform_filter(vh, size=scale)
        vh_sq_mean = uniform_filter(vh**2, size=scale)
        vh_std = np.sqrt(np.maximum(vh_sq_mean - vh_mean**2, 0))

        features.extend([vh_mean, vh_std])
        names.extend([f"VH_mean_s{scale}", f"VH_std_s{scale}"])

        # Local min (water is dark, so low min)
        if scale <= 9:  # Only for smaller scales
            from scipy.ndimage import minimum_filter

            vv_min = minimum_filter(vv, size=scale)
            vh_min = minimum_filter(vh, size=scale)
            features.extend([vv_min, vh_min])
            names.extend([f"VV_min_s{scale}", f"VH_min_s{scale}"])

    # =========================================================================
    # 5. SPECKLE STATISTICS
    # =========================================================================
    for band, band_name in [(vv, "VV"), (vh, "VH")]:
        # ENL
        enl = compute_enl(band, window=9)
        features.append(enl)
        names.append(f"{band_name}_ENL")

        # Coefficient of variation
        cv = compute_coefficient_of_variation(band, window=9)
        features.append(cv)
        names.append(f"{band_name}_CV")

    # =========================================================================
    # 6. TEXTURE FEATURES (Fast GLCM approximation)
    # =========================================================================
    for band, band_name in [(vv, "VV"), (vh, "VH")]:
        glcm_feats = compute_glcm_fast(band, window=11)
        for prop_name, prop_arr in glcm_feats.items():
            features.append(prop_arr)
            names.append(f"{band_name}_glcm_{prop_name}")

    # =========================================================================
    # 7. MORPHOLOGICAL FEATURES
    # =========================================================================
    morph_feats = compute_morphological_features(vh)
    for morph_name, morph_arr in morph_feats.items():
        features.append(morph_arr)
        names.append(f"VH_{morph_name}")

    # =========================================================================
    # 8. LINEAR FEATURE DETECTION
    # =========================================================================
    line_response = detect_linear_features(vh, n_orientations=8, line_length=15)
    features.append(line_response)
    names.append("line_response")

    # =========================================================================
    # 9. ADAPTIVE THRESHOLDS
    # =========================================================================
    # Otsu thresholds
    vv_otsu = compute_otsu_threshold(vv)
    vh_otsu = compute_otsu_threshold(vh)

    features.append(vv - vv_otsu)
    features.append(vh - vh_otsu)
    names.extend(["VV_otsu_diff", "VH_otsu_diff"])

    features.append((vv < vv_otsu).astype(np.float32))
    features.append((vh < vh_otsu).astype(np.float32))
    names.extend(["VV_below_otsu", "VH_below_otsu"])

    # Kapur thresholds
    vv_kapur = compute_kapur_threshold(vv)
    vh_kapur = compute_kapur_threshold(vh)

    features.append((vv < vv_kapur).astype(np.float32))
    features.append((vh < vh_kapur).astype(np.float32))
    names.extend(["VV_below_kapur", "VH_below_kapur"])

    # =========================================================================
    # 10. PHYSICS-BASED COMPOSITE SCORES
    # =========================================================================
    # Water probability based on physics priors
    hand_score = np.clip(1 - hand / CONFIG["physics"]["hand_threshold_init"], 0, 1)
    slope_score = np.clip(1 - slope / CONFIG["physics"]["slope_threshold_init"], 0, 1)
    vh_score = np.clip((CONFIG["physics"]["vh_dark_threshold_init"] - vh) / 10, 0, 1)
    twi_score = np.clip((twi - 5) / 10, 0, 1)

    physics_score = (
        hand_score * slope_score * vh_score * twi_score
    ) ** 0.25  # Geometric mean
    features.append(physics_score)
    names.append("physics_composite")

    # Individual physics scores
    features.extend([hand_score, slope_score, vh_score, twi_score])
    names.extend(["hand_score", "slope_score", "vh_score", "twi_score"])

    # =========================================================================
    # 11. EDGE/GRADIENT FEATURES
    # =========================================================================
    vh_dx = ndimage.sobel(vh, axis=1)
    vh_dy = ndimage.sobel(vh, axis=0)
    vh_grad = np.sqrt(vh_dx**2 + vh_dy**2)
    features.append(vh_grad)
    names.append("VH_gradient")

    vv_dx = ndimage.sobel(vv, axis=1)
    vv_dy = ndimage.sobel(vv, axis=0)
    vv_grad = np.sqrt(vv_dx**2 + vv_dy**2)
    features.append(vv_grad)
    names.append("VV_gradient")

    # Laplacian (second derivative)
    vh_lap = ndimage.laplace(vh)
    features.append(vh_lap)
    names.append("VH_laplacian")

    # =========================================================================
    # STACK AND RETURN
    # =========================================================================
    feature_stack = np.stack(features, axis=0).astype(np.float32)

    # Replace NaN/Inf
    feature_stack = np.nan_to_num(feature_stack, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_stack, names


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute gradient of Lovasz extension."""
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if len(jaccard) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Lovasz-Softmax loss for binary segmentation.

    Directly optimizes Jaccard index (IoU).
    """
    if probas.numel() == 0:
        return probas * 0.0

    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - probas * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]

    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / (union + 1e-10)

    if len(jaccard) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]

    loss = torch.dot(F.relu(errors_sorted), jaccard)
    return loss


class LovaszLoss(nn.Module):
    """Lovasz-Softmax loss."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Clamp predictions to [0, 1] to avoid numerical issues
        pred_clamped = torch.clamp(pred, 0.0, 1.0)
        pred_flat = pred_clamped.view(-1)
        target_flat = target.view(-1)
        return lovasz_softmax_flat(pred_flat, target_flat)


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for handling class imbalance.

    TI = TP / (TP + α*FN + β*FP)
    L = (1 - TI)^γ

    α > β: penalize false negatives more (better recall)
    α < β: penalize false positives more (better precision)
    γ > 1: focus on hard examples
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # True positives, false negatives, false positives
        tp = (pred_flat * target_flat).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()

        tversky_index = tp / (tp + self.alpha * fn + self.beta * fp + 1e-10)
        focal_tversky = (1 - tversky_index) ** self.gamma

        return focal_tversky


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss that penalizes errors near edges more heavily.
    """

    def __init__(self, weight_decay: float = 5.0):
        super().__init__()
        self.weight_decay = weight_decay

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute edge mask using Sobel
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
            device=target.device,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
            device=target.device,
        ).view(1, 1, 3, 3)

        # Edge magnitude
        edge_x = F.conv2d(target, sobel_x, padding=1)
        edge_y = F.conv2d(target, sobel_y, padding=1)
        edge_mag = torch.sqrt(edge_x**2 + edge_y**2 + 1e-10)

        # Weight: higher near edges
        weight = 1 + self.weight_decay * edge_mag

        # Weighted BCE - clamp pred to valid range
        pred_clamped = torch.clamp(pred, 1e-7, 1 - 1e-7)
        bce = F.binary_cross_entropy(pred_clamped, target, reduction="none")
        weighted_bce = (bce * weight).mean()

        return weighted_bce


class PhysicsConsistencyLoss(nn.Module):
    """
    Loss that penalizes predictions inconsistent with physics priors.

    - Water should be at low HAND
    - Water should be on flat slopes
    - Water should have low VH backscatter
    """

    def __init__(
        self,
        hand_threshold: float = 10.0,
        slope_threshold: float = 15.0,
        vh_threshold: float = -18.0,
    ):
        super().__init__()
        self.hand_threshold = hand_threshold
        self.slope_threshold = slope_threshold
        self.vh_threshold = vh_threshold

    def forward(
        self,
        pred: torch.Tensor,
        hand: torch.Tensor,
        slope: torch.Tensor,
        vh: torch.Tensor,
    ) -> torch.Tensor:
        # Penalize water predictions where physics says it's unlikely
        unlikely_hand = (hand > self.hand_threshold).float()
        unlikely_slope = (slope > self.slope_threshold).float()
        unlikely_vh = (vh > self.vh_threshold).float()

        # Combined physics violation mask
        physics_violation = unlikely_hand * unlikely_slope * unlikely_vh

        # Penalty for predicting water in physics-unlikely areas
        penalty = (pred * physics_violation).mean()

        return penalty


class CombinedLoss(nn.Module):
    """Combined loss with all components."""

    def __init__(self, weights: Dict[str, float] = None):
        super().__init__()
        self.weights = weights or CONFIG["loss_weights"]

        self.bce = nn.BCELoss()
        self.lovasz = LovaszLoss()
        self.focal_tversky = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
        self.boundary = BoundaryLoss(weight_decay=5.0)
        self.physics = PhysicsConsistencyLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        hand: torch.Tensor = None,
        slope: torch.Tensor = None,
        vh: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses = {}

        # Clamp BOTH predictions and targets to valid range for BCE
        pred = torch.clamp(pred, 1e-7, 1 - 1e-7)
        target = torch.clamp(target, 0, 1)

        # Replace any NaN with 0
        pred = torch.nan_to_num(pred, nan=0.5)
        target = torch.nan_to_num(target, nan=0.0)

        # BCE - use functional with reduction to avoid issues
        losses["bce"] = F.binary_cross_entropy(pred, target)

        # Dice loss instead of Lovasz (more stable on CUDA)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + 1e-7) / (
            pred_flat.sum() + target_flat.sum() + 1e-7
        )
        losses["lovasz"] = 1 - dice  # Use dice loss in place of lovasz

        # Focal Tversky
        losses["focal_tversky"] = self.focal_tversky(pred, target)

        # Boundary
        losses["boundary"] = self.boundary(pred, target)

        # Physics (if inputs provided)
        if hand is not None and slope is not None and vh is not None:
            losses["physics"] = self.physics(pred, hand, slope, vh)
        else:
            losses["physics"] = torch.tensor(0.0, device=pred.device)

        # Weighted sum
        total = sum(self.weights[k] * losses[k] for k in losses)

        # Convert to floats for logging
        losses_float = {k: v.item() for k, v in losses.items()}

        return total, losses_float


# =============================================================================
# ATTENTION MODULES
# =============================================================================


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (from CBAM).

    Learns which channels are important using global pooling.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()

        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))

        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (from CBAM).

    Learns where to focus using channel-wise pooling.
    """

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
    """
    Convolutional Block Attention Module.

    Combines channel and spatial attention.
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class LearnablePhysicsAttention(nn.Module):
    """
    Physics-guided attention with LEARNABLE thresholds.

    Initialized with domain knowledge but learned during training.
    """

    def __init__(self):
        super().__init__()
        # Learnable thresholds (initialized with physics priors)
        self.hand_threshold = nn.Parameter(
            torch.tensor(CONFIG["physics"]["hand_threshold_init"])
        )
        self.slope_threshold = nn.Parameter(
            torch.tensor(CONFIG["physics"]["slope_threshold_init"])
        )
        self.vh_threshold = nn.Parameter(
            torch.tensor(CONFIG["physics"]["vh_dark_threshold_init"])
        )
        self.twi_threshold = nn.Parameter(
            torch.tensor(CONFIG["physics"]["twi_threshold_init"])
        )

        # Learnable temperature for soft thresholding
        self.temperature = nn.Parameter(torch.tensor(3.0))

        # Learnable weights for combining physics scores
        self.weights = nn.Parameter(torch.ones(4) / 4)

    def forward(
        self,
        hand: torch.Tensor,
        slope: torch.Tensor,
        vh: torch.Tensor,
        twi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute physics attention map.

        Returns attention values in [0, 1] where 1 = likely water.
        """
        temp = torch.abs(self.temperature) + 0.1  # Ensure positive

        # Soft thresholding using sigmoid
        hand_attn = torch.sigmoid(-(hand - self.hand_threshold) / temp)
        slope_attn = torch.sigmoid(-(slope - self.slope_threshold) / temp)
        vh_attn = torch.sigmoid(-(vh - self.vh_threshold) / temp)
        twi_attn = torch.sigmoid((twi - self.twi_threshold) / temp)

        # Weighted combination
        weights = F.softmax(self.weights, dim=0)
        combined = (
            weights[0] * hand_attn
            + weights[1] * slope_attn
            + weights[2] * vh_attn
            + weights[3] * twi_attn
        )

        return combined

    def get_learned_params(self) -> Dict[str, float]:
        """Return learned parameters for analysis."""
        weights = F.softmax(self.weights, dim=0).detach().cpu().numpy()
        return {
            "hand_threshold": self.hand_threshold.item(),
            "slope_threshold": self.slope_threshold.item(),
            "vh_threshold": self.vh_threshold.item(),
            "twi_threshold": self.twi_threshold.item(),
            "temperature": self.temperature.item(),
            "weight_hand": weights[0],
            "weight_slope": weights[1],
            "weight_vh": weights[2],
            "weight_twi": weights[3],
        }


# =============================================================================
# U-NET WITH CBAM AND PHYSICS ATTENTION
# =============================================================================


class DepthwiseSeparableConv(nn.Module):
    """Efficient convolution: depthwise + pointwise."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class ConvBlock(nn.Module):
    """Double convolution block with optional CBAM."""

    def __init__(self, in_ch: int, out_ch: int, use_cbam: bool = True):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_ch, out_ch)
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch)
        self.cbam = CBAM(out_ch) if use_cbam else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.cbam(x)
        return x


class PhysicsGuidedUNetV4(nn.Module):
    """
    Physics-Guided U-Net with:
    - CBAM attention at each level
    - Learnable physics attention at bottleneck
    - Multi-task output (mask + edge + uncertainty)
    - ~500K parameters
    """

    def __init__(self, in_channels: int = 6):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 32, use_cbam=True)
        self.enc2 = ConvBlock(32, 64, use_cbam=True)
        self.enc3 = ConvBlock(64, 128, use_cbam=True)
        self.enc4 = ConvBlock(128, 256, use_cbam=True)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(256, 256, use_cbam=True)

        # Physics attention
        self.physics_attention = LearnablePhysicsAttention()

        # Decoder (with skip connections)
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = ConvBlock(128 + 256, 128, use_cbam=True)  # Fixed: 128 + 256 = 384

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = ConvBlock(64 + 128, 64, use_cbam=True)  # 64 + 128 = 192

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = ConvBlock(32 + 64, 32, use_cbam=True)  # 32 + 64 = 96

        self.up1 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.dec1 = ConvBlock(32 + 32, 32, use_cbam=True)  # 32 + 32 = 64

        # Multi-task heads
        self.mask_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(16, 1, 1)
        )

        self.edge_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(16, 1, 1)
        )

        # Uncertainty head (for Monte Carlo Dropout)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Softplus(),  # Ensure positive uncertainty
        )

        # Dropout for uncertainty estimation
        self.dropout = nn.Dropout2d(0.1)

    def forward(
        self,
        x: torch.Tensor,
        hand: torch.Tensor = None,
        slope: torch.Tensor = None,
        vh: torch.Tensor = None,
        twi: torch.Tensor = None,
        return_uncertainty: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.

        Args:
            x: Input features (B, C, H, W)
            hand, slope, vh, twi: Physics inputs for attention (B, 1, H, W)
            return_uncertainty: Whether to return uncertainty map

        Returns:
            mask: Segmentation mask (B, 1, H, W)
            edge: Edge prediction (B, 1, H, W)
            uncertainty: (optional) Uncertainty map (B, 1, H, W)
        """
        # Encoder
        e1 = self.enc1(x)  # (B, 32, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, 64, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 128, H/4, W/4)
        e4 = self.enc4(self.pool(e3))  # (B, 256, H/8, W/8)

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # (B, 256, H/16, W/16)

        # Apply physics attention at bottleneck
        if (
            hand is not None
            and slope is not None
            and vh is not None
            and twi is not None
        ):
            b_size = b.shape[2:]
            hand_down = F.interpolate(
                hand, size=b_size, mode="bilinear", align_corners=False
            )
            slope_down = F.interpolate(
                slope, size=b_size, mode="bilinear", align_corners=False
            )
            vh_down = F.interpolate(
                vh, size=b_size, mode="bilinear", align_corners=False
            )
            twi_down = F.interpolate(
                twi, size=b_size, mode="bilinear", align_corners=False
            )

            physics_attn = self.physics_attention(
                hand_down.squeeze(1),
                slope_down.squeeze(1),
                vh_down.squeeze(1),
                twi_down.squeeze(1),
            ).unsqueeze(1)

            # Modulate bottleneck features
            b = b * (0.5 + 0.5 * physics_attn)

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d4 = self.dropout(d4)

        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d3 = self.dropout(d3)

        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))

        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        # Output heads
        mask = torch.sigmoid(self.mask_head(d1))
        edge = torch.sigmoid(self.edge_head(d1))

        if return_uncertainty:
            uncertainty = self.uncertainty_head(d1)
            return mask, edge, uncertainty

        return mask, edge

    def get_learned_physics_params(self) -> Dict[str, float]:
        """Return learned physics parameters."""
        return self.physics_attention.get_learned_params()


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# DATASET
# =============================================================================


class SARDatasetV4(Dataset):
    """Dataset with comprehensive feature extraction."""

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
        truth = (
            chip[CONFIG["bands"]["TRUTH"]]
            if chip.shape[0] > 7
            else (vh < -18).astype(np.float32)
        )

        # Stack features for U-Net input
        # Using 6 channels: VV, VH, DEM, HAND, SLOPE, TWI
        features = np.stack([vv, vh, dem, hand, slope, twi], axis=0)
        mask = truth[np.newaxis]

        # Resize
        features = self._resize(features, self.image_size)
        mask = self._resize(mask, self.image_size)
        hand_resized = self._resize(hand[np.newaxis], self.image_size)
        slope_resized = self._resize(slope[np.newaxis], self.image_size)
        vh_resized = self._resize(vh[np.newaxis], self.image_size)
        twi_resized = self._resize(twi[np.newaxis], self.image_size)

        # Augmentation
        if self.augment:
            features, mask, hand_resized, slope_resized, vh_resized, twi_resized = (
                self._augment(
                    features, mask, hand_resized, slope_resized, vh_resized, twi_resized
                )
            )

        # Normalize features
        features = self._normalize(features)

        # Binarize mask
        mask = (mask > 0.5).astype(np.float32)

        # Compute edge for multi-task learning
        edge = self._compute_edge(mask[0])

        return {
            "features": torch.from_numpy(features),
            "mask": torch.from_numpy(mask),
            "edge": torch.from_numpy(edge[np.newaxis]),
            "hand": torch.from_numpy(hand_resized),
            "slope": torch.from_numpy(slope_resized),
            "vh": torch.from_numpy(vh_resized),
            "twi": torch.from_numpy(twi_resized),
            "name": self.names[idx],
        }

    def _resize(self, arr: np.ndarray, size: int) -> np.ndarray:
        """Resize array to target size."""
        from scipy.ndimage import zoom

        if arr.ndim == 2:
            h, w = arr.shape
            factors = (size / h, size / w)
        else:
            c, h, w = arr.shape
            factors = (1, size / h, size / w)

        return zoom(arr, factors, order=1)

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize features per-channel."""
        result = np.zeros_like(features)
        for i in range(features.shape[0]):
            channel = features[i]
            mean = np.nanmean(channel)
            std = np.nanstd(channel) + 1e-10
            result[i] = (channel - mean) / std
        return result.astype(np.float32)

    def _augment(self, features, mask, hand, slope, vh, twi):
        """Apply augmentation."""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            features = np.flip(features, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()
            hand = np.flip(hand, axis=2).copy()
            slope = np.flip(slope, axis=2).copy()
            vh = np.flip(vh, axis=2).copy()
            twi = np.flip(twi, axis=2).copy()

        # Random vertical flip
        if np.random.rand() > 0.5:
            features = np.flip(features, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
            hand = np.flip(hand, axis=1).copy()
            slope = np.flip(slope, axis=1).copy()
            vh = np.flip(vh, axis=1).copy()
            twi = np.flip(twi, axis=1).copy()

        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        features = np.rot90(features, k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k, axes=(1, 2)).copy()
        hand = np.rot90(hand, k, axes=(1, 2)).copy()
        slope = np.rot90(slope, k, axes=(1, 2)).copy()
        vh = np.rot90(vh, k, axes=(1, 2)).copy()
        twi = np.rot90(twi, k, axes=(1, 2)).copy()

        # Random noise (only on SAR channels)
        if np.random.rand() > 0.5:
            noise_std = 0.1 * np.random.rand()
            features[:2] += (
                np.random.randn(*features[:2].shape).astype(np.float32) * noise_std
            )

        return features, mask, hand, slope, vh, twi

    def _compute_edge(self, mask: np.ndarray) -> np.ndarray:
        """Compute edge map from mask."""
        from scipy.ndimage import binary_dilation, binary_erosion

        dilated = binary_dilation(mask > 0.5, iterations=2)
        eroded = binary_erosion(mask > 0.5, iterations=2)
        edge = (dilated & ~eroded).astype(np.float32)

        return edge


# =============================================================================
# DATA LOADING
# =============================================================================


def load_all_chips() -> Tuple[List[np.ndarray], List[str]]:
    """Load all chips from configured directories."""
    chips = []
    names = []

    for chip_dir in CONFIG["chip_dirs"]:
        if not chip_dir.exists():
            logger.warning(f"Chip directory not found: {chip_dir}")
            continue

        # Load .npy files
        for f in chip_dir.glob("*.npy"):
            try:
                chip = np.load(f)
                if chip.shape[0] >= 8 and np.nansum(chip[7]) > 0:  # Has valid truth
                    chips.append(chip)
                    names.append(f.stem)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")

        # Load .tif files
        for f in chip_dir.glob("*.tif"):
            try:
                with rasterio.open(f) as src:
                    chip = src.read()
                if chip.shape[0] >= 8 and np.nansum(chip[7]) > 0:
                    chips.append(chip)
                    names.append(f.stem)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")

    logger.info(f"Loaded {len(chips)} total chips")
    return chips, names


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute segmentation metrics."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    pred_binary = (pred_flat > 0.5).astype(int)
    target_binary = (target_flat > 0.5).astype(int)

    # Intersection and Union for IoU
    intersection = np.sum(pred_binary & target_binary)
    union = np.sum(pred_binary | target_binary)
    iou = intersection / (union + 1e-10)

    return {
        "accuracy": accuracy_score(target_binary, pred_binary),
        "f1": f1_score(target_binary, pred_binary, zero_division=0),
        "precision": precision_score(target_binary, pred_binary, zero_division=0),
        "recall": recall_score(target_binary, pred_binary, zero_division=0),
        "iou": iou,
        "auc_roc": roc_auc_score(target_binary, pred_flat)
        if len(np.unique(target_binary)) > 1
        else 0.5,
    }


def train_lightgbm_v4(chips: List[np.ndarray], names: List[str]) -> Dict[str, Any]:
    """
    Train LightGBM with comprehensive features.
    """
    logger.info("=" * 70)
    logger.info("TRAINING MODEL 1: LightGBM v4 (Comprehensive Features)")
    logger.info("=" * 70)

    start_time = time.time()

    # Extract features from all chips
    all_features = []
    all_labels = []
    feature_names = None

    for i, (chip, name) in enumerate(zip(chips, names)):
        if (i + 1) % 20 == 0:
            logger.info(f"  Extracting features from {i + 1}/{len(chips)} chips")

        try:
            features, feat_names = extract_comprehensive_features(chip, name)
            truth = chip[CONFIG["bands"]["TRUTH"]]

            if feature_names is None:
                feature_names = feat_names

            # Sample pixels
            h, w = truth.shape
            n_samples = int(h * w * CONFIG["sample_rate"])
            indices = np.random.choice(h * w, size=n_samples, replace=False)

            rows, cols = np.unravel_index(indices, (h, w))

            for r, c in zip(rows, cols):
                if not np.isnan(truth[r, c]):
                    all_features.append(features[:, r, c])
                    all_labels.append(truth[r, c])
        except Exception as e:
            logger.warning(f"Failed to extract features from {name}: {e}")

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32)
    y = (y > 0.5).astype(int)

    logger.info(f"Total samples: {len(X):,}")
    logger.info(f"Class balance: {100 * y.mean():.1f}% water")
    logger.info(f"Features: {len(feature_names)}")

    # 5-fold cross-validation
    logger.info("\nRunning 5-fold cross-validation...")

    kfold = StratifiedKFold(
        n_splits=CONFIG["n_folds"], shuffle=True, random_state=CONFIG["random_seed"]
    )

    cv_results = {"iou": [], "f1": [], "accuracy": []}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            CONFIG["lgb_params"],
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        y_pred = model.predict(X_val)
        metrics = compute_metrics(y_pred, y_val)

        cv_results["iou"].append(metrics["iou"])
        cv_results["f1"].append(metrics["f1"])
        cv_results["accuracy"].append(metrics["accuracy"])

        logger.info(f"  Fold {fold + 1}: IoU={metrics['iou']:.4f}")

    cv_mean_iou = np.mean(cv_results["iou"])
    cv_std_iou = np.std(cv_results["iou"])
    logger.info(f"\nCV Results: IoU = {cv_mean_iou:.4f} (+/- {cv_std_iou:.4f})")

    # Train final model on all data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=CONFIG["random_seed"]
    )

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    final_model = lgb.train(
        CONFIG["lgb_params"],
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # Test metrics
    y_pred_test = final_model.predict(X_test)
    test_results = compute_metrics(y_pred_test, y_test)

    logger.info(f"\nTest Results:")
    logger.info(f"  IoU:       {test_results['iou']:.4f}")
    logger.info(f"  F1:        {test_results['f1']:.4f}")
    logger.info(f"  AUC-ROC:   {test_results['auc_roc']:.4f}")

    # Feature importance
    importance = dict(zip(feature_names, final_model.feature_importance().tolist()))
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    logger.info(f"\nTop 15 Features:")
    for name, imp in sorted_importance[:15]:
        logger.info(f"  {name}: {imp}")

    # Save model
    model_path = CONFIG["model_dir"] / "lightgbm_v4_comprehensive.txt"
    final_model.save_model(str(model_path))

    return {
        "model": "LightGBM_v4_Comprehensive",
        "sar_only": CONFIG["sar_only"],
        "cv": {
            "mean_iou": float(cv_mean_iou),
            "std_iou": float(cv_std_iou),
            "fold_scores": cv_results,
        },
        "test": test_results,
        "feature_importance": importance,
        "top_features": [name for name, _ in sorted_importance[:20]],
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "training_time_seconds": time.time() - start_time,
        "n_samples": len(X),
        "model_path": str(model_path),
    }


def train_unet_v4(chips: List[np.ndarray], names: List[str]) -> Dict[str, Any]:
    """
    Train Physics-Guided U-Net v4 with all improvements.
    """
    logger.info("=" * 70)
    logger.info("TRAINING MODEL 2: Physics-Guided U-Net v4 (CBAM + Physics Attention)")
    logger.info("=" * 70)

    start_time = time.time()
    device = torch.device(CONFIG["device"])

    # Split data
    train_chips, test_chips, train_names, test_names = train_test_split(
        chips, names, test_size=0.15, random_state=CONFIG["random_seed"]
    )
    train_chips, val_chips, train_names, val_names = train_test_split(
        train_chips, train_names, test_size=0.15, random_state=CONFIG["random_seed"]
    )

    logger.info(
        f"Train: {len(train_chips)}, Val: {len(val_chips)}, Test: {len(test_chips)}"
    )

    # Create datasets
    train_dataset = SARDatasetV4(
        train_chips, train_names, image_size=CONFIG["unet"]["image_size"], augment=True
    )
    val_dataset = SARDatasetV4(
        val_chips, val_names, image_size=CONFIG["unet"]["image_size"], augment=False
    )
    test_dataset = SARDatasetV4(
        test_chips, test_names, image_size=CONFIG["unet"]["image_size"], augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["unet"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["unet"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["unet"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    model = PhysicsGuidedUNetV4(in_channels=6).to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")

    # Loss and optimizer
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["unet"]["learning_rate"],
        weight_decay=CONFIG["unet"]["weight_decay"],
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # Training loop
    best_val_iou = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_iou": []}

    for epoch in range(CONFIG["unet"]["epochs"]):
        # Training
        model.train()
        train_losses = []

        for batch in train_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            edge = batch["edge"].to(device)
            hand = batch["hand"].to(device)
            slope = batch["slope"].to(device)
            vh = batch["vh"].to(device)
            twi = batch["twi"].to(device)

            optimizer.zero_grad()

            pred_mask, pred_edge = model(features, hand, slope, vh, twi)

            # Clamp predictions to valid BCE range
            pred_mask = torch.clamp(pred_mask, 1e-7, 1 - 1e-7)
            pred_edge = torch.clamp(pred_edge, 1e-7, 1 - 1e-7)
            mask = torch.clamp(mask, 0, 1)
            edge = torch.clamp(edge, 0, 1)

            # Mask loss
            mask_loss, _ = criterion(pred_mask, mask, hand, slope, vh)

            # Edge loss (simple BCE)
            edge_loss = F.binary_cross_entropy(pred_edge, edge)

            # Total loss
            loss = mask_loss + 0.2 * edge_loss

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
                hand = batch["hand"].to(device)
                slope = batch["slope"].to(device)
                vh = batch["vh"].to(device)
                twi = batch["twi"].to(device)

                pred_mask, pred_edge = model(features, hand, slope, vh, twi)

                loss, _ = criterion(pred_mask, mask, hand, slope, vh)
                val_losses.append(loss.item())

                val_preds.append(pred_mask.cpu().numpy())
                val_targets.append(mask.cpu().numpy())

        # Compute validation metrics
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_metrics = compute_metrics(val_preds.flatten(), val_targets.flatten())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_iou = val_metrics["iou"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        if (epoch + 1) % 10 == 0 or val_iou > best_val_iou:
            logger.info(
                f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, "
                f"Val Loss={val_loss:.4f}, Val IoU={val_iou:.4f}"
            )

        # Early stopping
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            # Save best model
            model_path = CONFIG["model_dir"] / "unet_v4_physics_best.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_iou": val_iou,
                    "physics_params": model.get_learned_physics_params(),
                },
                model_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["unet"]["patience"]:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    # Load best model for testing
    checkpoint = torch.load(CONFIG["model_dir"] / "unet_v4_physics_best.pth")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Test evaluation
    model.eval()
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            mask = batch["mask"]
            hand = batch["hand"].to(device)
            slope = batch["slope"].to(device)
            vh = batch["vh"].to(device)
            twi = batch["twi"].to(device)

            pred_mask, _ = model(features, hand, slope, vh, twi)

            test_preds.append(pred_mask.cpu().numpy())
            test_targets.append(mask.numpy())

    test_preds = np.concatenate(test_preds)
    test_targets = np.concatenate(test_targets)
    test_metrics = compute_metrics(test_preds.flatten(), test_targets.flatten())

    logger.info(f"\nTest Results:")
    logger.info(f"  IoU:       {test_metrics['iou']:.4f}")
    logger.info(f"  F1:        {test_metrics['f1']:.4f}")
    logger.info(f"  AUC-ROC:   {test_metrics['auc_roc']:.4f}")

    # Get learned physics parameters
    physics_params = model.get_learned_physics_params()
    logger.info(f"\nLearned Physics Parameters:")
    for k, v in physics_params.items():
        logger.info(f"  {k}: {v:.4f}")

    return {
        "model": "PhysicsGuidedUNet_v4",
        "parameters": count_parameters(model),
        "test": test_metrics,
        "best_val_iou": best_val_iou,
        "epochs_trained": epoch + 1,
        "physics_params": physics_params,
        "training_time_seconds": time.time() - start_time,
        "model_path": str(CONFIG["model_dir"] / "unet_v4_physics_best.pth"),
        "history": history,
    }


def create_ensemble_predictions(
    chips: List[np.ndarray], names: List[str], lgb_model_path: str, unet_model_path: str
) -> Dict[str, Any]:
    """
    Create ensemble predictions combining LightGBM and U-Net.

    Fusion methods:
    1. Simple average
    2. Weighted average (based on confidence)
    3. Stacking (learn optimal weights)
    """
    logger.info("=" * 70)
    logger.info("CREATING ENSEMBLE PREDICTIONS")
    logger.info("=" * 70)

    # This would be implemented after both models are trained
    # For now, return placeholder
    return {
        "status": "pending",
        "note": "Ensemble will be created after individual models are trained",
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Main training pipeline."""
    logger.info("=" * 80)
    logger.info("MASTER SAR WATER DETECTION TRAINING PIPELINE v4")
    logger.info("=" * 80)
    logger.info(f"Device: {CONFIG['device']}")
    logger.info(f"SAR-only mode: {CONFIG['sar_only']}")
    logger.info(f"Started: {datetime.now().isoformat()}")

    # Set random seeds
    np.random.seed(CONFIG["random_seed"])
    torch.manual_seed(CONFIG["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG["random_seed"])

    # Load data
    chips, names = load_all_chips()

    if len(chips) == 0:
        logger.error("No chips loaded! Check data paths.")
        return

    # Results dictionary
    all_results = {
        "version": CONFIG["version"],
        "timestamp": datetime.now().isoformat(),
        "config": {
            "sar_only": CONFIG["sar_only"],
            "n_chips": len(chips),
            "scales": CONFIG["scales"],
            "loss_weights": CONFIG["loss_weights"],
        },
    }

    # Train LightGBM
    try:
        all_results["lightgbm_v4"] = train_lightgbm_v4(chips, names)
    except Exception as e:
        logger.error(f"LightGBM training failed: {e}")
        import traceback

        traceback.print_exc()
        all_results["lightgbm_v4"] = {"status": "failed", "error": str(e)}

    # Train U-Net
    try:
        all_results["unet_v4"] = train_unet_v4(chips, names)
    except Exception as e:
        logger.error(f"U-Net training failed: {e}")
        import traceback

        traceback.print_exc()
        all_results["unet_v4"] = {"status": "failed", "error": str(e)}

    # Save results
    results_path = CONFIG["results_dir"] / "training_results_v4.json"
    with open(results_path, "w") as f:
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        json.dump(convert(all_results), f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("=" * 80)

    if "lightgbm_v4" in all_results and "test" in all_results["lightgbm_v4"]:
        lgb_results = all_results["lightgbm_v4"]
        logger.info(f"\nLightGBM v4:")
        logger.info(f"  Test IoU: {lgb_results['test']['iou']:.4f}")
        logger.info(
            f"  CV IoU:   {lgb_results['cv']['mean_iou']:.4f} (+/- {lgb_results['cv']['std_iou']:.4f})"
        )
        logger.info(f"  Features: {lgb_results['n_features']}")

    if "unet_v4" in all_results and "test" in all_results["unet_v4"]:
        unet_results = all_results["unet_v4"]
        logger.info(f"\nU-Net v4:")
        logger.info(f"  Test IoU: {unet_results['test']['iou']:.4f}")
        logger.info(f"  Parameters: {unet_results['parameters']:,}")
        logger.info(f"  Learned Physics:")
        for k, v in unet_results["physics_params"].items():
            logger.info(f"    {k}: {v:.4f}")

    return all_results


if __name__ == "__main__":
    main()
