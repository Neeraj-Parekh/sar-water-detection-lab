#!/usr/bin/env python3
"""
================================================================================
RETRAIN V8 - Full Ensemble Training with MNDWI Support
================================================================================

Features:
1. LightGBM training on combined chips/ (fixed) + chips_expanded_npy/
2. Optional MNDWI feature with graceful fallback
3. Data validation and correction
4. Physics constraints as features
5. Ensemble-ready output

Key Design:
- Model works WITHOUT MNDWI (for SAR-only deployment)
- MNDWI improves accuracy when available
- All data validated before training

Author: SAR Water Detection Project
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
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter, minimum_filter, maximum_filter, laplace
from scipy.ndimage import grey_opening, grey_closing
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "version": "8.0",
    "random_seed": 42,
    "chip_dirs": [
        Path("/home/mit-aoe/sar_water_detection/chips"),  # Original (will fix SLOPE)
        Path("/home/mit-aoe/sar_water_detection/chips_expanded_npy"),  # Clean
    ],
    "model_dir": Path("/home/mit-aoe/sar_water_detection/models"),
    "results_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    "test_size": 0.2,
    "sample_rate": 0.10,  # 10% of pixels for faster training
    "lgb_params": {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "max_depth": 10,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 100,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    },
    "num_boost_round": 1000,
    "early_stopping_rounds": 100,
}

# Feature names - MNDWI features are at the end (optional)
FEATURE_NAMES_BASE = [
    "VV",
    "VH",
    "VV_VH_ratio",
    "VV_VH_diff",
    "NDWI_like",
    "RVI",
    "VV_mean_s3",
    "VV_std_s3",
    "VV_min_s3",
    "VV_max_s3",
    "VH_mean_s3",
    "VH_std_s3",
    "VH_min_s3",
    "VH_max_s3",
    "VV_mean_s5",
    "VV_std_s5",
    "VV_min_s5",
    "VV_max_s5",
    "VH_mean_s5",
    "VH_std_s5",
    "VH_min_s5",
    "VH_max_s5",
    "VV_mean_s9",
    "VV_std_s9",
    "VV_min_s9",
    "VV_max_s9",
    "VH_mean_s9",
    "VH_std_s9",
    "VH_min_s9",
    "VH_max_s9",
    "VV_mean_s15",
    "VV_std_s15",
    "VV_min_s15",
    "VV_max_s15",
    "VH_mean_s15",
    "VH_std_s15",
    "VH_min_s15",
    "VH_max_s15",
    "VV_mean_s21",
    "VV_std_s21",
    "VV_min_s21",
    "VV_max_s21",
    "VH_mean_s21",
    "VH_std_s21",
    "VH_min_s21",
    "VH_max_s21",
    "VV_gradient_mag",
    "VH_gradient_mag",
    "VV_laplacian",
    "VH_laplacian",
    "VV_opened",
    "VV_closed",
    "VH_opened",
    "VH_closed",
    "VV_otsu_diff",
    "VH_otsu_diff",
    "local_contrast_vv",
    "local_contrast_vh",
    "VV_glcm_contrast",
    "VV_glcm_homogeneity",
    "VH_glcm_contrast",
    "VH_glcm_homogeneity",
    "pseudo_entropy",
    "DEM",
    "SLOPE",
    "HAND",
    "TWI",
    "hand_score",
    "slope_score",
    "twi_score",
]

FEATURE_NAMES_MNDWI = ["MNDWI", "MNDWI_water_mask", "MNDWI_mean_s5", "MNDWI_std_s5"]


# =============================================================================
# DATA VALIDATION
# =============================================================================


@dataclass
class DataQualityReport:
    """Report on data quality issues."""

    is_valid: bool
    issues: List[str]
    corrections: Dict[str, str]


def validate_and_fix_data(
    vv: np.ndarray,
    vh: np.ndarray,
    dem: np.ndarray,
    slope: np.ndarray,
    hand: np.ndarray,
    twi: np.ndarray,
    mndwi: Optional[np.ndarray] = None,
) -> Tuple[Dict, DataQualityReport]:
    """
    Validate and fix common data quality issues.
    """
    issues = []
    corrections = {}

    # Fix VV NaN
    if np.any(np.isnan(vv)):
        issues.append("VV_NAN")
        vv = np.nan_to_num(vv, nan=-20.0)
        corrections["VV"] = "NaN replaced with -20"

    # Fix VH NaN
    if np.any(np.isnan(vh)):
        issues.append("VH_NAN")
        vh = np.nan_to_num(vh, nan=-25.0)
        corrections["VH"] = "NaN replaced with -25"

    # Fix SLOPE (must be 0-90)
    if np.any(slope > 90):
        max_slope = slope.max()
        issues.append(f"SLOPE_OVERFLOW: max={max_slope:.0f}")
        # Check if it's a units issue
        if max_slope > 900:
            # Likely stored as degrees * 10 or similar
            slope = np.clip(slope / 10, 0, 90)
            corrections["SLOPE"] = "Divided by 10 and clipped"
        else:
            slope = np.clip(slope, 0, 90)
            corrections["SLOPE"] = "Clipped to 0-90"

    if np.any(slope < 0):
        issues.append("SLOPE_NEGATIVE")
        slope = np.maximum(slope, 0)
        corrections["SLOPE"] = corrections.get("SLOPE", "") + ", negatives set to 0"

    # Fix HAND
    if np.any(np.isnan(hand)):
        issues.append("HAND_NAN")
        hand = np.nan_to_num(hand, nan=100.0)  # High = unlikely water
        corrections["HAND"] = "NaN replaced with 100"

    if np.any(hand > 500):
        issues.append(f"HAND_HIGH: max={hand.max():.0f}")
        hand = np.clip(hand, 0, 500)
        corrections["HAND"] = "Clipped to 0-500"

    # Fix TWI (should be ~0-30, but sometimes higher)
    if np.any(np.isnan(twi)):
        issues.append("TWI_NAN")
        twi = np.nan_to_num(twi, nan=5.0)
        corrections["TWI"] = "NaN replaced with 5"

    if twi.max() > 50:
        issues.append(f"TWI_HIGH: max={twi.max():.0f}")
        twi = np.clip(twi, 0, 30)
        corrections["TWI"] = "Clipped to 0-30"

    # Fix MNDWI if present
    if mndwi is not None:
        if np.any(np.isnan(mndwi)):
            issues.append("MNDWI_NAN")
            mndwi = np.nan_to_num(mndwi, nan=0.0)
            corrections["MNDWI"] = "NaN replaced with 0"

        if np.any(np.abs(mndwi) > 1.1):
            issues.append(f"MNDWI_RANGE: {mndwi.min():.2f} to {mndwi.max():.2f}")
            mndwi = np.clip(mndwi, -1, 1)
            corrections["MNDWI"] = "Clipped to -1 to 1"

    is_valid = len([i for i in issues if "NAN" not in i and "HIGH" not in i]) == 0

    report = DataQualityReport(
        is_valid=is_valid, issues=issues, corrections=corrections
    )

    data = {
        "vv": vv.astype(np.float32),
        "vh": vh.astype(np.float32),
        "dem": dem.astype(np.float32),
        "slope": slope.astype(np.float32),
        "hand": hand.astype(np.float32),
        "twi": twi.astype(np.float32),
    }

    if mndwi is not None:
        data["mndwi"] = mndwi.astype(np.float32)

    return data, report


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================


def compute_otsu_threshold(data: np.ndarray) -> float:
    """Compute Otsu's threshold."""
    data_flat = data[np.isfinite(data)].flatten()
    if len(data_flat) < 100:
        return float(np.nanmedian(data_flat))

    hist, bin_edges = np.histogram(data_flat, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    if total == 0:
        return float(np.nanmedian(data_flat))

    current_max, threshold = 0, bin_centers[0]
    sum_total = (hist * bin_centers).sum()
    sum_background, weight_background = 0, 0

    for i, (count, center) in enumerate(zip(hist, bin_centers)):
        weight_background += count
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += count * center
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance_between = (
            weight_background
            * weight_foreground
            * (mean_background - mean_foreground) ** 2
        )
        if variance_between > current_max:
            current_max = variance_between
            threshold = center

    return float(threshold)


def extract_features(
    data: Dict[str, np.ndarray], include_mndwi: bool = True
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract comprehensive feature set.

    Args:
        data: Dict with 'vv', 'vh', 'dem', 'slope', 'hand', 'twi', optionally 'mndwi'
        include_mndwi: Whether to include MNDWI features if available

    Returns:
        features: (H, W, N_features) array
        feature_names: List of feature names
    """
    vv = data["vv"]
    vh = data["vh"]
    dem = data["dem"]
    slope = data["slope"]
    hand = data["hand"]
    twi = data["twi"]
    mndwi = data.get("mndwi", None)

    h, w = vv.shape
    features = []
    feature_names = []

    # 1. Basic SAR features
    features.append(vv)
    feature_names.append("VV")
    features.append(vh)
    feature_names.append("VH")

    # Safe ratio computation
    with np.errstate(divide="ignore", invalid="ignore"):
        vv_vh_ratio = np.where(np.abs(vh) > 0.01, vv / vh, 0)
        vv_vh_ratio = np.clip(vv_vh_ratio, -10, 10)
    features.append(vv_vh_ratio)
    feature_names.append("VV_VH_ratio")

    features.append(vv - vh)
    feature_names.append("VV_VH_diff")

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = vv + vh
        ndwi = np.where(np.abs(denom) > 0.01, (vv - vh) / denom, 0)
    features.append(ndwi)
    feature_names.append("NDWI_like")

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = vv + vh
        rvi = np.where(np.abs(denom) > 0.01, 4 * vh / denom, 0)
    features.append(rvi)
    feature_names.append("RVI")

    # 2. Multi-scale texture features
    scales = [3, 5, 9, 15, 21]
    for scale in scales:
        for arr, name in [(vv, "VV"), (vh, "VH")]:
            arr_mean = uniform_filter(arr, size=scale)
            arr_sq_mean = uniform_filter(arr**2, size=scale)
            arr_var = np.maximum(arr_sq_mean - arr_mean**2, 0)  # Prevent negative
            arr_std = np.sqrt(arr_var)
            arr_min = minimum_filter(arr, size=scale)  # CORRECT: using minimum_filter
            arr_max = maximum_filter(arr, size=scale)

            features.extend([arr_mean, arr_std, arr_min, arr_max])
            feature_names.extend(
                [
                    f"{name}_mean_s{scale}",
                    f"{name}_std_s{scale}",
                    f"{name}_min_s{scale}",
                    f"{name}_max_s{scale}",
                ]
            )

    # 3. Gradient features
    gy_vv, gx_vv = np.gradient(vv)
    vv_grad_mag = np.sqrt(gx_vv**2 + gy_vv**2)
    features.append(vv_grad_mag)
    feature_names.append("VV_gradient_mag")

    gy_vh, gx_vh = np.gradient(vh)
    vh_grad_mag = np.sqrt(gx_vh**2 + gy_vh**2)
    features.append(vh_grad_mag)
    feature_names.append("VH_gradient_mag")

    features.append(np.abs(laplace(vv)))
    feature_names.append("VV_laplacian")
    features.append(np.abs(laplace(vh)))
    feature_names.append("VH_laplacian")

    # 4. Morphological features
    vv_opened = grey_opening(vv, size=5)
    vv_closed = grey_closing(vv, size=5)
    vh_opened = grey_opening(vh, size=5)
    vh_closed = grey_closing(vh, size=5)

    features.extend([vv_opened, vv_closed, vh_opened, vh_closed])
    feature_names.extend(["VV_opened", "VV_closed", "VH_opened", "VH_closed"])

    # 5. Otsu-based features
    vv_otsu = compute_otsu_threshold(vv)
    vh_otsu = compute_otsu_threshold(vh)
    features.append(vv - vv_otsu)
    feature_names.append("VV_otsu_diff")
    features.append(vh - vh_otsu)
    feature_names.append("VH_otsu_diff")

    # 6. Local contrast
    vv_local_mean = uniform_filter(vv, size=9)
    vh_local_mean = uniform_filter(vh, size=9)
    features.append(vv - vv_local_mean)
    feature_names.append("local_contrast_vv")
    features.append(vh - vh_local_mean)
    feature_names.append("local_contrast_vh")

    # 7. GLCM-like texture (fast approximation)
    for arr, name in [(vv, "VV"), (vh, "VH")]:
        arr_mean = uniform_filter(arr, size=5)
        arr_sq_mean = uniform_filter(arr**2, size=5)
        arr_var = np.maximum(arr_sq_mean - arr_mean**2, 0)
        contrast = np.sqrt(arr_var)

        arr_max = maximum_filter(arr, size=5)
        arr_min = minimum_filter(arr, size=5)
        arr_range = arr_max - arr_min
        homogeneity = 1.0 / (1.0 + arr_range)

        features.extend([contrast, homogeneity])
        feature_names.extend([f"{name}_glcm_contrast", f"{name}_glcm_homogeneity"])

    # 8. Pseudo-entropy
    vv_norm = vv - vv.min()
    vv_range = vv.max() - vv.min()
    if vv_range > 0:
        vv_prob = vv_norm / vv_range
    else:
        vv_prob = np.zeros_like(vv) + 0.5
    vv_prob = np.clip(vv_prob, 1e-10, 1 - 1e-10)
    pseudo_entropy = -vv_prob * np.log2(vv_prob) - (1 - vv_prob) * np.log2(1 - vv_prob)
    features.append(pseudo_entropy)
    feature_names.append("pseudo_entropy")

    # 9. DEM-derived features
    features.extend([dem, slope, hand, twi])
    feature_names.extend(["DEM", "SLOPE", "HAND", "TWI"])

    # 10. Physics-based scores (with safe exp)
    hand_exp = np.clip((hand - 10) / 3.0, -50, 50)
    hand_score = 1.0 / (1.0 + np.exp(hand_exp))

    slope_exp = np.clip((slope - 8) / 3.0, -50, 50)
    slope_score = 1.0 / (1.0 + np.exp(slope_exp))

    twi_exp = np.clip((8 - twi) / 2.0, -50, 50)
    twi_score = 1.0 / (1.0 + np.exp(twi_exp))

    features.extend([hand_score, slope_score, twi_score])
    feature_names.extend(["hand_score", "slope_score", "twi_score"])

    # 11. MNDWI features (OPTIONAL - with guards)
    if include_mndwi and mndwi is not None:
        # MNDWI raw
        features.append(mndwi)
        feature_names.append("MNDWI")

        # MNDWI water mask (simple threshold)
        mndwi_water = (mndwi > 0).astype(np.float32)
        features.append(mndwi_water)
        feature_names.append("MNDWI_water_mask")

        # MNDWI texture
        mndwi_mean = uniform_filter(mndwi, size=5)
        mndwi_sq_mean = uniform_filter(mndwi**2, size=5)
        mndwi_var = np.maximum(mndwi_sq_mean - mndwi_mean**2, 0)
        mndwi_std = np.sqrt(mndwi_var)

        features.extend([mndwi_mean, mndwi_std])
        feature_names.extend(["MNDWI_mean_s5", "MNDWI_std_s5"])

    # Stack all features
    feature_stack = np.stack(features, axis=-1)

    # Replace NaN/Inf with 0 (final safety)
    feature_stack = np.nan_to_num(feature_stack, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_stack.astype(np.float32), feature_names


# =============================================================================
# DATA LOADING
# =============================================================================


def load_chip(chip_path: Path) -> Optional[Dict]:
    """Load a chip file (.npy) with validation."""
    try:
        data = np.load(chip_path, allow_pickle=True)

        if isinstance(data, np.ndarray) and data.ndim == 3:
            n_channels = data.shape[2]

            if n_channels >= 7:
                # Format: VV, VH, DEM, SLOPE, HAND, TWI, label, [MNDWI]
                vv = data[:, :, 0]
                vh = data[:, :, 1]
                dem = data[:, :, 2]
                slope = data[:, :, 3]
                hand = data[:, :, 4]
                twi = data[:, :, 5]
                label = data[:, :, 6]
                mndwi = data[:, :, 7] if n_channels > 7 else None

                # Validate and fix
                validated, report = validate_and_fix_data(
                    vv, vh, dem, slope, hand, twi, mndwi
                )

                validated["label"] = (label > 0).astype(np.float32)
                validated["name"] = chip_path.stem
                validated["quality_report"] = report

                if report.issues:
                    logger.debug(f"  {chip_path.name}: {report.issues}")

                return validated

        logger.warning(f"Unexpected format in {chip_path}")
        return None

    except Exception as e:
        logger.error(f"Error loading {chip_path}: {e}")
        return None


def load_all_chips(chip_dirs: List[Path]) -> List[Dict]:
    """Load all chips from multiple directories."""
    all_chips = []

    for chip_dir in chip_dirs:
        if not chip_dir.exists():
            logger.warning(f"Directory not found: {chip_dir}")
            continue

        # Find all .npy files
        npy_files = sorted(chip_dir.glob("*_with_truth.npy"))
        logger.info(f"Found {len(npy_files)} chips in {chip_dir}")

        for chip_path in npy_files:
            chip_data = load_chip(chip_path)
            if chip_data is not None:
                chip_data["source_dir"] = chip_dir.name
                all_chips.append(chip_data)

    logger.info(f"Successfully loaded {len(all_chips)} chips total")
    return all_chips


# =============================================================================
# METRICS
# =============================================================================


def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute IoU score."""
    pred_bin = pred > 0.5
    target_bin = target > 0.5

    intersection = np.logical_and(pred_bin, target_bin).sum()
    union = np.logical_or(pred_bin, target_bin).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection) / float(union)


# =============================================================================
# MAIN TRAINING
# =============================================================================


def main():
    """Main training pipeline."""
    logger.info("=" * 70)
    logger.info("RETRAIN V8 - Full Ensemble Training")
    logger.info("=" * 70)

    # Create output directories
    CONFIG["model_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["results_dir"].mkdir(parents=True, exist_ok=True)

    # Load all chips
    logger.info("Loading chips from all directories...")
    chips = load_all_chips(CONFIG["chip_dirs"])

    if len(chips) == 0:
        logger.error("No chips found!")
        return

    # Check MNDWI availability
    mndwi_available = sum(1 for c in chips if "mndwi" in c)
    logger.info(f"Chips with MNDWI: {mndwi_available}/{len(chips)}")

    # Use MNDWI only if > 50% of chips have it
    use_mndwi = mndwi_available > len(chips) * 0.5
    logger.info(f"Using MNDWI features: {use_mndwi}")

    # Split into train/test (chip-level)
    np.random.seed(CONFIG["random_seed"])
    indices = np.random.permutation(len(chips))
    n_test = max(1, int(len(chips) * CONFIG["test_size"]))
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    train_chips = [chips[i] for i in train_indices]
    test_chips = [chips[i] for i in test_indices]

    logger.info(f"Train: {len(train_chips)} chips, Test: {len(test_chips)} chips")

    # Extract features for training
    logger.info("Extracting features for training...")
    X_train_list = []
    y_train_list = []
    feature_names = None

    for i, chip in enumerate(train_chips):
        if (i + 1) % 20 == 0:
            logger.info(f"  Processing train chip {i + 1}/{len(train_chips)}")

        features, feat_names = extract_features(chip, include_mndwi=use_mndwi)

        if feature_names is None:
            feature_names = feat_names

        h, w, n_features = features.shape
        X_flat = features.reshape(-1, n_features)
        y_flat = chip["label"].flatten()

        # Sample pixels
        n_samples = len(y_flat)
        sample_size = int(n_samples * CONFIG["sample_rate"])
        sample_idx = np.random.choice(n_samples, size=sample_size, replace=False)

        X_train_list.append(X_flat[sample_idx])
        y_train_list.append(y_flat[sample_idx])

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)

    logger.info(
        f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features"
    )
    logger.info(
        f"Class distribution: {y_train.sum():.0f} water, {len(y_train) - y_train.sum():.0f} non-water"
    )
    logger.info(f"Features: {feature_names}")

    # Train LightGBM
    logger.info("Training LightGBM...")
    start_time = time.time()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=CONFIG["random_seed"]
    )

    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
    val_data = lgb.Dataset(
        X_val, label=y_val, feature_name=feature_names, reference=train_data
    )

    model = lgb.train(
        CONFIG["lgb_params"],
        train_data,
        num_boost_round=CONFIG["num_boost_round"],
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=CONFIG["early_stopping_rounds"]),
            lgb.log_evaluation(period=100),
        ],
    )

    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.1f}s")

    # Save model
    model_name = "lightgbm_v8_ensemble" + ("_mndwi" if use_mndwi else "")
    model_path = CONFIG["model_dir"] / f"{model_name}.txt"
    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = []
    all_preds = []
    all_labels = []

    for i, chip in enumerate(test_chips):
        features, _ = extract_features(chip, include_mndwi=use_mndwi)

        h, w, n_features = features.shape
        X_test = features.reshape(-1, n_features)
        y_true = chip["label"].flatten()

        # Predict
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(np.float32)

        # Compute metrics
        iou = compute_iou(y_pred, y_true)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        test_results.append(
            {
                "chip": chip["name"],
                "source": chip.get("source_dir", "unknown"),
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "water_fraction": float(y_true.mean()),
                "has_mndwi": "mndwi" in chip,
            }
        )

        all_preds.extend(y_pred.tolist())
        all_labels.extend(y_true.tolist())

        logger.info(
            f"  {chip['name']}: IoU={iou:.4f}, P={precision:.4f}, R={recall:.4f}"
        )

    # Overall metrics
    overall_iou = compute_iou(np.array(all_preds), np.array(all_labels))
    overall_precision = precision_score(all_labels, all_preds, zero_division=0)
    overall_recall = recall_score(all_labels, all_preds, zero_division=0)
    overall_f1 = f1_score(all_labels, all_preds, zero_division=0)

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feature_importance = dict(zip(feature_names, importance.tolist()))
    sorted_importance = sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True
    )

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Overall IoU:       {overall_iou:.4f}")
    logger.info(f"Overall Precision: {overall_precision:.4f}")
    logger.info(f"Overall Recall:    {overall_recall:.4f}")
    logger.info(f"Overall F1:        {overall_f1:.4f}")
    logger.info(f"\nUsed MNDWI: {use_mndwi}")
    logger.info(f"Total features: {len(feature_names)}")
    logger.info(f"\nTop 15 Features:")
    for feat, imp in sorted_importance[:15]:
        logger.info(f"  {feat}: {imp:.0f}")

    # Save results
    results = {
        "version": CONFIG["version"],
        "timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "used_mndwi": use_mndwi,
        "feature_names": feature_names,
        "train_chips": [c["name"] for c in train_chips],
        "test_chips": [c["name"] for c in test_chips],
        "training_time_seconds": train_time,
        "overall_metrics": {
            "iou": overall_iou,
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
        },
        "per_chip_results": test_results,
        "feature_importance": dict(sorted_importance),
        "config": {
            "sample_rate": CONFIG["sample_rate"],
            "test_size": CONFIG["test_size"],
            "lgb_params": CONFIG["lgb_params"],
        },
    }

    results_path = (
        CONFIG["results_dir"]
        / f"retrain_v8_results{'_mndwi' if use_mndwi else ''}.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)

    return model, results


if __name__ == "__main__":
    main()
