#!/usr/bin/env python3
"""
================================================================================
RETRAIN V6 - LightGBM Training (NPY only, no rasterio)
================================================================================
Trains LightGBM on .npy chips from chips/ directory only.
Avoids rasterio dependency to work around numpy version conflict.

Usage:
    python retrain_v6.py
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
from scipy.ndimage import uniform_filter, minimum_filter, maximum_filter, laplace
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "version": "6.0",
    "random_seed": 42,
    "chip_dir": Path("/home/mit-aoe/sar_water_detection/chips"),
    "model_dir": Path("/home/mit-aoe/sar_water_detection/models"),
    "results_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    "test_size": 0.2,
    "sample_rate": 0.15,  # Sample 15% of pixels for faster training
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
        "min_child_samples": 100,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    },
    "num_boost_round": 500,
    "early_stopping_rounds": 50,
}

# Feature names matching v4 training
FEATURE_NAMES = [
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


def compute_otsu_threshold(data: np.ndarray) -> float:
    """Compute Otsu's threshold."""
    data_flat = data[np.isfinite(data)].flatten()
    if len(data_flat) < 100:
        return np.nanmedian(data_flat)

    hist, bin_edges = np.histogram(data_flat, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
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

    return threshold


def compute_glcm_features(
    data: np.ndarray, window_size: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute simplified GLCM-like texture features."""
    h, w = data.shape
    contrast = np.zeros((h, w), dtype=np.float32)
    homogeneity = np.zeros((h, w), dtype=np.float32)

    pad = window_size // 2
    padded = np.pad(data, pad, mode="reflect")

    for i in range(h):
        for j in range(w):
            window = padded[i : i + window_size, j : j + window_size]
            local_std = np.std(window)
            local_range = np.ptp(window)
            contrast[i, j] = local_std
            homogeneity[i, j] = 1.0 / (1.0 + local_range) if local_range > 0 else 1.0

    return contrast, homogeneity


def compute_glcm_fast(
    data: np.ndarray, window_size: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Fast approximate GLCM features using uniform filter."""
    mean = uniform_filter(data, size=window_size)
    sq_mean = uniform_filter(data**2, size=window_size)
    variance = sq_mean - mean**2
    variance = np.maximum(variance, 0)
    contrast = np.sqrt(variance)

    local_max = maximum_filter(data, size=window_size)
    local_min = minimum_filter(data, size=window_size)
    local_range = local_max - local_min
    homogeneity = 1.0 / (1.0 + local_range)

    return contrast.astype(np.float32), homogeneity.astype(np.float32)


def extract_features(
    vv: np.ndarray,
    vh: np.ndarray,
    dem: np.ndarray,
    slope: np.ndarray,
    hand: np.ndarray,
    twi: np.ndarray,
) -> np.ndarray:
    """Extract comprehensive feature set matching v4 training."""
    h, w = vv.shape
    features = []

    # 1. Basic SAR features
    features.append(vv)  # VV
    features.append(vh)  # VH

    with np.errstate(divide="ignore", invalid="ignore"):
        vv_vh_ratio = np.where(vh != 0, vv / vh, 0)
        vv_vh_ratio = np.clip(vv_vh_ratio, -10, 10)
    features.append(vv_vh_ratio)  # VV_VH_ratio
    features.append(vv - vh)  # VV_VH_diff

    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = np.where((vv + vh) != 0, (vv - vh) / (vv + vh + 1e-10), 0)
    features.append(ndwi)  # NDWI_like

    with np.errstate(divide="ignore", invalid="ignore"):
        rvi = np.where(vv != 0, 4 * vh / (vv + vh + 1e-10), 0)
    features.append(rvi)  # RVI

    # 2. Multi-scale texture features
    scales = [3, 5, 9, 15, 21]
    for scale in scales:
        # VV statistics - FIXED: using minimum_filter instead of uniform_filter(minimum)
        vv_mean = uniform_filter(vv, size=scale)
        vv_std = np.sqrt(np.maximum(uniform_filter(vv**2, size=scale) - vv_mean**2, 0))
        vv_min = minimum_filter(vv, size=scale)
        vv_max = maximum_filter(vv, size=scale)

        features.append(vv_mean)
        features.append(vv_std)
        features.append(vv_min)
        features.append(vv_max)

        # VH statistics - FIXED: using minimum_filter
        vh_mean = uniform_filter(vh, size=scale)
        vh_std = np.sqrt(np.maximum(uniform_filter(vh**2, size=scale) - vh_mean**2, 0))
        vh_min = minimum_filter(vh, size=scale)
        vh_max = maximum_filter(vh, size=scale)

        features.append(vh_mean)
        features.append(vh_std)
        features.append(vh_min)
        features.append(vh_max)

    # 3. Gradient features
    gy_vv, gx_vv = np.gradient(vv)
    vv_grad_mag = np.sqrt(gx_vv**2 + gy_vv**2)
    features.append(vv_grad_mag)

    gy_vh, gx_vh = np.gradient(vh)
    vh_grad_mag = np.sqrt(gx_vh**2 + gy_vh**2)
    features.append(vh_grad_mag)

    features.append(np.abs(laplace(vv)))  # VV_laplacian
    features.append(np.abs(laplace(vh)))  # VH_laplacian

    # 4. Morphological features
    from scipy.ndimage import grey_opening, grey_closing

    vv_opened = grey_opening(vv, size=5)
    vv_closed = grey_closing(vv, size=5)
    vh_opened = grey_opening(vh, size=5)
    vh_closed = grey_closing(vh, size=5)

    features.append(vv_opened)
    features.append(vv_closed)
    features.append(vh_opened)
    features.append(vh_closed)

    # 5. Otsu-based features
    vv_otsu = compute_otsu_threshold(vv)
    vh_otsu = compute_otsu_threshold(vh)
    features.append(vv - vv_otsu)  # VV_otsu_diff
    features.append(vh - vh_otsu)  # VH_otsu_diff

    # 6. Local contrast
    vv_local_mean = uniform_filter(vv, size=9)
    vh_local_mean = uniform_filter(vh, size=9)
    features.append(vv - vv_local_mean)  # local_contrast_vv
    features.append(vh - vh_local_mean)  # local_contrast_vh

    # 7. GLCM-like texture features (fast version)
    vv_contrast, vv_homogeneity = compute_glcm_fast(vv)
    vh_contrast, vh_homogeneity = compute_glcm_fast(vh)
    features.append(vv_contrast)
    features.append(vv_homogeneity)
    features.append(vh_contrast)
    features.append(vh_homogeneity)

    # 8. Pseudo-entropy
    vv_prob = (vv - vv.min()) / (vv.max() - vv.min() + 1e-10)
    vv_prob = np.clip(vv_prob, 1e-10, 1 - 1e-10)
    pseudo_entropy = -vv_prob * np.log2(vv_prob) - (1 - vv_prob) * np.log2(1 - vv_prob)
    features.append(pseudo_entropy)

    # 9. DEM-derived features
    features.append(dem)
    features.append(slope)
    features.append(hand)
    features.append(twi)

    # 10. Physics-based scores
    hand_score = 1.0 / (1.0 + np.exp((hand - 10) / 3.0))
    slope_score = 1.0 / (1.0 + np.exp((slope - 8) / 3.0))
    twi_score = 1.0 / (1.0 + np.exp((8 - twi) / 2.0))

    features.append(hand_score)
    features.append(slope_score)
    features.append(twi_score)

    # Stack all features
    feature_stack = np.stack(features, axis=-1)

    # Replace NaN/Inf with 0
    feature_stack = np.nan_to_num(feature_stack, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_stack.astype(np.float32)


def load_chip(chip_path: Path) -> Optional[Dict]:
    """Load a .npy chip file."""
    try:
        data = np.load(chip_path, allow_pickle=True)

        if isinstance(data, np.ndarray):
            if data.ndim == 3 and data.shape[2] >= 7:
                # Format: VV, VH, DEM, SLOPE, HAND, TWI, label, ...
                return {
                    "vv": data[:, :, 0].astype(np.float32),
                    "vh": data[:, :, 1].astype(np.float32),
                    "dem": data[:, :, 2].astype(np.float32),
                    "slope": data[:, :, 3].astype(np.float32),
                    "hand": data[:, :, 4].astype(np.float32),
                    "twi": data[:, :, 5].astype(np.float32),
                    "label": (data[:, :, 6] > 0).astype(np.float32),
                    "name": chip_path.stem,
                }

        logger.warning(f"Unexpected format in {chip_path}")
        return None

    except Exception as e:
        logger.error(f"Error loading {chip_path}: {e}")
        return None


def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute IoU score."""
    pred_bin = pred > 0.5
    target_bin = target > 0.5

    intersection = np.logical_and(pred_bin, target_bin).sum()
    union = np.logical_or(pred_bin, target_bin).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection) / float(union)


def main():
    """Main training pipeline."""
    logger.info("=" * 70)
    logger.info("RETRAIN V6 - LightGBM Training (NPY only)")
    logger.info("=" * 70)

    # Create output directories
    CONFIG["model_dir"].mkdir(parents=True, exist_ok=True)
    CONFIG["results_dir"].mkdir(parents=True, exist_ok=True)

    # Find all .npy chips
    chip_dir = CONFIG["chip_dir"]
    npy_files = sorted(chip_dir.glob("*_with_truth.npy"))

    logger.info(f"Found {len(npy_files)} .npy chips in {chip_dir}")

    if len(npy_files) == 0:
        logger.error("No chips found!")
        return

    # Load all chips
    logger.info("Loading chips...")
    chips = []
    for chip_path in npy_files:
        chip_data = load_chip(chip_path)
        if chip_data is not None:
            chips.append(chip_data)
            logger.info(f"  Loaded: {chip_data['name']} ({chip_data['label'].shape})")

    logger.info(f"Successfully loaded {len(chips)} chips")

    # Split into train/test (chip-level)
    np.random.seed(CONFIG["random_seed"])
    indices = np.random.permutation(len(chips))
    n_test = int(len(chips) * CONFIG["test_size"])
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    train_chips = [chips[i] for i in train_indices]
    test_chips = [chips[i] for i in test_indices]

    logger.info(f"Train: {len(train_chips)} chips, Test: {len(test_chips)} chips")

    # Extract features and labels for training
    logger.info("Extracting features for training...")
    X_train_list = []
    y_train_list = []

    for i, chip in enumerate(train_chips):
        logger.info(
            f"  Processing train chip {i + 1}/{len(train_chips)}: {chip['name']}"
        )

        features = extract_features(
            chip["vv"],
            chip["vh"],
            chip["dem"],
            chip["slope"],
            chip["hand"],
            chip["twi"],
        )

        # Flatten
        h, w, n_features = features.shape
        X_flat = features.reshape(-1, n_features)
        y_flat = chip["label"].flatten()

        # Random sampling to reduce memory
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
        f"Class distribution: {y_train.sum()} water, {len(y_train) - y_train.sum()} non-water"
    )

    # Train LightGBM
    logger.info("Training LightGBM...")
    start_time = time.time()

    # Create validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=CONFIG["random_seed"]
    )

    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(
        X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_data
    )

    model = lgb.train(
        CONFIG["lgb_params"],
        train_data,
        num_boost_round=CONFIG["num_boost_round"],
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=CONFIG["early_stopping_rounds"]),
            lgb.log_evaluation(period=50),
        ],
    )

    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.1f}s")

    # Save model
    model_path = CONFIG["model_dir"] / "lightgbm_v6_retrained.txt"
    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = []
    all_preds = []
    all_labels = []

    for i, chip in enumerate(test_chips):
        logger.info(f"  Testing chip {i + 1}/{len(test_chips)}: {chip['name']}")

        features = extract_features(
            chip["vv"],
            chip["vh"],
            chip["dem"],
            chip["slope"],
            chip["hand"],
            chip["twi"],
        )

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
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "water_fraction": float(y_true.mean()),
            }
        )

        all_preds.extend(y_pred.tolist())
        all_labels.extend(y_true.tolist())

        logger.info(
            f"    IoU: {iou:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}"
        )

    # Overall metrics
    overall_iou = compute_iou(np.array(all_preds), np.array(all_labels))
    overall_precision = precision_score(all_labels, all_preds, zero_division=0)
    overall_recall = recall_score(all_labels, all_preds, zero_division=0)
    overall_f1 = f1_score(all_labels, all_preds, zero_division=0)

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feature_importance = dict(zip(FEATURE_NAMES, importance.tolist()))
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
    logger.info(f"\nTop 10 Features:")
    for feat, imp in sorted_importance[:10]:
        logger.info(f"  {feat}: {imp:.0f}")

    # Save results
    results = {
        "version": CONFIG["version"],
        "timestamp": datetime.now().isoformat(),
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

    results_path = CONFIG["results_dir"] / "retrain_v6_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
