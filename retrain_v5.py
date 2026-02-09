#!/usr/bin/env python3
"""
================================================================================
RETRAIN V5 - Full Dataset Training
================================================================================
Trains LightGBM and U-Net on ALL available data:
- 86 chips from chips/ (with_truth.npy)
- 99 chips from chips_expanded/ (.tif)

Uses proper 80/20 train/test split with stratification.
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

# Try to import rasterio for .tif files, but don't fail if unavailable
try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not available, skipping .tif files")
from sklearn.model_selection import train_test_split
import lightgbm as lgb

warnings.filterwarnings("ignore")

# Configuration
CONFIG = {
    "version": "5.0",
    "random_seed": 42,
    "chip_dirs": [
        Path("/home/mit-aoe/sar_water_detection/chips"),
        Path("/home/mit-aoe/sar_water_detection/chips_expanded"),
    ],
    "model_dir": Path("/home/mit-aoe/sar_water_detection/models"),
    "results_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    "test_size": 0.2,
    "sample_rate": 0.1,  # Sample 10% of pixels (for memory)
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
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_69_features(vv, vh, dem, hand, slope, twi):
    """Compute 69 features matching LightGBM v4 training."""
    h, w = vv.shape

    # Handle NaN
    vv = np.nan_to_num(vv, nan=-20)
    vh = np.nan_to_num(vh, nan=-25)
    dem = np.nan_to_num(dem, nan=100)
    hand = np.nan_to_num(hand, nan=50)
    slope = np.nan_to_num(slope, nan=30)
    twi = np.nan_to_num(twi, nan=5)

    vv_lin = 10 ** (vv / 10)
    vh_lin = 10 ** (vh / 10)

    features = {}

    # 1-4: Core SAR
    features["VV"] = vv
    features["VH"] = vh
    features["VV_minus_VH"] = vv - vh
    features["VV_over_VH_log"] = np.log10(np.maximum(vv_lin / (vh_lin + 1e-10), 1e-10))

    # 5-8: Physics
    features["HAND"] = hand
    features["SLOPE"] = slope
    features["TWI"] = twi
    features["DEM"] = dem

    # 9-10: DEM derivatives
    gy_dem, gx_dem = np.gradient(dem)
    features["DEM_gradient"] = np.sqrt(gx_dem**2 + gy_dem**2)
    features["DEM_aspect"] = np.arctan2(gy_dem, gx_dem + 1e-10)

    # 11-14: Polarimetric
    features["pseudo_entropy"] = -np.abs(vv - vh) / (np.abs(vv) + np.abs(vh) + 1e-10)
    features["pseudo_alpha"] = np.arctan2(np.abs(vh), np.abs(vv) + 1e-10)
    features["RVI"] = 4 * vh_lin / (vv_lin + vh_lin + 1e-10)
    features["span"] = vv_lin + vh_lin

    # 15-32: Multi-scale texture
    for scale in [3, 5, 9, 15]:
        vv_mean = uniform_filter(vv, size=scale)
        vv_sq = uniform_filter(vv**2, size=scale)
        vv_std = np.sqrt(np.maximum(vv_sq - vv_mean**2, 0))
        vv_min = minimum_filter(vv, size=scale)

        vh_mean = uniform_filter(vh, size=scale)
        vh_sq = uniform_filter(vh**2, size=scale)
        vh_std = np.sqrt(np.maximum(vh_sq - vh_mean**2, 0))
        vh_min = minimum_filter(vh, size=scale)

        features[f"VV_mean_s{scale}"] = vv_mean
        features[f"VV_std_s{scale}"] = vv_std
        features[f"VH_mean_s{scale}"] = vh_mean
        features[f"VH_std_s{scale}"] = vh_std
        if scale <= 9:
            features[f"VV_min_s{scale}"] = vv_min
            features[f"VH_min_s{scale}"] = vh_min

    # 37-40: Scale 21
    for scale in [21]:
        features[f"VV_mean_s{scale}"] = uniform_filter(vv, size=scale)
        features[f"VV_std_s{scale}"] = np.sqrt(
            np.maximum(uniform_filter(vv**2, scale) - uniform_filter(vv, scale) ** 2, 0)
        )
        features[f"VH_mean_s{scale}"] = uniform_filter(vh, size=scale)
        features[f"VH_std_s{scale}"] = np.sqrt(
            np.maximum(uniform_filter(vh**2, scale) - uniform_filter(vh, scale) ** 2, 0)
        )

    # 41-44: ENL and CV
    features["VV_ENL"] = uniform_filter(vv, 9) ** 2 / (
        uniform_filter(vv**2, 9) - uniform_filter(vv, 9) ** 2 + 1e-10
    )
    features["VV_CV"] = np.sqrt(uniform_filter(vv**2, 9)) / (
        np.abs(uniform_filter(vv, 9)) + 1e-10
    )
    features["VH_ENL"] = uniform_filter(vh, 9) ** 2 / (
        uniform_filter(vh**2, 9) - uniform_filter(vh, 9) ** 2 + 1e-10
    )
    features["VH_CV"] = np.sqrt(uniform_filter(vh**2, 9)) / (
        np.abs(uniform_filter(vh, 9)) + 1e-10
    )

    # 45-50: GLCM approximations
    gy_vv, gx_vv = np.gradient(vv)
    gy_vh, gx_vh = np.gradient(vh)
    contrast_vv = uniform_filter(gx_vv**2 + gy_vv**2, 7)
    contrast_vh = uniform_filter(gx_vh**2 + gy_vh**2, 7)
    features["VV_glcm_contrast"] = contrast_vv
    features["VV_glcm_homogeneity"] = 1.0 / (1.0 + contrast_vv)
    features["VV_glcm_energy"] = uniform_filter(vv**2, 7)
    features["VH_glcm_contrast"] = contrast_vh
    features["VH_glcm_homogeneity"] = 1.0 / (1.0 + contrast_vh)
    features["VH_glcm_energy"] = uniform_filter(vh**2, 7)

    # 51-54: Morphological
    dilated = maximum_filter(vh, size=5)
    eroded = minimum_filter(vh, size=5)
    opened = maximum_filter(eroded, size=5)
    closed = minimum_filter(dilated, size=5)
    features["VH_opened"] = opened
    features["VH_closed"] = closed
    features["VH_white_tophat"] = vh - opened
    features["VH_black_tophat"] = closed - vh

    # 55: Line response
    features["line_response"] = np.abs(laplace(vh))

    # 56-61: Otsu/Kapur
    vv_otsu = np.median(vv)
    vh_otsu = np.median(vh)
    features["VV_otsu_diff"] = vv - vv_otsu
    features["VH_otsu_diff"] = vh - vh_otsu
    features["VV_below_otsu"] = (vv < vv_otsu).astype(float)
    features["VH_below_otsu"] = (vh < vh_otsu).astype(float)
    features["VV_below_kapur"] = (vv < np.percentile(vv, 25)).astype(float)
    features["VH_below_kapur"] = (vh < np.percentile(vh, 25)).astype(float)

    # 62-66: Physics scores
    hand_score = 1.0 / (1.0 + np.exp((hand - 10) / 3.0))
    slope_score = 1.0 / (1.0 + np.exp((slope - 8) / 3.0))
    vh_score = 1.0 / (1.0 + np.exp((vh + 18) / 2.0))
    twi_score = 1.0 / (1.0 + np.exp((8 - twi) / 2.0))
    features["physics_composite"] = hand_score * slope_score * vh_score * twi_score
    features["hand_score"] = hand_score
    features["slope_score"] = slope_score
    features["vh_score"] = vh_score
    features["twi_score"] = twi_score

    # 67-69: Gradients
    features["VH_gradient"] = np.sqrt(gx_vh**2 + gy_vh**2)
    features["VV_gradient"] = np.sqrt(gx_vv**2 + gy_vv**2)
    features["VH_laplacian"] = laplace(vh)

    return features


# Feature order (must match exactly)
FEATURE_ORDER = [
    "VV",
    "VH",
    "VV_minus_VH",
    "VV_over_VH_log",
    "HAND",
    "SLOPE",
    "TWI",
    "DEM",
    "DEM_gradient",
    "DEM_aspect",
    "pseudo_entropy",
    "pseudo_alpha",
    "RVI",
    "span",
    "VV_mean_s3",
    "VV_std_s3",
    "VH_mean_s3",
    "VH_std_s3",
    "VV_min_s3",
    "VH_min_s3",
    "VV_mean_s5",
    "VV_std_s5",
    "VH_mean_s5",
    "VH_std_s5",
    "VV_min_s5",
    "VH_min_s5",
    "VV_mean_s9",
    "VV_std_s9",
    "VH_mean_s9",
    "VH_std_s9",
    "VV_min_s9",
    "VH_min_s9",
    "VV_mean_s15",
    "VV_std_s15",
    "VH_mean_s15",
    "VH_std_s15",
    "VV_mean_s21",
    "VV_std_s21",
    "VH_mean_s21",
    "VH_std_s21",
    "VV_ENL",
    "VV_CV",
    "VH_ENL",
    "VH_CV",
    "VV_glcm_contrast",
    "VV_glcm_homogeneity",
    "VV_glcm_energy",
    "VH_glcm_contrast",
    "VH_glcm_homogeneity",
    "VH_glcm_energy",
    "VH_opened",
    "VH_closed",
    "VH_white_tophat",
    "VH_black_tophat",
    "line_response",
    "VV_otsu_diff",
    "VH_otsu_diff",
    "VV_below_otsu",
    "VH_below_otsu",
    "VV_below_kapur",
    "VH_below_kapur",
    "physics_composite",
    "hand_score",
    "slope_score",
    "vh_score",
    "twi_score",
    "VH_gradient",
    "VV_gradient",
    "VH_laplacian",
]


def load_chip_npy(filepath):
    """Load .npy chip with 8 bands."""
    data = np.load(filepath)
    if data.shape[0] < data.shape[2]:
        data = np.transpose(data, (1, 2, 0))
    if data.shape[2] < 8:
        return None, None

    vv = data[:, :, 0]
    vh = data[:, :, 1]
    dem = data[:, :, 3]
    hand = data[:, :, 4]
    slope = data[:, :, 5]
    twi = data[:, :, 6]
    truth = data[:, :, 7]

    return (vv, vh, dem, hand, slope, twi), truth


def load_chip_tif(filepath):
    """Load .tif chip."""
    if not HAS_RASTERIO:
        return None, None
    try:
        with rasterio.open(filepath) as src:
            data = src.read()

        if data.shape[0] < 8:
            return None, None

        vv = data[0]
        vh = data[1]
        dem = data[3]
        hand = data[4]
        slope = data[5]
        twi = data[6]
        truth = data[7]

        return (vv, vh, dem, hand, slope, twi), truth
    except Exception as e:
        logger.warning(f"Failed to load {filepath}: {e}")
        return None, None


def load_all_chips():
    """Load all chips from both directories."""
    all_chips = []

    for chip_dir in CONFIG["chip_dirs"]:
        if not chip_dir.exists():
            logger.warning(f"Directory not found: {chip_dir}")
            continue

        # Load .npy files
        for f in chip_dir.glob("*_with_truth.npy"):
            bands, truth = load_chip_npy(f)
            if bands is not None:
                all_chips.append((f.stem, bands, truth))

        # Load .tif files
        for f in chip_dir.glob("*.tif"):
            bands, truth = load_chip_tif(f)
            if bands is not None:
                all_chips.append((f.stem, bands, truth))

    logger.info(f"Loaded {len(all_chips)} chips total")
    return all_chips


def prepare_training_data(chips, sample_rate=0.1):
    """Prepare training data with random sampling."""
    np.random.seed(CONFIG["random_seed"])

    all_X = []
    all_y = []
    chip_names = []

    for name, bands, truth in chips:
        vv, vh, dem, hand, slope, twi = bands
        h, w = vv.shape

        # Compute features
        features = compute_69_features(vv, vh, dem, hand, slope, twi)

        # Stack features in order
        X_chip = np.column_stack(
            [
                np.nan_to_num(features[fname], nan=0, posinf=0, neginf=0).flatten()
                for fname in FEATURE_ORDER
            ]
        )
        y_chip = truth.flatten()

        # Random sample
        n_samples = int(len(y_chip) * sample_rate)
        indices = np.random.choice(len(y_chip), size=n_samples, replace=False)

        all_X.append(X_chip[indices])
        all_y.append(y_chip[indices])
        chip_names.append(name)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    logger.info(f"Training data: {X.shape[0]:,} samples, {X.shape[1]} features")
    logger.info(f"Water ratio: {np.mean(y > 0.5):.2%}")

    return X, y, chip_names


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM model."""
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_ORDER)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_ORDER)

    params = CONFIG["lgb_params"].copy()
    n_estimators = params.pop("n_estimators")
    early_stopping = params.pop("early_stopping_rounds")

    model = lgb.train(
        params,
        train_data,
        num_boost_round=n_estimators,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    return model


def compute_metrics(y_true, y_pred, threshold=0.5):
    """Compute evaluation metrics."""
    y_pred_bin = y_pred > threshold
    y_true_bin = y_true > threshold

    tp = np.sum(y_pred_bin & y_true_bin)
    fp = np.sum(y_pred_bin & ~y_true_bin)
    fn = np.sum(~y_pred_bin & y_true_bin)
    tn = np.sum(~y_pred_bin & ~y_true_bin)

    iou = tp / (tp + fp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def main():
    logger.info("=" * 80)
    logger.info("RETRAIN V5 - Full Dataset Training")
    logger.info("=" * 80)

    # Load all chips
    chips = load_all_chips()

    if len(chips) == 0:
        logger.error("No chips found!")
        return

    # Split into train/test by chip
    np.random.seed(CONFIG["random_seed"])
    np.random.shuffle(chips)

    n_test = int(len(chips) * CONFIG["test_size"])
    test_chips = chips[:n_test]
    train_chips = chips[n_test:]

    logger.info(f"Train chips: {len(train_chips)}")
    logger.info(f"Test chips: {len(test_chips)}")

    # Prepare training data
    logger.info("Preparing training data...")
    X_train, y_train, train_names = prepare_training_data(
        train_chips, CONFIG["sample_rate"]
    )

    # Split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=CONFIG["random_seed"]
    )

    logger.info(f"Train: {len(y_train):,}, Val: {len(y_val):,}")

    # Train LightGBM
    logger.info("Training LightGBM...")
    start_time = time.time()
    model = train_lightgbm(X_train, y_train, X_val, y_val)
    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.1f}s")

    # Evaluate on validation
    y_val_pred = model.predict(X_val)
    val_metrics = compute_metrics(y_val, y_val_pred)
    logger.info(f"Validation IoU: {val_metrics['iou']:.4f}")

    # Evaluate on test chips (full resolution)
    logger.info("Evaluating on test chips...")
    test_results = []

    for name, bands, truth in test_chips:
        vv, vh, dem, hand, slope, twi = bands
        h, w = vv.shape

        features = compute_69_features(vv, vh, dem, hand, slope, twi)
        X_test = np.column_stack(
            [
                np.nan_to_num(features[fname], nan=0, posinf=0, neginf=0).flatten()
                for fname in FEATURE_ORDER
            ]
        )

        y_pred = model.predict(X_test).reshape(h, w)
        metrics = compute_metrics(truth.flatten(), y_pred.flatten())
        metrics["chip"] = name
        test_results.append(metrics)

        logger.info(f"  {name}: IoU={metrics['iou']:.4f}")

    # Summary
    test_ious = [r["iou"] for r in test_results]
    mean_test_iou = np.mean(test_ious)
    logger.info(f"\nTest Mean IoU: {mean_test_iou:.4f}")

    # Save model
    model_path = CONFIG["model_dir"] / "lightgbm_v5_full.txt"
    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")

    # Save results
    results = {
        "version": CONFIG["version"],
        "timestamp": datetime.now().isoformat(),
        "n_train_chips": len(train_chips),
        "n_test_chips": len(test_chips),
        "n_train_samples": len(y_train),
        "train_time_seconds": train_time,
        "validation_metrics": val_metrics,
        "test_mean_iou": mean_test_iou,
        "test_results": test_results,
        "feature_importance": dict(
            zip(FEATURE_ORDER, [int(x) for x in model.feature_importance()])
        ),
    }

    results_path = CONFIG["results_dir"] / "training_results_v5.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
