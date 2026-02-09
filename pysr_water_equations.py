#!/usr/bin/env python3
"""
================================================================================
PySR SYMBOLIC REGRESSION FOR WATER DETECTION EQUATIONS
================================================================================
Discovers interpretable mathematical equations for SAR water detection.

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
from typing import Dict, List, Tuple, Optional

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter, sobel, minimum_filter, maximum_filter
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "random_seed": 42,
    "chip_dirs": [
        Path("/home/mit-aoe/sar_water_detection/chips"),
        Path("/home/mit-aoe/sar_water_detection/chips_expanded"),
    ],
    "results_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    "sample_rate": 0.02,  # 2% of pixels for speed
    "max_samples": 100000,  # Maximum samples for PySR
    "bands": {"VV": 0, "VH": 1, "DEM": 3, "HAND": 4, "SLOPE": 5, "TWI": 6, "TRUTH": 7},
}

CONFIG["results_dir"].mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(CONFIG["results_dir"] / "pysr_equations.log"),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def db_to_linear(db: np.ndarray) -> np.ndarray:
    return np.power(10, db / 10)


def load_chips() -> List[np.ndarray]:
    """Load all chips."""
    chips = []
    for chip_dir in CONFIG["chip_dirs"]:
        if not chip_dir.exists():
            continue
        for f in chip_dir.glob("*.npy"):
            try:
                chip = np.load(f).astype(np.float32)
                if chip.shape[0] >= 8 and np.nansum(chip[7]) > 0:
                    chips.append(chip)
            except:
                pass
        for f in chip_dir.glob("*.tif"):
            try:
                with rasterio.open(f) as src:
                    chip = src.read().astype(np.float32)
                if chip.shape[0] >= 8 and np.nansum(chip[7]) > 0:
                    chips.append(chip)
            except:
                pass
    logger.info(f"Loaded {len(chips)} chips")
    return chips


def extract_pysr_features(chip: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Extract features for PySR - focused on physically meaningful combinations.
    Keep it simple for interpretable equations.
    """
    vv = chip[0]
    vh = chip[1]
    dem = chip[3]
    hand = chip[4]
    slope = chip[5]
    twi = chip[6]

    features = []
    names = []

    # Basic SAR (dB)
    features.append(vv)
    names.append("VV")
    features.append(vh)
    names.append("VH")

    # VV-VH difference
    features.append(vv - vh)
    names.append("VV_VH_diff")

    # VV/VH ratio in linear scale
    vv_lin = db_to_linear(vv)
    vh_lin = db_to_linear(vh)
    ratio = np.log10(vv_lin / (vh_lin + 1e-10) + 1e-10)
    features.append(ratio)
    names.append("VV_VH_ratio")

    # Topographic
    features.append(hand)
    names.append("HAND")
    features.append(slope)
    names.append("SLOPE")
    features.append(twi)
    names.append("TWI")

    # Local texture (CV)
    local_mean = uniform_filter(vh, size=5)
    local_sq = uniform_filter(vh**2, size=5)
    local_std = np.sqrt(np.maximum(local_sq - local_mean**2, 0))
    cv = local_std / (np.abs(local_mean) + 1e-10)
    features.append(cv)
    names.append("VH_CV")

    # Local minimum (dark water detection)
    vh_min = minimum_filter(vh, size=5)
    features.append(vh_min)
    names.append("VH_min")

    # Stack
    feature_stack = np.stack(features, axis=0).astype(np.float32)
    feature_stack = np.nan_to_num(feature_stack, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_stack, names


def sample_data(chips: List[np.ndarray], sample_rate: float, max_samples: int):
    """Sample pixels from chips for PySR."""
    all_X = []
    all_y = []
    feature_names = None

    for chip in chips:
        features, names = extract_pysr_features(chip)
        if feature_names is None:
            feature_names = names

        truth = chip[7]
        h, w = truth.shape

        # Sample pixels
        n_samples = int(h * w * sample_rate)
        indices = np.random.choice(h * w, size=n_samples, replace=False)

        # Extract samples
        X = features.reshape(features.shape[0], -1).T[indices]
        y = truth.flatten()[indices]

        all_X.append(X)
        all_y.append(y)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    # Remove NaN values
    valid_mask = ~np.isnan(y) & np.all(np.isfinite(X), axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    # Limit total samples
    if len(y) > max_samples:
        idx = np.random.choice(len(y), size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    logger.info(
        f"Sampled {len(y)} pixels with {X.shape[1]} features (after removing NaN)"
    )
    return X, y, feature_names


# =============================================================================
# PySR SYMBOLIC REGRESSION
# =============================================================================


def run_pysr_regression(X: np.ndarray, y: np.ndarray, feature_names: List[str]):
    """Run PySR to discover water detection equations."""
    try:
        from pysr import PySRRegressor
    except ImportError:
        logger.error("PySR not installed. Installing...")
        os.system("pip install pysr")
        from pysr import PySRRegressor

    logger.info("Starting PySR symbolic regression...")
    logger.info(f"Features: {feature_names}")
    logger.info(f"Samples: {len(y)}, Water fraction: {y.mean():.3f}")

    # Create PySR model
    model = PySRRegressor(
        # Model complexity
        maxsize=20,
        maxdepth=5,
        niterations=50,  # Increase for better results
        # Operators
        binary_operators=["+", "-", "*", "/", "max", "min"],
        unary_operators=[
            "exp",
            "log",
            "sqrt",
            "abs",
            "square",
            "neg",
        ],
        # Constraints
        constraints={
            "/": (-1, 9),
            "log": 5,
            "sqrt": 5,
            "exp": 5,
        },
        nested_constraints={
            "log": {"log": 0, "exp": 0, "sqrt": 0},
            "sqrt": {"sqrt": 0, "log": 0},
            "exp": {"exp": 0, "log": 0},
        },
        # Complexity penalties
        complexity_of_operators={
            "/": 2,
            "log": 2,
            "sqrt": 1.5,
            "exp": 2,
        },
        complexity_of_constants=1,
        # Loss for binary classification (MSE works, can also try log loss)
        elementwise_loss="loss(pred, target) = (pred - target)^2",
        # Feature selection
        select_k_features=min(7, len(feature_names)),
        # Regularization
        weight_optimize=0.001,
        # Parallel
        populations=15,
        population_size=33,
        parsimony=0.003,
        # Batching for large datasets
        batching=True,
        batch_size=min(5000, len(y)),
        # Output
        temp_equation_file=str(CONFIG["results_dir"] / "water_equations.csv"),
        progress=True,
        verbosity=1,
        # Early stopping
        timeout_in_seconds=3600 * 8,  # 8 hours max
    )

    # Fit
    start_time = time.time()
    model.fit(X, y, variable_names=feature_names)
    elapsed = time.time() - start_time

    logger.info(f"PySR completed in {elapsed / 3600:.2f} hours")

    return model


def evaluate_equations(model, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
    """Evaluate discovered equations."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(y),
        "water_fraction": float(y.mean()),
        "features": feature_names,
        "equations": [],
    }

    try:
        equations = model.equations_
        logger.info(f"\nDiscovered {len(equations)} equations:")

        for idx, row in equations.iterrows():
            eq_str = str(row["equation"])
            complexity = int(row["complexity"])
            loss = float(row["loss"])

            # Evaluate IoU
            pred = model.predict(X, index=idx)
            pred_binary = (pred > 0.5).astype(int)
            y_binary = (y > 0.5).astype(int)

            tp = np.sum(pred_binary & y_binary)
            fp = np.sum(pred_binary & ~y_binary)
            fn = np.sum(~pred_binary & y_binary)
            iou = tp / (tp + fp + fn + 1e-10)

            eq_result = {
                "index": int(idx),
                "equation": eq_str,
                "complexity": complexity,
                "loss": loss,
                "iou": float(iou),
            }
            results["equations"].append(eq_result)

            logger.info(f"  [{complexity:2d}] IoU={iou:.4f} | {eq_str}")

        # Best equation by IoU
        best_eq = max(results["equations"], key=lambda x: x["iou"])
        results["best_equation"] = best_eq
        logger.info(f"\nBest equation (IoU {best_eq['iou']:.4f}):")
        logger.info(f"  {best_eq['equation']}")

    except Exception as e:
        logger.error(f"Error evaluating equations: {e}")

    return results


def discover_category_equations(chips: List[np.ndarray]):
    """
    Discover equations for different water body categories.
    Categories based on water fraction and characteristics.
    """
    categories = {
        "small_water": [],  # < 20% water
        "medium_water": [],  # 20-50% water
        "large_water": [],  # > 50% water
    }

    for chip in chips:
        water_fraction = np.nanmean(chip[7])
        if water_fraction < 0.2:
            categories["small_water"].append(chip)
        elif water_fraction < 0.5:
            categories["medium_water"].append(chip)
        else:
            categories["large_water"].append(chip)

    all_results = {}

    for cat_name, cat_chips in categories.items():
        if len(cat_chips) < 5:
            logger.info(f"Skipping {cat_name}: only {len(cat_chips)} chips")
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"CATEGORY: {cat_name} ({len(cat_chips)} chips)")
        logger.info("=" * 60)

        X, y, names = sample_data(
            cat_chips, CONFIG["sample_rate"], CONFIG["max_samples"]
        )
        model = run_pysr_regression(X, y, names)
        results = evaluate_equations(model, X, y, names)

        all_results[cat_name] = results

        # Save intermediate results
        with open(CONFIG["results_dir"] / f"pysr_{cat_name}.json", "w") as f:
            json.dump(results, f, indent=2)

    return all_results


# =============================================================================
# MAIN
# =============================================================================


def main():
    logger.info("=" * 80)
    logger.info("PySR SYMBOLIC REGRESSION FOR WATER DETECTION")
    logger.info("=" * 80)
    logger.info(f"Started: {datetime.now().isoformat()}")

    np.random.seed(CONFIG["random_seed"])

    # Load data
    chips = load_chips()

    # Split train/test
    train_chips, test_chips = train_test_split(
        chips, test_size=0.15, random_state=CONFIG["random_seed"]
    )

    # Option 1: Global equation discovery
    logger.info("\n" + "=" * 60)
    logger.info("GLOBAL EQUATION DISCOVERY")
    logger.info("=" * 60)

    X_train, y_train, feature_names = sample_data(
        train_chips, CONFIG["sample_rate"], CONFIG["max_samples"]
    )

    model = run_pysr_regression(X_train, y_train, feature_names)
    results = evaluate_equations(model, X_train, y_train, feature_names)

    # Save global results
    with open(CONFIG["results_dir"] / "pysr_global_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Test on held-out data
    logger.info("\n" + "=" * 60)
    logger.info("TESTING ON HELD-OUT DATA")
    logger.info("=" * 60)

    X_test, y_test, _ = sample_data(
        test_chips, CONFIG["sample_rate"], CONFIG["max_samples"] // 2
    )
    test_results = evaluate_equations(model, X_test, y_test, feature_names)

    with open(CONFIG["results_dir"] / "pysr_test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    # Option 2: Category-specific equations
    logger.info("\n" + "=" * 60)
    logger.info("CATEGORY-SPECIFIC EQUATION DISCOVERY")
    logger.info("=" * 60)

    category_results = discover_category_equations(train_chips)

    # Save all results
    all_results = {
        "global": results,
        "test": test_results,
        "categories": category_results,
    }

    with open(CONFIG["results_dir"] / "pysr_all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nCompleted: {datetime.now().isoformat()}")
    logger.info(f"Results saved to: {CONFIG['results_dir']}")

    return all_results


if __name__ == "__main__":
    main()
