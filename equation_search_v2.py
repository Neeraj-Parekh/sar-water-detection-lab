#!/usr/bin/env python3
"""
PySR Equation Search for SAR Water Detection
=============================================
Discovers interpretable mathematical equations for water detection.

This is a separate script that runs for ~10 hours with checkpointing.

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
import numpy as np

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("equation_search.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "chip_dir": Path("/home/mit-aoe/sar_water_detection/chips"),
    "output_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    "random_seed": 42,
    "max_runtime_hours": 10,
    "checkpoint_interval": 100,  # Save every N iterations
}


def load_chips(chip_dir: Path):
    """Load all chip files."""
    chips = []
    for f in sorted(chip_dir.glob("*.npy")):
        try:
            data = np.load(f, allow_pickle=True)
            if isinstance(data, np.lib.npyio.NpzFile):
                data = data["arr_0"]
            chips.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")
    logger.info(f"Loaded {len(chips)} chips")
    return chips


def prepare_data(chips, samples_per_chip=2000):
    """
    Prepare pixel-level data for symbolic regression.

    Features: VV, VH, VV-VH, VV/VH, HAND, SLOPE, TWI
    Target: Water probability (from MNDWI or truth mask)
    """
    logger.info("Preparing data for equation search...")

    X_all = []
    y_all = []

    for chip_idx, chip in enumerate(chips):
        if chip.shape[0] < 7:
            continue

        vv = chip[0]
        vh = chip[1]
        mndwi = chip[2]
        hand = chip[4]
        slope = chip[5]
        twi = chip[6]

        # Use 8th band as truth if available, otherwise MNDWI
        if chip.shape[0] >= 8:
            truth = chip[7]
        else:
            truth = (mndwi > 0).astype(float)

        h, w = vv.shape
        n_pixels = h * w

        # Sample pixels (stratified by water/non-water if possible)
        water_mask = truth > 0.5
        water_idx = np.where(water_mask.flatten())[0]
        nonwater_idx = np.where(~water_mask.flatten())[0]

        n_water = min(len(water_idx), samples_per_chip // 2)
        n_nonwater = min(len(nonwater_idx), samples_per_chip // 2)

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

        sample_idx = np.concatenate([water_samples, nonwater_samples])

        for idx in sample_idx:
            i, j = idx // w, idx % w

            vv_val = vv[i, j]
            vh_val = vh[i, j]

            # Skip invalid pixels
            if not np.isfinite(vv_val) or not np.isfinite(vh_val):
                continue

            X_all.append(
                [
                    vv_val,  # VV backscatter
                    vh_val,  # VH backscatter
                    vv_val - vh_val,  # VV-VH difference
                    vv_val / (vh_val + 1e-6),  # VV/VH ratio
                    np.abs(vh_val),  # |VH| (magnitude)
                    hand[i, j],  # HAND
                    slope[i, j],  # Slope
                    twi[i, j],  # TWI
                ]
            )

            y_all.append(float(truth[i, j] > 0.5))

        if (chip_idx + 1) % 20 == 0:
            logger.info(
                f"Processed {chip_idx + 1}/{len(chips)} chips, {len(X_all)} samples so far"
            )

    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.float32)

    # Clean data
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]

    logger.info(f"Final dataset: {len(X)} samples, {X.shape[1]} features")
    logger.info(f"Class balance: {np.mean(y):.2%} water")

    return X, y


def run_equation_search(X, y, feature_names):
    """
    Run PySR symbolic regression.
    """
    try:
        from pysr import PySRRegressor
    except ImportError:
        logger.error("PySR not installed! Run: pip install pysr")
        logger.error(
            "Note: PySR requires Julia. Follow instructions at https://astroautomata.com/PySR/"
        )
        return None

    logger.info("=" * 60)
    logger.info("STARTING EQUATION SEARCH")
    logger.info("=" * 60)
    logger.info(f"Feature names: {feature_names}")
    logger.info(f"Sample size: {len(X)}")
    logger.info(f"Max runtime: {CONFIG['max_runtime_hours']} hours")

    # Subsample if too large
    max_samples = 100000
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X = X[idx]
        y = y[idx]
        logger.info(f"Subsampled to {max_samples} samples")

    start_time = time.time()

    # Configure PySR for water detection
    model = PySRRegressor(
        # Search parameters
        niterations=1000,
        populations=30,
        population_size=100,
        # Operators (physics-informed selection)
        binary_operators=[
            "+",
            "-",
            "*",
            "/",
            "greater",  # For threshold-like operations
        ],
        unary_operators=[
            "abs",
            "square",
            "sqrt",
            "neg",
        ],
        # Complexity control
        maxsize=25,
        parsimony=0.005,
        # Constraints
        constraints={
            "square": (1, 1),
            "sqrt": (1, 1),
            "/": (-1, 1),
        },
        nested_constraints={
            "square": {"square": 0, "sqrt": 1},
            "sqrt": {"square": 1, "sqrt": 0},
        },
        # Loss function (for binary classification)
        loss="L2DistLoss()",  # MSE for probability-like output
        # Runtime
        timeout_in_seconds=CONFIG["max_runtime_hours"] * 3600,
        # Parallelism
        procs=8,
        multithreading=True,
        # Output and checkpointing
        progress=True,
        verbosity=1,
        temp_equation_file=True,
        equation_file="water_equations.csv",
        # Reproducibility
        random_state=CONFIG["random_seed"],
        deterministic=False,  # Faster
    )

    # Fit model
    logger.info("Fitting symbolic regression model...")
    model.fit(X, y, variable_names=feature_names)

    elapsed_time = time.time() - start_time
    logger.info(f"Equation search completed in {elapsed_time / 3600:.2f} hours")

    # Extract results
    results = {
        "runtime_seconds": elapsed_time,
        "n_samples": len(X),
        "feature_names": feature_names,
        "equations": [],
    }

    if hasattr(model, "equations_") and model.equations_ is not None:
        for i, eq in enumerate(model.equations_):
            eq_dict = {
                "index": i,
                "equation": str(eq.equation) if hasattr(eq, "equation") else str(eq),
                "complexity": int(eq.complexity) if hasattr(eq, "complexity") else 0,
                "loss": float(eq.loss) if hasattr(eq, "loss") else 0,
                "score": float(eq.score) if hasattr(eq, "score") else 0,
            }
            results["equations"].append(eq_dict)

        # Get best equation
        if hasattr(model, "sympy"):
            results["best_equation"] = str(model.sympy())
        else:
            results["best_equation"] = (
                results["equations"][-1]["equation"] if results["equations"] else "None"
            )

    return results


def validate_equations(equations, X, y, feature_names):
    """
    Validate discovered equations on data.
    """
    from sympy import symbols, lambdify, sympify

    results = []

    # Create symbols
    syms = symbols(feature_names)

    for eq_dict in equations:
        eq_str = eq_dict["equation"]

        try:
            # Parse equation
            expr = sympify(eq_str)
            func = lambdify(syms, expr, modules=["numpy"])

            # Evaluate
            preds = func(*[X[:, i] for i in range(len(feature_names))])
            preds = np.clip(preds, 0, 1)  # Clip to valid probability range

            # Threshold at 0.5
            binary_preds = (preds > 0.5).astype(int)

            # Calculate metrics
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
            )

            eq_dict["validation"] = {
                "accuracy": accuracy_score(y, binary_preds),
                "f1": f1_score(y, binary_preds, zero_division=0),
                "precision": precision_score(y, binary_preds, zero_division=0),
                "recall": recall_score(y, binary_preds, zero_division=0),
            }

            results.append(eq_dict)

        except Exception as e:
            logger.warning(f"Failed to validate equation '{eq_str}': {e}")
            eq_dict["validation"] = {"error": str(e)}
            results.append(eq_dict)

    return results


def main():
    """Main execution."""
    logger.info("=" * 80)
    logger.info("SAR WATER DETECTION - EQUATION SEARCH")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("=" * 80)

    # Create output directory
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

    # Load data
    chips = load_chips(CONFIG["chip_dir"])

    if not chips:
        logger.error("No chips found!")
        return

    # Prepare data
    feature_names = [
        "VV",
        "VH",
        "VV_minus_VH",
        "VV_div_VH",
        "abs_VH",
        "HAND",
        "SLOPE",
        "TWI",
    ]
    X, y = prepare_data(chips, samples_per_chip=2000)

    # Run equation search
    results = run_equation_search(X, y, feature_names)

    if results is None:
        logger.error("Equation search failed!")
        return

    # Validate equations
    if results.get("equations"):
        logger.info("Validating discovered equations...")
        results["equations"] = validate_equations(
            results["equations"], X, y, feature_names
        )

    # Save results
    output_file = CONFIG["output_dir"] / "equation_search_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to: {output_file}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EQUATION SEARCH SUMMARY")
    logger.info("=" * 60)

    if results.get("equations"):
        logger.info(f"Found {len(results['equations'])} equations")
        logger.info(f"\nBest equation: {results.get('best_equation', 'N/A')}")

        # Sort by F1 score
        validated = [
            e
            for e in results["equations"]
            if e.get("validation", {}).get("f1") is not None
        ]
        validated.sort(key=lambda x: x["validation"].get("f1", 0), reverse=True)

        logger.info("\nTop 5 equations by F1 score:")
        for i, eq in enumerate(validated[:5]):
            val = eq["validation"]
            logger.info(f"\n{i + 1}. {eq['equation']}")
            logger.info(f"   Complexity: {eq['complexity']}, Loss: {eq['loss']:.4f}")
            logger.info(
                f"   F1: {val.get('f1', 0):.4f}, Accuracy: {val.get('accuracy', 0):.4f}"
            )
    else:
        logger.warning("No equations found!")

    logger.info("\n" + "=" * 80)
    logger.info(f"Finished: {datetime.now().isoformat()}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
