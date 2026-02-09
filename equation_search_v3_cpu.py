#!/usr/bin/env python3
"""
Equation Search v3 - CPU Version
=================================
Pure NumPy implementation for reliable execution.

Author: SAR Water Detection Lab
Date: 2026-01-25
"""

import os
import sys
import json
import time
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from itertools import product

import numpy as np
from scipy.ndimage import uniform_filter

warnings.filterwarnings("ignore")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("equation_search_v3.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "chip_dir": Path("/home/mit-aoe/sar_water_detection/chips"),
    "output_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    "max_equations_per_template": 500,
    "checkpoint_interval": 200,
}


# =============================================================================
# Terrain Profiles
# =============================================================================

TERRAIN_PROFILES = {
    "flat_lowland": {"hand_max": 15.0, "slope_max": 10.0, "twi_min": 7.0},
    "hilly": {"hand_max": 25.0, "slope_max": 20.0, "twi_min": 6.0},
    "mountainous": {"hand_max": 100.0, "slope_max": 30.0, "twi_min": 4.0},
    "arid": {"hand_max": 8.0, "slope_max": 5.0, "twi_min": 5.0},
    "urban": {"hand_max": 8.0, "slope_max": 10.0, "twi_min": 6.0},
    "wetland": {"hand_max": 20.0, "slope_max": 8.0, "twi_min": 10.0},
}


# =============================================================================
# Equation Templates
# =============================================================================

EQUATION_TEMPLATES = {
    # Basic thresholds
    "vh_simple": {
        "template": "(vh < T_vh)",
        "params": {"T_vh": np.arange(-26, -12, 2.0)},
    },
    # HAND-constrained
    "hand_constrained": {
        "template": "(vh < T_vh) & (hand < T_hand)",
        "params": {"T_vh": np.arange(-24, -14, 2.0), "T_hand": np.arange(5, 25, 5.0)},
    },
    # Triple-lock
    "triple_lock": {
        "template": "(vh < T_vh) & (hand < T_hand) & (slope < T_slope)",
        "params": {
            "T_vh": np.arange(-22, -14, 2.0),
            "T_hand": np.arange(5, 20, 5.0),
            "T_slope": np.arange(5, 20, 5.0),
        },
    },
    # Bright water (relaxed VH, strict physics)
    "bright_water": {
        "template": "(vh < T_vh) & (hand < T_hand) & (slope < T_slope)",
        "params": {
            "T_vh": np.arange(-14, -8, 2.0),
            "T_hand": np.arange(2, 8, 2.0),
            "T_slope": np.arange(1, 5, 1.0),
        },
    },
    # TWI-based
    "twi_based": {
        "template": "(vh < T_vh) & (twi > T_twi)",
        "params": {"T_vh": np.arange(-22, -14, 2.0), "T_twi": np.arange(6, 14, 2.0)},
    },
    # Full physics
    "full_physics": {
        "template": "(vh < T_vh) & (hand < T_hand) & (slope < T_slope) & (twi > T_twi)",
        "params": {
            "T_vh": np.arange(-22, -14, 4.0),
            "T_hand": np.arange(5, 20, 5.0),
            "T_slope": np.arange(5, 15, 5.0),
            "T_twi": np.arange(5, 12, 3.0),
        },
    },
    # Urban exclusion
    "urban_exclusion": {
        "template": "(vh < T_vh) & (hand < T_hand) & ~((vv > T_vv_urban) & ((vv - vh) > T_ratio))",
        "params": {
            "T_vh": np.arange(-22, -14, 2.0),
            "T_hand": np.arange(5, 15, 5.0),
            "T_vv_urban": np.arange(-12, -6, 2.0),
            "T_ratio": np.arange(6, 12, 2.0),
        },
    },
    # Hysteresis
    "hysteresis": {
        "template": "(vh < T_vh_low) | ((vh < T_vh_high) & (hand < T_hand))",
        "params": {
            "T_vh_low": np.arange(-24, -18, 2.0),
            "T_vh_high": np.arange(-18, -12, 2.0),
            "T_hand": np.arange(5, 15, 5.0),
        },
    },
}


# =============================================================================
# Chip Loading
# =============================================================================


def load_chip(chip_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load chip and extract features."""
    try:
        data = np.load(chip_path)

        # Handle different formats
        if len(data.shape) == 3:
            if data.shape[0] < data.shape[2]:
                data = np.transpose(data, (1, 2, 0))

        n_bands = data.shape[2] if len(data.shape) == 3 else 1

        features = {}

        # Extract standard bands
        if n_bands >= 2:
            features["vv"] = np.nan_to_num(data[:, :, 0], nan=-20).astype(np.float32)
            features["vh"] = np.nan_to_num(data[:, :, 1], nan=-25).astype(np.float32)
        else:
            return None

        # Optional bands
        if n_bands >= 4:
            features["dem"] = np.nan_to_num(data[:, :, 3], nan=100).astype(np.float32)
        else:
            features["dem"] = np.zeros_like(features["vv"])

        if n_bands >= 5:
            features["hand"] = np.nan_to_num(data[:, :, 4], nan=50).astype(np.float32)
        else:
            features["hand"] = np.zeros_like(features["vv"])

        if n_bands >= 6:
            features["slope"] = np.nan_to_num(data[:, :, 5], nan=15).astype(np.float32)
        else:
            features["slope"] = np.zeros_like(features["vv"])

        if n_bands >= 7:
            features["twi"] = np.nan_to_num(data[:, :, 6], nan=5).astype(np.float32)
        else:
            features["twi"] = np.zeros_like(features["vv"])

        # Truth band
        features["truth"] = None
        if n_bands == 8:
            features["truth"] = data[:, :, 7]
        elif n_bands == 7:
            last_band = data[:, :, 6]
            unique = np.unique(last_band[~np.isnan(last_band)])
            if len(unique) <= 3 and np.all(unique <= 1) and np.all(unique >= 0):
                features["truth"] = last_band

        return features

    except Exception as e:
        logger.debug(f"Failed to load {chip_path}: {e}")
        return None


def classify_terrain(features: Dict[str, np.ndarray], chip_name: str) -> str:
    """Classify terrain type."""
    chip_lower = chip_name.lower()

    # Name-based
    if any(x in chip_lower for x in ["urban", "mumbai", "delhi", "bangalore"]):
        return "urban"
    if any(x in chip_lower for x in ["wetland", "kolleru", "loktak", "chilika"]):
        return "wetland"
    if any(x in chip_lower for x in ["arid", "rann", "thar", "sambhar"]):
        return "arid"
    if any(x in chip_lower for x in ["mountain", "pangong", "bhakra"]):
        return "mountainous"

    # Feature-based
    dem = features.get("dem", np.zeros((1,)))
    slope = features.get("slope", np.zeros((1,)))

    if np.nanmean(dem) > 2000:
        return "mountainous"
    if np.nanmean(slope) > 12:
        return "hilly"

    return "flat_lowland"


# =============================================================================
# Equation Evaluation
# =============================================================================


def evaluate_equation(
    template_name: str,
    template: str,
    params: Dict[str, float],
    features: Dict[str, np.ndarray],
) -> Optional[Dict]:
    """Evaluate single equation."""
    try:
        # Build local variables
        local_vars = {
            "vv": features["vv"],
            "vh": features["vh"],
            "hand": features["hand"],
            "slope": features["slope"],
            "twi": features["twi"],
            "dem": features["dem"],
        }

        # Substitute parameters
        eq = template
        for k, v in params.items():
            eq = eq.replace(k, str(v))

        # Evaluate
        pred = eval(eq, {"__builtins__": {}}, local_vars)
        pred = np.asarray(pred).astype(bool)

        # Get truth
        truth = features.get("truth")
        if truth is None:
            return None

        truth_bool = truth > 0.5

        # Compute metrics
        tp = np.sum(pred & truth_bool)
        fp = np.sum(pred & ~truth_bool)
        fn = np.sum(~pred & truth_bool)

        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        intersection = np.sum(pred & truth_bool)
        union = np.sum(pred | truth_bool)

        if union == 0:
            iou = 0.0
        else:
            iou = intersection / union

        # Physics compliance
        profile = TERRAIN_PROFILES.get("flat_lowland")
        hand_violation = np.sum(pred & (features["hand"] > profile["hand_max"]))
        slope_violation = np.sum(pred & (features["slope"] > profile["slope_max"]))

        total_pred = np.sum(pred)
        if total_pred > 0:
            physics_score = (
                1.0
                - 0.5 * (hand_violation / total_pred)
                - 0.5 * (slope_violation / total_pred)
            )
        else:
            physics_score = 1.0

        return {
            "template": template_name,
            "equation": eq,
            "params": params,
            "iou": float(iou),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "physics_score": float(max(0, physics_score)),
            "combined_score": float(iou * max(0, physics_score)),
            "water_fraction_pred": float(pred.sum() / pred.size),
            "water_fraction_true": float(truth_bool.sum() / truth_bool.size),
        }

    except Exception as e:
        logger.debug(f"Eval failed: {e}")
        return None


# =============================================================================
# Main Search
# =============================================================================


def run_search():
    """Run equation search."""
    logger.info("=" * 60)
    logger.info("EQUATION SEARCH v3 (CPU)")
    logger.info("=" * 60)

    # Load chips
    chip_dir = CONFIG["chip_dir"]
    chip_files = list(chip_dir.glob("*.npy"))
    logger.info(f"Found {len(chip_files)} chip files")

    chips = []
    for cf in chip_files:
        features = load_chip(cf)
        if features and features.get("truth") is not None:
            terrain = classify_terrain(features, cf.stem)
            chips.append({"name": cf.stem, "features": features, "terrain": terrain})
            logger.info(f"  Loaded {cf.stem} - terrain: {terrain}")

    logger.info(f"\nLoaded {len(chips)} chips with ground truth")

    if not chips:
        logger.error("No valid chips!")
        return

    # Run search
    all_results = {}
    total_evals = 0
    start_time = time.time()

    for template_name, template_config in EQUATION_TEMPLATES.items():
        template = template_config["template"]
        param_ranges = template_config["params"]

        # Generate combinations
        keys = list(param_ranges.keys())
        values = [param_ranges[k] for k in keys]
        combinations = list(product(*values))

        # Limit
        max_combos = CONFIG["max_equations_per_template"]
        if len(combinations) > max_combos:
            indices = np.random.choice(len(combinations), max_combos, replace=False)
            combinations = [combinations[i] for i in indices]

        logger.info(f"\n{template_name}: {len(combinations)} combinations")

        template_results = []

        for combo in combinations:
            params = dict(zip(keys, combo))

            # Evaluate on all chips
            chip_metrics = []

            for chip in chips:
                result = evaluate_equation(
                    template_name, template, params, chip["features"]
                )
                if result:
                    chip_metrics.append(result)

            if chip_metrics:
                mean_iou = np.mean([m["iou"] for m in chip_metrics])
                mean_physics = np.mean([m["physics_score"] for m in chip_metrics])

                if mean_iou > 0.1:
                    template_results.append(
                        {
                            "template": template_name,
                            "params": {k: float(v) for k, v in params.items()},
                            "mean_iou": float(mean_iou),
                            "std_iou": float(np.std([m["iou"] for m in chip_metrics])),
                            "mean_physics": float(mean_physics),
                            "combined_score": float(mean_iou * mean_physics),
                            "n_chips": len(chip_metrics),
                        }
                    )

            total_evals += 1

            if total_evals % 50 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"  Evaluated {total_evals} equations ({total_evals / elapsed:.1f}/sec)"
                )

        # Sort by combined score
        template_results.sort(key=lambda x: x["combined_score"], reverse=True)
        all_results[template_name] = template_results[:50]

        if template_results:
            best = template_results[0]
            logger.info(
                f"  Best: IoU={best['mean_iou']:.4f}, Physics={best['mean_physics']:.4f}"
            )
            logger.info(f"  Params: {best['params']}")

    # Save results
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

    # Per-template
    for template_name, results in all_results.items():
        if results:
            output_path = CONFIG["output_dir"] / f"equations_{template_name}.json"
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    # Summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_chips": len(chips),
        "total_evaluations": total_evals,
        "runtime_seconds": time.time() - start_time,
        "top_10": [],
    }

    # Collect all best
    all_best = []
    for template_name, results in all_results.items():
        if results:
            all_best.append(results[0])

    all_best.sort(key=lambda x: x["combined_score"], reverse=True)
    summary["top_10"] = all_best[:10]

    summary_path = CONFIG["output_dir"] / "equation_search_v3_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TOP 10 EQUATIONS")
    logger.info("=" * 60)

    for i, eq in enumerate(all_best[:10]):
        logger.info(
            f"{i + 1}. {eq['template']}: IoU={eq['mean_iou']:.4f}, Physics={eq['mean_physics']:.4f}"
        )
        logger.info(f"   Params: {eq['params']}")

    logger.info(f"\nCompleted in {time.time() - start_time:.1f} seconds")
    logger.info(f"Results saved to {summary_path}")


if __name__ == "__main__":
    run_search()
