#!/usr/bin/env python3
"""
GPU-Accelerated Equation Search v3 - CLEAN DATA ONLY
=====================================================
Uses chips_expanded_npy/ (validated, clean data)

Goal: Find a raw mathematical formula (e.g., `Water = VH * Slope^2`)
that runs without heavy AI and is interpretable by scientists.

Key Features:
- Full GPU utilization with batch processing
- Uses ONLY clean chips_expanded_npy/ data
- Physics-aware equation templates
- Includes MNDWI if available

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
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from itertools import product
import numpy as np

warnings.filterwarnings("ignore")

# GPU imports
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage

    GPU_AVAILABLE = True
    print(
        f"CuPy available. GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}"
    )
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("CuPy not available, using CPU mode")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            "/home/mit-aoe/sar_water_detection/results/equation_search_v3.log"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # ONLY clean data
    "chip_dir": Path("/home/mit-aoe/sar_water_detection/chips_expanded_npy"),
    "output_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    "batch_size": 100000,  # Pixels per GPU batch
    "max_equations_per_template": 1000,  # Grid search limit per template
    "top_k": 20,  # Keep top K equations per template
}


# =============================================================================
# Data Loading
# =============================================================================


@dataclass
class ChipData:
    """Container for chip data."""

    name: str
    vv: np.ndarray
    vh: np.ndarray
    dem: np.ndarray
    slope: np.ndarray
    hand: np.ndarray
    twi: np.ndarray
    mndwi: Optional[np.ndarray]
    label: np.ndarray


def load_clean_chips() -> List[ChipData]:
    """Load all chips from clean directory."""
    chips = []
    chip_files = sorted(CONFIG["chip_dir"].glob("*_with_truth.npy"))

    logger.info(f"Found {len(chip_files)} chips in {CONFIG['chip_dir']}")

    for chip_path in chip_files:
        try:
            data = np.load(chip_path)
            n_channels = data.shape[2]

            chip = ChipData(
                name=chip_path.stem.replace("_with_truth", ""),
                vv=data[:, :, 0].astype(np.float32),
                vh=data[:, :, 1].astype(np.float32),
                dem=data[:, :, 2].astype(np.float32),
                slope=np.clip(data[:, :, 3].astype(np.float32), 0, 90),
                hand=np.clip(data[:, :, 4].astype(np.float32), 0, 500),
                twi=np.clip(data[:, :, 5].astype(np.float32), 0, 30),
                mndwi=np.clip(data[:, :, 7].astype(np.float32), -1, 1)
                if n_channels > 7
                else None,
                label=(data[:, :, 6] > 0).astype(np.float32),
            )

            # Fix NaN
            chip.vv = np.nan_to_num(chip.vv, nan=-20.0)
            chip.vh = np.nan_to_num(chip.vh, nan=-25.0)
            chip.hand = np.nan_to_num(chip.hand, nan=100.0)
            chip.twi = np.nan_to_num(chip.twi, nan=5.0)
            if chip.mndwi is not None:
                chip.mndwi = np.nan_to_num(chip.mndwi, nan=0.0)

            chips.append(chip)

        except Exception as e:
            logger.warning(f"Failed to load {chip_path}: {e}")

    logger.info(f"Loaded {len(chips)} chips successfully")
    return chips


def flatten_data(chips: List[ChipData]) -> Dict[str, np.ndarray]:
    """Flatten all chips into 1D arrays."""
    data = {
        "vv": [],
        "vh": [],
        "dem": [],
        "slope": [],
        "hand": [],
        "twi": [],
        "mndwi": [],
        "label": [],
    }

    has_mndwi = all(c.mndwi is not None for c in chips)

    for chip in chips:
        data["vv"].append(chip.vv.flatten())
        data["vh"].append(chip.vh.flatten())
        data["dem"].append(chip.dem.flatten())
        data["slope"].append(chip.slope.flatten())
        data["hand"].append(chip.hand.flatten())
        data["twi"].append(chip.twi.flatten())
        if has_mndwi:
            data["mndwi"].append(chip.mndwi.flatten())
        data["label"].append(chip.label.flatten())

    result = {}
    for key in data:
        if data[key]:
            result[key] = np.concatenate(data[key])

    logger.info(f"Flattened data: {len(result['label'])} pixels")
    logger.info(
        f"Water pixels: {result['label'].sum():.0f} ({result['label'].mean() * 100:.1f}%)"
    )

    return result


# =============================================================================
# Metrics
# =============================================================================


def compute_iou_gpu(pred: cp.ndarray, target: cp.ndarray) -> float:
    """Compute IoU on GPU."""
    pred_bin = pred > 0.5
    target_bin = target > 0.5

    intersection = cp.logical_and(pred_bin, target_bin).sum()
    union = cp.logical_or(pred_bin, target_bin).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


def compute_precision_recall_gpu(
    pred: cp.ndarray, target: cp.ndarray
) -> Tuple[float, float]:
    """Compute precision and recall on GPU."""
    pred_bin = pred > 0.5
    target_bin = target > 0.5

    tp = cp.logical_and(pred_bin, target_bin).sum()
    fp = cp.logical_and(pred_bin, ~target_bin).sum()
    fn = cp.logical_and(~pred_bin, target_bin).sum()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    return precision, recall


# =============================================================================
# Equation Templates
# =============================================================================

EQUATION_TEMPLATES = [
    # Basic threshold equations
    {
        "name": "VH_threshold",
        "formula": "1 / (1 + exp((vh - T) / S))",
        "params": {"T": np.linspace(-25, -12, 20), "S": np.linspace(0.5, 5, 10)},
        "func": lambda vh, vv, dem, slope, hand, twi, mndwi, T, S: 1.0
        / (1.0 + cp.exp(cp.clip((vh - T) / S, -50, 50))),
    },
    {
        "name": "VH_with_physics",
        "formula": "sigmoid(vh - T) * hand_score * slope_score",
        "params": {
            "T": np.linspace(-22, -15, 10),
            "S": np.linspace(1, 4, 5),
            "H": np.linspace(10, 30, 5),  # HAND threshold
            "SL": np.linspace(8, 20, 5),  # Slope threshold
        },
        "func": lambda vh, vv, dem, slope, hand, twi, mndwi, T, S, H, SL: (
            1.0 / (1.0 + cp.exp(cp.clip((vh - T) / S, -50, 50)))
        )
        * (1.0 / (1.0 + cp.exp(cp.clip((hand - H) / 5, -50, 50))))
        * (1.0 / (1.0 + cp.exp(cp.clip((slope - SL) / 5, -50, 50)))),
    },
    {
        "name": "VH_min_local",
        "formula": "VH_min < T where HAND < H and SLOPE < SL",
        "params": {
            "T": np.linspace(-22, -14, 15),
            "H": np.linspace(15, 50, 8),
            "SL": np.linspace(10, 30, 8),
        },
        "func": lambda vh, vv, dem, slope, hand, twi, mndwi, T, H, SL: (
            (vh < T) & (hand < H) & (slope < SL)
        ).astype(cp.float32),
    },
    {
        "name": "VV_VH_ratio",
        "formula": "sigmoid(VV/VH - R) * physics",
        "params": {
            "R": np.linspace(0.5, 2.0, 10),
            "H": np.linspace(15, 40, 5),
        },
        "func": lambda vh, vv, dem, slope, hand, twi, mndwi, R, H: (
            1.0
            / (
                1.0
                + cp.exp(
                    cp.clip(
                        (cp.where(cp.abs(vh) > 0.01, vv / vh, 0) - R) / 0.3, -50, 50
                    )
                )
            )
        )
        * (1.0 / (1.0 + cp.exp(cp.clip((hand - H) / 5, -50, 50)))),
    },
    {
        "name": "TWI_based",
        "formula": "VH < T AND TWI > TW AND HAND < H",
        "params": {
            "T": np.linspace(-22, -14, 10),
            "TW": np.linspace(6, 12, 6),
            "H": np.linspace(15, 40, 6),
        },
        "func": lambda vh, vv, dem, slope, hand, twi, mndwi, T, TW, H: (
            (vh < T) & (twi > TW) & (hand < H)
        ).astype(cp.float32),
    },
]

# MNDWI templates (only if MNDWI available)
MNDWI_TEMPLATES = [
    {
        "name": "MNDWI_only",
        "formula": "MNDWI > M",
        "params": {"M": np.linspace(-0.2, 0.3, 20)},
        "func": lambda vh, vv, dem, slope, hand, twi, mndwi, M: (mndwi > M).astype(
            cp.float32
        )
        if mndwi is not None
        else cp.zeros_like(vh),
    },
    {
        "name": "MNDWI_with_VH",
        "formula": "(MNDWI > M) OR (VH < T AND MNDWI > M2)",
        "params": {
            "M": np.linspace(0.0, 0.3, 8),
            "T": np.linspace(-22, -16, 6),
            "M2": np.linspace(-0.3, 0.0, 6),
        },
        "func": lambda vh, vv, dem, slope, hand, twi, mndwi, M, T, M2: (
            (mndwi > M) | ((vh < T) & (mndwi > M2))
        ).astype(cp.float32)
        if mndwi is not None
        else cp.zeros_like(vh),
    },
    {
        "name": "MNDWI_SAR_fusion",
        "formula": "0.6 * sigmoid(MNDWI - M) + 0.4 * sigmoid(T - VH)",
        "params": {
            "M": np.linspace(-0.1, 0.2, 10),
            "T": np.linspace(-20, -14, 10),
        },
        "func": lambda vh, vv, dem, slope, hand, twi, mndwi, M, T: (
            0.6 * (1.0 / (1.0 + cp.exp(cp.clip((M - mndwi) / 0.1, -50, 50))))
            + 0.4 * (1.0 / (1.0 + cp.exp(cp.clip((vh - T) / 2, -50, 50))))
        )
        if mndwi is not None
        else cp.zeros_like(vh),
    },
]


# =============================================================================
# Search Engine
# =============================================================================


@dataclass
class EquationResult:
    """Result of equation evaluation."""

    template_name: str
    formula: str
    params: Dict[str, float]
    iou: float
    precision: float
    recall: float
    f1: float


def evaluate_equation(
    template: Dict,
    params: Dict[str, float],
    data_gpu: Dict[str, cp.ndarray],
) -> Optional[EquationResult]:
    """Evaluate a single equation on GPU."""
    try:
        # Call the equation function
        pred = template["func"](
            data_gpu["vh"],
            data_gpu["vv"],
            data_gpu["dem"],
            data_gpu["slope"],
            data_gpu["hand"],
            data_gpu["twi"],
            data_gpu.get("mndwi"),
            **params,
        )

        # Compute metrics
        iou = compute_iou_gpu(pred, data_gpu["label"])
        precision, recall = compute_precision_recall_gpu(pred, data_gpu["label"])
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return EquationResult(
            template_name=template["name"],
            formula=template["formula"],
            params={k: float(v) for k, v in params.items()},
            iou=iou,
            precision=precision,
            recall=recall,
            f1=f1,
        )

    except Exception as e:
        return None


def grid_search_template(
    template: Dict,
    data_gpu: Dict[str, cp.ndarray],
    max_evals: int = 1000,
) -> List[EquationResult]:
    """Grid search over template parameters."""
    results = []

    # Generate parameter grid
    param_names = list(template["params"].keys())
    param_values = [template["params"][k] for k in param_names]

    # Limit combinations
    grid = list(product(*param_values))
    if len(grid) > max_evals:
        indices = np.random.choice(len(grid), max_evals, replace=False)
        grid = [grid[i] for i in indices]

    logger.info(
        f"  Evaluating {len(grid)} parameter combinations for {template['name']}"
    )

    for param_combo in grid:
        params = dict(zip(param_names, param_combo))
        result = evaluate_equation(template, params, data_gpu)
        if result is not None:
            results.append(result)

    # Sort by IoU
    results.sort(key=lambda x: x.iou, reverse=True)

    return results


def run_search():
    """Main search function."""
    logger.info("=" * 70)
    logger.info("GPU EQUATION SEARCH v3 - CLEAN DATA ONLY")
    logger.info("=" * 70)

    # Load data
    logger.info("Loading clean chips...")
    chips = load_clean_chips()

    if len(chips) == 0:
        logger.error("No chips loaded!")
        return

    # Flatten
    data = flatten_data(chips)

    # Check MNDWI availability
    has_mndwi = "mndwi" in data and len(data["mndwi"]) > 0
    logger.info(f"MNDWI available: {has_mndwi}")

    # Transfer to GPU
    logger.info("Transferring data to GPU...")
    data_gpu = {}
    for key, arr in data.items():
        data_gpu[key] = cp.asarray(arr)

    # Memory info
    if GPU_AVAILABLE:
        mempool = cp.get_default_memory_pool()
        logger.info(f"GPU memory used: {mempool.used_bytes() / 1e9:.2f} GB")

    # Run search
    all_results = []
    templates = EQUATION_TEMPLATES.copy()
    if has_mndwi:
        templates.extend(MNDWI_TEMPLATES)

    logger.info(f"Searching {len(templates)} equation templates...")

    for i, template in enumerate(templates):
        logger.info(f"\n[{i + 1}/{len(templates)}] Template: {template['name']}")
        logger.info(f"  Formula: {template['formula']}")

        results = grid_search_template(
            template, data_gpu, max_evals=CONFIG["max_equations_per_template"]
        )

        if results:
            best = results[0]
            logger.info(
                f"  Best: IoU={best.iou:.4f}, P={best.precision:.4f}, R={best.recall:.4f}"
            )
            logger.info(f"  Params: {best.params}")
            all_results.extend(results[: CONFIG["top_k"]])

    # Sort all results
    all_results.sort(key=lambda x: x.iou, reverse=True)

    # Save results
    logger.info("\n" + "=" * 70)
    logger.info("TOP 20 EQUATIONS")
    logger.info("=" * 70)

    for i, r in enumerate(all_results[:20]):
        logger.info(f"\n{i + 1}. {r.template_name}")
        logger.info(f"   Formula: {r.formula}")
        logger.info(f"   Params: {r.params}")
        logger.info(
            f"   IoU: {r.iou:.4f}, P: {r.precision:.4f}, R: {r.recall:.4f}, F1: {r.f1:.4f}"
        )

    # Save to JSON
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_chips": len(chips),
        "n_pixels": len(data["label"]),
        "has_mndwi": has_mndwi,
        "top_equations": [asdict(r) for r in all_results[:50]],
    }

    output_path = CONFIG["output_dir"] / "equation_search_v3_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")

    # Best equation summary
    if all_results:
        best = all_results[0]
        logger.info("\n" + "=" * 70)
        logger.info("BEST EQUATION FOUND")
        logger.info("=" * 70)
        logger.info(f"Template: {best.template_name}")
        logger.info(f"Formula: {best.formula}")
        logger.info(f"Parameters: {best.params}")
        logger.info(f"IoU: {best.iou:.4f}")
        logger.info(f"Precision: {best.precision:.4f}")
        logger.info(f"Recall: {best.recall:.4f}")
        logger.info(f"F1: {best.f1:.4f}")

        # Compare with LightGBM v9
        logger.info(f"\nComparison with LightGBM v9:")
        logger.info(f"  LightGBM v9 IoU: 0.807")
        logger.info(f"  Best Equation IoU: {best.iou:.3f}")
        logger.info(
            f"  Gap: {(0.807 - best.iou) * 100:.1f}% (equation is simpler but less accurate)"
        )

    logger.info("\n" + "=" * 70)
    logger.info("EQUATION SEARCH COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_search()
