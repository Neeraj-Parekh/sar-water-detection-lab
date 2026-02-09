#!/usr/bin/env python3
"""
================================================================================
FULL DETECTION PIPELINE v10 - Production-Ready Water Detection
================================================================================

Complete pipeline combining:
1. LightGBM v9 (SAR + MNDWI features)
2. Physics constraints (VETO + soft scoring)
3. Edge case handlers (bright water, urban shadows)
4. Post-processing (morphological cleanup)
5. Difference map generation

Usage:
    python full_pipeline_v10.py --chip_dir /path/to/chips --output_dir /path/to/output

Author: SAR Water Detection Project
Date: 2026-01-25
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from scipy.ndimage import (
    uniform_filter,
    minimum_filter,
    maximum_filter,
    laplace,
    grey_opening,
    grey_closing,
    label as scipy_label,
    binary_dilation,
    binary_erosion,
    generate_binary_structure,
)
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "lgb_model_path": "/home/mit-aoe/sar_water_detection/models/lightgbm_v9_clean_mndwi.txt",
    "output_dir": Path("/home/mit-aoe/sar_water_detection/results/full_pipeline"),
    # Detection thresholds
    "water_threshold": 0.5,
    "min_region_size": 50,
    # Physics constraints
    "hand_veto": 100,  # HAND > this = no water
    "slope_veto": 45,  # Slope > this = no water
    "combined_hand": 30,  # HAND > this AND slope > 20 = no water
    "combined_slope": 20,
    # Bright water handler
    "vh_calm": -18.0,
    "vh_bright_max": -10.0,
    "texture_threshold": 1.5,
    # Urban handler
    "vv_vh_ratio_threshold": 6.0,
    "urban_variance_threshold": 8.0,
}


# =============================================================================
# FEATURE EXTRACTION (must match training)
# =============================================================================


def extract_features(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Extract features for LightGBM prediction."""
    vv = data["vv"]
    vh = data["vh"]
    dem = data["dem"]
    slope = data["slope"]
    hand = data["hand"]
    twi = data["twi"]
    mndwi = data.get("mndwi", None)

    features = []

    # Basic
    features.append(vv)
    features.append(vh)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(vh) > 0.01, vv / vh, 0)
        ratio = np.clip(ratio, -10, 10)
    features.append(ratio)
    features.append(vv - vh)

    denom = vv + vh
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = np.where(np.abs(denom) > 0.01, (vv - vh) / denom, 0)
        rvi = np.where(np.abs(denom) > 0.01, 4 * vh / denom, 0)
    features.append(ndwi)
    features.append(rvi)

    # Multi-scale texture
    for scale in [3, 5, 9, 15, 21]:
        for arr in [vv, vh]:
            arr_mean = uniform_filter(arr, size=scale)
            arr_sq = uniform_filter(arr**2, size=scale)
            arr_var = np.maximum(arr_sq - arr_mean**2, 0)
            arr_std = np.sqrt(arr_var)
            arr_min = minimum_filter(arr, size=scale)
            arr_max = maximum_filter(arr, size=scale)
            features.extend([arr_mean, arr_std, arr_min, arr_max])

    # Gradients
    gy_vv, gx_vv = np.gradient(vv)
    gy_vh, gx_vh = np.gradient(vh)
    features.append(np.sqrt(gx_vv**2 + gy_vv**2))
    features.append(np.sqrt(gx_vh**2 + gy_vh**2))
    features.append(np.abs(laplace(vv)))
    features.append(np.abs(laplace(vh)))

    # Morphological
    features.extend(
        [
            grey_opening(vv, size=5),
            grey_closing(vv, size=5),
            grey_opening(vh, size=5),
            grey_closing(vh, size=5),
        ]
    )

    # Otsu-like
    vv_med = np.median(vv)
    vh_med = np.median(vh)
    features.append(vv - vv_med)
    features.append(vh - vh_med)

    # Local contrast
    features.append(vv - uniform_filter(vv, size=9))
    features.append(vh - uniform_filter(vh, size=9))

    # GLCM-like
    for arr in [vv, vh]:
        arr_mean = uniform_filter(arr, size=5)
        arr_sq = uniform_filter(arr**2, size=5)
        arr_var = np.maximum(arr_sq - arr_mean**2, 0)
        contrast = np.sqrt(arr_var)
        arr_range = maximum_filter(arr, size=5) - minimum_filter(arr, size=5)
        homogeneity = 1.0 / (1.0 + arr_range)
        features.extend([contrast, homogeneity])

    # Pseudo-entropy
    vv_norm = vv - vv.min()
    vv_range = vv.max() - vv.min() + 1e-10
    vv_prob = np.clip(vv_norm / vv_range, 1e-10, 1 - 1e-10)
    entropy = -vv_prob * np.log2(vv_prob) - (1 - vv_prob) * np.log2(1 - vv_prob)
    entropy = np.nan_to_num(entropy, nan=0.0)
    features.append(entropy)

    # DEM features
    features.extend([dem, slope, hand, twi])

    # Physics scores
    hand_exp = np.clip((hand - 10) / 3.0, -50, 50)
    slope_exp = np.clip((slope - 8) / 3.0, -50, 50)
    twi_exp = np.clip((8 - twi) / 2.0, -50, 50)
    features.append(1.0 / (1.0 + np.exp(hand_exp)))
    features.append(1.0 / (1.0 + np.exp(slope_exp)))
    features.append(1.0 / (1.0 + np.exp(twi_exp)))

    # MNDWI features
    if mndwi is not None:
        features.append(mndwi)
        features.append((mndwi > 0).astype(np.float32))
        mndwi_mean = uniform_filter(mndwi, size=5)
        mndwi_sq = uniform_filter(mndwi**2, size=5)
        mndwi_var = np.maximum(mndwi_sq - mndwi_mean**2, 0)
        features.append(mndwi_mean)
        features.append(np.sqrt(mndwi_var))
    else:
        h, w = vv.shape
        features.extend([np.zeros((h, w), dtype=np.float32)] * 4)

    feature_stack = np.stack(features, axis=-1)
    feature_stack = np.nan_to_num(feature_stack, nan=0.0, posinf=0.0, neginf=0.0)
    return feature_stack.astype(np.float32)


# =============================================================================
# PHYSICS CONSTRAINTS
# =============================================================================


def compute_physics(
    hand: np.ndarray, slope: np.ndarray, twi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute physics score and VETO mask."""
    # Hard VETO
    veto = np.zeros_like(hand, dtype=bool)
    veto |= hand > CONFIG["hand_veto"]
    veto |= slope > CONFIG["slope_veto"]
    veto |= (hand > CONFIG["combined_hand"]) & (slope > CONFIG["combined_slope"])

    # Soft score
    hand_exp = np.clip((hand - 15) / 5.0, -50, 50)
    hand_score = 1.0 / (1.0 + np.exp(hand_exp))

    slope_exp = np.clip((slope - 12) / 4.0, -50, 50)
    slope_score = 1.0 / (1.0 + np.exp(slope_exp))

    twi_exp = np.clip((7 - twi) / 2.0, -50, 50)
    twi_score = 1.0 / (1.0 + np.exp(twi_exp))

    physics_score = 0.4 * hand_score + 0.4 * slope_score + 0.2 * twi_score

    return physics_score.astype(np.float32), veto


# =============================================================================
# BRIGHT WATER HANDLER
# =============================================================================


def apply_bright_water_correction(
    water_mask: np.ndarray,
    vh: np.ndarray,
    lgb_proba: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Adaptive region growing for wind-roughened water."""
    # Compute texture
    vh_mean = uniform_filter(vh.astype(np.float64), size=5)
    vh_sq_mean = uniform_filter(vh.astype(np.float64) ** 2, size=5)
    vh_variance = np.maximum(vh_sq_mean - vh_mean**2, 0)

    # Seeds: high confidence water
    seeds = (water_mask > 0.5) & (lgb_proba > 0.7)

    # Ambiguous: bright but smooth
    ambiguous = (
        (vh >= CONFIG["vh_calm"])
        & (vh < CONFIG["vh_bright_max"])
        & (vh_variance < CONFIG["texture_threshold"])
        & (~water_mask.astype(bool))
    )

    # Region growing
    current_water = seeds.copy()
    struct = generate_binary_structure(2, 1)
    pixels_added = 0

    for _ in range(10):
        dilated = binary_dilation(current_water, structure=struct)
        adjacent = dilated & ambiguous & (~current_water)

        if not adjacent.any():
            break

        current_water = current_water | adjacent
        ambiguous = ambiguous & (~adjacent)
        pixels_added += int(adjacent.sum())

    corrected = water_mask.astype(bool) | current_water
    return corrected.astype(np.float32), pixels_added


# =============================================================================
# URBAN SHADOW HANDLER
# =============================================================================


def apply_urban_mask(
    water_mask: np.ndarray,
    vv: np.ndarray,
    vh: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Remove urban shadow false positives."""
    # VV - VH difference (in dB, high = double bounce)
    vv_vh_diff = vv - vh

    # Compute variance
    vv_mean = uniform_filter(vv.astype(np.float64), size=7)
    vv_sq_mean = uniform_filter(vv.astype(np.float64) ** 2, size=7)
    vv_variance = np.maximum(vv_sq_mean - vv_mean**2, 0)

    vh_mean = uniform_filter(vh.astype(np.float64), size=7)
    vh_sq_mean = uniform_filter(vh.astype(np.float64) ** 2, size=7)
    vh_variance = np.maximum(vh_sq_mean - vh_mean**2, 0)

    combined_variance = (vv_variance + vh_variance) / 2

    # Bright neighbors
    vv_max_local = maximum_filter(vv, size=9)
    has_bright = vv_max_local > -8.0

    # Urban shadow detection
    urban = (
        (vv_vh_diff > CONFIG["vv_vh_ratio_threshold"])
        | (combined_variance > CONFIG["urban_variance_threshold"])
    ) & has_bright

    # Remove small clusters
    labeled, num = scipy_label(urban)
    for i in range(1, num + 1):
        if (labeled == i).sum() < 100:
            urban = urban & (labeled != i)

    # Dilate
    urban = binary_dilation(urban, iterations=2)

    # Remove from water
    corrected = water_mask.astype(bool) & (~urban)
    pixels_removed = int((water_mask > 0.5).sum() - corrected.sum())

    return corrected.astype(np.float32), pixels_removed


# =============================================================================
# MORPHOLOGICAL CLEANUP
# =============================================================================


def morphological_cleanup(mask: np.ndarray, min_size: int = 50) -> np.ndarray:
    """Remove small regions and fill holes."""
    # Remove small regions
    labeled, num = scipy_label(mask > 0.5)
    cleaned = np.zeros_like(mask, dtype=bool)

    for i in range(1, num + 1):
        region = labeled == i
        if region.sum() >= min_size:
            cleaned |= region

    # Fill small holes
    filled = binary_dilation(cleaned, iterations=1)
    filled = binary_erosion(filled, iterations=1)

    return filled.astype(np.float32)


# =============================================================================
# DIFFERENCE MAP
# =============================================================================


def generate_difference_map(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    output_path: Path,
) -> Dict:
    """Generate RGB difference map."""
    pred = prediction > 0.5
    truth = ground_truth > 0.5

    tp = pred & truth
    fp = pred & (~truth)
    fn = (~pred) & truth
    tn = (~pred) & (~truth)

    h, w = pred.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    rgb[tn] = [200, 200, 200]  # Gray
    rgb[tp] = [0, 255, 0]  # Green
    rgb[fp] = [255, 0, 0]  # Red
    rgb[fn] = [0, 0, 255]  # Blue

    try:
        from PIL import Image

        img = Image.fromarray(rgb)
        img.save(output_path)
    except ImportError:
        np.save(str(output_path).replace(".png", ".npy"), rgb)

    stats = {
        "tp": int(tp.sum()),
        "fp": int(fp.sum()),
        "fn": int(fn.sum()),
        "tn": int(tn.sum()),
    }

    if stats["tp"] + stats["fn"] > 0:
        stats["recall"] = stats["tp"] / (stats["tp"] + stats["fn"])
    else:
        stats["recall"] = 0.0

    if stats["tp"] + stats["fp"] > 0:
        stats["precision"] = stats["tp"] / (stats["tp"] + stats["fp"])
    else:
        stats["precision"] = 0.0

    if stats["tp"] + stats["fp"] + stats["fn"] > 0:
        stats["iou"] = stats["tp"] / (stats["tp"] + stats["fp"] + stats["fn"])
    else:
        stats["iou"] = 1.0

    return stats


# =============================================================================
# FULL PIPELINE
# =============================================================================


class FullDetectionPipeline:
    """Complete water detection pipeline."""

    def __init__(self, lgb_model_path: str = None):
        self.lgb_model = None

        if lgb_model_path is None:
            lgb_model_path = CONFIG["lgb_model_path"]

        try:
            import lightgbm as lgb

            self.lgb_model = lgb.Booster(model_file=lgb_model_path)
            logger.info(f"Loaded LightGBM model: {lgb_model_path}")
        except Exception as e:
            logger.error(f"Failed to load LightGBM: {e}")

    def detect(
        self, data: Dict[str, np.ndarray], name: str = "chip"
    ) -> Tuple[np.ndarray, Dict]:
        """
        Full detection pipeline.

        Args:
            data: Dict with vv, vh, dem, slope, hand, twi, optional mndwi
            name: Chip name for logging

        Returns:
            final_mask: Binary water mask
            stats: Detailed statistics
        """
        stats = {"name": name, "steps": {}}

        # Validate input
        for key in ["vv", "vh", "dem", "slope", "hand", "twi"]:
            if key not in data:
                raise ValueError(f"Missing: {key}")
            data[key] = np.nan_to_num(data[key].astype(np.float32))

        data["slope"] = np.clip(data["slope"], 0, 90)
        data["hand"] = np.clip(data["hand"], 0, 500)
        data["twi"] = np.clip(data["twi"], 0, 30)
        if "mndwi" in data:
            data["mndwi"] = np.clip(np.nan_to_num(data["mndwi"]), -1, 1)

        # Step 1: LightGBM prediction
        features = extract_features(data)
        h, w, n = features.shape
        X = features.reshape(-1, n)
        lgb_proba = self.lgb_model.predict(X).reshape(h, w).astype(np.float32)
        stats["steps"]["lgb"] = {"mean_proba": float(lgb_proba.mean())}

        # Step 2: Physics
        physics_score, veto = compute_physics(data["hand"], data["slope"], data["twi"])
        stats["steps"]["physics"] = {"veto_fraction": float(veto.mean())}

        # Step 3: Combine LGB + Physics
        combined = 0.9 * lgb_proba + 0.1 * physics_score
        combined = np.where(veto, 0.0, combined)

        # Step 4: Initial threshold
        water_mask = (combined > CONFIG["water_threshold"]).astype(np.float32)
        stats["steps"]["initial"] = {"water_fraction": float(water_mask.mean())}

        # Step 5: Bright water correction
        water_mask, bright_added = apply_bright_water_correction(
            water_mask, data["vh"], lgb_proba
        )
        stats["steps"]["bright_water"] = {"pixels_added": bright_added}

        # Step 6: Urban shadow removal
        water_mask, urban_removed = apply_urban_mask(water_mask, data["vv"], data["vh"])
        stats["steps"]["urban"] = {"pixels_removed": urban_removed}

        # Step 7: Morphological cleanup
        water_mask = morphological_cleanup(water_mask, CONFIG["min_region_size"])
        stats["final_water_fraction"] = float(water_mask.mean())

        return water_mask, stats


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Full Water Detection Pipeline")
    parser.add_argument(
        "--chip_dir",
        type=str,
        default="/home/mit-aoe/sar_water_detection/chips_expanded_npy",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/mit-aoe/sar_water_detection/results/full_pipeline",
    )
    parser.add_argument("--max_chips", type=int, default=None)
    args = parser.parse_args()

    chip_dir = Path(args.chip_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "difference_maps").mkdir(exist_ok=True)

    logger.info("=" * 70)
    logger.info("FULL DETECTION PIPELINE v10")
    logger.info("=" * 70)

    # Initialize pipeline
    pipeline = FullDetectionPipeline()

    # Load chips
    chip_files = sorted(chip_dir.glob("*_with_truth.npy"))
    if args.max_chips:
        chip_files = chip_files[: args.max_chips]

    logger.info(f"Processing {len(chip_files)} chips...")

    all_results = []
    all_preds = []
    all_labels = []

    for chip_path in chip_files:
        name = chip_path.stem.replace("_with_truth", "")

        try:
            data_raw = np.load(chip_path)
            data = {
                "vv": data_raw[:, :, 0],
                "vh": data_raw[:, :, 1],
                "dem": data_raw[:, :, 2],
                "slope": data_raw[:, :, 3],
                "hand": data_raw[:, :, 4],
                "twi": data_raw[:, :, 5],
            }
            if data_raw.shape[2] > 7:
                data["mndwi"] = data_raw[:, :, 7]

            label = (data_raw[:, :, 6] > 0).astype(np.float32)

            # Detect
            pred, stats = pipeline.detect(data, name)

            # Generate difference map
            diff_stats = generate_difference_map(
                pred, label, output_dir / "difference_maps" / f"{name}_diff.png"
            )
            stats["metrics"] = diff_stats

            all_results.append(stats)
            all_preds.append(pred.flatten())
            all_labels.append(label.flatten())

            logger.info(
                f"  {name}: IoU={diff_stats['iou']:.4f}, "
                f"P={diff_stats['precision']:.4f}, R={diff_stats['recall']:.4f}"
            )

        except Exception as e:
            logger.error(f"  {name}: ERROR - {e}")
            import traceback

            traceback.print_exc()

    # Overall metrics
    if all_preds:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        pred_bin = all_preds > 0.5
        label_bin = all_labels > 0.5

        tp = (pred_bin & label_bin).sum()
        fp = (pred_bin & ~label_bin).sum()
        fn = (~pred_bin & label_bin).sum()

        overall = {
            "iou": float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0,
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        }

        logger.info("\n" + "=" * 70)
        logger.info("OVERALL RESULTS")
        logger.info("=" * 70)
        logger.info(f"IoU:       {overall['iou']:.4f}")
        logger.info(f"Precision: {overall['precision']:.4f}")
        logger.info(f"Recall:    {overall['recall']:.4f}")

        # Save results
        output = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()
            },
            "overall_metrics": overall,
            "per_chip_results": all_results,
        }

        results_path = output_dir / "full_pipeline_results.json"
        with open(results_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"\nResults saved to {results_path}")
        logger.info(f"Difference maps saved to {output_dir / 'difference_maps'}")

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
