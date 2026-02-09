#!/usr/bin/env python3
"""
================================================================================
PER-CLASS BENCHMARK - Rivers vs Lakes IoU Analysis
================================================================================

This script evaluates model performance separately for:
1. Rivers (thin, elongated structures)
2. Lakes (compact, blob-like structures)

This helps identify if the model is struggling with rivers (topology) or lakes
(area coverage) and guides loss function selection.

Classification is based on morphological analysis:
- Shape factor (perimeter^2 / area) - Rivers have high values
- Aspect ratio of bounding box
- Solidity (area / convex_hull_area)

Author: SAR Water Detection Project
Date: 2026-01-26
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import ndimage
from scipy.ndimage import label, find_objects
from skimage.measure import regionprops
from skimage.morphology import skeletonize

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "chip_dir": Path("/home/mit-aoe/sar_water_detection/chips_expanded_npy"),
    "results_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    "val_split": 0.2,
    "random_seed": 42,
    # Classification thresholds
    "river_shape_factor_min": 50,  # Rivers have high perimeter^2/area
    "river_solidity_max": 0.4,  # Rivers are not compact
    "lake_shape_factor_max": 30,  # Lakes are more compact
    "lake_solidity_min": 0.6,  # Lakes fill their convex hull
    # Minimum region size (pixels) to analyze
    "min_region_size": 100,
}


# =============================================================================
# WATER BODY CLASSIFICATION
# =============================================================================


def classify_water_bodies(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Classify water pixels into rivers vs lakes based on morphology.

    Returns:
        river_mask: Binary mask of river pixels
        lake_mask: Binary mask of lake pixels
        stats: Classification statistics
    """
    # Label connected components
    labeled, n_components = label(mask > 0.5)

    if n_components == 0:
        return (
            np.zeros_like(mask),
            np.zeros_like(mask),
            {"n_rivers": 0, "n_lakes": 0, "n_ambiguous": 0},
        )

    river_mask = np.zeros_like(mask, dtype=bool)
    lake_mask = np.zeros_like(mask, dtype=bool)

    n_rivers = 0
    n_lakes = 0
    n_ambiguous = 0

    props = regionprops(labeled)

    for region in props:
        if region.area < CONFIG["min_region_size"]:
            continue

        # Compute morphological features
        perimeter = region.perimeter
        area = region.area

        # Shape factor: circularity inverse (higher = more elongated)
        shape_factor = (perimeter**2) / (4 * np.pi * area + 1e-8)

        # Solidity: how much the shape fills its convex hull
        solidity = region.solidity

        # Eccentricity: how elongated (1 = line, 0 = circle)
        eccentricity = region.eccentricity

        # Aspect ratio of bounding box
        bbox = region.bbox
        bbox_height = bbox[2] - bbox[0]
        bbox_width = bbox[3] - bbox[1]
        aspect_ratio = max(bbox_height, bbox_width) / (
            min(bbox_height, bbox_width) + 1e-8
        )

        # Skeleton ratio (skeleton length / area) - high for rivers
        region_mask = labeled == region.label
        try:
            skeleton = skeletonize(region_mask)
            skeleton_length = np.sum(skeleton)
            skeleton_ratio = skeleton_length / (area + 1e-8)
        except:
            skeleton_ratio = 0

        # Classification logic
        is_river = (
            (shape_factor > CONFIG["river_shape_factor_min"])
            or (solidity < CONFIG["river_solidity_max"])
            or (eccentricity > 0.9)
            or (skeleton_ratio > 0.05)
            or (aspect_ratio > 5)
        )

        is_lake = (
            (shape_factor < CONFIG["lake_shape_factor_max"])
            and (solidity > CONFIG["lake_solidity_min"])
            and (eccentricity < 0.7)
            and (aspect_ratio < 3)
        )

        if is_river and not is_lake:
            river_mask[region_mask] = True
            n_rivers += 1
        elif is_lake and not is_river:
            lake_mask[region_mask] = True
            n_lakes += 1
        else:
            # Ambiguous - use skeleton ratio as tiebreaker
            if skeleton_ratio > 0.03:
                river_mask[region_mask] = True
                n_rivers += 1
            else:
                lake_mask[region_mask] = True
                n_lakes += 1
            n_ambiguous += 1

    stats = {
        "n_rivers": n_rivers,
        "n_lakes": n_lakes,
        "n_ambiguous": n_ambiguous,
        "river_pixels": int(np.sum(river_mask)),
        "lake_pixels": int(np.sum(lake_mask)),
    }

    return river_mask.astype(np.float32), lake_mask.astype(np.float32), stats


# =============================================================================
# METRICS
# =============================================================================


def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute IoU between prediction and target."""
    pred_binary = pred > 0.5
    target_binary = target > 0.5

    intersection = np.sum(pred_binary & target_binary)
    union = np.sum(pred_binary | target_binary)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


def compute_class_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    target_river_mask: np.ndarray,
    target_lake_mask: np.ndarray,
) -> Dict[str, float]:
    """Compute per-class metrics."""
    # Overall IoU
    overall_iou = compute_iou(pred, target)

    # River IoU
    if np.sum(target_river_mask) > 0:
        river_pred = pred * target_river_mask  # Only evaluate river regions
        river_target = target * target_river_mask
        river_iou = compute_iou(river_pred, river_target)
    else:
        river_iou = None

    # Lake IoU
    if np.sum(target_lake_mask) > 0:
        lake_pred = pred * target_lake_mask
        lake_target = target * target_lake_mask
        lake_iou = compute_iou(lake_pred, lake_target)
    else:
        lake_iou = None

    # Topology metrics (for rivers)
    if np.sum(target_river_mask) > 0:
        # Skeleton recall: how much of the true skeleton is captured
        try:
            true_skeleton = skeletonize(target_river_mask > 0.5)
            pred_binary = pred > 0.5
            skeleton_recall = np.sum(pred_binary[true_skeleton]) / (
                np.sum(true_skeleton) + 1e-8
            )
        except:
            skeleton_recall = None
    else:
        skeleton_recall = None

    return {
        "overall_iou": overall_iou,
        "river_iou": river_iou,
        "lake_iou": lake_iou,
        "skeleton_recall": skeleton_recall,
    }


# =============================================================================
# MODEL LOADING
# =============================================================================


def load_model(model_path: Path, model_type: str):
    """Load a trained model."""
    if model_type == "lightgbm":
        import lightgbm as lgb

        return lgb.Booster(model_file=str(model_path))

    elif model_type == "pytorch":
        import torch

        try:
            from train_with_cldice_v10 import AttentionUNetV10
        except ImportError:
            from attention_unet_v9_sota import AttentionUNet as AttentionUNetV10

        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(model_path, map_location=device)

        config = checkpoint.get("config", {})
        model = AttentionUNetV10(
            in_channels=config.get("in_channels", 9),
            out_channels=1,
            base_filters=config.get("base_filters", 32),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model, device

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def predict_chip(model, chip: np.ndarray, model_type: str) -> np.ndarray:
    """Generate prediction for a chip."""
    from scipy.ndimage import uniform_filter
    from skimage.filters import frangi

    # Extract features
    vv = chip[:, :, 0]
    vh = chip[:, :, 1]
    dem = chip[:, :, 2]
    slope = chip[:, :, 3]
    hand = chip[:, :, 4]
    twi = chip[:, :, 5]
    mndwi = chip[:, :, 7] if chip.shape[2] > 7 else np.zeros_like(vv)

    vh_texture = uniform_filter(vh**2, size=5) - uniform_filter(vh, size=5) ** 2
    vh_texture = np.sqrt(np.maximum(vh_texture, 0))

    vh_norm = (vh - vh.min()) / (vh.max() - vh.min() + 1e-8)
    try:
        frangi_feat = frangi(1.0 - vh_norm, sigmas=[1, 2, 3], black_ridges=False)
        frangi_feat = frangi_feat / (frangi_feat.max() + 1e-8)
    except:
        frangi_feat = np.zeros_like(vh)

    def normalize(x, vmin=None, vmax=None):
        if vmin is None:
            vmin = np.nanpercentile(x, 1)
        if vmax is None:
            vmax = np.nanpercentile(x, 99)
        return np.clip((x - vmin) / (vmax - vmin + 1e-8), 0, 1)

    features = np.stack(
        [
            normalize(vv, -30, 0),
            normalize(vh, -35, -5),
            normalize(dem, 0, 2000),
            normalize(slope, 0, 45),
            normalize(hand, 0, 100),
            normalize(twi, 0, 20),
            normalize(mndwi, -1, 1),
            normalize(vh_texture),
            frangi_feat,
        ],
        axis=-1,
    )
    features = np.nan_to_num(features, nan=0.0)

    if model_type == "lightgbm":
        h, w = features.shape[:2]
        X = features.reshape(-1, features.shape[-1])
        pred = model.predict(X).reshape(h, w)

    elif model_type == "pytorch":
        import torch

        model_obj, device = model
        x = (
            torch.from_numpy(features.transpose(2, 0, 1))
            .unsqueeze(0)
            .float()
            .to(device)
        )
        with torch.no_grad():
            pred = torch.sigmoid(model_obj(x)).squeeze().cpu().numpy()

    return pred


# =============================================================================
# MAIN BENCHMARK
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Per-class benchmark (Rivers vs Lakes)"
    )
    parser.add_argument("--model_path", type=str, help="Path to model file")
    parser.add_argument(
        "--model_type", type=str, default="lightgbm", choices=["pytorch", "lightgbm"]
    )
    parser.add_argument("--chip_dir", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PER-CLASS BENCHMARK: RIVERS vs LAKES")
    logger.info("=" * 60)

    # Setup paths
    chip_dir = Path(args.chip_dir) if args.chip_dir else CONFIG["chip_dir"]

    if not chip_dir.exists():
        logger.error(f"Chip directory not found: {chip_dir}")
        return

    # Load chips
    all_chips = sorted(chip_dir.glob("*.npy"))
    np.random.seed(CONFIG["random_seed"])
    np.random.shuffle(all_chips)

    n_val = max(1, int(len(all_chips) * CONFIG["val_split"]))
    val_chips = all_chips[:n_val]

    if args.n_samples:
        val_chips = val_chips[: args.n_samples]

    logger.info(f"Evaluating on {len(val_chips)} validation chips")

    # Load model
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        if args.model_type == "lightgbm":
            model_path = Path(
                "/home/mit-aoe/sar_water_detection/models/lightgbm_v9.txt"
            )
        else:
            model_path = Path(
                "/home/mit-aoe/sar_water_detection/models/attention_unet_v10_cldice_best.pth"
            )

    if not model_path.exists():
        logger.warning(f"Model not found: {model_path}")
        logger.info("Running analysis without model predictions (ground truth only)")
        model = None
    else:
        logger.info(f"Loading model: {model_path}")
        model = load_model(model_path, args.model_type)

    # Evaluate chips
    all_metrics = []
    river_ious = []
    lake_ious = []
    skeleton_recalls = []
    total_river_pixels = 0
    total_lake_pixels = 0

    for i, chip_path in enumerate(val_chips):
        chip = np.load(chip_path)
        target = chip[:, :, 6]

        # Classify water bodies
        river_mask, lake_mask, class_stats = classify_water_bodies(target)
        total_river_pixels += class_stats["river_pixels"]
        total_lake_pixels += class_stats["lake_pixels"]

        if model is not None:
            # Generate prediction
            pred = predict_chip(model, chip, args.model_type)

            # Compute metrics
            metrics = compute_class_metrics(pred, target, river_mask, lake_mask)
            all_metrics.append(metrics)

            if metrics["river_iou"] is not None:
                river_ious.append(metrics["river_iou"])
            if metrics["lake_iou"] is not None:
                lake_ious.append(metrics["lake_iou"])
            if metrics["skeleton_recall"] is not None:
                skeleton_recalls.append(metrics["skeleton_recall"])

        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(val_chips)} chips")

    # Aggregate results
    results = {
        "model_path": str(model_path) if model is not None else None,
        "model_type": args.model_type,
        "n_chips": len(val_chips),
        "total_river_pixels": total_river_pixels,
        "total_lake_pixels": total_lake_pixels,
        "river_lake_ratio": total_river_pixels / (total_lake_pixels + 1e-8),
    }

    if model is not None:
        results["overall_iou"] = np.mean([m["overall_iou"] for m in all_metrics])
        results["river_iou"] = np.mean(river_ious) if river_ious else None
        results["lake_iou"] = np.mean(lake_ious) if lake_ious else None
        results["skeleton_recall"] = (
            np.mean(skeleton_recalls) if skeleton_recalls else None
        )
        results["n_chips_with_rivers"] = len(river_ious)
        results["n_chips_with_lakes"] = len(lake_ious)

    # Print results
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Total river pixels: {total_river_pixels:,}")
    logger.info(f"  Total lake pixels:  {total_lake_pixels:,}")
    logger.info(f"  River/Lake ratio:   {results['river_lake_ratio']:.2f}")

    if model is not None:
        logger.info(f"\nOverall Performance:")
        logger.info(f"  Overall IoU: {results['overall_iou']:.4f}")

        logger.info(f"\nPer-Class Performance:")
        if results["river_iou"] is not None:
            logger.info(
                f"  River IoU:       {results['river_iou']:.4f} ({len(river_ious)} chips)"
            )
        else:
            logger.info(f"  River IoU:       N/A (no rivers in validation set)")

        if results["lake_iou"] is not None:
            logger.info(
                f"  Lake IoU:        {results['lake_iou']:.4f} ({len(lake_ious)} chips)"
            )
        else:
            logger.info(f"  Lake IoU:        N/A (no lakes in validation set)")

        if results["skeleton_recall"] is not None:
            logger.info(f"  Skeleton Recall: {results['skeleton_recall']:.4f}")

        # Analysis
        logger.info(f"\nAnalysis:")
        if results["river_iou"] is not None and results["lake_iou"] is not None:
            gap = results["lake_iou"] - results["river_iou"]
            if gap > 0.05:
                logger.info(f"  Model struggles with RIVERS (gap: {gap:.4f})")
                logger.info(
                    f"  Recommendation: Use clDice + FocalTversky loss for river topology"
                )
            elif gap < -0.05:
                logger.info(f"  Model struggles with LAKES (gap: {abs(gap):.4f})")
                logger.info(f"  Recommendation: Use standard Dice + BCE loss")
            else:
                logger.info(f"  Model performs similarly on rivers and lakes")

    # Save results
    CONFIG["results_dir"].mkdir(parents=True, exist_ok=True)
    results_path = (
        CONFIG["results_dir"]
        / f"per_class_benchmark_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {results_path}")

    # Summary table
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 60)
    if model is not None:
        logger.info(f"{'Metric':<25} {'Value':<15} {'Note':<30}")
        logger.info("-" * 70)
        logger.info(f"{'Overall IoU':<25} {results['overall_iou']:.4f}")
        if results["river_iou"] is not None:
            logger.info(f"{'River IoU':<25} {results['river_iou']:.4f}")
        if results["lake_iou"] is not None:
            logger.info(f"{'Lake IoU':<25} {results['lake_iou']:.4f}")
        if results["skeleton_recall"] is not None:
            logger.info(
                f"{'Skeleton Recall':<25} {results['skeleton_recall']:.4f}{'<-- Critical for rivers':<30}"
            )


if __name__ == "__main__":
    main()
