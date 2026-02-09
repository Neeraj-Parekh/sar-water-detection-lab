#!/usr/bin/env python3
"""
================================================================================
EDGE CASE HANDLERS - Bright Water, Urban Shadows, and Visualization
================================================================================

Implements three specialized handlers for edge cases:

1. Bright Water Handler (Wind Correction)
   - Adaptive region growing from high-confidence water
   - Uses texture to identify wind-roughened water

2. Urban Mask Handler (Shadow Removal)
   - VV/VH ratio check for urban double-bounce
   - Local variance check for building shadows

3. Difference Map Generator
   - Green: Correct Water (TP)
   - Red: False Positive (FP)
   - Blue: False Negative (FN)
   - White: Correct Land (TN)

Author: SAR Water Detection Project
Date: 2026-01-25
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from scipy.ndimage import (
    uniform_filter,
    minimum_filter,
    maximum_filter,
    binary_dilation,
    binary_erosion,
    label as scipy_label,
    generate_binary_structure,
)
from scipy import ndimage
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. BRIGHT WATER HANDLER (Wind Correction)
# =============================================================================


class BrightWaterHandler:
    """
    Handles wind-roughened water that appears brighter than calm water.

    Algorithm:
    1. Start with high-confidence water pixels
    2. Look at neighbors in a 3x3 window
    3. If neighbor is "ambiguous" (bright) but has low texture -> water
    4. Repeat until convergence

    Physics:
    - Wind creates capillary waves (Bragg scattering)
    - Increases backscatter by 3-6 dB
    - But texture remains LOW (water is still smoother than land)
    """

    def __init__(
        self,
        vh_calm_threshold: float = -18.0,  # Calm water VH
        vh_bright_min: float = -14.0,  # Brightest possible water
        vh_bright_max: float = -10.0,  # Beyond this is definitely land
        texture_threshold: float = 1.5,  # Max variance for water
        max_iterations: int = 10,  # Region growing iterations
    ):
        self.vh_calm_threshold = vh_calm_threshold
        self.vh_bright_min = vh_bright_min
        self.vh_bright_max = vh_bright_max
        self.texture_threshold = texture_threshold
        self.max_iterations = max_iterations

    def compute_local_variance(
        self, arr: np.ndarray, window_size: int = 5
    ) -> np.ndarray:
        """Compute local variance (texture measure)."""
        arr_mean = uniform_filter(arr.astype(np.float64), size=window_size)
        arr_sq_mean = uniform_filter(arr.astype(np.float64) ** 2, size=window_size)
        variance = np.maximum(arr_sq_mean - arr_mean**2, 0)
        return variance.astype(np.float32)

    def apply(
        self,
        water_mask: np.ndarray,
        vh: np.ndarray,
        confidence: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply bright water correction via adaptive region growing.

        Args:
            water_mask: Initial binary water mask
            vh: VH backscatter in dB
            confidence: Optional confidence scores (0-1)

        Returns:
            corrected_mask: Water mask with bright water included
            stats: Statistics about corrections made
        """
        # Compute texture
        vh_texture = self.compute_local_variance(vh, window_size=5)

        # Identify high-confidence water seeds
        if confidence is not None:
            seeds = (water_mask > 0.5) & (confidence > 0.7)
        else:
            # Use calm water as seeds
            seeds = (water_mask > 0.5) & (vh < self.vh_calm_threshold)

        # Identify ambiguous bright pixels (potential windy water)
        ambiguous = (
            (vh >= self.vh_calm_threshold)  # Brighter than calm water
            & (vh < self.vh_bright_max)  # But not too bright (land)
            & (vh_texture < self.texture_threshold)  # Low texture (smooth)
            & (~water_mask.astype(bool))  # Not already classified as water
        )

        # Region growing
        current_water = seeds.copy()
        struct = generate_binary_structure(2, 1)  # 4-connectivity

        stats = {
            "initial_water_pixels": int(seeds.sum()),
            "ambiguous_pixels": int(ambiguous.sum()),
            "iterations": 0,
            "pixels_added": 0,
        }

        for iteration in range(self.max_iterations):
            # Dilate current water
            dilated = binary_dilation(current_water, structure=struct)

            # Find ambiguous pixels adjacent to current water
            adjacent_ambiguous = dilated & ambiguous & (~current_water)

            if not adjacent_ambiguous.any():
                break

            # Add these pixels to water
            current_water = current_water | adjacent_ambiguous
            ambiguous = ambiguous & (~adjacent_ambiguous)  # Remove from ambiguous

            stats["iterations"] = iteration + 1
            stats["pixels_added"] += int(adjacent_ambiguous.sum())

        stats["final_water_pixels"] = int(current_water.sum())

        # Combine with original mask
        corrected = water_mask.astype(bool) | current_water

        return corrected.astype(np.float32), stats


# =============================================================================
# 2. URBAN MASK HANDLER (Shadow Removal)
# =============================================================================


class UrbanMaskHandler:
    """
    Removes false positive water detections in urban areas.

    Urban characteristics:
    1. High VV/VH ratio (double-bounce from buildings)
    2. High local variance (heterogeneous texture)
    3. Bright pixels adjacent to dark (building + shadow)

    Water characteristics:
    1. Low VV/VH ratio (similar backscatter in both)
    2. Low local variance (homogeneous)
    3. Consistently dark across larger areas
    """

    def __init__(
        self,
        vv_vh_ratio_threshold: float = 6.0,  # Urban has high ratio
        variance_threshold: float = 8.0,  # Urban has high variance
        bright_neighbor_threshold: float = -8.0,  # Bright building nearby
        min_urban_size: int = 100,  # Min urban cluster size
    ):
        self.vv_vh_ratio_threshold = vv_vh_ratio_threshold
        self.variance_threshold = variance_threshold
        self.bright_neighbor_threshold = bright_neighbor_threshold
        self.min_urban_size = min_urban_size

    def compute_local_variance(
        self, arr: np.ndarray, window_size: int = 7
    ) -> np.ndarray:
        """Compute local variance."""
        arr_mean = uniform_filter(arr.astype(np.float64), size=window_size)
        arr_sq_mean = uniform_filter(arr.astype(np.float64) ** 2, size=window_size)
        variance = np.maximum(arr_sq_mean - arr_mean**2, 0)
        return variance.astype(np.float32)

    def apply(
        self,
        water_mask: np.ndarray,
        vv: np.ndarray,
        vh: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Remove urban false positives from water mask.

        Args:
            water_mask: Binary water mask
            vv: VV backscatter in dB
            vh: VH backscatter in dB

        Returns:
            corrected_mask: Water mask with urban shadows removed
            stats: Statistics about corrections made
        """
        # Compute VV/VH ratio (handle division carefully)
        with np.errstate(divide="ignore", invalid="ignore"):
            # In dB, ratio is subtraction
            vv_vh_ratio_db = vv - vh  # High positive = urban double-bounce
            vv_vh_ratio_db = np.nan_to_num(vv_vh_ratio_db, nan=0.0)

        # Compute local variance
        vv_variance = self.compute_local_variance(vv, window_size=7)
        vh_variance = self.compute_local_variance(vh, window_size=7)
        combined_variance = (vv_variance + vh_variance) / 2

        # Check for bright neighbors (buildings)
        vv_max_local = maximum_filter(vv, size=9)
        has_bright_neighbor = vv_max_local > self.bright_neighbor_threshold

        # Urban shadow detection logic
        # A pixel is likely an urban shadow if:
        # 1. High VV-VH difference (double-bounce signature nearby)
        # 2. OR high local variance (heterogeneous urban texture)
        # 3. AND has bright neighbor (building causing shadow)

        urban_shadow = (
            (vv_vh_ratio_db > self.vv_vh_ratio_threshold)
            | (combined_variance > self.variance_threshold)
        ) & has_bright_neighbor

        # Remove small urban clusters (noise)
        if self.min_urban_size > 0:
            labeled, num_features = scipy_label(urban_shadow)
            for i in range(1, num_features + 1):
                region = labeled == i
                if region.sum() < self.min_urban_size:
                    urban_shadow = urban_shadow & (~region)

        # Dilate urban mask slightly to catch edges
        urban_shadow = binary_dilation(urban_shadow, iterations=2)

        stats = {
            "urban_shadow_pixels": int(urban_shadow.sum()),
            "water_before": int((water_mask > 0.5).sum()),
        }

        # Remove urban shadows from water mask
        corrected = water_mask.astype(bool) & (~urban_shadow)

        stats["water_after"] = int(corrected.sum())
        stats["pixels_removed"] = stats["water_before"] - stats["water_after"]

        return corrected.astype(np.float32), stats


# =============================================================================
# 3. DIFFERENCE MAP GENERATOR
# =============================================================================


class DifferenceMapGenerator:
    """
    Generates visualization maps showing prediction vs ground truth.

    Color coding:
    - Green: True Positive (correct water)
    - Red: False Positive (incorrectly predicted water)
    - Blue: False Negative (missed water)
    - White/Gray: True Negative (correct land)
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("./difference_maps")
        self.output_dir.mkdir(exist_ok=True)

    def generate(
        self,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        name: str,
        save: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate a difference map.

        Args:
            prediction: Binary prediction mask
            ground_truth: Binary ground truth mask
            name: Name for the output file
            save: Whether to save the image

        Returns:
            rgb_map: RGB image (H, W, 3)
            stats: Statistics
        """
        pred = prediction > 0.5
        truth = ground_truth > 0.5

        # Compute confusion matrix components
        tp = pred & truth  # True Positive (Green)
        fp = pred & (~truth)  # False Positive (Red)
        fn = (~pred) & truth  # False Negative (Blue)
        tn = (~pred) & (~truth)  # True Negative (Gray)

        # Create RGB image
        h, w = pred.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Assign colors
        rgb[tn] = [200, 200, 200]  # Gray - correct land
        rgb[tp] = [0, 255, 0]  # Green - correct water
        rgb[fp] = [255, 0, 0]  # Red - false positive
        rgb[fn] = [0, 0, 255]  # Blue - false negative

        # Compute statistics
        stats = {
            "true_positive": int(tp.sum()),
            "false_positive": int(fp.sum()),
            "false_negative": int(fn.sum()),
            "true_negative": int(tn.sum()),
        }

        total_positives = stats["true_positive"] + stats["false_negative"]
        total_negatives = stats["true_negative"] + stats["false_positive"]

        if total_positives > 0:
            stats["recall"] = stats["true_positive"] / total_positives
        else:
            stats["recall"] = 0.0

        if stats["true_positive"] + stats["false_positive"] > 0:
            stats["precision"] = stats["true_positive"] / (
                stats["true_positive"] + stats["false_positive"]
            )
        else:
            stats["precision"] = 0.0

        if (
            stats["true_positive"] + stats["false_positive"] + stats["false_negative"]
            > 0
        ):
            stats["iou"] = stats["true_positive"] / (
                stats["true_positive"]
                + stats["false_positive"]
                + stats["false_negative"]
            )
        else:
            stats["iou"] = 1.0 if stats["true_negative"] > 0 else 0.0

        # Save image
        if save:
            try:
                from PIL import Image

                img = Image.fromarray(rgb)
                img_path = self.output_dir / f"{name}_diff_map.png"
                img.save(img_path)
                stats["saved_to"] = str(img_path)
                logger.info(f"Saved difference map: {img_path}")
            except ImportError:
                # Fallback to numpy save
                np_path = self.output_dir / f"{name}_diff_map.npy"
                np.save(np_path, rgb)
                stats["saved_to"] = str(np_path)
                logger.warning(f"PIL not available, saved as numpy: {np_path}")

        return rgb, stats

    def generate_batch(
        self,
        predictions: List[np.ndarray],
        ground_truths: List[np.ndarray],
        names: List[str],
    ) -> List[Dict]:
        """Generate difference maps for a batch of predictions."""
        all_stats = []
        for pred, truth, name in zip(predictions, ground_truths, names):
            _, stats = self.generate(pred, truth, name)
            stats["name"] = name
            all_stats.append(stats)
        return all_stats


# =============================================================================
# 4. COMBINED POST-PROCESSOR
# =============================================================================


class WaterPostProcessor:
    """
    Combined post-processing pipeline:
    1. Apply bright water correction
    2. Apply urban shadow removal
    3. Generate difference maps
    """

    def __init__(
        self,
        bright_water_handler: Optional[BrightWaterHandler] = None,
        urban_mask_handler: Optional[UrbanMaskHandler] = None,
        diff_map_generator: Optional[DifferenceMapGenerator] = None,
    ):
        self.bright_water = bright_water_handler or BrightWaterHandler()
        self.urban_mask = urban_mask_handler or UrbanMaskHandler()
        self.diff_map = diff_map_generator or DifferenceMapGenerator()

    def process(
        self,
        water_mask: np.ndarray,
        vv: np.ndarray,
        vh: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        name: str = "chip",
        confidence: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply full post-processing pipeline.

        Args:
            water_mask: Initial water prediction
            vv: VV backscatter
            vh: VH backscatter
            ground_truth: Optional ground truth for evaluation
            name: Chip name for saving
            confidence: Optional confidence scores

        Returns:
            final_mask: Post-processed water mask
            stats: Detailed statistics
        """
        all_stats = {
            "original_water_fraction": float((water_mask > 0.5).mean()),
        }

        # Step 1: Bright water correction
        mask_after_bright, bright_stats = self.bright_water.apply(
            water_mask, vh, confidence
        )
        all_stats["bright_water"] = bright_stats
        all_stats["after_bright_fraction"] = float((mask_after_bright > 0.5).mean())

        # Step 2: Urban shadow removal
        final_mask, urban_stats = self.urban_mask.apply(mask_after_bright, vv, vh)
        all_stats["urban_shadow"] = urban_stats
        all_stats["final_water_fraction"] = float((final_mask > 0.5).mean())

        # Step 3: Difference map (if ground truth available)
        if ground_truth is not None:
            # Generate for original
            _, orig_stats = self.diff_map.generate(
                water_mask, ground_truth, f"{name}_original", save=True
            )
            all_stats["original_metrics"] = orig_stats

            # Generate for final
            _, final_stats = self.diff_map.generate(
                final_mask, ground_truth, f"{name}_corrected", save=True
            )
            all_stats["corrected_metrics"] = final_stats

            # Improvement
            all_stats["iou_improvement"] = final_stats["iou"] - orig_stats["iou"]

        return final_mask, all_stats


# =============================================================================
# TEST FUNCTION
# =============================================================================


def test_on_chips():
    """Test edge case handlers on chip data."""
    logger.info("=" * 70)
    logger.info("EDGE CASE HANDLER TEST")
    logger.info("=" * 70)

    chip_dir = Path("/home/mit-aoe/sar_water_detection/chips_expanded_npy")
    output_dir = Path("/home/mit-aoe/sar_water_detection/results/difference_maps")
    output_dir.mkdir(exist_ok=True)

    # Initialize
    processor = WaterPostProcessor(
        diff_map_generator=DifferenceMapGenerator(output_dir)
    )

    # Load a sample of chips
    chip_files = sorted(chip_dir.glob("*_with_truth.npy"))[:10]

    # Load LightGBM predictions (simulate with threshold)
    results = []

    for chip_path in chip_files:
        name = chip_path.stem.replace("_with_truth", "")
        logger.info(f"Processing {name}...")

        try:
            data = np.load(chip_path)
            vv = data[:, :, 0].astype(np.float32)
            vh = data[:, :, 1].astype(np.float32)
            label = (data[:, :, 6] > 0).astype(np.float32)

            # Simple threshold prediction (simulate model output)
            # In practice, use actual model predictions
            initial_mask = (vh < -16).astype(np.float32)

            # Apply post-processing
            final_mask, stats = processor.process(
                water_mask=initial_mask,
                vv=vv,
                vh=vh,
                ground_truth=label,
                name=name,
            )

            stats["chip"] = name
            results.append(stats)

            if "iou_improvement" in stats:
                logger.info(
                    f"  Original IoU: {stats['original_metrics']['iou']:.4f}, "
                    f"Corrected IoU: {stats['corrected_metrics']['iou']:.4f}, "
                    f"Improvement: {stats['iou_improvement']:+.4f}"
                )

        except Exception as e:
            logger.error(f"  Error: {e}")

    # Save results
    results_path = output_dir / "edge_case_test_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to {results_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    test_on_chips()
