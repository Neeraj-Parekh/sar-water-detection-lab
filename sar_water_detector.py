#!/usr/bin/env python3
"""
================================================================================
SAR WATER DETECTOR - Production Package
================================================================================

Physics-Guided SAR Water Detection for India
Model: LightGBM v9 (IoU: 0.882)

USAGE:
    from sar_water_detector import SARWaterDetector

    detector = SARWaterDetector()
    result = detector.predict(chip)  # chip: (H, W, 8) numpy array

    # Access results
    water_mask = result['water_mask']        # Binary mask
    water_proba = result['water_probability'] # Probability map
    confidence = result['confidence']         # Per-pixel confidence

CHIP FORMAT (8 channels):
    0: VV backscatter (dB)
    1: VH backscatter (dB)
    2: DEM (meters)
    3: Slope (degrees)
    4: HAND (meters)
    5: TWI (Topographic Wetness Index)
    6: Label (ground truth, optional for inference)
    7: MNDWI (Modified Normalized Difference Water Index)

Author: SAR Water Detection Project
Version: 1.0.0
Date: 2026-01-26
"""

import os
import gc
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np

# Feature computation imports
from scipy.ndimage import uniform_filter, minimum_filter, maximum_filter, laplace
from scipy.ndimage import grey_opening, grey_closing, label as scipy_label
from skimage.morphology import remove_small_objects, remove_small_holes

import warnings

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    """Configuration for SAR Water Detector."""

    model_path: str = "models/lightgbm_v9_clean_mndwi.txt"
    min_water_size: int = 50  # Minimum water body size in pixels
    min_hole_size: int = 50  # Minimum hole size to fill
    physics_weight: float = 0.1  # Weight for physics score
    lgb_weight: float = 0.9  # Weight for LightGBM prediction
    threshold: float = 0.5  # Binary threshold
    use_physics_veto: bool = True  # Apply physics-based veto


class FeatureExtractor:
    """
    Extracts 74 features from SAR chip for LightGBM prediction.

    Features:
        0-1: VV, VH (raw backscatter)
        2-5: SAR ratios (VV/VH, VV-VH, NDWI-like, RVI)
        6-45: Multi-scale texture (5 scales x 2 bands x 4 stats)
        46-49: Gradient features (magnitude, Laplacian)
        50-53: Morphological features (opening, closing)
        54-57: Otsu-like and local contrast
        58-62: GLCM-like features + entropy
        63-66: Terrain features (DEM, Slope, HAND, TWI)
        67-69: Physics-based scores
        70-73: MNDWI features
    """

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
        "MNDWI",
        "MNDWI_water_mask",
        "MNDWI_mean_s5",
        "MNDWI_std_s5",
    ]

    def __init__(self):
        self.n_features = 74

    def extract(self, chip: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extract 74 features from chip.

        Args:
            chip: (H, W, 8) numpy array

        Returns:
            features: (H*W, 74) feature array
            shape: (H, W) original shape
        """
        h, w = chip.shape[:2]

        # Extract channels
        vv = chip[:, :, 0].astype(np.float32)
        vh = chip[:, :, 1].astype(np.float32)
        dem = chip[:, :, 2].astype(np.float32)
        slope = np.clip(chip[:, :, 3].astype(np.float32), 0, 90)
        hand = np.clip(chip[:, :, 4].astype(np.float32), 0, 500)
        twi = np.clip(chip[:, :, 5].astype(np.float32), 0, 30)
        mndwi = (
            chip[:, :, 7].astype(np.float32) if chip.shape[2] > 7 else np.zeros_like(vv)
        )

        features = []

        # 0-1: Raw backscatter
        features.extend([vv, vh])

        # 2-5: SAR ratios
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(vh) > 0.01, vv / vh, 0)
            ratio = np.clip(ratio, -10, 10)
        features.append(ratio)
        features.append(vv - vh)

        denom = vv + vh
        with np.errstate(divide="ignore", invalid="ignore"):
            ndwi = np.where(np.abs(denom) > 0.01, (vv - vh) / denom, 0)
            rvi = np.where(np.abs(denom) > 0.01, 4 * vh / denom, 0)
        features.extend([ndwi, rvi])

        # 6-45: Multi-scale texture
        for scale in [3, 5, 9, 15, 21]:
            for arr in [vv, vh]:
                arr_mean = uniform_filter(arr, size=scale)
                arr_sq = uniform_filter(arr**2, size=scale)
                arr_var = np.maximum(arr_sq - arr_mean**2, 0)
                features.extend(
                    [
                        arr_mean,
                        np.sqrt(arr_var),
                        minimum_filter(arr, size=scale),
                        maximum_filter(arr, size=scale),
                    ]
                )

        # 46-49: Gradients
        gy_vv, gx_vv = np.gradient(vv)
        gy_vh, gx_vh = np.gradient(vh)
        features.extend(
            [
                np.sqrt(gx_vv**2 + gy_vv**2),
                np.sqrt(gx_vh**2 + gy_vh**2),
                np.abs(laplace(vv)),
                np.abs(laplace(vh)),
            ]
        )

        # 50-53: Morphological
        features.extend(
            [
                grey_opening(vv, 5),
                grey_closing(vv, 5),
                grey_opening(vh, 5),
                grey_closing(vh, 5),
            ]
        )

        # 54-55: Otsu-like
        features.extend([vv - np.median(vv), vh - np.median(vh)])

        # 56-57: Local contrast
        features.extend([vv - uniform_filter(vv, 9), vh - uniform_filter(vh, 9)])

        # 58-61: GLCM-like
        for arr in [vv, vh]:
            arr_mean = uniform_filter(arr, 5)
            arr_sq = uniform_filter(arr**2, 5)
            arr_var = np.maximum(arr_sq - arr_mean**2, 0)
            arr_range = maximum_filter(arr, 5) - minimum_filter(arr, 5)
            features.extend([np.sqrt(arr_var), 1.0 / (1.0 + arr_range)])

        # 62: Pseudo-entropy
        vv_norm = vv - vv.min()
        vv_range = vv.max() - vv.min() + 1e-10
        vv_prob = np.clip(vv_norm / vv_range, 1e-10, 1 - 1e-10)
        entropy = -vv_prob * np.log2(vv_prob) - (1 - vv_prob) * np.log2(1 - vv_prob)
        features.append(entropy)

        # 63-66: Terrain
        features.extend([dem, slope, hand, twi])

        # 67-69: Physics scores
        features.append(1.0 / (1.0 + np.exp(np.clip((hand - 10) / 3.0, -50, 50))))
        features.append(1.0 / (1.0 + np.exp(np.clip((slope - 8) / 3.0, -50, 50))))
        features.append(1.0 / (1.0 + np.exp(np.clip((8 - twi) / 2.0, -50, 50))))

        # 70-73: MNDWI
        features.append(mndwi)
        features.append((mndwi > 0).astype(np.float32))
        mndwi_mean = uniform_filter(mndwi, 5)
        mndwi_sq = uniform_filter(mndwi**2, 5)
        mndwi_var = np.maximum(mndwi_sq - mndwi_mean**2, 0)
        features.extend([mndwi_mean, np.sqrt(mndwi_var)])

        # Stack and clean
        feature_stack = np.stack(features, axis=-1)
        feature_stack = np.nan_to_num(feature_stack, nan=0.0, posinf=0.0, neginf=0.0)

        return feature_stack.reshape(-1, self.n_features).astype(np.float32), (h, w)


class PhysicsEngine:
    """Physics-based constraints and scoring for water detection."""

    @staticmethod
    def compute_veto_mask(chip: np.ndarray) -> np.ndarray:
        """
        Compute physics-based veto mask (areas that CANNOT be water).

        Rules:
        - HAND > 100m: Too high above drainage
        - Slope > 45 deg: Too steep for water
        - HAND > 30m AND Slope > 20 deg: Combined constraint
        """
        slope = chip[:, :, 3]
        hand = chip[:, :, 4]

        veto = (hand > 100) | (slope > 45) | ((hand > 30) & (slope > 20))
        return veto

    @staticmethod
    def compute_water_score(chip: np.ndarray) -> np.ndarray:
        """
        Compute physics-based water likelihood score.

        Based on:
        - Low HAND (Height Above Nearest Drainage)
        - Low Slope
        - High TWI (Topographic Wetness Index)
        """
        slope = chip[:, :, 3]
        hand = chip[:, :, 4]
        twi = chip[:, :, 5]

        # Sigmoid-based scoring
        hand_score = 1.0 / (1.0 + np.exp((hand - 15) / 5.0))
        slope_score = 1.0 / (1.0 + np.exp((slope - 10) / 4.0))
        twi_score = 1.0 / (1.0 + np.exp((7 - twi) / 2.0))

        # Weighted combination
        physics_score = 0.4 * hand_score + 0.4 * slope_score + 0.2 * twi_score

        return physics_score


class PostProcessor:
    """Post-processing operations for water mask refinement."""

    def __init__(self, min_water_size: int = 50, min_hole_size: int = 50):
        self.min_water_size = min_water_size
        self.min_hole_size = min_hole_size

    def process(self, water_proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Apply post-processing to probability map.

        Steps:
        1. Threshold to binary
        2. Remove small objects
        3. Fill small holes
        """
        water_binary = water_proba > threshold

        # Remove small water bodies
        water_binary = remove_small_objects(water_binary, min_size=self.min_water_size)

        # Fill small holes
        water_binary = remove_small_holes(
            water_binary, area_threshold=self.min_hole_size
        )

        return water_binary.astype(np.float32)


class SARWaterDetector:
    """
    Main SAR Water Detection class.

    Combines LightGBM ML model with physics-based constraints.

    Example:
        detector = SARWaterDetector()
        result = detector.predict(chip)
        water_mask = result['water_mask']
    """

    def __init__(
        self, config: Optional[DetectorConfig] = None, model_path: Optional[str] = None
    ):
        """
        Initialize detector.

        Args:
            config: DetectorConfig object (optional)
            model_path: Path to LightGBM model file (overrides config)
        """
        self.config = config or DetectorConfig()

        if model_path:
            self.config.model_path = model_path

        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.physics_engine = PhysicsEngine()
        self.post_processor = PostProcessor(
            min_water_size=self.config.min_water_size,
            min_hole_size=self.config.min_hole_size,
        )

        # Load model
        self._load_model()

    def _load_model(self):
        """Load LightGBM model."""
        import lightgbm as lgb

        model_path = Path(self.config.model_path)
        if not model_path.exists():
            # Try relative to package
            package_dir = Path(__file__).parent.parent
            model_path = package_dir / self.config.model_path

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.config.model_path}")

        self.model = lgb.Booster(model_file=str(model_path))
        logger.info(
            f"Loaded model: {model_path.name} ({self.model.num_feature()} features)"
        )

    def predict(self, chip: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict water mask from SAR chip.

        Args:
            chip: (H, W, 8) numpy array with SAR and auxiliary data

        Returns:
            Dictionary with:
                - water_mask: Binary water mask (H, W)
                - water_probability: Probability map (H, W)
                - confidence: Confidence scores (H, W)
                - physics_score: Physics-based score (H, W)
                - veto_mask: Physics veto mask (H, W)
        """
        # Validate input
        if chip.ndim != 3 or chip.shape[2] < 7:
            raise ValueError(f"Expected chip shape (H, W, >=7), got {chip.shape}")

        h, w = chip.shape[:2]

        # Extract features
        features, shape = self.feature_extractor.extract(chip)

        # LightGBM prediction
        lgb_proba = self.model.predict(features).reshape(shape)

        # Physics constraints
        veto_mask = self.physics_engine.compute_veto_mask(chip)
        physics_score = self.physics_engine.compute_water_score(chip)

        # Combine predictions
        combined_proba = (
            self.config.lgb_weight * lgb_proba
            + self.config.physics_weight * physics_score
        )

        # Apply physics veto
        if self.config.use_physics_veto:
            combined_proba = np.where(veto_mask, 0.0, combined_proba)

        # Post-process
        water_mask = self.post_processor.process(combined_proba, self.config.threshold)

        # Compute confidence (how certain the model is)
        confidence = np.abs(combined_proba - 0.5) * 2  # 0 = uncertain, 1 = certain

        return {
            "water_mask": water_mask,
            "water_probability": combined_proba,
            "confidence": confidence,
            "physics_score": physics_score,
            "veto_mask": veto_mask.astype(np.float32),
            "lgb_raw": lgb_proba,
        }

    def predict_batch(self, chips: list) -> list:
        """Predict on multiple chips."""
        return [self.predict(chip) for chip in chips]

    def evaluate(self, chip: np.ndarray, label: Optional[np.ndarray] = None) -> Dict:
        """
        Predict and evaluate against ground truth.

        Args:
            chip: (H, W, 8) numpy array
            label: (H, W) ground truth mask (optional, uses chip[:,:,6] if None)

        Returns:
            Dictionary with prediction and metrics
        """
        result = self.predict(chip)

        if label is None:
            label = chip[:, :, 6]

        # Compute metrics
        pred_bin = result["water_mask"] > 0.5
        label_bin = label > 0.5

        intersection = np.logical_and(pred_bin, label_bin).sum()
        union = np.logical_or(pred_bin, label_bin).sum()

        tp = intersection
        fp = np.logical_and(pred_bin, ~label_bin).sum()
        fn = np.logical_and(~pred_bin, label_bin).sum()

        result["metrics"] = {
            "iou": float(intersection) / float(union) if union > 0 else 0.0,
            "precision": float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0,
            "recall": float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0,
            "water_fraction_pred": float(pred_bin.mean()),
            "water_fraction_true": float(label_bin.mean()),
        }

        return result


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """Command-line interface for SAR Water Detector."""
    import argparse

    parser = argparse.ArgumentParser(description="SAR Water Detector")
    parser.add_argument(
        "--input", type=str, required=True, help="Input chip .npy file or directory"
    )
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument(
        "--model",
        type=str,
        default="models/lightgbm_v9_clean_mndwi.txt",
        help="Model path",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Detection threshold"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate against ground truth"
    )
    args = parser.parse_args()

    # Initialize detector
    config = DetectorConfig(model_path=args.model, threshold=args.threshold)
    detector = SARWaterDetector(config)

    # Process input
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    if input_path.is_file():
        chip_files = [input_path]
    else:
        chip_files = sorted(input_path.glob("*.npy"))

    results = []

    for chip_file in chip_files:
        logger.info(f"Processing {chip_file.name}...")

        try:
            chip = np.load(chip_file)

            if args.evaluate:
                result = detector.evaluate(chip)
                metrics = result["metrics"]
                logger.info(
                    f"  IoU: {metrics['iou']:.4f}, P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f}"
                )
                results.append({"chip": chip_file.name, **metrics})
            else:
                result = detector.predict(chip)

            # Save output
            np.save(output_dir / f"{chip_file.stem}_mask.npy", result["water_mask"])
            np.save(
                output_dir / f"{chip_file.stem}_proba.npy", result["water_probability"]
            )

        except Exception as e:
            logger.error(f"Error processing {chip_file.name}: {e}")

    # Summary
    if results:
        avg_iou = np.mean([r["iou"] for r in results])
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Average IoU: {avg_iou:.4f} over {len(results)} chips")
        logger.info(f"{'=' * 60}")

        # Save results
        with open(output_dir / "evaluation_results.json", "w") as f:
            json.dump({"average_iou": avg_iou, "per_chip": results}, f, indent=2)


if __name__ == "__main__":
    main()
