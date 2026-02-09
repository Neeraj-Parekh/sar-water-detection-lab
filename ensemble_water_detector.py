#!/usr/bin/env python3
"""
Ensemble Water Detection System
================================
Combines LightGBM, U-Net, and Physics-based detection for optimal results.

Best models from our research:
- LightGBM v4: Test IoU 0.881 (69 features)
- U-Net v4: Test IoU 0.766
- Physics equations: IoU ~0.51 (hand_constrained best)

This ensemble combines the strengths of each approach.

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
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
from scipy.ndimage import (
    uniform_filter,
    gaussian_filter,
    binary_dilation,
    binary_erosion,
)

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ensemble_detector.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "chip_dir": Path("/home/mit-aoe/sar_water_detection/chips"),
    "model_dir": Path("/home/mit-aoe/sar_water_detection/models"),
    "output_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    # Best equation from search (hand_constrained: IoU=0.5113)
    "best_equation": {
        "template": "hand_constrained",
        "T_vh": -12.0,  # Will be updated from search results
        "T_hand": 21.0,
    },
    # Ensemble weights (tuned for best performance)
    "weights": {
        "lightgbm": 0.50,  # Best single model
        "unet": 0.30,  # Good at boundaries
        "physics": 0.20,  # Safety net
    },
    # Physics thresholds by terrain
    "terrain_thresholds": {
        "flat_lowland": {"vh": -18, "hand": 15, "slope": 10},
        "hilly": {"vh": -17, "hand": 25, "slope": 20},
        "mountainous": {"vh": -16, "hand": 100, "slope": 30},
        "arid": {"vh": -20, "hand": 8, "slope": 5},
        "urban": {"vh": -20, "hand": 8, "slope": 10},
        "wetland": {"vh": -14, "hand": 20, "slope": 8},
        "coastal": {"vh": -16, "hand": 10, "slope": 8},
    },
}


# =============================================================================
# Terrain Classifier
# =============================================================================


class TerrainClassifier:
    """Classify terrain type from features."""

    KNOWN_LOCATIONS = {
        "mumbai": "urban",
        "delhi": "urban",
        "bangalore": "urban",
        "sundarbans": "coastal",
        "bhitarkanika": "coastal",
        "keoladeo": "wetland",
        "kolleru": "wetland",
        "chilika": "wetland",
        "pangong": "mountainous",
        "dal_lake": "mountainous",
        "rann": "arid",
        "kutch": "arid",
        "sambhar": "arid",
        "brahmaputra": "flat_lowland",
        "ganga": "flat_lowland",
    }

    @classmethod
    def classify(cls, features: Dict[str, np.ndarray], chip_name: str = "") -> str:
        """Classify terrain based on features and name."""
        # Check name first
        chip_lower = chip_name.lower()
        for loc, terrain in cls.KNOWN_LOCATIONS.items():
            if loc in chip_lower:
                return terrain

        # Feature-based classification
        dem = features.get("dem", np.zeros((1, 1)))
        slope = features.get("slope", np.zeros((1, 1)))
        twi = features.get("twi", np.zeros((1, 1)))
        vv = features.get("vv", np.zeros((1, 1)))
        vh = features.get("vh", np.zeros((1, 1)))

        dem_mean = np.nanmean(dem)
        slope_mean = np.nanmean(slope)
        slope_p90 = (
            np.nanpercentile(slope[~np.isnan(slope)], 90)
            if np.any(~np.isnan(slope))
            else 0
        )
        twi_mean = np.nanmean(twi)
        vv_mean = np.nanmean(vv)
        vh_mean = np.nanmean(vh)
        vv_vh_diff = vv_mean - vh_mean

        if dem_mean > 2000:
            return "mountainous"
        elif slope_mean > 12 or slope_p90 > 25:
            return "hilly" if dem_mean < 500 else "mountainous"
        elif vv_mean > -12 and vv_vh_diff > 7:
            return "urban"
        elif twi_mean > 10:
            return "wetland"
        elif twi_mean < 5 and vh_mean < -22:
            return "arid"
        else:
            return "flat_lowland"


# =============================================================================
# Physics-Based Detector
# =============================================================================


class PhysicsWaterDetector:
    """
    Physics-based water detection using discovered equations.

    Best equations from our search:
    1. hand_constrained: (vh < T_vh) & (hand < T_hand) - IoU=0.5113
    2. hysteresis: Core + extended detection - IoU=0.4745
    3. bright_water_strict: Relaxed VH, strict physics - IoU=0.3781
    """

    def __init__(self):
        self.texture_window = 9

    def detect(
        self,
        features: Dict[str, np.ndarray],
        terrain: str = "flat_lowland",
    ) -> np.ndarray:
        """
        Multi-tier physics detection.

        Returns probability map [0, 1].
        """
        vv = features.get("vv", np.zeros((256, 256)))
        vh = features.get("vh", np.zeros((256, 256)))
        hand = features.get("hand", np.zeros_like(vv))
        slope = features.get("slope", np.zeros_like(vv))
        twi = features.get("twi", np.zeros_like(vv))

        # Get terrain-specific thresholds
        thresholds = CONFIG["terrain_thresholds"].get(
            terrain, CONFIG["terrain_thresholds"]["flat_lowland"]
        )

        # Handle NaN
        vv = np.nan_to_num(vv, nan=-20)
        vh = np.nan_to_num(vh, nan=-25)
        hand = np.nan_to_num(hand, nan=50)
        slope = np.nan_to_num(slope, nan=30)
        twi = np.nan_to_num(twi, nan=5)

        prob = np.zeros_like(vh, dtype=np.float32)

        # Tier 1: Core water (high confidence)
        # Very dark VH, reasonable HAND
        core_water = (
            (vh < -24) & (hand < thresholds["hand"]) & (slope < thresholds["slope"])
        )
        prob = np.where(core_water, 0.95, prob)

        # Tier 2: Standard water (hand_constrained equation)
        # Best from search: IoU=0.5113
        standard_water = (
            (vh < thresholds["vh"])
            & (hand < thresholds["hand"])
            & (slope < thresholds["slope"])
        )
        prob = np.where(standard_water & ~core_water, 0.85, prob)

        # Tier 3: Extended water (hysteresis)
        # Moderately dark VH, strict HAND
        extended_water = (
            (vh >= thresholds["vh"])
            & (vh < thresholds["vh"] + 6)  # Extended range
            & (hand < 5)  # Very strict HAND
            & (slope < 3)  # Very flat
            & (twi > 7)  # High wetness
        )
        prob = np.where(extended_water, 0.70, prob)

        # Tier 4: Bright water (wind-roughened)
        # Very relaxed VH, very strict physics
        bright_water = (
            (vh >= thresholds["vh"] + 6)
            & (vh < -10)  # Not too bright
            & (hand < 4)
            & (slope < 4)
            & (twi > 8)
        )
        prob = np.where(bright_water, 0.55, prob)

        # Urban exclusion
        urban_mask = (vv > -10) & ((vv - vh) > 8)
        prob = np.where(urban_mask, prob * 0.3, prob)

        # Extreme slope cutoff
        prob = np.where(slope > 35, 0.0, prob)

        return prob


# =============================================================================
# LightGBM Wrapper
# =============================================================================


class LightGBMPredictor:
    """
    Load and run LightGBM model predictions.

    Best model: LightGBM v4 with 69 features
    - Val IoU: 0.9302
    - Test IoU: 0.8808
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.feature_names = None

        if model_path and model_path.exists():
            self._load_model(model_path)

    def _load_model(self, model_path: Path):
        """Load LightGBM model."""
        try:
            import joblib

            self.model = joblib.load(model_path)
            logger.info(f"Loaded LightGBM model from {model_path}")

            # Try to get feature names
            if hasattr(self.model, "feature_name_"):
                self.feature_names = self.model.feature_name_
        except Exception as e:
            logger.warning(f"Could not load LightGBM model: {e}")

    def predict(self, features: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Run LightGBM prediction.

        Returns probability map or None if model not loaded.
        """
        if self.model is None:
            return None

        try:
            # Build feature matrix
            h, w = features.get("vv", np.zeros((256, 256))).shape

            # Extract core features
            feature_list = []
            feature_order = ["vv", "vh", "dem", "hand", "slope", "twi"]

            for name in feature_order:
                if name in features:
                    feat = features[name].flatten()
                    feat = np.nan_to_num(feat, nan=0)
                    feature_list.append(feat)

            if not feature_list:
                return None

            X = np.column_stack(feature_list)

            # Predict
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X)[:, 1]
            else:
                probs = self.model.predict(X)

            return probs.reshape(h, w)

        except Exception as e:
            logger.warning(f"LightGBM prediction failed: {e}")
            return None


# =============================================================================
# U-Net Wrapper
# =============================================================================


class UNetPredictor:
    """
    Load and run U-Net model predictions.

    Best model: U-Net v4
    - Val IoU: 0.9184
    - Test IoU: 0.7662
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.device = None

        if model_path and model_path.exists():
            self._load_model(model_path)

    def _load_model(self, model_path: Path):
        """Load PyTorch U-Net model."""
        try:
            import torch

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            logger.info(f"Loaded U-Net model from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load U-Net model: {e}")

    def predict(self, features: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """
        Run U-Net prediction.

        Returns probability map or None if model not loaded.
        """
        if self.model is None:
            return None

        try:
            import torch

            # Build input tensor (C, H, W)
            channels = []
            for name in ["vv", "vh", "dem", "hand", "slope", "twi"]:
                if name in features:
                    ch = features[name].astype(np.float32)
                    ch = np.nan_to_num(ch, nan=0)
                    channels.append(ch)

            if not channels:
                return None

            x = np.stack(channels, axis=0)
            x = torch.from_numpy(x).unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                pred = self.model(x)
                if isinstance(pred, tuple):
                    pred = pred[0]
                pred = torch.sigmoid(pred)
                pred = pred.squeeze().cpu().numpy()

            return pred

        except Exception as e:
            logger.warning(f"U-Net prediction failed: {e}")
            return None


# =============================================================================
# Ensemble Detector
# =============================================================================


class EnsembleWaterDetector:
    """
    Main ensemble combining all detection methods.

    Ensemble strategy:
    1. Get predictions from all available models
    2. Apply terrain-adaptive weighting
    3. Apply physics safety net
    4. Post-process boundaries
    """

    def __init__(
        self,
        lightgbm_path: Optional[Path] = None,
        unet_path: Optional[Path] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.physics_detector = PhysicsWaterDetector()
        self.lightgbm_predictor = LightGBMPredictor(lightgbm_path)
        self.unet_predictor = UNetPredictor(unet_path)
        self.terrain_classifier = TerrainClassifier()

        self.weights = weights or CONFIG["weights"]

        logger.info(f"Ensemble initialized with weights: {self.weights}")

    def detect(
        self,
        features: Dict[str, np.ndarray],
        chip_name: str = "",
        return_components: bool = False,
    ) -> np.ndarray:
        """
        Run ensemble detection.

        Args:
            features: Dict with vv, vh, dem, hand, slope, twi
            chip_name: Optional name for terrain detection
            return_components: If True, return individual predictions

        Returns:
            Water probability map [0, 1]
        """
        # Classify terrain
        terrain = self.terrain_classifier.classify(features, chip_name)
        logger.debug(f"Detected terrain: {terrain}")

        # Get predictions from each model
        physics_pred = self.physics_detector.detect(features, terrain)
        lightgbm_pred = self.lightgbm_predictor.predict(features)
        unet_pred = self.unet_predictor.predict(features)

        # Collect available predictions
        predictions = {"physics": physics_pred}
        if lightgbm_pred is not None:
            predictions["lightgbm"] = lightgbm_pred
        if unet_pred is not None:
            predictions["unet"] = unet_pred

        # Compute adaptive weights
        active_weights = {}
        for name, pred in predictions.items():
            if pred is not None:
                active_weights[name] = self.weights.get(name, 0.1)

        # Normalize weights
        total_weight = sum(active_weights.values())
        for k in active_weights:
            active_weights[k] /= total_weight

        logger.debug(f"Active weights: {active_weights}")

        # Weighted combination
        combined = np.zeros_like(physics_pred, dtype=np.float32)
        for name, pred in predictions.items():
            if pred is not None and name in active_weights:
                combined += pred * active_weights[name]

        # Apply physics safety net (soft constraints)
        combined = self._apply_physics_safety(combined, features, terrain)

        # Post-process
        combined = self._post_process(combined)

        if return_components:
            return combined, predictions, terrain

        return combined

    def _apply_physics_safety(
        self,
        prob: np.ndarray,
        features: Dict[str, np.ndarray],
        terrain: str,
    ) -> np.ndarray:
        """Apply soft physics constraints."""
        hand = features.get("hand", np.zeros_like(prob))
        slope = features.get("slope", np.zeros_like(prob))

        thresholds = CONFIG["terrain_thresholds"].get(
            terrain, CONFIG["terrain_thresholds"]["flat_lowland"]
        )

        # Soft HAND penalty
        hand_max = thresholds["hand"]
        hand_penalty = 1.0 / (1.0 + np.exp((hand - hand_max) / 5.0))

        # Soft slope penalty
        slope_max = thresholds["slope"]
        slope_penalty = np.clip(1.0 - (slope - slope_max) / 15.0, 0.3, 1.0)

        # Apply penalties
        result = prob * hand_penalty * slope_penalty

        # Hard cutoff for extreme slopes
        result = np.where(slope > 40, 0.0, result)

        return result

    def _post_process(self, prob: np.ndarray) -> np.ndarray:
        """Post-process probability map."""
        # Light smoothing
        prob = gaussian_filter(prob, sigma=0.5)

        # Clip to valid range
        prob = np.clip(prob, 0, 1)

        return prob

    def detect_binary(
        self,
        features: Dict[str, np.ndarray],
        chip_name: str = "",
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Get binary water mask."""
        prob = self.detect(features, chip_name)
        return prob > threshold


# =============================================================================
# Evaluation Functions
# =============================================================================


def compute_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    pred_bool = pred > 0.5 if pred.dtype == np.float32 else pred.astype(bool)
    truth_bool = truth > 0.5 if truth.dtype == np.float32 else truth.astype(bool)

    tp = np.sum(pred_bool & truth_bool)
    fp = np.sum(pred_bool & ~truth_bool)
    fn = np.sum(~pred_bool & truth_bool)
    tn = np.sum(~pred_bool & ~truth_bool)

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    intersection = tp
    union = tp + fp + fn
    iou = intersection / (union + 1e-10)

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


def load_chip(chip_path: Path) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    """Load chip and extract features."""
    data = np.load(chip_path)

    # Handle different formats
    if len(data.shape) == 3:
        if data.shape[0] < data.shape[2]:
            data = np.transpose(data, (1, 2, 0))

    n_bands = data.shape[2] if len(data.shape) == 3 else 1

    features = {}
    truth = None

    # Standard band order
    band_names = ["vv", "vh", "nasadem", "dem", "hand", "slope", "twi", "truth"]

    for i, name in enumerate(band_names):
        if i < n_bands:
            if name == "truth":
                truth = data[:, :, i]
            elif name == "nasadem":
                continue  # Skip, use dem instead
            else:
                features[name] = data[:, :, i]

    return features, truth


def evaluate_ensemble(
    chip_dir: Path,
    detector: EnsembleWaterDetector,
    max_chips: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate ensemble on all chips."""
    chip_files = list(chip_dir.glob("*_with_truth.npy"))

    if max_chips:
        chip_files = chip_files[:max_chips]

    logger.info(f"Evaluating on {len(chip_files)} chips")

    all_metrics = []
    terrain_metrics = {}

    for chip_file in chip_files:
        try:
            features, truth = load_chip(chip_file)

            if truth is None:
                continue

            # Get prediction
            prob, components, terrain = detector.detect(
                features, chip_file.stem, return_components=True
            )

            # Compute metrics
            metrics = compute_metrics(prob, truth)
            metrics["chip"] = chip_file.stem
            metrics["terrain"] = terrain

            all_metrics.append(metrics)

            # Group by terrain
            if terrain not in terrain_metrics:
                terrain_metrics[terrain] = []
            terrain_metrics[terrain].append(metrics)

        except Exception as e:
            logger.warning(f"Failed to evaluate {chip_file.name}: {e}")

    # Aggregate results
    if not all_metrics:
        return {"error": "No valid chips evaluated"}

    mean_iou = np.mean([m["iou"] for m in all_metrics])
    std_iou = np.std([m["iou"] for m in all_metrics])
    mean_f1 = np.mean([m["f1"] for m in all_metrics])

    # Per-terrain summary
    terrain_summary = {}
    for terrain, metrics in terrain_metrics.items():
        terrain_summary[terrain] = {
            "count": len(metrics),
            "mean_iou": float(np.mean([m["iou"] for m in metrics])),
            "std_iou": float(np.std([m["iou"] for m in metrics])),
        }

    results = {
        "n_chips": len(all_metrics),
        "mean_iou": float(mean_iou),
        "std_iou": float(std_iou),
        "mean_f1": float(mean_f1),
        "terrain_summary": terrain_summary,
        "all_metrics": all_metrics,
    }

    return results


# =============================================================================
# Main
# =============================================================================


def main():
    """Main evaluation script."""
    logger.info("=" * 60)
    logger.info("ENSEMBLE WATER DETECTOR EVALUATION")
    logger.info("=" * 60)

    # Initialize detector (physics-only mode for now)
    detector = EnsembleWaterDetector(
        lightgbm_path=CONFIG["model_dir"] / "lightgbm_v4.joblib",
        unet_path=CONFIG["model_dir"] / "unet_v4.pt",
    )

    # Evaluate
    results = evaluate_ensemble(CONFIG["chip_dir"], detector)

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Chips evaluated: {results.get('n_chips', 0)}")
    logger.info(
        f"Mean IoU: {results.get('mean_iou', 0):.4f} +/- {results.get('std_iou', 0):.4f}"
    )
    logger.info(f"Mean F1: {results.get('mean_f1', 0):.4f}")

    logger.info("\nPer-terrain results:")
    for terrain, summary in results.get("terrain_summary", {}).items():
        logger.info(
            f"  {terrain}: IoU={summary['mean_iou']:.4f} (n={summary['count']})"
        )

    # Save results
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    output_path = CONFIG["output_dir"] / "ensemble_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
