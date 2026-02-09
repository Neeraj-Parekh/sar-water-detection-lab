#!/usr/bin/env python3
"""
================================================================================
PHYSICS-VISION HYBRID DETECTOR V7
================================================================================
Combines:
1. LightGBM (data-driven SAR features)
2. Soft physics constraints (terrain feasibility)
3. Signature-adaptive detection (handles bright water)

Key improvements:
- Soft physics: Only extreme cases penalized, not multiplicative
- Signature classification: Detects bright vs dark water
- Adaptive thresholds: Based on local statistics
- U-Net refinement: For edge cases
"""

import numpy as np
from scipy.ndimage import (
    uniform_filter,
    minimum_filter,
    maximum_filter,
    label as scipy_label,
)
from typing import Tuple, Optional, Dict
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PhysicsVisionHybrid:
    """Hybrid water detector combining physics and ML."""

    def __init__(
        self,
        lgb_model_path: Optional[str] = None,
        unet_model_path: Optional[str] = None,
    ):
        self.lgb_model = None
        self.unet_model = None

        if lgb_model_path:
            self.load_lgb(lgb_model_path)
        if unet_model_path:
            self.load_unet(unet_model_path)

    def load_lgb(self, path: str):
        """Load LightGBM model."""
        try:
            import lightgbm as lgb

            self.lgb_model = lgb.Booster(model_file=path)
            logger.info(f"Loaded LightGBM from {path}")
        except Exception as e:
            logger.error(f"Failed to load LightGBM: {e}")

    def load_unet(self, path: str):
        """Load U-Net model."""
        try:
            import torch

            self.unet_model = torch.load(path, map_location="cpu")
            self.unet_model.eval()
            logger.info(f"Loaded U-Net from {path}")
        except Exception as e:
            logger.error(f"Failed to load U-Net: {e}")

    def classify_signature(self, vv: np.ndarray, vh: np.ndarray) -> Dict:
        """
        Classify the SAR signature type in the image.

        Returns:
            dict with 'type': 'dark_water', 'bright_water', 'mixed', 'ambiguous'
        """
        # Statistics
        vv_mean = np.nanmean(vv)
        vv_std = np.nanstd(vv)
        vh_mean = np.nanmean(vh)

        # Bimodal check (water + land separation)
        vv_p10 = np.nanpercentile(vv, 10)
        vv_p90 = np.nanpercentile(vv, 90)
        bimodal_gap = vv_p90 - vv_p10

        result = {
            "vv_mean": vv_mean,
            "vv_std": vv_std,
            "bimodal_gap": bimodal_gap,
            "type": "unknown",
        }

        if vv_mean < -18:
            # Very dark overall - likely dark water dominant
            result["type"] = "dark_water"
            result["water_expected"] = "dark"
        elif vv_mean > -12:
            # Very bright overall - urban/rough surface
            result["type"] = "bright_surface"
            result["water_expected"] = "bright"  # Urban water = double bounce
        elif bimodal_gap > 6:
            # Good separation - mixed surface types
            result["type"] = "mixed"
            result["water_expected"] = "dark"  # Assume typical water
        else:
            # Low contrast - ambiguous
            result["type"] = "ambiguous"
            result["water_expected"] = "unknown"

        return result

    def compute_soft_physics(
        self, hand: np.ndarray, slope: np.ndarray, twi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute SOFT physics constraints.

        Unlike hard multiplicative physics, this only penalizes EXTREME cases.
        Returns:
            feasibility: 0-1 score (1 = feasible, 0 = impossible)
            confidence: How confident the physics is (low in ambiguous areas)
        """
        # Clip slope to valid range (data corruption fix)
        slope_clean = np.clip(slope, 0, 90)
        hand_clean = np.clip(hand, 0, 500)

        # Initialize as fully feasible
        feasibility = np.ones_like(hand, dtype=np.float32)
        confidence = np.ones_like(hand, dtype=np.float32)

        # Only penalize EXTREME cases (not typical conditions)

        # 1. Very high HAND (> 50m) - water very unlikely
        high_hand = hand_clean > 50
        feasibility = np.where(high_hand, 0.2, feasibility)
        confidence = np.where(high_hand, 0.9, confidence)

        # 2. Very steep slope (> 30°) - water pools unlikely
        steep_slope = slope_clean > 30
        feasibility = np.where(steep_slope, 0.3, feasibility)
        confidence = np.where(steep_slope, 0.8, confidence)

        # 3. Moderate HAND (20-50m) - reduce but don't eliminate
        moderate_hand = (hand_clean > 20) & (hand_clean <= 50)
        feasibility = np.where(moderate_hand, 0.6, feasibility)
        confidence = np.where(moderate_hand, 0.5, confidence)

        # 4. Low TWI (< 5) in non-river areas - less likely water
        low_twi = twi < 5
        feasibility = np.where(low_twi, feasibility * 0.8, feasibility)

        # 5. Boost confidence where physics is very clear
        clear_water_physics = (hand_clean < 5) & (slope_clean < 5) & (twi > 10)
        confidence = np.where(clear_water_physics, 0.95, confidence)
        feasibility = np.where(clear_water_physics, 1.0, feasibility)

        return feasibility, confidence

    def compute_adaptive_threshold(self, vv: np.ndarray, signature_type: str) -> float:
        """Compute adaptive VV threshold based on scene."""
        if signature_type == "dark_water":
            # Use Otsu on dark side
            dark_pixels = vv[vv < np.percentile(vv, 50)]
            threshold = (
                np.percentile(dark_pixels, 70) if len(dark_pixels) > 100 else -18
            )
        elif signature_type == "bright_surface":
            # For urban scenes, water might be brighter
            threshold = np.percentile(vv, 30)  # Look for relatively dark areas
        else:
            # Default: standard dark water threshold
            threshold = -18

        return float(threshold)

    def detect_water_simple(
        self,
        vv: np.ndarray,
        vh: np.ndarray,
        hand: np.ndarray,
        slope: np.ndarray,
        twi: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Simple water detection without ML models.
        Uses signature-adaptive thresholding + soft physics.
        """
        # Classify scene
        signature = self.classify_signature(vv, vh)

        # Compute soft physics
        feasibility, physics_conf = self.compute_soft_physics(hand, slope, twi)

        # Adaptive SAR-based detection
        if signature["type"] == "dark_water" or signature["water_expected"] == "dark":
            # Standard: dark = water
            threshold = self.compute_adaptive_threshold(vv, signature["type"])
            sar_water = vv < threshold
            sar_score = 1.0 / (1.0 + np.exp((vv - threshold) / 2.0))
        elif signature["type"] == "bright_surface":
            # Urban: look for local dark areas (relative)
            local_mean = uniform_filter(vv, size=21)
            local_dark = vv < (local_mean - 3)  # 3dB darker than surroundings
            sar_water = local_dark
            sar_score = np.where(local_dark, 0.7, 0.3)
        else:
            # Ambiguous: use conservative threshold
            sar_water = vv < -20
            sar_score = 1.0 / (1.0 + np.exp((vv + 20) / 2.0))

        # Combine SAR + Physics (weighted by confidence)
        # High physics confidence: use physics to constrain
        # Low physics confidence: trust SAR more
        combined = sar_score * (0.5 + 0.5 * feasibility)

        # Binary threshold
        water_mask = combined > 0.5

        # Post-processing: remove small regions
        water_mask = self.remove_small_regions(water_mask, min_size=50)

        return water_mask.astype(np.float32), {
            "signature": signature,
            "threshold": threshold if "threshold" in dir() else -18,
            "sar_score_mean": float(sar_score.mean()),
            "physics_feasibility_mean": float(feasibility.mean()),
            "combined_score_mean": float(combined.mean()),
        }

    def remove_small_regions(self, mask: np.ndarray, min_size: int = 50) -> np.ndarray:
        """Remove small connected components."""
        labeled, num_features = scipy_label(mask)
        if num_features == 0:
            return mask

        cleaned = np.zeros_like(mask)
        for i in range(1, num_features + 1):
            region = labeled == i
            if region.sum() >= min_size:
                cleaned = np.logical_or(cleaned, region)

        return cleaned

    def detect_water(
        self,
        vv: np.ndarray,
        vh: np.ndarray,
        dem: np.ndarray,
        slope: np.ndarray,
        hand: np.ndarray,
        twi: np.ndarray,
        use_ml: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Full hybrid water detection.

        Args:
            vv, vh: SAR polarizations (dB)
            dem, slope, hand, twi: Terrain features
            use_ml: Whether to use ML models if available

        Returns:
            water_mask: Binary water mask
            metadata: Detection statistics
        """
        metadata = {"method": "simple"}

        # Always compute physics and signature
        signature = self.classify_signature(vv, vh)
        feasibility, physics_conf = self.compute_soft_physics(hand, slope, twi)
        metadata["signature"] = signature

        # If LightGBM available, use it
        if use_ml and self.lgb_model is not None:
            try:
                from .retrain_v6 import extract_features  # Import feature extractor

                features = extract_features(vv, vh, dem, slope, hand, twi)
                h, w, n_feat = features.shape
                X = features.reshape(-1, n_feat)
                proba = self.lgb_model.predict(X).reshape(h, w)

                # Apply soft physics as post-filter
                combined = proba * (0.7 + 0.3 * feasibility)
                water_mask = combined > 0.5
                metadata["method"] = "lgb+physics"
                metadata["lgb_mean_proba"] = float(proba.mean())
            except Exception as e:
                logger.warning(f"LightGBM failed, falling back to simple: {e}")
                water_mask, simple_meta = self.detect_water_simple(
                    vv, vh, hand, slope, twi
                )
                metadata.update(simple_meta)
        else:
            # Simple detection
            water_mask, simple_meta = self.detect_water_simple(vv, vh, hand, slope, twi)
            metadata.update(simple_meta)

        # Post-process
        water_mask = self.remove_small_regions(water_mask.astype(bool), min_size=50)

        return water_mask.astype(np.float32), metadata


def analyze_bright_water_cases(chips_dir: str, output_file: str = None):
    """
    Analyze chips with bright water signatures.
    Helps understand when vision alone fails and physics is needed.
    """
    import glob

    results = []
    chips = sorted(glob.glob(f"{chips_dir}/*_with_truth.npy"))

    detector = PhysicsVisionHybrid()

    for chip_path in chips[:20]:  # First 20 for quick analysis
        try:
            data = np.load(chip_path)
            name = chip_path.split("/")[-1]

            vv, vh = data[:, :, 0], data[:, :, 1]
            slope, hand, twi = data[:, :, 3], data[:, :, 4], data[:, :, 5]
            label = data[:, :, 6]

            # Get signature
            sig = detector.classify_signature(vv, vh)

            # Compute metrics
            water_mask = label > 0
            if water_mask.sum() > 100:
                water_vv = vv[water_mask].mean()
                land_vv = vv[~water_mask].mean() if (~water_mask).sum() > 0 else 0

                results.append(
                    {
                        "chip": name,
                        "signature_type": sig["type"],
                        "water_vv": water_vv,
                        "land_vv": land_vv,
                        "vv_separation": abs(water_vv - land_vv),
                        "water_fraction": water_mask.mean(),
                        "bright_water": water_vv > -15,
                    }
                )
        except Exception as e:
            logger.error(f"Error processing {chip_path}: {e}")

    # Print summary
    print("\n=== BRIGHT WATER ANALYSIS ===\n")
    bright_count = sum(1 for r in results if r["bright_water"])
    print(f"Chips with bright water (VV > -15dB): {bright_count}/{len(results)}")

    print("\nBright water chips:")
    for r in results:
        if r["bright_water"]:
            print(
                f"  {r['chip'][:40]}: VV={r['water_vv']:.1f}dB, type={r['signature_type']}"
            )

    return results


if __name__ == "__main__":
    # Test on sample chip
    import sys

    if len(sys.argv) > 1:
        chip_path = sys.argv[1]
    else:
        chip_path = "/home/mit-aoe/sar_water_detection/chips/chip_001_large_lakes_with_truth.npy"

    print(f"Testing on {chip_path}")

    data = np.load(chip_path)
    vv, vh = data[:, :, 0], data[:, :, 1]
    dem, slope, hand, twi = data[:, :, 2], data[:, :, 3], data[:, :, 4], data[:, :, 5]
    label = data[:, :, 6]

    # Test detector
    detector = PhysicsVisionHybrid()
    pred, meta = detector.detect_water(vv, vh, dem, slope, hand, twi, use_ml=False)

    # Compute IoU
    intersection = np.logical_and(pred > 0.5, label > 0.5).sum()
    union = np.logical_or(pred > 0.5, label > 0.5).sum()
    iou = intersection / union if union > 0 else 0

    print(f"\nResults:")
    print(f"  Method: {meta['method']}")
    print(f"  Signature: {meta['signature']['type']}")
    print(f"  IoU: {iou:.4f}")
    print(f"  Water pixels predicted: {(pred > 0.5).sum()}")
    print(f"  Water pixels actual: {(label > 0.5).sum()}")
