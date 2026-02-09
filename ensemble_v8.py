#!/usr/bin/env python3
"""
================================================================================
ENSEMBLE V8 - LightGBM + U-Net + Physics Ensemble
================================================================================

Combines:
1. LightGBM (feature-based, fast, accurate on bulk)
2. U-Net (spatial context, good on edges)
3. Physics constraints (hard VETO + soft weighting)

Key features:
- Graceful fallback when U-Net unavailable
- MNDWI optional (works without it)
- Data validation built-in
- Configurable weights

Author: SAR Water Detection Project
Date: 2026-01-25
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from scipy.ndimage import uniform_filter, minimum_filter, maximum_filter, laplace
from scipy.ndimage import grey_opening, grey_closing, label as scipy_label, sobel
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnsembleDetectorV8:
    """
    Unified ensemble detector combining LightGBM, U-Net, and physics.
    """

    def __init__(
        self,
        lgb_model_path: Optional[str] = None,
        unet_model_path: Optional[str] = None,
        lgb_weight: float = 0.6,
        unet_weight: float = 0.3,
        physics_weight: float = 0.1,
    ):
        """
        Initialize ensemble detector.

        Args:
            lgb_model_path: Path to LightGBM model
            unet_model_path: Path to U-Net model
            lgb_weight: Weight for LightGBM predictions
            unet_weight: Weight for U-Net predictions
            physics_weight: Weight for physics score
        """
        self.lgb_model = None
        self.unet_model = None
        self.lgb_weight = lgb_weight
        self.unet_weight = unet_weight
        self.physics_weight = physics_weight
        self.feature_names = None

        if lgb_model_path:
            self._load_lgb(lgb_model_path)
        if unet_model_path:
            self._load_unet(unet_model_path)

    def _load_lgb(self, path: str):
        """Load LightGBM model."""
        try:
            import lightgbm as lgb

            self.lgb_model = lgb.Booster(model_file=path)
            self.feature_names = self.lgb_model.feature_name()
            logger.info(f"Loaded LightGBM: {path} ({len(self.feature_names)} features)")
        except Exception as e:
            logger.warning(f"Could not load LightGBM: {e}")

    def _load_unet(self, path: str):
        """Load U-Net model."""
        try:
            import torch

            # Check if it's a full model or state dict
            checkpoint = torch.load(path, map_location="cpu")

            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                # Assume it's already a model
                self.unet_model = checkpoint
                self.unet_model.eval()
                logger.info(f"Loaded U-Net model directly: {path}")
                return

            # Create model architecture
            from unet_v6_fixed import CBAMUNet

            self.unet_model = CBAMUNet(in_channels=6, out_channels=1)
            self.unet_model.load_state_dict(state_dict)
            self.unet_model.eval()
            logger.info(f"Loaded U-Net: {path}")

        except Exception as e:
            logger.warning(f"Could not load U-Net: {e}")

    def validate_data(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Validate and fix input data."""
        validated = {}

        for key in ["vv", "vh", "dem", "slope", "hand", "twi"]:
            if key not in data:
                raise ValueError(f"Missing required field: {key}")

            arr = data[key].astype(np.float32)

            # Fix NaN
            if np.any(np.isnan(arr)):
                if key == "vv":
                    arr = np.nan_to_num(arr, nan=-20.0)
                elif key == "vh":
                    arr = np.nan_to_num(arr, nan=-25.0)
                elif key == "hand":
                    arr = np.nan_to_num(arr, nan=100.0)
                elif key == "twi":
                    arr = np.nan_to_num(arr, nan=5.0)
                else:
                    arr = np.nan_to_num(arr, nan=0.0)

            # Fix range
            if key == "slope":
                arr = np.clip(arr, 0, 90)
            elif key == "hand":
                arr = np.clip(arr, 0, 500)
            elif key == "twi":
                arr = np.clip(arr, 0, 30)

            validated[key] = arr

        # Optional MNDWI
        if "mndwi" in data:
            mndwi = data["mndwi"].astype(np.float32)
            mndwi = np.nan_to_num(mndwi, nan=0.0)
            mndwi = np.clip(mndwi, -1, 1)
            validated["mndwi"] = mndwi

        return validated

    def compute_physics_score(
        self, hand: np.ndarray, slope: np.ndarray, twi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute physics feasibility score and VETO mask.

        Returns:
            physics_score: 0-1 soft score
            veto_mask: True where water is impossible
        """
        # Hard VETO conditions
        veto = np.zeros_like(hand, dtype=bool)
        veto |= hand > 100  # Very high above drainage
        veto |= slope > 45  # Very steep
        veto |= (hand > 30) & (slope > 20)  # Combined constraint

        # Soft physics score
        hand_exp = np.clip((hand - 15) / 5.0, -50, 50)
        hand_score = 1.0 / (1.0 + np.exp(hand_exp))

        slope_exp = np.clip((slope - 12) / 4.0, -50, 50)
        slope_score = 1.0 / (1.0 + np.exp(slope_exp))

        twi_exp = np.clip((7 - twi) / 2.0, -50, 50)
        twi_score = 1.0 / (1.0 + np.exp(twi_exp))

        physics_score = 0.4 * hand_score + 0.4 * slope_score + 0.2 * twi_score

        return physics_score.astype(np.float32), veto

    def extract_features(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract features for LightGBM."""
        vv = data["vv"]
        vh = data["vh"]
        dem = data["dem"]
        slope = data["slope"]
        hand = data["hand"]
        twi = data["twi"]
        mndwi = data.get("mndwi", None)

        features = []

        # Basic features
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

        # Otsu
        vv_otsu = np.median(vv)
        vh_otsu = np.median(vh)
        features.append(vv - vv_otsu)
        features.append(vh - vh_otsu)

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

        # MNDWI features (if available and model expects them)
        has_mndwi_features = self.feature_names and "MNDWI" in self.feature_names

        if has_mndwi_features:
            if mndwi is not None:
                features.append(mndwi)
                features.append((mndwi > 0).astype(np.float32))
                mndwi_mean = uniform_filter(mndwi, size=5)
                mndwi_sq = uniform_filter(mndwi**2, size=5)
                mndwi_var = np.maximum(mndwi_sq - mndwi_mean**2, 0)
                features.append(mndwi_mean)
                features.append(np.sqrt(mndwi_var))
            else:
                # Fill with zeros if MNDWI not available but model expects it
                logger.warning("MNDWI not available, using zeros")
                h, w = vv.shape
                features.extend([np.zeros((h, w), dtype=np.float32)] * 4)

        feature_stack = np.stack(features, axis=-1)
        feature_stack = np.nan_to_num(feature_stack, nan=0.0, posinf=0.0, neginf=0.0)

        return feature_stack.astype(np.float32)

    def predict_lgb(self, features: np.ndarray) -> np.ndarray:
        """Run LightGBM prediction."""
        if self.lgb_model is None:
            raise ValueError("LightGBM model not loaded")

        h, w, n = features.shape
        X = features.reshape(-1, n)
        proba = self.lgb_model.predict(X).reshape(h, w)
        return proba.astype(np.float32)

    def predict_unet(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Run U-Net prediction."""
        if self.unet_model is None:
            raise ValueError("U-Net model not loaded")

        import torch

        vv = data["vv"]
        vh = data["vh"]
        dem = data["dem"]
        slope = data["slope"]
        hand = data["hand"]
        twi = data["twi"]

        h, w = vv.shape

        # Pad to multiple of 16
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16

        inputs = np.stack([vv, vh, dem, slope, hand, twi], axis=0)
        if pad_h > 0 or pad_w > 0:
            inputs = np.pad(inputs, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

        # Normalize each channel
        for i in range(6):
            mean = inputs[i].mean()
            std = inputs[i].std() + 1e-6
            inputs[i] = (inputs[i] - mean) / std

        # Inference
        with torch.no_grad():
            x = torch.from_numpy(inputs[np.newaxis]).float()
            out = self.unet_model(x)
            proba = torch.sigmoid(out).numpy()[0, 0]

        # Remove padding
        proba = proba[:h, :w]

        return proba.astype(np.float32)

    def detect_edges(self, proba: np.ndarray, threshold: float = 0.2) -> np.ndarray:
        """Detect edge regions where U-Net should be weighted more."""
        grad_x = sobel(proba, axis=0)
        grad_y = sobel(proba, axis=1)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        return (gradient > threshold).astype(np.float32)

    def remove_small_regions(self, mask: np.ndarray, min_size: int = 50) -> np.ndarray:
        """Remove small connected components."""
        labeled, num = scipy_label(mask)
        if num == 0:
            return mask

        cleaned = np.zeros_like(mask, dtype=bool)
        for i in range(1, num + 1):
            region = labeled == i
            if region.sum() >= min_size:
                cleaned |= region

        return cleaned.astype(np.float32)

    def detect(
        self,
        data: Dict[str, np.ndarray],
        use_unet: bool = True,
        threshold: float = 0.5,
        min_region_size: int = 50,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Full ensemble detection.

        Args:
            data: Dict with 'vv', 'vh', 'dem', 'slope', 'hand', 'twi', optional 'mndwi'
            use_unet: Whether to use U-Net if available
            threshold: Binary threshold
            min_region_size: Minimum water region size in pixels

        Returns:
            water_mask: Binary water mask
            metadata: Detection statistics
        """
        metadata = {"method": []}

        # Validate data
        validated = self.validate_data(data)

        # Compute physics
        physics_score, veto_mask = self.compute_physics_score(
            validated["hand"], validated["slope"], validated["twi"]
        )
        metadata["physics_veto_fraction"] = float(veto_mask.mean())

        # LightGBM prediction
        lgb_proba = None
        if self.lgb_model is not None:
            features = self.extract_features(validated)
            lgb_proba = self.predict_lgb(features)
            metadata["method"].append("lgb")
            metadata["lgb_mean"] = float(lgb_proba.mean())

        # U-Net prediction
        unet_proba = None
        if use_unet and self.unet_model is not None:
            try:
                unet_proba = self.predict_unet(validated)
                metadata["method"].append("unet")
                metadata["unet_mean"] = float(unet_proba.mean())
            except Exception as e:
                logger.warning(f"U-Net failed: {e}")

        # Ensemble
        if lgb_proba is not None and unet_proba is not None:
            # Adaptive weighting: trust U-Net more on edges
            edge_mask = self.detect_edges(lgb_proba)

            # On edges: 40% LGB, 50% UNet, 10% physics
            # On bulk: 60% LGB, 30% UNet, 10% physics
            lgb_w = np.where(edge_mask > 0.5, 0.4, self.lgb_weight)
            unet_w = np.where(edge_mask > 0.5, 0.5, self.unet_weight)
            phys_w = self.physics_weight

            combined = lgb_w * lgb_proba + unet_w * unet_proba + phys_w * physics_score
            metadata["method"].append("ensemble")
        elif lgb_proba is not None:
            combined = (
                1 - self.physics_weight
            ) * lgb_proba + self.physics_weight * physics_score
        elif unet_proba is not None:
            combined = (
                1 - self.physics_weight
            ) * unet_proba + self.physics_weight * physics_score
        else:
            # Fallback to simple thresholding
            vv = validated["vv"]
            combined = 1.0 / (1.0 + np.exp((vv + 18) / 2.0))
            metadata["method"].append("threshold_fallback")

        # Apply physics VETO
        combined = np.where(veto_mask, 0.0, combined)

        # Threshold
        water_mask = (combined > threshold).astype(np.float32)

        # Remove small regions
        water_mask = self.remove_small_regions(water_mask, min_region_size)

        metadata["water_fraction"] = float(water_mask.mean())
        metadata["combined_mean"] = float(combined.mean())

        return water_mask, metadata


def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute IoU score."""
    pred_bin = pred > 0.5
    target_bin = target > 0.5
    intersection = np.logical_and(pred_bin, target_bin).sum()
    union = np.logical_or(pred_bin, target_bin).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def main():
    """Test ensemble on sample chips."""
    import glob

    logger.info("=" * 70)
    logger.info("ENSEMBLE V8 - Testing")
    logger.info("=" * 70)

    # Load models
    lgb_path = "/home/mit-aoe/sar_water_detection/models/lightgbm_v7_clean.txt"
    unet_path = "/home/mit-aoe/sar_water_detection/models/unet_v4_best.pth"

    detector = EnsembleDetectorV8(
        lgb_model_path=lgb_path,
        unet_model_path=unet_path,
        lgb_weight=0.6,
        unet_weight=0.3,
        physics_weight=0.1,
    )

    # Test on clean chips
    chips = sorted(
        glob.glob(
            "/home/mit-aoe/sar_water_detection/chips_expanded_npy/*_with_truth.npy"
        )
    )[:10]

    results = []
    for chip_path in chips:
        name = chip_path.split("/")[-1].replace("_with_truth.npy", "")

        try:
            data = np.load(chip_path)
            chip_data = {
                "vv": data[:, :, 0],
                "vh": data[:, :, 1],
                "dem": data[:, :, 2],
                "slope": data[:, :, 3],
                "hand": data[:, :, 4],
                "twi": data[:, :, 5],
            }
            if data.shape[2] > 7:
                chip_data["mndwi"] = data[:, :, 7]

            label = data[:, :, 6]

            # Detect
            pred, meta = detector.detect(chip_data, use_unet=True)

            # Compute metrics
            iou = compute_iou(pred, label)

            results.append(
                {
                    "chip": name,
                    "iou": iou,
                    "method": meta["method"],
                    "water_fraction": meta["water_fraction"],
                }
            )

            logger.info(f"{name}: IoU={iou:.4f}, method={meta['method']}")

        except Exception as e:
            logger.error(f"{name}: ERROR - {e}")

    # Summary
    if results:
        avg_iou = np.mean([r["iou"] for r in results])
        logger.info(f"\nAverage IoU: {avg_iou:.4f}")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
