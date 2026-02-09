#!/usr/bin/env python3
"""
Full Ensemble Evaluation Script
===============================
Loads LightGBM and U-Net models and evaluates the combined ensemble.

Models:
- LightGBM v4: lightgbm_v4_comprehensive.txt (Test IoU 0.881)
- U-Net v4: unet_v4_best.pth (Test IoU 0.766)
- Physics: Best equations from search

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
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("full_ensemble_eval.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "chip_dir": Path("/home/mit-aoe/sar_water_detection/chips"),
    "model_dir": Path("/home/mit-aoe/sar_water_detection/models"),
    "output_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    # Model paths
    "lightgbm_model": "lightgbm_v4_comprehensive.txt",
    "unet_model": "unet_v4_best.pth",
    # Ensemble weights
    "weights": {
        "lightgbm": 0.50,
        "unet": 0.30,
        "physics": 0.20,
    },
    # Terrain thresholds
    "terrain_thresholds": {
        "flat_lowland": {"vh": -18, "hand": 15, "slope": 10},
        "hilly": {"vh": -17, "hand": 25, "slope": 20},
        "mountainous": {"vh": -16, "hand": 100, "slope": 30},
        "arid": {"vh": -20, "hand": 8, "slope": 5},
        "urban": {"vh": -20, "hand": 8, "slope": 10},
        "wetland": {"vh": -14, "hand": 20, "slope": 8},
    },
}


# =============================================================================
# Feature Engineering (same as training)
# =============================================================================


def compute_features(
    data: np.ndarray, chip_name: str = ""
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Extract features matching LightGBM v4 training.

    Returns: X (features), valid_mask, feature_dict
    """
    h, w = data.shape[:2]
    n_bands = data.shape[2] if len(data.shape) == 3 else 1

    # Extract bands
    vv = data[:, :, 0] if n_bands > 0 else np.zeros((h, w))
    vh = data[:, :, 1] if n_bands > 1 else np.zeros((h, w))
    dem = data[:, :, 3] if n_bands > 3 else np.zeros((h, w))
    hand = data[:, :, 4] if n_bands > 4 else np.zeros((h, w))
    slope = data[:, :, 5] if n_bands > 5 else np.zeros((h, w))
    twi = data[:, :, 6] if n_bands > 6 else np.zeros((h, w))

    features = {}

    # Core SAR features
    features["vv"] = vv.flatten()
    features["vh"] = vh.flatten()
    features["vv_vh_ratio"] = (vv - vh).flatten()

    # Terrain features
    features["dem"] = dem.flatten()
    features["hand"] = hand.flatten()
    features["slope"] = slope.flatten()
    features["twi"] = twi.flatten()

    # Multi-scale texture features
    for scale in [3, 5, 9, 15, 21]:
        # VH statistics
        vh_mean = uniform_filter(vh, size=scale)
        vh_sq_mean = uniform_filter(vh**2, size=scale)
        vh_std = np.sqrt(np.maximum(vh_sq_mean - vh_mean**2, 0))

        features[f"vh_mean_s{scale}"] = vh_mean.flatten()
        features[f"vh_std_s{scale}"] = vh_std.flatten()

        # VV statistics
        vv_mean = uniform_filter(vv, size=scale)
        vv_sq_mean = uniform_filter(vv**2, size=scale)
        vv_std = np.sqrt(np.maximum(vv_sq_mean - vv_mean**2, 0))

        features[f"vv_mean_s{scale}"] = vv_mean.flatten()
        features[f"vv_std_s{scale}"] = vv_std.flatten()

    # Morphological features (using uniform filter as proxy)
    vh_closed = uniform_filter(np.maximum(vh, uniform_filter(vh, 5)), 5)
    vh_opened = uniform_filter(np.minimum(vh, uniform_filter(vh, 5)), 5)
    features["vh_closed"] = vh_closed.flatten()
    features["vh_opened"] = vh_opened.flatten()
    features["vh_morph_gradient"] = (vh_closed - vh_opened).flatten()

    # Edge features
    gy, gx = np.gradient(vh)
    features["vh_gradient_mag"] = np.sqrt(gx**2 + gy**2).flatten()

    # HAND-based features
    features["hand_log"] = np.log1p(np.maximum(hand, 0)).flatten()
    features["hand_water_prob"] = (1.0 / (1.0 + np.exp((hand - 10) / 3.0))).flatten()

    # Slope features
    features["slope_water_prob"] = (1.0 / (1.0 + np.exp((slope - 8) / 3.0))).flatten()

    # TWI features
    features["twi_norm"] = ((twi - np.nanmean(twi)) / (np.nanstd(twi) + 1e-6)).flatten()

    # DEM features
    dem_local = uniform_filter(dem, 15)
    features["dem_local_relief"] = (dem - dem_local).flatten()

    # Create feature matrix
    feature_names = list(features.keys())
    X = np.column_stack([features[name] for name in feature_names])

    # Handle NaN
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Valid mask
    valid_mask = ~np.isnan(vv.flatten()) & ~np.isnan(vh.flatten())

    return X, valid_mask, {"names": feature_names, "shape": (h, w)}


# =============================================================================
# Terrain Classifier
# =============================================================================


class TerrainClassifier:
    """Classify terrain from features."""

    @staticmethod
    def classify(
        vv: np.ndarray,
        vh: np.ndarray,
        dem: np.ndarray,
        slope: np.ndarray,
        twi: np.ndarray,
        chip_name: str = "",
    ) -> str:
        """Classify terrain type."""
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

        if dem_mean > 2000:
            return "mountainous"
        elif slope_mean > 12 or slope_p90 > 25:
            return "hilly"
        elif vv_mean > -12 and (vv_mean - vh_mean) > 7:
            return "urban"
        elif twi_mean > 10:
            return "wetland"
        elif twi_mean < 5 and vh_mean < -22:
            return "arid"
        else:
            return "flat_lowland"


# =============================================================================
# Physics Detector
# =============================================================================


class PhysicsDetector:
    """Physics-based water detection."""

    def detect(
        self,
        vv: np.ndarray,
        vh: np.ndarray,
        hand: np.ndarray,
        slope: np.ndarray,
        twi: np.ndarray,
        terrain: str,
    ) -> np.ndarray:
        """Multi-tier physics detection."""
        thresholds = CONFIG["terrain_thresholds"].get(
            terrain, CONFIG["terrain_thresholds"]["flat_lowland"]
        )

        prob = np.zeros_like(vh, dtype=np.float32)

        # Tier 1: Core water
        core = (vh < -24) & (hand < thresholds["hand"]) & (slope < thresholds["slope"])
        prob = np.where(core, 0.95, prob)

        # Tier 2: Standard water
        standard = (
            (vh < thresholds["vh"])
            & (hand < thresholds["hand"])
            & (slope < thresholds["slope"])
        )
        prob = np.where(standard & ~core, 0.85, prob)

        # Tier 3: Extended (bright water)
        extended = (
            (vh >= thresholds["vh"])
            & (vh < thresholds["vh"] + 6)
            & (hand < 5)
            & (slope < 3)
            & (twi > 7)
        )
        prob = np.where(extended, 0.70, prob)

        # Urban exclusion
        urban = (vv > -10) & ((vv - vh) > 8)
        prob = np.where(urban, prob * 0.3, prob)

        return prob


# =============================================================================
# LightGBM Predictor
# =============================================================================


class LightGBMPredictor:
    """LightGBM model predictor."""

    def __init__(self, model_path: Path):
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load LightGBM booster."""
        try:
            import lightgbm as lgb

            self.model = lgb.Booster(model_file=str(self.model_path))
            logger.info(f"Loaded LightGBM from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load LightGBM: {e}")

    def predict(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predict water probability."""
        if self.model is None:
            return None
        try:
            probs = self.model.predict(X)
            return probs
        except Exception as e:
            logger.warning(f"LightGBM prediction failed: {e}")
            return None


# =============================================================================
# U-Net Predictor
# =============================================================================


class UNetPredictor:
    """U-Net model predictor."""

    def __init__(self, model_path: Path):
        self.model = None
        self.device = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load PyTorch U-Net."""
        try:
            import torch
            import torch.nn as nn

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Build model architecture
            self.model = self._build_unet()

            # Load weights
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded U-Net from {self.model_path} on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load U-Net: {e}")
            import traceback

            traceback.print_exc()

    def _build_unet(self):
        """Build U-Net architecture matching training."""
        import torch
        import torch.nn as nn

        class DoubleConv(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x):
                return self.conv(x)

        class UNet(nn.Module):
            def __init__(self, in_channels=6, out_channels=1):
                super().__init__()
                self.enc1 = DoubleConv(in_channels, 64)
                self.enc2 = DoubleConv(64, 128)
                self.enc3 = DoubleConv(128, 256)
                self.enc4 = DoubleConv(256, 512)

                self.pool = nn.MaxPool2d(2)

                self.bottleneck = DoubleConv(512, 1024)

                self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
                self.dec4 = DoubleConv(1024, 512)
                self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
                self.dec3 = DoubleConv(512, 256)
                self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
                self.dec2 = DoubleConv(256, 128)
                self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                self.dec1 = DoubleConv(128, 64)

                self.out = nn.Conv2d(64, out_channels, 1)

            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                e4 = self.enc4(self.pool(e3))

                b = self.bottleneck(self.pool(e4))

                d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
                d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
                d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
                d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

                return self.out(d1)

        return UNet(in_channels=6, out_channels=1)

    def predict(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Predict water probability."""
        if self.model is None:
            return None

        try:
            import torch

            h, w = data.shape[:2]
            n_bands = data.shape[2] if len(data.shape) == 3 else 1

            # Extract 6 channels: vv, vh, dem, hand, slope, twi
            channels = []
            for i in [0, 1, 3, 4, 5, 6]:  # Skip band 2 (nasadem)
                if i < n_bands:
                    ch = data[:, :, i].astype(np.float32)
                    ch = np.nan_to_num(ch, nan=0)
                    channels.append(ch)

            while len(channels) < 6:
                channels.append(np.zeros((h, w), dtype=np.float32))

            # Stack and normalize
            x = np.stack(channels, axis=0)

            # Simple normalization
            for i in range(6):
                mean = np.mean(x[i])
                std = np.std(x[i]) + 1e-6
                x[i] = (x[i] - mean) / std

            # To tensor
            x = torch.from_numpy(x).unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                pred = self.model(x)
                pred = torch.sigmoid(pred)
                pred = pred.squeeze().cpu().numpy()

            return pred

        except Exception as e:
            logger.warning(f"U-Net prediction failed: {e}")
            import traceback

            traceback.print_exc()
            return None


# =============================================================================
# Ensemble Evaluator
# =============================================================================


class EnsembleEvaluator:
    """Full ensemble evaluation."""

    def __init__(self):
        self.physics = PhysicsDetector()
        self.lightgbm = LightGBMPredictor(
            CONFIG["model_dir"] / CONFIG["lightgbm_model"]
        )
        self.unet = UNetPredictor(CONFIG["model_dir"] / CONFIG["unet_model"])
        self.weights = CONFIG["weights"]

    def predict(self, data: np.ndarray, chip_name: str = "") -> np.ndarray:
        """Combined ensemble prediction."""
        h, w = data.shape[:2]
        n_bands = data.shape[2] if len(data.shape) == 3 else 1

        # Extract bands
        vv = data[:, :, 0] if n_bands > 0 else np.zeros((h, w))
        vh = data[:, :, 1] if n_bands > 1 else np.zeros((h, w))
        dem = data[:, :, 3] if n_bands > 3 else np.zeros((h, w))
        hand = data[:, :, 4] if n_bands > 4 else np.zeros((h, w))
        slope = data[:, :, 5] if n_bands > 5 else np.zeros((h, w))
        twi = data[:, :, 6] if n_bands > 6 else np.zeros((h, w))

        # Handle NaN
        vv = np.nan_to_num(vv, nan=-20)
        vh = np.nan_to_num(vh, nan=-25)
        hand = np.nan_to_num(hand, nan=50)
        slope = np.nan_to_num(slope, nan=30)
        twi = np.nan_to_num(twi, nan=5)
        dem = np.nan_to_num(dem, nan=100)

        # Classify terrain
        terrain = TerrainClassifier.classify(vv, vh, dem, slope, twi, chip_name)

        # Get predictions
        predictions = {}
        weights = {}

        # Physics prediction
        physics_pred = self.physics.detect(vv, vh, hand, slope, twi, terrain)
        predictions["physics"] = physics_pred
        weights["physics"] = self.weights["physics"]

        # LightGBM prediction
        X, valid_mask, meta = compute_features(data, chip_name)
        lgb_pred = self.lightgbm.predict(X)
        if lgb_pred is not None:
            predictions["lightgbm"] = lgb_pred.reshape(h, w)
            weights["lightgbm"] = self.weights["lightgbm"]

        # U-Net prediction
        unet_pred = self.unet.predict(data)
        if unet_pred is not None:
            predictions["unet"] = unet_pred
            weights["unet"] = self.weights["unet"]

        # Normalize weights
        total = sum(weights.values())
        for k in weights:
            weights[k] /= total

        # Combine
        combined = np.zeros((h, w), dtype=np.float32)
        for name, pred in predictions.items():
            combined += pred * weights[name]

        # Apply physics safety
        thresholds = CONFIG["terrain_thresholds"].get(
            terrain, CONFIG["terrain_thresholds"]["flat_lowland"]
        )
        hand_penalty = 1.0 / (1.0 + np.exp((hand - thresholds["hand"]) / 5.0))
        slope_penalty = np.clip(1.0 - (slope - thresholds["slope"]) / 15.0, 0.3, 1.0)
        combined = combined * hand_penalty * slope_penalty

        # Hard cutoff
        combined = np.where(slope > 40, 0.0, combined)

        return np.clip(combined, 0, 1)


def compute_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """Compute metrics."""
    pred_bool = pred > 0.5
    truth_bool = truth > 0.5

    tp = np.sum(pred_bool & truth_bool)
    fp = np.sum(pred_bool & ~truth_bool)
    fn = np.sum(~pred_bool & truth_bool)

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    iou = tp / (tp + fp + fn + 1e-10)

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main():
    """Main evaluation."""
    logger.info("=" * 60)
    logger.info("FULL ENSEMBLE EVALUATION")
    logger.info("=" * 60)

    # Initialize
    evaluator = EnsembleEvaluator()

    # Find chips
    chip_files = list(CONFIG["chip_dir"].glob("*_with_truth.npy"))
    logger.info(f"Found {len(chip_files)} chips")

    # Evaluate
    all_results = []
    terrain_results = {}

    for chip_file in chip_files:
        try:
            data = np.load(chip_file)
            if len(data.shape) == 3 and data.shape[0] < data.shape[2]:
                data = np.transpose(data, (1, 2, 0))

            n_bands = data.shape[2] if len(data.shape) == 3 else 1
            if n_bands < 7:
                continue

            truth = data[:, :, 7] if n_bands > 7 else data[:, :, 6]

            # Predict
            pred = evaluator.predict(data, chip_file.stem)

            # Metrics
            metrics = compute_metrics(pred, truth)
            metrics["chip"] = chip_file.stem

            # Classify terrain
            vv = data[:, :, 0]
            vh = data[:, :, 1]
            dem = data[:, :, 3]
            slope = data[:, :, 5]
            twi = data[:, :, 6]
            terrain = TerrainClassifier.classify(vv, vh, dem, slope, twi)
            metrics["terrain"] = terrain

            all_results.append(metrics)

            if terrain not in terrain_results:
                terrain_results[terrain] = []
            terrain_results[terrain].append(metrics)

            logger.info(f"{chip_file.stem}: IoU={metrics['iou']:.4f} [{terrain}]")

        except Exception as e:
            logger.warning(f"Failed {chip_file.name}: {e}")

    # Summary
    if all_results:
        mean_iou = np.mean([r["iou"] for r in all_results])
        std_iou = np.std([r["iou"] for r in all_results])
        mean_f1 = np.mean([r["f1"] for r in all_results])

        logger.info("\n" + "=" * 60)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total chips: {len(all_results)}")
        logger.info(f"Mean IoU: {mean_iou:.4f} +/- {std_iou:.4f}")
        logger.info(f"Mean F1: {mean_f1:.4f}")

        logger.info("\nPer-terrain:")
        for terrain, results in terrain_results.items():
            t_iou = np.mean([r["iou"] for r in results])
            logger.info(f"  {terrain}: IoU={t_iou:.4f} (n={len(results)})")

        # Save
        CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
        output = {
            "mean_iou": mean_iou,
            "std_iou": std_iou,
            "mean_f1": mean_f1,
            "n_chips": len(all_results),
            "per_terrain": {
                t: {"mean_iou": np.mean([r["iou"] for r in rs]), "count": len(rs)}
                for t, rs in terrain_results.items()
            },
            "all_results": all_results,
        }

        with open(CONFIG["output_dir"] / "full_ensemble_results.json", "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nSaved to {CONFIG['output_dir'] / 'full_ensemble_results.json'}")


if __name__ == "__main__":
    main()
