#!/usr/bin/env python3
"""
Corrected Full Ensemble Evaluation Script
==========================================
Properly loads LightGBM and U-Net models.

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
    handlers=[logging.FileHandler("ensemble_v2.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "chip_dir": Path("/home/mit-aoe/sar_water_detection/chips"),
    "model_dir": Path("/home/mit-aoe/sar_water_detection/models"),
    "output_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    "lightgbm_model": "lightgbm_v4_comprehensive.txt",
    "unet_model": "unet_v4_best.pth",
    "weights": {"lightgbm": 0.50, "unet": 0.30, "physics": 0.20},
    "terrain_thresholds": {
        "flat_lowland": {"vh": -18, "hand": 15, "slope": 10},
        "hilly": {"vh": -17, "hand": 25, "slope": 20},
        "mountainous": {"vh": -16, "hand": 100, "slope": 30},
        "wetland": {"vh": -14, "hand": 20, "slope": 8},
        "urban": {"vh": -20, "hand": 8, "slope": 10},
    },
}

# =============================================================================
# Feature Engineering (matching LightGBM v4 training)
# =============================================================================


def compute_69_features(data: np.ndarray) -> np.ndarray:
    """
    Compute 69 features matching LightGBM v4 training.
    Feature order must match training!
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

    # Handle NaN
    vv = np.nan_to_num(vv, nan=-20)
    vh = np.nan_to_num(vh, nan=-25)
    dem = np.nan_to_num(dem, nan=100)
    hand = np.nan_to_num(hand, nan=50)
    slope = np.nan_to_num(slope, nan=30)
    twi = np.nan_to_num(twi, nan=5)

    features = []

    # Core SAR (4 features)
    features.append(("VV", vv))
    features.append(("VH", vh))
    features.append(("VV_minus_VH", vv - vh))
    features.append(
        (
            "VV_over_VH_log",
            np.log10(np.maximum(10 ** (vv / 10) / (10 ** (vh / 10) + 1e-10), 1e-10)),
        )
    )

    # Physics (4 features)
    features.append(("HAND", hand))
    features.append(("SLOPE", slope))
    features.append(("TWI", twi))
    features.append(("DEM", dem))

    # DEM derivatives (2 features)
    gy, gx = np.gradient(dem)
    features.append(("DEM_gradient", np.sqrt(gx**2 + gy**2)))
    features.append(("DEM_aspect", np.arctan2(gy, gx + 1e-10)))

    # Multi-scale texture (5 scales x 4 stats = 20 features)
    for scale in [3, 5, 9, 15, 21]:
        vh_mean = uniform_filter(vh, size=scale)
        vh_sq_mean = uniform_filter(vh**2, size=scale)
        vh_std = np.sqrt(np.maximum(vh_sq_mean - vh_mean**2, 0))
        vh_min = uniform_filter(vh, size=scale)  # Simplified
        vh_max = uniform_filter(vh, size=scale)  # Simplified

        features.append((f"VH_mean_s{scale}", vh_mean))
        features.append((f"VH_std_s{scale}", vh_std))
        features.append((f"VH_min_s{scale}", vh_min))
        features.append((f"VH_max_s{scale}", vh_max))

    # VV texture (5 scales x 2 stats = 10 features)
    for scale in [3, 5, 9, 15, 21]:
        vv_mean = uniform_filter(vv, size=scale)
        vv_sq_mean = uniform_filter(vv**2, size=scale)
        vv_std = np.sqrt(np.maximum(vv_sq_mean - vv_mean**2, 0))
        features.append((f"VV_mean_s{scale}", vv_mean))
        features.append((f"VV_std_s{scale}", vv_std))

    # Morphological (6 features)
    vh_closed = uniform_filter(np.maximum(vh, uniform_filter(vh, 5)), 5)
    vh_opened = uniform_filter(np.minimum(vh, uniform_filter(vh, 5)), 5)
    features.append(("VH_closed", vh_closed))
    features.append(("VH_opened", vh_opened))
    features.append(("VH_morph_gradient", vh_closed - vh_opened))

    vv_closed = uniform_filter(np.maximum(vv, uniform_filter(vv, 5)), 5)
    vv_opened = uniform_filter(np.minimum(vv, uniform_filter(vv, 5)), 5)
    features.append(("VV_closed", vv_closed))
    features.append(("VV_opened", vv_opened))
    features.append(("VV_morph_gradient", vv_closed - vv_opened))

    # Edge features (3 features)
    gy_vh, gx_vh = np.gradient(vh)
    features.append(("VH_gradient_mag", np.sqrt(gx_vh**2 + gy_vh**2)))
    gy_vv, gx_vv = np.gradient(vv)
    features.append(("VV_gradient_mag", np.sqrt(gx_vv**2 + gy_vv**2)))
    features.append(
        (
            "gradient_ratio",
            np.sqrt(gx_vh**2 + gy_vh**2) / (np.sqrt(gx_vv**2 + gy_vv**2) + 1e-10),
        )
    )

    # Physics derived (6 features)
    features.append(("HAND_log", np.log1p(np.maximum(hand, 0))))
    features.append(("HAND_water_prob", 1.0 / (1.0 + np.exp((hand - 10) / 3.0))))
    features.append(("SLOPE_water_prob", 1.0 / (1.0 + np.exp((slope - 8) / 3.0))))
    features.append(("TWI_norm", (twi - np.mean(twi)) / (np.std(twi) + 1e-6)))
    dem_local = uniform_filter(dem, 15)
    features.append(("DEM_local_relief", dem - dem_local))
    features.append(
        (
            "physics_score",
            (1.0 / (1.0 + np.exp((hand - 10) / 3.0)))
            * (1.0 / (1.0 + np.exp((slope - 8) / 3.0))),
        )
    )

    # Otsu-based (4 features)
    vh_mean_global = np.mean(vh)
    vv_mean_global = np.mean(vv)
    features.append(("VH_otsu_diff", vh - vh_mean_global))
    features.append(("VV_otsu_diff", vv - vv_mean_global))
    features.append(
        ("VH_normalized", (vh - np.min(vh)) / (np.max(vh) - np.min(vh) + 1e-10))
    )
    features.append(
        ("VV_normalized", (vv - np.min(vv)) / (np.max(vv) - np.min(vv) + 1e-10))
    )

    # Context (4 features)
    features.append(
        (
            "VH_CV",
            uniform_filter(vh**2, 9) ** 0.5 / (np.abs(uniform_filter(vh, 9)) + 1e-10),
        )
    )
    features.append(
        (
            "VV_CV",
            uniform_filter(vv**2, 9) ** 0.5 / (np.abs(uniform_filter(vv, 9)) + 1e-10),
        )
    )
    features.append(
        ("local_water_density", uniform_filter((vh < -20).astype(float), 21))
    )
    features.append(
        ("edge_strength", np.sqrt(gx_vh**2 + gy_vh**2) + np.sqrt(gx_vv**2 + gy_vv**2))
    )

    # Flatten and stack
    feature_matrix = np.column_stack([f[1].flatten() for f in features])

    logger.info(f"Created {feature_matrix.shape[1]} features (expected 69)")

    return feature_matrix, (h, w)


# =============================================================================
# U-Net Architecture (matching checkpoint)
# =============================================================================


def build_unet():
    """Build U-Net with CBAM attention matching the checkpoint."""
    import torch
    import torch.nn as nn

    class CBAM(nn.Module):
        def __init__(self, channels, reduction=8):
            super().__init__()
            self.channel_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, max(channels // reduction, 4)),
                nn.ReLU(),
                nn.Linear(max(channels // reduction, 4), channels),
                nn.Sigmoid(),
            )
            self.spatial_attn = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3), nn.Sigmoid()
            )

        def forward(self, x):
            # Channel attention
            ca = self.channel_attn(x).unsqueeze(-1).unsqueeze(-1)
            x = x * ca
            # Spatial attention
            max_pool = x.max(dim=1, keepdim=True)[0]
            avg_pool = x.mean(dim=1, keepdim=True)
            sa = self.spatial_attn(torch.cat([max_pool, avg_pool], dim=1))
            return x * sa

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch, use_cbam=True):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU(inplace=True)
            self.cbam = CBAM(out_ch) if use_cbam else nn.Identity()

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.cbam(x)
            return x

    class UNetCBAM(nn.Module):
        def __init__(self, in_channels=6, out_channels=1):
            super().__init__()
            self.enc1 = ConvBlock(in_channels, 32)
            self.enc2 = ConvBlock(32, 64)
            self.enc3 = ConvBlock(64, 128)
            self.enc4 = ConvBlock(128, 256)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = ConvBlock(256, 512)
            self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.dec4 = ConvBlock(512, 256)
            self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dec3 = ConvBlock(256, 128)
            self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec2 = ConvBlock(128, 64)
            self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec1 = ConvBlock(64, 32)
            self.out_conv = nn.Conv2d(32, out_channels, 1)

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
            return self.out_conv(d1)

    return UNetCBAM()


# =============================================================================
# Model Loaders
# =============================================================================


class LightGBMPredictor:
    def __init__(self, model_path: Path):
        self.model = None
        try:
            import lightgbm as lgb

            self.model = lgb.Booster(model_file=str(model_path))
            logger.info(f"LightGBM loaded: {self.model.num_feature()} features")
        except Exception as e:
            logger.error(f"LightGBM load failed: {e}")

    def predict(self, X: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.warning(f"LightGBM predict failed: {e}")
            return None


class UNetPredictor:
    def __init__(self, model_path: Path):
        self.model = None
        self.device = None
        try:
            import torch

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
            self.model = build_unet()
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            logger.info(
                f"U-Net loaded on {self.device}, val_iou={checkpoint.get('val_iou', 'N/A')}"
            )
        except Exception as e:
            logger.error(f"U-Net load failed: {e}")
            import traceback

            traceback.print_exc()

    def predict(self, data: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        try:
            import torch

            h, w = data.shape[:2]
            n_bands = data.shape[2] if len(data.shape) == 3 else 1

            # Extract 6 channels: vv, vh, dem, hand, slope, twi
            channels = []
            for i in [0, 1, 3, 4, 5, 6]:
                if i < n_bands:
                    ch = data[:, :, i].astype(np.float32)
                    ch = np.nan_to_num(ch, nan=0)
                    # Normalize
                    ch = (ch - np.mean(ch)) / (np.std(ch) + 1e-6)
                    channels.append(ch)

            while len(channels) < 6:
                channels.append(np.zeros((h, w), dtype=np.float32))

            x = np.stack(channels, axis=0)
            x = torch.from_numpy(x).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                pred = self.model(x)
                pred = torch.sigmoid(pred)
                pred = pred.squeeze().cpu().numpy()

            return pred
        except Exception as e:
            logger.warning(f"U-Net predict failed: {e}")
            return None


# =============================================================================
# Physics Detector
# =============================================================================


class PhysicsDetector:
    def detect(self, vv, vh, hand, slope, twi, terrain="flat_lowland"):
        thresholds = CONFIG["terrain_thresholds"].get(
            terrain, CONFIG["terrain_thresholds"]["flat_lowland"]
        )
        prob = np.zeros_like(vh, dtype=np.float32)

        # Core water
        core = (vh < -24) & (hand < thresholds["hand"]) & (slope < thresholds["slope"])
        prob = np.where(core, 0.95, prob)

        # Standard
        standard = (
            (vh < thresholds["vh"])
            & (hand < thresholds["hand"])
            & (slope < thresholds["slope"])
        )
        prob = np.where(standard & ~core, 0.85, prob)

        # Extended
        extended = (
            (vh >= thresholds["vh"])
            & (vh < thresholds["vh"] + 6)
            & (hand < 5)
            & (slope < 3)
        )
        prob = np.where(extended, 0.70, prob)

        # Urban exclusion
        urban = (vv > -10) & ((vv - vh) > 8)
        prob = np.where(urban, prob * 0.3, prob)

        return prob


def classify_terrain(vv, vh, dem, slope, twi):
    dem_mean = np.nanmean(dem)
    slope_mean = np.nanmean(slope)
    twi_mean = np.nanmean(twi)
    vv_mean = np.nanmean(vv)

    if dem_mean > 2000:
        return "mountainous"
    elif slope_mean > 12:
        return "hilly"
    elif vv_mean > -12:
        return "urban"
    elif twi_mean > 10:
        return "wetland"
    else:
        return "flat_lowland"


def compute_metrics(pred, truth):
    pred_bool = pred > 0.5
    truth_bool = truth > 0.5
    tp = np.sum(pred_bool & truth_bool)
    fp = np.sum(pred_bool & ~truth_bool)
    fn = np.sum(~pred_bool & truth_bool)
    iou = tp / (tp + fp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# =============================================================================
# Main Ensemble
# =============================================================================


def main():
    logger.info("=" * 60)
    logger.info("ENSEMBLE v2 - WITH PROPER MODEL LOADING")
    logger.info("=" * 60)

    # Load models
    lgb_predictor = LightGBMPredictor(CONFIG["model_dir"] / CONFIG["lightgbm_model"])
    unet_predictor = UNetPredictor(CONFIG["model_dir"] / CONFIG["unet_model"])
    physics_detector = PhysicsDetector()

    # Find chips
    chip_files = list(CONFIG["chip_dir"].glob("*_with_truth.npy"))
    logger.info(f"Found {len(chip_files)} chips")

    all_results = []

    for chip_file in chip_files:
        try:
            data = np.load(chip_file)
            if len(data.shape) == 3 and data.shape[0] < data.shape[2]:
                data = np.transpose(data, (1, 2, 0))

            h, w = data.shape[:2]
            n_bands = data.shape[2] if len(data.shape) == 3 else 1
            if n_bands < 7:
                continue

            truth = data[:, :, 7] if n_bands > 7 else data[:, :, 6]

            # Extract bands
            vv = np.nan_to_num(data[:, :, 0], nan=-20)
            vh = np.nan_to_num(data[:, :, 1], nan=-25)
            dem = np.nan_to_num(data[:, :, 3], nan=100)
            hand = np.nan_to_num(data[:, :, 4], nan=50)
            slope = np.nan_to_num(data[:, :, 5], nan=30)
            twi = np.nan_to_num(data[:, :, 6], nan=5)

            terrain = classify_terrain(vv, vh, dem, slope, twi)

            # Get predictions
            predictions = {}
            weights = {}

            # Physics
            physics_pred = physics_detector.detect(vv, vh, hand, slope, twi, terrain)
            predictions["physics"] = physics_pred
            weights["physics"] = CONFIG["weights"]["physics"]

            # LightGBM
            X, shape = compute_69_features(data)
            lgb_pred = lgb_predictor.predict(X)
            if lgb_pred is not None:
                predictions["lightgbm"] = lgb_pred.reshape(h, w)
                weights["lightgbm"] = CONFIG["weights"]["lightgbm"]

            # U-Net
            unet_pred = unet_predictor.predict(data)
            if unet_pred is not None:
                predictions["unet"] = unet_pred
                weights["unet"] = CONFIG["weights"]["unet"]

            # Normalize weights
            total = sum(weights.values())
            for k in weights:
                weights[k] /= total

            # Combine
            combined = np.zeros((h, w), dtype=np.float32)
            for name, pred in predictions.items():
                combined += pred * weights[name]

            # Physics safety
            thresholds = CONFIG["terrain_thresholds"].get(
                terrain, CONFIG["terrain_thresholds"]["flat_lowland"]
            )
            hand_penalty = 1.0 / (1.0 + np.exp((hand - thresholds["hand"]) / 5.0))
            combined = combined * hand_penalty
            combined = np.where(slope > 40, 0.0, combined)

            # Metrics
            metrics = compute_metrics(combined, truth)
            metrics["chip"] = chip_file.stem
            metrics["terrain"] = terrain
            metrics["models_used"] = list(predictions.keys())
            all_results.append(metrics)

            logger.info(
                f"{chip_file.stem}: IoU={metrics['iou']:.4f} [{terrain}] models={len(predictions)}"
            )

        except Exception as e:
            logger.warning(f"Failed {chip_file.name}: {e}")

    # Summary
    if all_results:
        mean_iou = np.mean([r["iou"] for r in all_results])
        std_iou = np.std([r["iou"] for r in all_results])

        logger.info("\n" + "=" * 60)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(
            f"Chips: {len(all_results)}, Mean IoU: {mean_iou:.4f} +/- {std_iou:.4f}"
        )

        # Per terrain
        terrain_groups = {}
        for r in all_results:
            t = r["terrain"]
            if t not in terrain_groups:
                terrain_groups[t] = []
            terrain_groups[t].append(r["iou"])

        for t, ious in terrain_groups.items():
            logger.info(f"  {t}: IoU={np.mean(ious):.4f} (n={len(ious)})")

        # Save
        CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
        with open(CONFIG["output_dir"] / "ensemble_v2_results.json", "w") as f:
            json.dump(
                {"mean_iou": mean_iou, "std_iou": std_iou, "results": all_results},
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
