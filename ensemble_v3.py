#!/usr/bin/env python3
"""
Ensemble v3 - Fixed Feature Engineering and Device Handling
============================================================
Properly computes all 69 features matching LightGBM v4 training.

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
from scipy.ndimage import (
    uniform_filter,
    gaussian_filter,
    minimum_filter,
    maximum_filter,
    laplace,
)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ensemble_v3.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

CONFIG = {
    "chip_dir": Path("/home/mit-aoe/sar_water_detection/chips"),
    "model_dir": Path("/home/mit-aoe/sar_water_detection/models"),
    "output_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    "weights": {"lightgbm": 0.65, "unet": 0.25, "physics": 0.10},
}


def compute_glcm_features(arr: np.ndarray, window: int = 7) -> Dict[str, np.ndarray]:
    """Approximate GLCM features using local statistics."""
    arr_safe = np.nan_to_num(arr, nan=0)

    # Contrast: variance of local differences
    gy, gx = np.gradient(arr_safe)
    contrast = uniform_filter(gx**2 + gy**2, window)

    # Homogeneity: inverse of contrast (normalized)
    homogeneity = 1.0 / (1.0 + contrast)

    # Energy: sum of squared values (local)
    sq_mean = uniform_filter(arr_safe**2, window)
    energy = sq_mean

    return {"contrast": contrast, "homogeneity": homogeneity, "energy": energy}


def compute_morphology(arr: np.ndarray, size: int = 5) -> Dict[str, np.ndarray]:
    """Compute morphological features."""
    arr_safe = np.nan_to_num(arr, nan=0)

    # Dilation/erosion with min/max filters
    dilated = maximum_filter(arr_safe, size=size)
    eroded = minimum_filter(arr_safe, size=size)

    opened = maximum_filter(eroded, size=size)
    closed = minimum_filter(dilated, size=size)

    white_tophat = arr_safe - opened
    black_tophat = closed - arr_safe

    return {
        "opened": opened,
        "closed": closed,
        "white_tophat": white_tophat,
        "black_tophat": black_tophat,
    }


def compute_otsu_threshold(arr: np.ndarray) -> float:
    """Compute Otsu threshold."""
    arr_flat = arr[~np.isnan(arr)].flatten()
    if len(arr_flat) == 0:
        return 0.0

    # Simple Otsu approximation using mean
    hist, bins = np.histogram(arr_flat, bins=256)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    total = len(arr_flat)
    current_max = 0
    threshold = 0
    sum_total = np.sum(bin_centers * hist)

    weight_bg = 0
    sum_bg = 0

    for i, (freq, center) in enumerate(zip(hist, bin_centers)):
        weight_bg += freq
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += freq * center
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        variance_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if variance_between > current_max:
            current_max = variance_between
            threshold = center

    return threshold


def compute_69_features(data: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Compute all 69 features matching LightGBM v4 training.
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

    features = {}

    # 1-4: Core SAR
    features["VV"] = vv
    features["VH"] = vh
    features["VV_minus_VH"] = vv - vh
    vv_lin = 10 ** (vv / 10)
    vh_lin = 10 ** (vh / 10)
    features["VV_over_VH_log"] = np.log10(np.maximum(vv_lin / (vh_lin + 1e-10), 1e-10))

    # 5-8: Physics
    features["HAND"] = hand
    features["SLOPE"] = slope
    features["TWI"] = twi
    features["DEM"] = dem

    # 9-10: DEM derivatives
    gy_dem, gx_dem = np.gradient(dem)
    features["DEM_gradient"] = np.sqrt(gx_dem**2 + gy_dem**2)
    features["DEM_aspect"] = np.arctan2(gy_dem, gx_dem + 1e-10)

    # 11-14: Polarimetric pseudo-features
    features["pseudo_entropy"] = -np.abs(vv - vh) / (np.abs(vv) + np.abs(vh) + 1e-10)
    features["pseudo_alpha"] = np.arctan2(np.abs(vh), np.abs(vv) + 1e-10)
    features["RVI"] = 4 * vh_lin / (vv_lin + vh_lin + 1e-10)
    features["span"] = vv_lin + vh_lin

    # 15-32: Multi-scale texture (scales 3, 5, 9, 15)
    for scale in [3, 5, 9, 15]:
        vv_mean = uniform_filter(vv, size=scale)
        vv_sq = uniform_filter(vv**2, size=scale)
        vv_std = np.sqrt(np.maximum(vv_sq - vv_mean**2, 0))
        vv_min = minimum_filter(vv, size=scale)

        vh_mean = uniform_filter(vh, size=scale)
        vh_sq = uniform_filter(vh**2, size=scale)
        vh_std = np.sqrt(np.maximum(vh_sq - vh_mean**2, 0))
        vh_min = minimum_filter(vh, size=scale)

        features[f"VV_mean_s{scale}"] = vv_mean
        features[f"VV_std_s{scale}"] = vv_std
        features[f"VH_mean_s{scale}"] = vh_mean
        features[f"VH_std_s{scale}"] = vh_std
        if scale <= 9:
            features[f"VV_min_s{scale}"] = vv_min
            features[f"VH_min_s{scale}"] = vh_min

    # 37-40: Scale 21
    scale = 21
    features[f"VV_mean_s{scale}"] = uniform_filter(vv, size=scale)
    features[f"VV_std_s{scale}"] = np.sqrt(
        np.maximum(uniform_filter(vv**2, scale) - uniform_filter(vv, scale) ** 2, 0)
    )
    features[f"VH_mean_s{scale}"] = uniform_filter(vh, size=scale)
    features[f"VH_std_s{scale}"] = np.sqrt(
        np.maximum(uniform_filter(vh**2, scale) - uniform_filter(vh, scale) ** 2, 0)
    )

    # 41-44: ENL and CV
    features["VV_ENL"] = uniform_filter(vv, 9) ** 2 / (
        uniform_filter(vv**2, 9) - uniform_filter(vv, 9) ** 2 + 1e-10
    )
    features["VV_CV"] = np.sqrt(uniform_filter(vv**2, 9)) / (
        np.abs(uniform_filter(vv, 9)) + 1e-10
    )
    features["VH_ENL"] = uniform_filter(vh, 9) ** 2 / (
        uniform_filter(vh**2, 9) - uniform_filter(vh, 9) ** 2 + 1e-10
    )
    features["VH_CV"] = np.sqrt(uniform_filter(vh**2, 9)) / (
        np.abs(uniform_filter(vh, 9)) + 1e-10
    )

    # 45-50: GLCM features
    vv_glcm = compute_glcm_features(vv)
    vh_glcm = compute_glcm_features(vh)
    features["VV_glcm_contrast"] = vv_glcm["contrast"]
    features["VV_glcm_homogeneity"] = vv_glcm["homogeneity"]
    features["VV_glcm_energy"] = vv_glcm["energy"]
    features["VH_glcm_contrast"] = vh_glcm["contrast"]
    features["VH_glcm_homogeneity"] = vh_glcm["homogeneity"]
    features["VH_glcm_energy"] = vh_glcm["energy"]

    # 51-54: Morphological features
    vh_morph = compute_morphology(vh)
    features["VH_opened"] = vh_morph["opened"]
    features["VH_closed"] = vh_morph["closed"]
    features["VH_white_tophat"] = vh_morph["white_tophat"]
    features["VH_black_tophat"] = vh_morph["black_tophat"]

    # 55: Line response (Laplacian-like)
    features["line_response"] = np.abs(laplace(vh))

    # 56-61: Otsu/Kapur thresholds
    vv_otsu = compute_otsu_threshold(vv)
    vh_otsu = compute_otsu_threshold(vh)
    features["VV_otsu_diff"] = vv - vv_otsu
    features["VH_otsu_diff"] = vh - vh_otsu
    features["VV_below_otsu"] = (vv < vv_otsu).astype(float)
    features["VH_below_otsu"] = (vh < vh_otsu).astype(float)
    features["VV_below_kapur"] = (vv < np.percentile(vv, 25)).astype(
        float
    )  # Approximate Kapur
    features["VH_below_kapur"] = (vh < np.percentile(vh, 25)).astype(float)

    # 62-66: Physics composite scores
    hand_score = 1.0 / (1.0 + np.exp((hand - 10) / 3.0))
    slope_score = 1.0 / (1.0 + np.exp((slope - 8) / 3.0))
    vh_score = 1.0 / (1.0 + np.exp((vh + 18) / 2.0))
    twi_score = 1.0 / (1.0 + np.exp((8 - twi) / 2.0))

    features["physics_composite"] = hand_score * slope_score * vh_score * twi_score
    features["hand_score"] = hand_score
    features["slope_score"] = slope_score
    features["vh_score"] = vh_score
    features["twi_score"] = twi_score

    # 67-69: Gradients
    gy_vh, gx_vh = np.gradient(vh)
    gy_vv, gx_vv = np.gradient(vv)
    features["VH_gradient"] = np.sqrt(gx_vh**2 + gy_vh**2)
    features["VV_gradient"] = np.sqrt(gx_vv**2 + gy_vv**2)
    features["VH_laplacian"] = laplace(vh)

    # Create feature matrix in exact order
    feature_order = [
        "VV",
        "VH",
        "VV_minus_VH",
        "VV_over_VH_log",
        "HAND",
        "SLOPE",
        "TWI",
        "DEM",
        "DEM_gradient",
        "DEM_aspect",
        "pseudo_entropy",
        "pseudo_alpha",
        "RVI",
        "span",
        "VV_mean_s3",
        "VV_std_s3",
        "VH_mean_s3",
        "VH_std_s3",
        "VV_min_s3",
        "VH_min_s3",
        "VV_mean_s5",
        "VV_std_s5",
        "VH_mean_s5",
        "VH_std_s5",
        "VV_min_s5",
        "VH_min_s5",
        "VV_mean_s9",
        "VV_std_s9",
        "VH_mean_s9",
        "VH_std_s9",
        "VV_min_s9",
        "VH_min_s9",
        "VV_mean_s15",
        "VV_std_s15",
        "VH_mean_s15",
        "VH_std_s15",
        "VV_mean_s21",
        "VV_std_s21",
        "VH_mean_s21",
        "VH_std_s21",
        "VV_ENL",
        "VV_CV",
        "VH_ENL",
        "VH_CV",
        "VV_glcm_contrast",
        "VV_glcm_homogeneity",
        "VV_glcm_energy",
        "VH_glcm_contrast",
        "VH_glcm_homogeneity",
        "VH_glcm_energy",
        "VH_opened",
        "VH_closed",
        "VH_white_tophat",
        "VH_black_tophat",
        "line_response",
        "VV_otsu_diff",
        "VH_otsu_diff",
        "VV_below_otsu",
        "VH_below_otsu",
        "VV_below_kapur",
        "VH_below_kapur",
        "physics_composite",
        "hand_score",
        "slope_score",
        "vh_score",
        "twi_score",
        "VH_gradient",
        "VV_gradient",
        "VH_laplacian",
    ]

    # Stack in order
    feature_arrays = []
    for name in feature_order:
        arr = features.get(name, np.zeros((h, w)))
        arr = np.nan_to_num(arr, nan=0, posinf=0, neginf=0)
        feature_arrays.append(arr.flatten())

    X = np.column_stack(feature_arrays)
    logger.info(f"Created {X.shape[1]} features")

    return X, (h, w)


def build_unet():
    """Build U-Net with CBAM attention."""
    import torch
    import torch.nn as nn

    class ChannelAttention(nn.Module):
        """Channel attention matching checkpoint: channel_attn.fc.0 and fc.2 (no bias)"""

        def __init__(self, channels, reduction=16):
            super().__init__()
            reduced = max(channels // reduction, 4)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channels, reduced, bias=False),  # fc.0
                nn.ReLU(),  # fc.1 (not saved)
                nn.Linear(reduced, channels, bias=False),  # fc.2
            )

        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return torch.sigmoid(y)

    class SpatialAttention(nn.Module):
        """Spatial attention matching checkpoint: spatial_attn.conv (no bias)"""

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

        def forward(self, x):
            max_pool = x.max(dim=1, keepdim=True)[0]
            avg_pool = x.mean(dim=1, keepdim=True)
            y = torch.cat([max_pool, avg_pool], dim=1)
            return torch.sigmoid(self.conv(y))

    class CBAM(nn.Module):
        def __init__(self, channels, reduction=16):
            super().__init__()
            self.channel_attn = ChannelAttention(channels, reduction)
            self.spatial_attn = SpatialAttention()

        def forward(self, x):
            x = x * self.channel_attn(x)
            x = x * self.spatial_attn(x)
            return x

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU(inplace=True)
            self.cbam = CBAM(out_ch)

        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            return self.cbam(x)

    class UNet(nn.Module):
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

    return UNet()


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
            logger.info(f"U-Net loaded on {self.device}")
        except Exception as e:
            logger.error(f"U-Net load failed: {e}")
            import traceback

            traceback.print_exc()

    def predict(self, data: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        try:
            import torch
            import torch.nn.functional as F

            h, w = data.shape[:2]
            n_bands = data.shape[2] if len(data.shape) == 3 else 1

            channels = []
            for i in [0, 1, 3, 4, 5, 6]:
                if i < n_bands:
                    ch = data[:, :, i].astype(np.float32)
                    ch = np.nan_to_num(ch, nan=0)
                    ch = (ch - np.mean(ch)) / (np.std(ch) + 1e-6)
                    channels.append(ch)

            while len(channels) < 6:
                channels.append(np.zeros((h, w), dtype=np.float32))

            x = np.stack(channels, axis=0)
            x = torch.from_numpy(x).unsqueeze(0).float().to(self.device)

            # Pad to multiple of 16 for U-Net
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

            with torch.no_grad():
                pred = self.model(x)
                pred = torch.sigmoid(pred)
                # Crop back to original size
                pred = pred[:, :, :h, :w]
                pred = pred.squeeze().cpu().numpy()

            return pred
        except Exception as e:
            logger.warning(f"U-Net predict failed: {e}")
            import traceback

            traceback.print_exc()
            return None


class PhysicsDetector:
    def detect(self, vv, vh, hand, slope, twi, terrain="flat_lowland"):
        thresholds = {"vh": -18, "hand": 15, "slope": 10}
        prob = np.zeros_like(vh, dtype=np.float32)

        core = (vh < -24) & (hand < thresholds["hand"]) & (slope < thresholds["slope"])
        prob = np.where(core, 0.95, prob)

        standard = (
            (vh < thresholds["vh"])
            & (hand < thresholds["hand"])
            & (slope < thresholds["slope"])
        )
        prob = np.where(standard & ~core, 0.85, prob)

        extended = (
            (vh >= thresholds["vh"])
            & (vh < thresholds["vh"] + 6)
            & (hand < 5)
            & (slope < 3)
        )
        prob = np.where(extended, 0.70, prob)

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
    return "flat_lowland"


def compute_metrics(pred, truth):
    pred_bool = pred > 0.5
    truth_bool = truth > 0.5
    tp = np.sum(pred_bool & truth_bool)
    fp = np.sum(pred_bool & ~truth_bool)
    fn = np.sum(~pred_bool & truth_bool)
    iou = tp / (tp + fp + fn + 1e-10)
    return {"iou": float(iou)}


def main():
    logger.info("=" * 60)
    logger.info("ENSEMBLE v3 - FIXED FEATURES")
    logger.info("=" * 60)

    lgb_pred = LightGBMPredictor(CONFIG["model_dir"] / "lightgbm_v4_comprehensive.txt")
    unet_pred = UNetPredictor(CONFIG["model_dir"] / "unet_v4_best.pth")
    physics = PhysicsDetector()

    chip_files = list(CONFIG["chip_dir"].glob("*_with_truth.npy"))
    logger.info(f"Found {len(chip_files)} chips")

    results = []

    for chip_file in chip_files:  # Test on ALL chips
        try:
            data = np.load(chip_file)
            if len(data.shape) == 3 and data.shape[0] < data.shape[2]:
                data = np.transpose(data, (1, 2, 0))

            h, w = data.shape[:2]
            n_bands = data.shape[2] if len(data.shape) == 3 else 1
            if n_bands < 7:
                continue

            truth = data[:, :, 7] if n_bands > 7 else data[:, :, 6]

            vv = np.nan_to_num(data[:, :, 0], nan=-20)
            vh = np.nan_to_num(data[:, :, 1], nan=-25)
            dem = np.nan_to_num(data[:, :, 3], nan=100)
            hand = np.nan_to_num(data[:, :, 4], nan=50)
            slope = np.nan_to_num(data[:, :, 5], nan=30)
            twi = np.nan_to_num(data[:, :, 6], nan=5)

            terrain = classify_terrain(vv, vh, dem, slope, twi)

            preds = {}
            weights = {}

            physics_prob = physics.detect(vv, vh, hand, slope, twi, terrain)
            preds["physics"] = physics_prob
            weights["physics"] = CONFIG["weights"]["physics"]

            X, shape = compute_69_features(data)
            lgb_prob = lgb_pred.predict(X)
            if lgb_prob is not None:
                preds["lightgbm"] = lgb_prob.reshape(h, w)
                weights["lightgbm"] = CONFIG["weights"]["lightgbm"]

            unet_prob = unet_pred.predict(data)
            if unet_prob is not None:
                preds["unet"] = unet_prob
                weights["unet"] = CONFIG["weights"]["unet"]

            total = sum(weights.values())
            for k in weights:
                weights[k] /= total

            combined = np.zeros((h, w), dtype=np.float32)
            for name, pred in preds.items():
                combined += pred * weights[name]

            # Soft physics safety - only extreme cases
            # Only penalize very high HAND (>50m) and very steep slopes (>45°)
            hand_penalty = 1.0 / (1.0 + np.exp((hand - 50) / 15.0))
            combined = combined * hand_penalty
            combined = np.where(slope > 45, combined * 0.3, combined)

            metrics = compute_metrics(combined, truth)
            metrics["chip"] = chip_file.stem
            metrics["terrain"] = terrain
            metrics["models"] = list(preds.keys())
            results.append(metrics)

            logger.info(
                f"{chip_file.stem}: IoU={metrics['iou']:.4f} [{terrain}] models={len(preds)}"
            )

        except Exception as e:
            logger.warning(f"Failed {chip_file.name}: {e}")

    if results:
        mean_iou = np.mean([r["iou"] for r in results])
        logger.info(f"\nMean IoU: {mean_iou:.4f} (n={len(results)})")

        CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
        with open(CONFIG["output_dir"] / "ensemble_v3_results.json", "w") as f:
            json.dump({"mean_iou": mean_iou, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
