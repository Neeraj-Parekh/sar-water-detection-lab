#!/usr/bin/env python3
"""
================================================================================
FINAL ENSEMBLE v11 - Three-Expert Water Detection System
================================================================================

Combines:
1. LightGBM v9 (74 features) - Pixel-level texture expert
2. Attention U-Net v7 (8 channels) - Contextual shape expert
3. Physics constraints - Domain knowledge expert

Fusion Strategy:
- Weighted average: 0.5 * LGB + 0.35 * UNet + 0.15 * Physics
- High-confidence U-Net recovery: If UNet > 0.9, accept even if LGB uncertain
- Physics VETO: HAND > 100m OR Slope > 45 = No water (absolute)

Target: IoU > 0.90

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
from typing import Dict, Tuple, List
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

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Models
    "lgb_model_path": Path(
        "/home/mit-aoe/sar_water_detection/models/lightgbm_v9_clean_mndwi.txt"
    ),
    "unet_model_path": Path(
        "/home/mit-aoe/sar_water_detection/models/attention_unet_v7_best.pth"
    ),
    "chip_dir": Path("/home/mit-aoe/sar_water_detection/chips_expanded_npy"),
    "output_dir": Path("/home/mit-aoe/sar_water_detection/results/final_ensemble_v11"),
    # Ensemble weights - REFINED based on individual performance
    # LGB (0.807) >> UNet (0.797), so weight LGB more heavily
    "lgb_weight": 0.75,  # Pixel expert (dominant)
    "unet_weight": 0.10,  # Shape expert (refinement only)
    "physics_weight": 0.15,  # Domain expert (VETO power)
    # High-confidence recovery - MORE CONSERVATIVE
    "unet_high_conf": 0.95,  # If UNet > this, accept water (very high threshold)
    "lgb_low_conf": 0.4,  # Even if LGB < this
    # Detection thresholds
    "water_threshold": 0.5,
    "min_region_size": 50,
    # Physics constraints
    "hand_veto": 100,
    "slope_veto": 45,
    "combined_hand": 30,
    "combined_slope": 20,
    # Edge case handlers
    "vh_calm": -18.0,
    "vh_bright_max": -10.0,
    "texture_threshold": 1.5,
    "vv_vh_ratio_threshold": 6.0,
    "urban_variance_threshold": 8.0,
    # U-Net normalization (must match training)
    "norm": {
        "vv": {"mean": -15.0, "std": 5.0},
        "vh": {"mean": -22.0, "std": 5.0},
        "dem": {"mean": 200.0, "std": 200.0},
        "slope": {"mean": 5.0, "std": 8.0},
        "hand": {"mean": 10.0, "std": 15.0},
        "twi": {"mean": 10.0, "std": 5.0},
        "mndwi": {"mean": 0.0, "std": 0.5},
        "vh_texture": {"mean": 0.0, "std": 1.0},
    },
}


# =============================================================================
# ATTENTION U-NET ARCHITECTURE (copy from v7)
# =============================================================================


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True), nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True), nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=8, out_channels=1, base_filters=32, dropout=0.3):
        super().__init__()
        f = base_filters

        # Encoder
        self.enc1 = ResidualBlock(in_channels, f, dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResidualBlock(f, f * 2, dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResidualBlock(f * 2, f * 4, dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResidualBlock(f * 4, f * 8, dropout)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(f * 8, f * 16, dropout)

        # Decoder with Attention
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.att4 = AttentionGate(F_g=f * 8, F_l=f * 8, F_int=f * 4)
        self.dec4 = ResidualBlock(f * 16, f * 8, dropout)

        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.att3 = AttentionGate(F_g=f * 4, F_l=f * 4, F_int=f * 2)
        self.dec3 = ResidualBlock(f * 8, f * 4, dropout)

        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.att2 = AttentionGate(F_g=f * 2, F_l=f * 2, F_int=f)
        self.dec2 = ResidualBlock(f * 4, f * 2, dropout)

        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.att1 = AttentionGate(F_g=f, F_l=f, F_int=f // 2)
        self.dec1 = ResidualBlock(f * 2, f, dropout)

        # Output
        self.out = nn.Conv2d(f, out_channels, 1)

        # Deep supervision (not used in inference)
        self.ds4 = nn.Conv2d(f * 8, out_channels, 1)
        self.ds3 = nn.Conv2d(f * 4, out_channels, 1)
        self.ds2 = nn.Conv2d(f * 2, out_channels, 1)

    def forward(self, x, deep_supervision=False):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, self.att4(d4, e4)], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, self.att3(d3, e3)], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self.att2(d2, e2)], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self.att1(d1, e1)], dim=1))

        return self.out(d1)


# =============================================================================
# FEATURE EXTRACTION FOR LIGHTGBM (74 features)
# =============================================================================


def extract_lgb_features(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Extract 74 features for LightGBM prediction."""
    vv = data["vv"]
    vh = data["vh"]
    dem = data["dem"]
    slope = data["slope"]
    hand = data["hand"]
    twi = data["twi"]
    mndwi = data.get("mndwi", None)

    features = []

    # Basic (6)
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

    # Multi-scale texture (40)
    for scale in [3, 5, 9, 15, 21]:
        for arr in [vv, vh]:
            arr_mean = uniform_filter(arr, size=scale)
            arr_sq = uniform_filter(arr**2, size=scale)
            arr_var = np.maximum(arr_sq - arr_mean**2, 0)
            arr_std = np.sqrt(arr_var)
            arr_min = minimum_filter(arr, size=scale)
            arr_max = maximum_filter(arr, size=scale)
            features.extend([arr_mean, arr_std, arr_min, arr_max])

    # Gradients (4)
    gy_vv, gx_vv = np.gradient(vv)
    gy_vh, gx_vh = np.gradient(vh)
    features.append(np.sqrt(gx_vv**2 + gy_vv**2))
    features.append(np.sqrt(gx_vh**2 + gy_vh**2))
    features.append(np.abs(laplace(vv)))
    features.append(np.abs(laplace(vh)))

    # Morphological (4)
    features.extend(
        [
            grey_opening(vv, size=5),
            grey_closing(vv, size=5),
            grey_opening(vh, size=5),
            grey_closing(vh, size=5),
        ]
    )

    # Otsu-like (2)
    features.append(vv - np.median(vv))
    features.append(vh - np.median(vh))

    # Local contrast (2)
    features.append(vv - uniform_filter(vv, size=9))
    features.append(vh - uniform_filter(vh, size=9))

    # GLCM-like (4)
    for arr in [vv, vh]:
        arr_mean = uniform_filter(arr, size=5)
        arr_sq = uniform_filter(arr**2, size=5)
        arr_var = np.maximum(arr_sq - arr_mean**2, 0)
        contrast = np.sqrt(arr_var)
        arr_range = maximum_filter(arr, size=5) - minimum_filter(arr, size=5)
        homogeneity = 1.0 / (1.0 + arr_range)
        features.extend([contrast, homogeneity])

    # Pseudo-entropy (1)
    vv_norm = vv - vv.min()
    vv_range = vv.max() - vv.min() + 1e-10
    vv_prob = np.clip(vv_norm / vv_range, 1e-10, 1 - 1e-10)
    entropy = -vv_prob * np.log2(vv_prob) - (1 - vv_prob) * np.log2(1 - vv_prob)
    entropy = np.nan_to_num(entropy, nan=0.0)
    features.append(entropy)

    # DEM features (4)
    features.extend([dem, slope, hand, twi])

    # Physics scores (3)
    hand_exp = np.clip((hand - 10) / 3.0, -50, 50)
    slope_exp = np.clip((slope - 8) / 3.0, -50, 50)
    twi_exp = np.clip((8 - twi) / 2.0, -50, 50)
    features.append(1.0 / (1.0 + np.exp(hand_exp)))
    features.append(1.0 / (1.0 + np.exp(slope_exp)))
    features.append(1.0 / (1.0 + np.exp(twi_exp)))

    # MNDWI features (4)
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

    # Total: 6 + 40 + 4 + 4 + 2 + 2 + 4 + 1 + 4 + 3 + 4 = 74 features
    feature_stack = np.stack(features, axis=-1)
    feature_stack = np.nan_to_num(feature_stack, nan=0.0, posinf=0.0, neginf=0.0)
    return feature_stack.astype(np.float32)


# =============================================================================
# U-NET INPUT PREPARATION (8 channels)
# =============================================================================


def prepare_unet_input(
    data: Dict[str, np.ndarray], device: torch.device
) -> torch.Tensor:
    """Prepare 8-channel input for U-Net with proper normalization."""
    vv = data["vv"].astype(np.float32)
    vh = data["vh"].astype(np.float32)
    dem = data["dem"].astype(np.float32)
    slope = np.clip(data["slope"].astype(np.float32), 0, 90)
    hand = np.clip(data["hand"].astype(np.float32), 0, 500)
    twi = np.clip(data["twi"].astype(np.float32), 0, 30)
    mndwi = data.get("mndwi", np.zeros_like(vv)).astype(np.float32)

    # Compute VH texture
    vh_mean = uniform_filter(vh, size=5)
    vh_sq_mean = uniform_filter(vh**2, size=5)
    vh_var = np.maximum(vh_sq_mean - vh_mean**2, 0)
    vh_texture = np.sqrt(vh_var).astype(np.float32)

    # Stack channels
    channels = np.stack([vv, vh, dem, slope, hand, twi, mndwi, vh_texture], axis=0)

    # Normalize
    norm = CONFIG["norm"]
    keys = ["vv", "vh", "dem", "slope", "hand", "twi", "mndwi", "vh_texture"]
    for i, key in enumerate(keys):
        channels[i] = (channels[i] - norm[key]["mean"]) / norm[key]["std"]
    channels = np.clip(channels, -5, 5)

    # To tensor (1, 8, H, W)
    tensor = torch.from_numpy(channels).unsqueeze(0).to(device)
    return tensor


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
# EDGE CASE HANDLERS
# =============================================================================


def apply_bright_water_correction(
    water_mask: np.ndarray, vh: np.ndarray, proba: np.ndarray
) -> Tuple[np.ndarray, int]:
    """Adaptive region growing for wind-roughened water."""
    vh_mean = uniform_filter(vh.astype(np.float64), size=5)
    vh_sq_mean = uniform_filter(vh.astype(np.float64) ** 2, size=5)
    vh_variance = np.maximum(vh_sq_mean - vh_mean**2, 0)

    seeds = (water_mask > 0.5) & (proba > 0.7)
    ambiguous = (
        (vh >= CONFIG["vh_calm"])
        & (vh < CONFIG["vh_bright_max"])
        & (vh_variance < CONFIG["texture_threshold"])
        & (~water_mask.astype(bool))
    )

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


def apply_urban_mask(
    water_mask: np.ndarray, vv: np.ndarray, vh: np.ndarray
) -> Tuple[np.ndarray, int]:
    """Remove urban shadow false positives."""
    vv_vh_diff = vv - vh

    vv_mean = uniform_filter(vv.astype(np.float64), size=7)
    vv_sq_mean = uniform_filter(vv.astype(np.float64) ** 2, size=7)
    vv_variance = np.maximum(vv_sq_mean - vv_mean**2, 0)

    vh_mean = uniform_filter(vh.astype(np.float64), size=7)
    vh_sq_mean = uniform_filter(vh.astype(np.float64) ** 2, size=7)
    vh_variance = np.maximum(vh_sq_mean - vh_mean**2, 0)

    combined_variance = (vv_variance + vh_variance) / 2
    vv_max_local = maximum_filter(vv, size=9)
    has_bright = vv_max_local > -8.0

    urban = (
        (vv_vh_diff > CONFIG["vv_vh_ratio_threshold"])
        | (combined_variance > CONFIG["urban_variance_threshold"])
    ) & has_bright

    labeled, num = scipy_label(urban)
    for i in range(1, num + 1):
        if (labeled == i).sum() < 100:
            urban = urban & (labeled != i)

    urban = binary_dilation(urban, iterations=2)
    corrected = water_mask.astype(bool) & (~urban)
    pixels_removed = int((water_mask > 0.5).sum() - corrected.sum())

    return corrected.astype(np.float32), pixels_removed


def morphological_cleanup(mask: np.ndarray, min_size: int = 50) -> np.ndarray:
    """Remove small regions and fill holes."""
    labeled, num = scipy_label(mask > 0.5)
    cleaned = np.zeros_like(mask, dtype=bool)

    for i in range(1, num + 1):
        region = labeled == i
        if region.sum() >= min_size:
            cleaned |= region

    filled = binary_dilation(cleaned, iterations=1)
    filled = binary_erosion(filled, iterations=1)

    return filled.astype(np.float32)


# =============================================================================
# DIFFERENCE MAP
# =============================================================================


def generate_difference_map(
    prediction: np.ndarray, ground_truth: np.ndarray, output_path: Path
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
    rgb[tn] = [200, 200, 200]
    rgb[tp] = [0, 255, 0]
    rgb[fp] = [255, 0, 0]
    rgb[fn] = [0, 0, 255]

    try:
        from PIL import Image

        img = Image.fromarray(rgb)
        img.save(output_path)
    except ImportError:
        np.save(str(output_path).replace(".png", ".npy"), rgb)

    # Compute metrics with edge case handling
    tp_sum = int(tp.sum())
    fp_sum = int(fp.sum())
    fn_sum = int(fn.sum())
    tn_sum = int(tn.sum())

    total_truth = tp_sum + fn_sum
    total_pred = tp_sum + fp_sum

    stats = {
        "tp": tp_sum,
        "fp": fp_sum,
        "fn": fn_sum,
        "tn": tn_sum,
    }

    # Handle edge case: no water in ground truth or prediction
    if total_truth == 0 and total_pred == 0:
        # Perfect: both agree there's no water
        stats["recall"] = 1.0
        stats["precision"] = 1.0
        stats["iou"] = 1.0
        stats["accuracy"] = 1.0
        stats["note"] = "No water in truth or prediction (perfect agreement)"
    elif total_truth == 0:
        # No water in truth but we predicted some = all FP
        stats["recall"] = 1.0  # vacuously true
        stats["precision"] = 0.0
        stats["iou"] = 0.0
        stats["accuracy"] = tn_sum / (tn_sum + fp_sum) if (tn_sum + fp_sum) > 0 else 0.0
        stats["note"] = f"No water in truth, but predicted {fp_sum} pixels"
    elif total_pred == 0:
        # No prediction but there's water = all FN
        stats["recall"] = 0.0
        stats["precision"] = 1.0  # vacuously true
        stats["iou"] = 0.0
        stats["accuracy"] = tn_sum / (tn_sum + fn_sum) if (tn_sum + fn_sum) > 0 else 0.0
        stats["note"] = f"Predicted no water, but truth has {fn_sum} pixels"
    else:
        # Normal case
        stats["recall"] = tp_sum / total_truth
        stats["precision"] = tp_sum / total_pred
        stats["iou"] = tp_sum / (tp_sum + fp_sum + fn_sum)
        stats["accuracy"] = (tp_sum + tn_sum) / (tp_sum + tn_sum + fp_sum + fn_sum)

    return stats


# =============================================================================
# FINAL ENSEMBLE PIPELINE
# =============================================================================


class FinalEnsemblePipeline:
    """Three-expert ensemble for water detection."""

    def __init__(self):
        self.lgb_model = None
        self.unet_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load LightGBM
        try:
            import lightgbm as lgb

            self.lgb_model = lgb.Booster(model_file=str(CONFIG["lgb_model_path"]))
            logger.info(f"Loaded LightGBM: {CONFIG['lgb_model_path']}")
        except Exception as e:
            logger.error(f"Failed to load LightGBM: {e}")

        # Load Attention U-Net
        try:
            self.unet_model = AttentionUNet(
                in_channels=8, out_channels=1, base_filters=32, dropout=0.3
            ).to(self.device)

            checkpoint = torch.load(
                CONFIG["unet_model_path"], map_location=self.device, weights_only=False
            )
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                self.unet_model.load_state_dict(checkpoint["state_dict"])
            else:
                self.unet_model.load_state_dict(checkpoint)

            self.unet_model.eval()
            logger.info(f"Loaded Attention U-Net: {CONFIG['unet_model_path']}")
        except Exception as e:
            logger.error(f"Failed to load U-Net: {e}")

    def detect(
        self, data: Dict[str, np.ndarray], name: str = "chip"
    ) -> Tuple[np.ndarray, Dict]:
        """Run three-expert ensemble detection."""
        stats = {"name": name, "experts": {}, "fusion": {}}

        # Validate and clean input
        for key in ["vv", "vh", "dem", "slope", "hand", "twi"]:
            if key not in data:
                raise ValueError(f"Missing: {key}")
            data[key] = np.nan_to_num(data[key].astype(np.float32))

        data["slope"] = np.clip(data["slope"], 0, 90)
        data["hand"] = np.clip(data["hand"], 0, 500)
        data["twi"] = np.clip(data["twi"], 0, 30)
        if "mndwi" in data:
            data["mndwi"] = np.clip(np.nan_to_num(data["mndwi"]), -1, 1)

        h, w = data["vv"].shape

        # =================================================================
        # EXPERT 1: LightGBM (Pixel Expert)
        # =================================================================
        features = extract_lgb_features(data)
        X = features.reshape(-1, features.shape[-1])
        lgb_proba = self.lgb_model.predict(X).reshape(h, w).astype(np.float32)
        stats["experts"]["lgb"] = {
            "mean_proba": float(lgb_proba.mean()),
            "water_fraction": float((lgb_proba > 0.5).mean()),
        }

        # =================================================================
        # EXPERT 2: Attention U-Net (Shape Expert)
        # =================================================================
        with torch.no_grad():
            # Pad to multiple of 16 for U-Net
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16

            padded_data = {
                k: np.pad(v, ((0, pad_h), (0, pad_w)), mode="reflect")
                for k, v in data.items()
            }

            unet_input = prepare_unet_input(padded_data, self.device)
            unet_output = self.unet_model(unet_input)
            unet_proba = torch.sigmoid(unet_output).squeeze().cpu().numpy()

            # Crop back to original size
            unet_proba = unet_proba[:h, :w].astype(np.float32)

        stats["experts"]["unet"] = {
            "mean_proba": float(unet_proba.mean()),
            "water_fraction": float((unet_proba > 0.5).mean()),
        }

        # =================================================================
        # EXPERT 3: Physics (Domain Expert)
        # =================================================================
        physics_score, veto = compute_physics(data["hand"], data["slope"], data["twi"])
        stats["experts"]["physics"] = {
            "mean_score": float(physics_score.mean()),
            "veto_fraction": float(veto.mean()),
        }

        # =================================================================
        # FUSION: Weighted Ensemble + High-Confidence Recovery
        # =================================================================

        # Base weighted combination
        combined = (
            CONFIG["lgb_weight"] * lgb_proba
            + CONFIG["unet_weight"] * unet_proba
            + CONFIG["physics_weight"] * physics_score
        )

        # High-confidence U-Net recovery (for thin rivers LGB might miss)
        high_conf_unet = (unet_proba > CONFIG["unet_high_conf"]) & (
            lgb_proba > CONFIG["lgb_low_conf"]
        )
        combined = np.maximum(combined, high_conf_unet.astype(np.float32) * 0.8)

        stats["fusion"]["high_conf_recovery_pixels"] = int(high_conf_unet.sum())

        # Apply physics VETO (absolute - mountains can't have water)
        combined = np.where(veto, 0.0, combined)

        # Initial threshold
        water_mask = (combined > CONFIG["water_threshold"]).astype(np.float32)
        stats["fusion"]["initial_water_fraction"] = float(water_mask.mean())

        # =================================================================
        # EDGE CASE HANDLERS
        # =================================================================

        # Use maximum of both expert probabilities for edge case handling
        max_proba = np.maximum(lgb_proba, unet_proba)

        # Bright water correction
        water_mask, bright_added = apply_bright_water_correction(
            water_mask, data["vh"], max_proba
        )
        stats["fusion"]["bright_water_added"] = bright_added

        # Urban shadow removal
        water_mask, urban_removed = apply_urban_mask(water_mask, data["vv"], data["vh"])
        stats["fusion"]["urban_shadow_removed"] = urban_removed

        # Morphological cleanup
        water_mask = morphological_cleanup(water_mask, CONFIG["min_region_size"])
        stats["fusion"]["final_water_fraction"] = float(water_mask.mean())

        return water_mask, stats


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Final Ensemble v11")
    parser.add_argument("--chip_dir", type=str, default=str(CONFIG["chip_dir"]))
    parser.add_argument("--output_dir", type=str, default=str(CONFIG["output_dir"]))
    parser.add_argument("--max_chips", type=int, default=None)
    args = parser.parse_args()

    chip_dir = Path(args.chip_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "difference_maps").mkdir(exist_ok=True)

    logger.info("=" * 70)
    logger.info("FINAL ENSEMBLE v11 - Three-Expert Water Detection")
    logger.info("=" * 70)
    logger.info(f"LightGBM weight: {CONFIG['lgb_weight']}")
    logger.info(f"U-Net weight:    {CONFIG['unet_weight']}")
    logger.info(f"Physics weight:  {CONFIG['physics_weight']}")
    logger.info("=" * 70)

    # Initialize pipeline
    pipeline = FinalEnsemblePipeline()

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

            note = f" ({diff_stats.get('note', '')})" if "note" in diff_stats else ""
            logger.info(
                f"  {name}: IoU={diff_stats['iou']:.4f}, "
                f"P={diff_stats['precision']:.4f}, R={diff_stats['recall']:.4f}{note}"
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
        tn = (~pred_bin & ~label_bin).sum()

        overall = {
            "iou": float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0,
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "accuracy": float((tp + tn) / (tp + tn + fp + fn))
            if (tp + tn + fp + fn) > 0
            else 0.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
        }

        logger.info("\n" + "=" * 70)
        logger.info("FINAL ENSEMBLE v11 RESULTS")
        logger.info("=" * 70)
        logger.info(f"IoU:       {overall['iou']:.4f}")
        logger.info(f"Precision: {overall['precision']:.4f}")
        logger.info(f"Recall:    {overall['recall']:.4f}")
        logger.info(f"F1:        {overall['f1']:.4f}")
        logger.info(f"Accuracy:  {overall['accuracy']:.4f}")
        logger.info("=" * 70)

        # Compare with previous best
        prev_best = 0.882  # LGB + Physics from earlier
        improvement = overall["iou"] - prev_best
        logger.info(f"\nComparison with previous best (LGB+Physics):")
        logger.info(f"  Previous IoU: {prev_best:.4f}")
        logger.info(f"  Current IoU:  {overall['iou']:.4f}")
        logger.info(
            f"  Change:       {improvement:+.4f} ({improvement / prev_best * 100:+.1f}%)"
        )

        # Save results
        output = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "lgb_weight": CONFIG["lgb_weight"],
                "unet_weight": CONFIG["unet_weight"],
                "physics_weight": CONFIG["physics_weight"],
                "unet_high_conf": CONFIG["unet_high_conf"],
                "lgb_low_conf": CONFIG["lgb_low_conf"],
                "water_threshold": CONFIG["water_threshold"],
            },
            "overall_metrics": overall,
            "comparison": {
                "previous_best": prev_best,
                "improvement": improvement,
            },
            "per_chip_results": all_results,
        }

        results_path = output_dir / "final_ensemble_results.json"
        with open(results_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"\nResults saved to {results_path}")
        logger.info(f"Difference maps saved to {output_dir / 'difference_maps'}")

    logger.info("=" * 70)
    logger.info("ENSEMBLE COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
