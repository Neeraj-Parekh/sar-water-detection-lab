#!/usr/bin/env python3
"""
================================================================================
TEST ENSEMBLE - LightGBM + U-Net + Physics Super Learner
================================================================================

Logic:
1. If LGB says Water AND UNet says Water -> HIGH CONFIDENCE WATER
2. If LGB says Water BUT UNet says Land -> Check Physics -> decide
3. If LGB says Land BUT UNet says Water -> Check Physics -> reject if physics agrees with Land
4. Physics VETO always applies (HAND>100 or Slope>45 = impossible)

This combines:
- LightGBM v9: Best pixel-level precision (IoU 0.807, Precision 0.901)
- U-Net v6: Best spatial context (edges, shape)
- Physics: Hard constraints + soft scoring

Author: SAR Water Detection Project
Date: 2026-01-25
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List
from scipy.ndimage import uniform_filter, minimum_filter, maximum_filter, laplace
from scipy.ndimage import grey_opening, grey_closing, label as scipy_label, sobel
from scipy.ndimage import binary_dilation, binary_erosion
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "lgb_model_path": "/home/mit-aoe/sar_water_detection/models/lightgbm_v9_clean_mndwi.txt",
    "unet_model_path": "/home/mit-aoe/sar_water_detection/models/unet_v6_best.pth",
    "chip_dir": Path("/home/mit-aoe/sar_water_detection/chips_expanded_npy"),
    "results_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    # Ensemble weights for bulk areas (non-edge)
    "lgb_weight_bulk": 0.55,
    "unet_weight_bulk": 0.35,
    "physics_weight": 0.10,
    # Ensemble weights for edge areas (where spatial context matters more)
    "lgb_weight_edge": 0.35,
    "unet_weight_edge": 0.55,
    # Thresholds
    "water_threshold": 0.5,
    "edge_threshold": 0.15,
    "min_region_size": 50,
    # Agreement logic
    "agreement_boost": 0.15,  # Boost when LGB and UNet agree
    "disagreement_physics_weight": 0.4,  # How much physics decides when they disagree
}


# =============================================================================
# U-NET ARCHITECTURE (from unet_v6_fixed)
# =============================================================================


class CBAM(object):
    """Placeholder - will load from unet_v6_fixed"""

    pass


def load_unet_model(model_path: str):
    """Load U-Net model with proper architecture."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Define CBAM inline
    class ChannelAttention(nn.Module):
        def __init__(self, channels, reduction=16):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channels, channels // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, channels, bias=False),
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            b, c, _, _ = x.size()
            avg_out = self.fc(self.avg_pool(x).view(b, c))
            max_out = self.fc(self.max_pool(x).view(b, c))
            return self.sigmoid(avg_out + max_out).view(b, c, 1, 1)

    class SpatialAttention(nn.Module):
        def __init__(self, kernel_size=7):
            super().__init__()
            self.conv = nn.Conv2d(
                2, 1, kernel_size, padding=kernel_size // 2, bias=False
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x = torch.cat([avg_out, max_out], dim=1)
            return self.sigmoid(self.conv(x))

    class CBAM(nn.Module):
        def __init__(self, channels, reduction=16, kernel_size=7):
            super().__init__()
            self.ca = ChannelAttention(channels, reduction)
            self.sa = SpatialAttention(kernel_size)

        def forward(self, x):
            x = x * self.ca(x)
            x = x * self.sa(x)
            return x

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch, dropout=0.2):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.conv(x)

    class CBAMUNet(nn.Module):
        def __init__(self, in_channels=6, out_channels=1, base_filters=32, dropout=0.3):
            super().__init__()
            f = base_filters

            # Encoder
            self.enc1 = ConvBlock(in_channels, f, dropout)
            self.cbam1 = CBAM(f)
            self.pool1 = nn.MaxPool2d(2)

            self.enc2 = ConvBlock(f, f * 2, dropout)
            self.cbam2 = CBAM(f * 2)
            self.pool2 = nn.MaxPool2d(2)

            self.enc3 = ConvBlock(f * 2, f * 4, dropout)
            self.cbam3 = CBAM(f * 4)
            self.pool3 = nn.MaxPool2d(2)

            self.enc4 = ConvBlock(f * 4, f * 8, dropout)
            self.cbam4 = CBAM(f * 8)
            self.pool4 = nn.MaxPool2d(2)

            # Bridge
            self.bridge = ConvBlock(f * 8, f * 16, dropout)

            # Decoder
            self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
            self.dec4 = ConvBlock(f * 16, f * 8, dropout)

            self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
            self.dec3 = ConvBlock(f * 8, f * 4, dropout)

            self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
            self.dec2 = ConvBlock(f * 4, f * 2, dropout)

            self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
            self.dec1 = ConvBlock(f * 2, f, dropout)

            self.out = nn.Conv2d(f, out_channels, 1)

        def forward(self, x):
            # Encoder
            e1 = self.cbam1(self.enc1(x))
            e2 = self.cbam2(self.enc2(self.pool1(e1)))
            e3 = self.cbam3(self.enc3(self.pool2(e2)))
            e4 = self.cbam4(self.enc4(self.pool3(e3)))

            # Bridge
            b = self.bridge(self.pool4(e4))

            # Decoder
            d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

            return self.out(d1)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CBAMUNet(in_channels=6, out_channels=1, base_filters=32)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    model = model.to(device)
    model.eval()
    return model, device


# =============================================================================
# SUPER LEARNER ENSEMBLE
# =============================================================================


class SuperLearnerEnsemble:
    """
    Ensemble that combines LightGBM + U-Net + Physics with smart logic.
    """

    def __init__(self):
        self.lgb_model = None
        self.unet_model = None
        self.device = None
        self.feature_names = None

    def load_models(self, lgb_path: str, unet_path: str):
        """Load both models."""
        # Load LightGBM
        try:
            import lightgbm as lgb

            self.lgb_model = lgb.Booster(model_file=lgb_path)
            self.feature_names = self.lgb_model.feature_name()
            logger.info(f"Loaded LightGBM: {len(self.feature_names)} features")
        except Exception as e:
            logger.error(f"Failed to load LightGBM: {e}")
            raise

        # Load U-Net
        try:
            self.unet_model, self.device = load_unet_model(unet_path)
            logger.info(f"Loaded U-Net on {self.device}")
        except Exception as e:
            logger.warning(f"Failed to load U-Net: {e}")
            self.unet_model = None

    def extract_features(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract features for LightGBM (must match training)."""
        vv = data["vv"]
        vh = data["vh"]
        dem = data["dem"]
        slope = data["slope"]
        hand = data["hand"]
        twi = data["twi"]
        mndwi = data.get("mndwi", None)

        features = []

        # Basic
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

        # Otsu-like (using median as proxy)
        vv_med = np.median(vv)
        vh_med = np.median(vh)
        features.append(vv - vv_med)
        features.append(vh - vh_med)

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

        # MNDWI features (if model expects them)
        if "MNDWI" in self.feature_names:
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

        feature_stack = np.stack(features, axis=-1)
        feature_stack = np.nan_to_num(feature_stack, nan=0.0, posinf=0.0, neginf=0.0)
        return feature_stack.astype(np.float32)

    def predict_lgb(self, features: np.ndarray) -> np.ndarray:
        """LightGBM prediction."""
        h, w, n = features.shape
        X = features.reshape(-1, n)
        proba = self.lgb_model.predict(X).reshape(h, w)
        return proba.astype(np.float32)

    def predict_unet(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """U-Net prediction."""
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

        with torch.no_grad():
            x = torch.from_numpy(inputs[np.newaxis]).float().to(self.device)
            out = self.unet_model(x)
            proba = torch.sigmoid(out).cpu().numpy()[0, 0]

        proba = proba[:h, :w]
        return proba.astype(np.float32)

    def compute_physics(
        self, hand: np.ndarray, slope: np.ndarray, twi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute physics score and VETO mask.

        Returns:
            physics_score: 0-1 soft score (higher = more likely water)
            veto_mask: True where water is IMPOSSIBLE
        """
        # Hard VETO - water cannot exist here
        veto = np.zeros_like(hand, dtype=bool)
        veto |= hand > 100  # Very high above drainage
        veto |= slope > 45  # Very steep
        veto |= (hand > 30) & (slope > 20)  # Combined

        # Soft physics score
        hand_exp = np.clip((hand - 15) / 5.0, -50, 50)
        hand_score = 1.0 / (1.0 + np.exp(hand_exp))

        slope_exp = np.clip((slope - 12) / 4.0, -50, 50)
        slope_score = 1.0 / (1.0 + np.exp(slope_exp))

        twi_exp = np.clip((7 - twi) / 2.0, -50, 50)
        twi_score = 1.0 / (1.0 + np.exp(twi_exp))

        physics_score = 0.4 * hand_score + 0.4 * slope_score + 0.2 * twi_score
        return physics_score.astype(np.float32), veto

    def detect_edges(self, proba: np.ndarray) -> np.ndarray:
        """Detect edge regions using gradient magnitude."""
        grad_x = sobel(proba, axis=0)
        grad_y = sobel(proba, axis=1)
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        edges = gradient > CONFIG["edge_threshold"]
        # Dilate edges slightly
        edges = binary_dilation(edges, iterations=2)
        return edges.astype(np.float32)

    def smart_ensemble(
        self,
        lgb_proba: np.ndarray,
        unet_proba: np.ndarray,
        physics_score: np.ndarray,
        veto_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Smart ensemble logic:
        1. Both agree -> high confidence
        2. Disagree -> let physics decide
        3. Always apply VETO
        """
        h, w = lgb_proba.shape
        combined = np.zeros((h, w), dtype=np.float32)

        # Detect edge regions
        edges = self.detect_edges(lgb_proba)

        # Agreement masks
        lgb_water = lgb_proba > 0.5
        unet_water = unet_proba > 0.5
        both_water = lgb_water & unet_water
        both_land = (~lgb_water) & (~unet_water)
        disagree = ~(both_water | both_land)

        # Case 1: Both agree on water -> boost confidence
        combined = np.where(
            both_water,
            np.minimum(1.0, (lgb_proba + unet_proba) / 2 + CONFIG["agreement_boost"]),
            combined,
        )

        # Case 2: Both agree on land -> low probability
        combined = np.where(
            both_land,
            (lgb_proba + unet_proba) / 2 * 0.5,  # Reduce even further
            combined,
        )

        # Case 3: Disagree -> weighted combination with physics tiebreaker
        lgb_w = np.where(
            edges > 0.5, CONFIG["lgb_weight_edge"], CONFIG["lgb_weight_bulk"]
        )
        unet_w = np.where(
            edges > 0.5, CONFIG["unet_weight_edge"], CONFIG["unet_weight_bulk"]
        )
        phys_w = CONFIG["disagreement_physics_weight"]

        # When they disagree, physics gets more say
        disagree_combined = (1 - phys_w) * (lgb_w * lgb_proba + unet_w * unet_proba) / (
            lgb_w + unet_w
        ) + phys_w * physics_score
        combined = np.where(disagree, disagree_combined, combined)

        # Apply physics VETO (overrides everything)
        combined = np.where(veto_mask, 0.0, combined)

        return combined

    def remove_small_regions(self, mask: np.ndarray, min_size: int) -> np.ndarray:
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

    def detect(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        """
        Full detection pipeline.

        Returns:
            water_mask: Binary water mask
            metadata: Detection statistics
        """
        metadata = {"components": []}

        # Validate data
        for key in ["vv", "vh", "dem", "slope", "hand", "twi"]:
            if key not in data:
                raise ValueError(f"Missing: {key}")
            data[key] = np.nan_to_num(data[key].astype(np.float32))

        if "slope" in data:
            data["slope"] = np.clip(data["slope"], 0, 90)
        if "hand" in data:
            data["hand"] = np.clip(data["hand"], 0, 500)
        if "twi" in data:
            data["twi"] = np.clip(data["twi"], 0, 30)
        if "mndwi" in data:
            data["mndwi"] = np.clip(np.nan_to_num(data["mndwi"]), -1, 1)

        # Physics
        physics_score, veto_mask = self.compute_physics(
            data["hand"], data["slope"], data["twi"]
        )
        metadata["veto_fraction"] = float(veto_mask.mean())
        metadata["components"].append("physics")

        # LightGBM
        features = self.extract_features(data)
        lgb_proba = self.predict_lgb(features)
        metadata["lgb_water_fraction"] = float((lgb_proba > 0.5).mean())
        metadata["components"].append("lgb")

        # U-Net (if available)
        if self.unet_model is not None:
            unet_proba = self.predict_unet(data)
            metadata["unet_water_fraction"] = float((unet_proba > 0.5).mean())
            metadata["components"].append("unet")

            # Smart ensemble
            combined = self.smart_ensemble(
                lgb_proba, unet_proba, physics_score, veto_mask
            )
            metadata["method"] = "ensemble"
        else:
            # LGB + Physics only
            combined = 0.9 * lgb_proba + 0.1 * physics_score
            combined = np.where(veto_mask, 0.0, combined)
            metadata["method"] = "lgb_physics"

        # Threshold
        water_mask = (combined > CONFIG["water_threshold"]).astype(np.float32)

        # Remove small regions
        water_mask = self.remove_small_regions(water_mask, CONFIG["min_region_size"])

        metadata["final_water_fraction"] = float(water_mask.mean())
        metadata["combined_mean"] = float(combined.mean())

        return water_mask, metadata


# =============================================================================
# METRICS
# =============================================================================


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute all metrics."""
    pred_bin = pred > 0.5
    target_bin = target > 0.5

    intersection = np.logical_and(pred_bin, target_bin).sum()
    union = np.logical_or(pred_bin, target_bin).sum()

    tp = intersection
    fp = np.logical_and(pred_bin, ~target_bin).sum()
    fn = np.logical_and(~pred_bin, target_bin).sum()

    iou = (
        float(intersection) / float(union)
        if union > 0
        else (1.0 if intersection == 0 else 0.0)
    )
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Test ensemble on all chips."""
    logger.info("=" * 70)
    logger.info("SUPER LEARNER ENSEMBLE TEST")
    logger.info("=" * 70)
    logger.info("Combining: LightGBM v9 + U-Net v6 + Physics")

    # Initialize
    ensemble = SuperLearnerEnsemble()
    ensemble.load_models(CONFIG["lgb_model_path"], CONFIG["unet_model_path"])

    # Load chips
    chip_files = sorted(CONFIG["chip_dir"].glob("*_with_truth.npy"))
    logger.info(f"Found {len(chip_files)} chips")

    results = []
    all_preds = []
    all_labels = []

    for chip_path in chip_files:
        name = chip_path.stem.replace("_with_truth", "")

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

            label = (data[:, :, 6] > 0).astype(np.float32)

            # Detect
            pred, meta = ensemble.detect(chip_data)

            # Metrics
            metrics = compute_metrics(pred, label)

            results.append(
                {
                    "chip": name,
                    **metrics,
                    "gt_water_fraction": float(label.mean()),
                    "pred_water_fraction": meta["final_water_fraction"],
                    "method": meta["method"],
                }
            )

            all_preds.append(pred.flatten())
            all_labels.append(label.flatten())

            logger.info(
                f"  {name}: IoU={metrics['iou']:.4f}, P={metrics['precision']:.4f}, "
                f"R={metrics['recall']:.4f}, method={meta['method']}"
            )

        except Exception as e:
            logger.error(f"  {name}: ERROR - {e}")
            import traceback

            traceback.print_exc()

    # Overall metrics
    if all_preds:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        overall = compute_metrics(all_preds, all_labels)

        logger.info("\n" + "=" * 70)
        logger.info("OVERALL RESULTS")
        logger.info("=" * 70)
        logger.info(f"IoU:       {overall['iou']:.4f}")
        logger.info(f"Precision: {overall['precision']:.4f}")
        logger.info(f"Recall:    {overall['recall']:.4f}")
        logger.info(f"F1:        {overall['f1']:.4f}")

        # Comparison with single models
        logger.info("\nComparison:")
        logger.info(f"  LightGBM v9 alone: IoU 0.807")
        logger.info(f"  Ensemble:          IoU {overall['iou']:.4f}")
        logger.info(f"  Improvement:       {(overall['iou'] - 0.807) * 100:+.2f}%")

        # Save results
        output = {
            "timestamp": datetime.now().isoformat(),
            "method": "SuperLearner Ensemble (LGB v9 + UNet v6 + Physics)",
            "config": {
                k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()
            },
            "overall_metrics": overall,
            "comparison": {
                "lgb_v9_alone": 0.807,
                "ensemble": overall["iou"],
                "improvement": overall["iou"] - 0.807,
            },
            "per_chip_results": results,
        }

        results_path = CONFIG["results_dir"] / "ensemble_test_results.json"
        with open(results_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nResults saved to {results_path}")

    logger.info("=" * 70)
    logger.info("ENSEMBLE TEST COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
