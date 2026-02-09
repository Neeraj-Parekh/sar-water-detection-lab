#!/usr/bin/env python3
"""
================================================================================
FINAL ENSEMBLE v12 - SOTA Post-Processing Integration
================================================================================

This is the PRODUCTION-READY ensemble that combines:

1. LightGBM v9 (IoU 0.882) - Best pixel-level classifier
2. U-Net v9 SOTA (IoU 0.685) - Spatial context with Frangi vesselness
3. SOTA Post-Processing:
   - MST River Connector (gap healing)
   - Mamdani Fuzzy Logic (soft decisions)
   - Physics VETO (hard constraints)

Expected Performance:
- Target IoU: >0.88 (beat LightGBM alone)
- River Connectivity: Significantly improved
- False Positives: Reduced via physics veto

Author: SAR Water Detection Project
Date: 2026-01-25
Version: 12.0
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Any
from scipy.ndimage import (
    uniform_filter,
    minimum_filter,
    maximum_filter,
    laplace,
    grey_opening,
    grey_closing,
    label as scipy_label,
    sobel,
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
)
from scipy.spatial import cKDTree
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Model paths (on server)
    "lgb_model_path": "/home/mit-aoe/sar_water_detection/models/lightgbm_v9_clean_mndwi.txt",
    "unet_v9_path": "/home/mit-aoe/sar_water_detection/models/attention_unet_v9_sota_best.pth",
    "chip_dir": Path("/home/mit-aoe/sar_water_detection/chips_expanded_npy"),
    "results_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    # Ensemble Strategy
    # LightGBM dominates because it has higher IoU, but U-Net helps with:
    # - Edge refinement
    # - Spatial coherence
    # - River connectivity
    "lgb_weight": 0.75,  # LightGBM gets most weight (it's 0.882 vs 0.685)
    "unet_weight": 0.20,  # U-Net for spatial context
    "physics_weight": 0.05,  # Physics for sanity checks
    # Thresholds
    "water_threshold": 0.5,
    "mst_max_gap": 15,  # Max pixels to bridge for river gaps
    "mst_vh_threshold": -16.0,  # Only bridge if intermediate pixels are dark
    "min_region_size": 50,  # Remove small blobs
    # Fuzzy logic parameters
    "fuzzy_enable": True,
    "fuzzy_edge_boost": 0.1,  # Boost confidence at water edges
    # Physics VETO thresholds
    "veto_hand_max": 100,  # No water above 100m drainage
    "veto_slope_max": 45,  # No water on 45+ degree slopes
}


# =============================================================================
# IMPORT SOTA MODULE
# =============================================================================

# Try to import our research-grade SOTA module
try:
    from sota_research_module import (
        FrangiVesselness,
        MSTRiverConnector,
        MamdaniFuzzyController,
        GMMAutoThreshold,
        CenterlineDiceLoss,
    )

    HAS_SOTA = True
    logger.info("SOTA research module loaded successfully")
except ImportError:
    HAS_SOTA = False
    logger.warning("SOTA module not found - using fallback implementations")


# =============================================================================
# FALLBACK IMPLEMENTATIONS (if SOTA module not available)
# =============================================================================


class SimpleMSTConnector:
    """Simplified MST gap healer for when full SOTA module is unavailable."""

    def __init__(self, max_gap_pixels: int = 15, vh_threshold: float = -16.0):
        self.max_gap = max_gap_pixels
        self.vh_threshold = vh_threshold

    def connect(self, water_mask: np.ndarray, vh: np.ndarray) -> Tuple[np.ndarray, int]:
        """Connect nearby water blobs if path is dark."""
        labeled, n_regions = scipy_label(water_mask > 0.5)
        if n_regions < 2:
            return water_mask, 0

        # Find region centroids
        centroids = []
        for i in range(1, n_regions + 1):
            region = labeled == i
            y_coords, x_coords = np.where(region)
            if len(y_coords) > 0:
                centroids.append((np.mean(y_coords), np.mean(x_coords), i))

        connections = 0
        result = water_mask.copy()

        # Try to connect nearby regions
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                c1, c2 = centroids[i], centroids[j]
                dist = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

                if dist < self.max_gap:
                    # Check if path is dark
                    n_points = int(dist) + 1
                    ys = np.linspace(c1[0], c2[0], n_points).astype(int)
                    xs = np.linspace(c1[1], c2[1], n_points).astype(int)

                    # Clip to image bounds
                    ys = np.clip(ys, 0, vh.shape[0] - 1)
                    xs = np.clip(xs, 0, vh.shape[1] - 1)

                    path_vh = vh[ys, xs]
                    if np.median(path_vh) < self.vh_threshold:
                        # Draw line
                        result[ys, xs] = 1.0
                        connections += 1

        return result, connections


class SimpleFuzzyController:
    """Simplified fuzzy logic for water detection."""

    def predict(
        self, vh: np.ndarray, slope: np.ndarray, hand: np.ndarray = None
    ) -> np.ndarray:
        """Return water probability based on fuzzy rules."""
        h, w = vh.shape
        prob = np.zeros((h, w), dtype=np.float32)

        # Rule 1: Very dark VH + flat terrain -> high water probability
        dark_mask = vh < -18  # dB
        flat_mask = slope < 5
        prob = np.where(dark_mask & flat_mask, 0.9, prob)

        # Rule 2: Medium dark + moderate terrain
        medium_dark = (vh >= -18) & (vh < -14)
        moderate_slope = (slope >= 5) & (slope < 15)
        prob = np.where(medium_dark & moderate_slope, 0.5, prob)

        # Rule 3: Apply HAND if available
        if hand is not None:
            low_hand = hand < 10
            prob = np.where(dark_mask & low_hand, np.maximum(prob, 0.85), prob)

        return prob


# =============================================================================
# U-NET V9 ARCHITECTURE
# =============================================================================


def load_unet_v9(model_path: str, device=None):
    """Load U-Net v9 with 9 input channels (includes Frangi)."""
    import torch
    import torch.nn as nn

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    class AttentionUNetV9(nn.Module):
        """U-Net v9 with 9 input channels (VV, VH, DEM, Slope, HAND, TWI, MNDWI, Physics, Frangi)."""

        def __init__(self, in_channels=9, out_channels=1, base_filters=32, dropout=0.3):
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

            # Decoder with skip connections
            d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

            return self.out(d1)

    # Create and load model
    model = AttentionUNetV9(in_channels=9, out_channels=1, base_filters=32)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            model = checkpoint

        model = model.to(device)
        model.eval()
        logger.info(f"Loaded U-Net v9 on {device}")
        return model, device
    except Exception as e:
        logger.error(f"Failed to load U-Net v9: {e}")
        return None, device


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================


def compute_frangi_vesselness(
    vh: np.ndarray, sigmas: List[float] = [1, 2, 3]
) -> np.ndarray:
    """Compute Frangi vesselness for river enhancement."""
    if HAS_SOTA:
        try:
            frangi = FrangiVesselness(sigmas=sigmas, beta=0.5, c=15.0)
            return frangi.compute(vh)
        except:
            pass

    # Fallback: simple dark-line detection
    h, w = vh.shape
    vesselness = np.zeros((h, w), dtype=np.float32)

    for sigma in sigmas:
        smoothed = gaussian_filter(vh, sigma)
        # Laplacian highlights edges
        lap = ndimage.laplace(smoothed)
        # Dark lines have negative Laplacian
        vesselness = np.maximum(vesselness, -lap)

    # Normalize to [0, 1]
    v_min, v_max = vesselness.min(), vesselness.max()
    if v_max > v_min:
        vesselness = (vesselness - v_min) / (v_max - v_min)

    return vesselness.astype(np.float32)


def compute_physics_score(
    hand: np.ndarray, slope: np.ndarray, twi: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute physics probability and VETO mask.

    Returns:
        physics_score: 0-1 probability (higher = more likely water)
        veto_mask: Boolean mask where water is IMPOSSIBLE
    """
    # VETO conditions (water cannot exist here)
    veto = np.zeros_like(hand, dtype=bool)
    veto |= hand > CONFIG["veto_hand_max"]
    veto |= slope > CONFIG["veto_slope_max"]
    veto |= (hand > 30) & (slope > 20)  # Combined threshold

    # Soft physics score using sigmoid functions
    hand_exp = np.clip((hand - 15) / 5.0, -50, 50)
    hand_score = 1.0 / (1.0 + np.exp(hand_exp))

    slope_exp = np.clip((slope - 10) / 4.0, -50, 50)
    slope_score = 1.0 / (1.0 + np.exp(slope_exp))

    twi_exp = np.clip((7 - twi) / 2.0, -50, 50)
    twi_score = 1.0 / (1.0 + np.exp(twi_exp))

    physics_score = 0.4 * hand_score + 0.4 * slope_score + 0.2 * twi_score

    return physics_score.astype(np.float32), veto


def extract_lgb_features(
    data: Dict[str, np.ndarray], feature_names: List[str]
) -> np.ndarray:
    """Extract features for LightGBM (matching training)."""
    vv = data["vv"]
    vh = data["vh"]
    dem = data.get("dem", np.zeros_like(vv))
    slope = data.get("slope", np.zeros_like(vv))
    hand = data.get("hand", np.zeros_like(vv))
    twi = data.get("twi", np.zeros_like(vv))
    mndwi = data.get("mndwi", None)

    features = []

    # Basic SAR features
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

    # Otsu-like
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

    # MNDWI if model uses it
    if "MNDWI" in feature_names:
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


# =============================================================================
# FINAL ENSEMBLE CLASS
# =============================================================================


class FinalEnsembleV12:
    """
    Production-ready ensemble combining:
    - LightGBM v9 (IoU 0.882)
    - U-Net v9 SOTA (IoU 0.685)
    - SOTA Post-Processing (MST, Fuzzy, Physics)
    """

    def __init__(self):
        self.lgb_model = None
        self.unet_model = None
        self.device = None
        self.feature_names = []

        # SOTA components
        if HAS_SOTA:
            self.mst_connector = MSTRiverConnector(
                max_gap_pixels=CONFIG["mst_max_gap"],
                vh_threshold=CONFIG["mst_vh_threshold"],
            )
            self.fuzzy = MamdaniFuzzyController()
        else:
            self.mst_connector = SimpleMSTConnector(
                max_gap_pixels=CONFIG["mst_max_gap"],
                vh_threshold=CONFIG["mst_vh_threshold"],
            )
            self.fuzzy = SimpleFuzzyController()

    def load_models(self, lgb_path: str = None, unet_path: str = None):
        """Load all models."""
        # LightGBM
        lgb_path = lgb_path or CONFIG["lgb_model_path"]
        try:
            import lightgbm as lgb

            self.lgb_model = lgb.Booster(model_file=lgb_path)
            self.feature_names = self.lgb_model.feature_name()
            logger.info(f"LightGBM loaded: {len(self.feature_names)} features")
        except Exception as e:
            logger.error(f"LightGBM load failed: {e}")
            raise

        # U-Net v9
        unet_path = unet_path or CONFIG["unet_v9_path"]
        try:
            self.unet_model, self.device = load_unet_v9(unet_path)
            if self.unet_model is not None:
                logger.info(f"U-Net v9 loaded on {self.device}")
        except Exception as e:
            logger.warning(f"U-Net v9 load failed: {e}")
            self.unet_model = None

    def predict_lgb(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """LightGBM prediction."""
        features = extract_lgb_features(data, self.feature_names)
        h, w, n = features.shape
        X = features.reshape(-1, n)
        proba = self.lgb_model.predict(X).reshape(h, w)
        return proba.astype(np.float32)

    def predict_unet(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """U-Net v9 prediction with 9 channels."""
        import torch

        if self.unet_model is None:
            return None

        vv = data["vv"]
        vh = data["vh"]
        dem = data.get("dem", np.zeros_like(vv))
        slope = data.get("slope", np.zeros_like(vv))
        hand = data.get("hand", np.zeros_like(vv))
        twi = data.get("twi", np.zeros_like(vv))
        mndwi = data.get("mndwi", np.zeros_like(vv))

        # Compute additional channels
        physics_score, _ = compute_physics_score(hand, slope, twi)
        frangi = compute_frangi_vesselness(vh)

        h, w = vv.shape

        # Pad to multiple of 16
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16

        # Stack 9 channels: VV, VH, DEM, Slope, HAND, TWI, MNDWI, Physics, Frangi
        inputs = np.stack(
            [vv, vh, dem, slope, hand, twi, mndwi, physics_score, frangi], axis=0
        )

        if pad_h > 0 or pad_w > 0:
            inputs = np.pad(inputs, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

        # Normalize each channel
        for i in range(9):
            mean = inputs[i].mean()
            std = inputs[i].std() + 1e-6
            inputs[i] = (inputs[i] - mean) / std

        with torch.no_grad():
            x = torch.from_numpy(inputs[np.newaxis]).float().to(self.device)
            out = self.unet_model(x)
            proba = torch.sigmoid(out).cpu().numpy()[0, 0]

        proba = proba[:h, :w]
        return proba.astype(np.float32)

    def apply_mst_gap_healing(
        self, water_mask: np.ndarray, vh: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """Apply MST-based gap healing for river connectivity."""
        healed, n_connections = self.mst_connector.connect(water_mask, vh)
        return healed, n_connections

    def apply_fuzzy_refinement(
        self, proba: np.ndarray, vh: np.ndarray, slope: np.ndarray, hand: np.ndarray
    ) -> np.ndarray:
        """Apply fuzzy logic for edge case handling."""
        if not CONFIG["fuzzy_enable"]:
            return proba

        fuzzy_prob = self.fuzzy.predict(vh, slope, hand)

        # Blend: use fuzzy to boost uncertain regions
        uncertain = (proba > 0.3) & (proba < 0.7)
        result = proba.copy()
        result = np.where(uncertain, 0.7 * proba + 0.3 * fuzzy_prob, result)

        return result

    def remove_small_regions(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        """Remove connected components smaller than min_size."""
        labeled, num = scipy_label(mask > 0.5)
        if num == 0:
            return mask

        cleaned = np.zeros_like(mask, dtype=np.float32)
        for i in range(1, num + 1):
            region = labeled == i
            if region.sum() >= min_size:
                cleaned = np.where(region, 1.0, cleaned)

        return cleaned

    def detect(self, data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Full detection pipeline.

        Returns:
            water_mask: Binary water mask
            metadata: Detection statistics and diagnostics
        """
        start_time = time.time()
        metadata = {"components": [], "timings": {}}

        # Validate and clean data
        required = ["vv", "vh"]
        optional = ["dem", "slope", "hand", "twi", "mndwi"]

        for key in required:
            if key not in data:
                raise ValueError(f"Missing required channel: {key}")
            data[key] = np.nan_to_num(data[key].astype(np.float32))

        for key in optional:
            if key in data:
                data[key] = np.nan_to_num(data[key].astype(np.float32))
            else:
                data[key] = np.zeros_like(data["vv"])

        # Clip to valid ranges
        data["slope"] = np.clip(data["slope"], 0, 90)
        data["hand"] = np.clip(data["hand"], 0, 500)
        data["twi"] = np.clip(data["twi"], 0, 30)
        data["mndwi"] = np.clip(data["mndwi"], -1, 1)

        h, w = data["vv"].shape

        # Step 1: Physics score and VETO
        t0 = time.time()
        physics_score, veto_mask = compute_physics_score(
            data["hand"], data["slope"], data["twi"]
        )
        metadata["timings"]["physics"] = time.time() - t0
        metadata["veto_fraction"] = float(veto_mask.mean())
        metadata["components"].append("physics")

        # Step 2: LightGBM prediction
        t0 = time.time()
        lgb_proba = self.predict_lgb(data)
        metadata["timings"]["lgb"] = time.time() - t0
        metadata["lgb_water_fraction"] = float((lgb_proba > 0.5).mean())
        metadata["components"].append("lgb")

        # Step 3: U-Net prediction (if available)
        if self.unet_model is not None:
            t0 = time.time()
            unet_proba = self.predict_unet(data)
            metadata["timings"]["unet"] = time.time() - t0
            metadata["unet_water_fraction"] = float((unet_proba > 0.5).mean())
            metadata["components"].append("unet")
        else:
            unet_proba = np.zeros_like(lgb_proba)
            metadata["unet_water_fraction"] = 0.0

        # Step 4: Ensemble combination
        t0 = time.time()

        # Weighted average with physics
        if self.unet_model is not None:
            combined = (
                CONFIG["lgb_weight"] * lgb_proba
                + CONFIG["unet_weight"] * unet_proba
                + CONFIG["physics_weight"] * physics_score
            )
        else:
            # LGB + Physics only
            combined = 0.9 * lgb_proba + 0.1 * physics_score

        # Apply physics VETO
        combined = np.where(veto_mask, 0.0, combined)

        metadata["timings"]["ensemble"] = time.time() - t0

        # Step 5: Fuzzy refinement
        if CONFIG["fuzzy_enable"]:
            t0 = time.time()
            combined = self.apply_fuzzy_refinement(
                combined, data["vh"], data["slope"], data["hand"]
            )
            metadata["timings"]["fuzzy"] = time.time() - t0
            metadata["components"].append("fuzzy")

        # Step 6: Threshold to binary
        water_mask = (combined > CONFIG["water_threshold"]).astype(np.float32)

        # Step 7: MST gap healing
        t0 = time.time()
        water_mask, n_connections = self.apply_mst_gap_healing(water_mask, data["vh"])
        metadata["timings"]["mst"] = time.time() - t0
        metadata["mst_connections"] = n_connections
        metadata["components"].append("mst")

        # Step 8: Remove small regions
        water_mask = self.remove_small_regions(water_mask, CONFIG["min_region_size"])

        # Final metadata
        metadata["final_water_fraction"] = float(water_mask.mean())
        metadata["total_time"] = time.time() - start_time

        return water_mask, metadata


# =============================================================================
# METRICS
# =============================================================================


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute IoU, Precision, Recall, F1."""
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

    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Test ensemble on all chips."""
    logger.info("=" * 70)
    logger.info("FINAL ENSEMBLE v12 - SOTA Integration Test")
    logger.info("=" * 70)
    logger.info(f"SOTA Module Available: {HAS_SOTA}")
    logger.info(f"Components: LightGBM v9 + U-Net v9 + MST + Fuzzy + Physics")

    # Initialize ensemble
    ensemble = FinalEnsembleV12()

    try:
        ensemble.load_models()
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.info("Please ensure you're running on the server with models available")
        return

    # Load chips
    chip_files = sorted(CONFIG["chip_dir"].glob("*_with_truth.npy"))
    logger.info(f"Found {len(chip_files)} chips")

    if len(chip_files) == 0:
        logger.error("No chips found! Check chip_dir path")
        return

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
                    "gt_water": float(label.mean()),
                    "pred_water": meta["final_water_fraction"],
                    "mst_connections": meta.get("mst_connections", 0),
                    "time": meta["total_time"],
                }
            )

            all_preds.append(pred.flatten())
            all_labels.append(label.flatten())

            logger.info(
                f"  {name}: IoU={metrics['iou']:.4f}, P={metrics['precision']:.4f}, "
                f"R={metrics['recall']:.4f}, MST={meta.get('mst_connections', 0)}"
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

        # Comparison
        logger.info("\n" + "-" * 40)
        logger.info("COMPARISON WITH BASELINES:")
        logger.info("-" * 40)
        logger.info(f"  LightGBM v9 alone:     IoU 0.882")
        logger.info(f"  U-Net v9 alone:        IoU 0.685")
        logger.info(f"  Ensemble v12:          IoU {overall['iou']:.4f}")
        logger.info(f"  vs LightGBM:           {(overall['iou'] - 0.882) * 100:+.2f}%")

        # Total MST connections
        total_mst = sum(r.get("mst_connections", 0) for r in results)
        logger.info(f"\nTotal MST gap connections: {total_mst}")

        # Save results
        output = {
            "timestamp": datetime.now().isoformat(),
            "version": "12.0",
            "method": "FinalEnsembleV12 (LGB v9 + UNet v9 + MST + Fuzzy + Physics)",
            "sota_module": HAS_SOTA,
            "config": {
                k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()
            },
            "overall_metrics": overall,
            "comparison": {
                "lgb_v9_alone": 0.882,
                "unet_v9_alone": 0.685,
                "ensemble_v12": overall["iou"],
                "improvement_vs_lgb": overall["iou"] - 0.882,
            },
            "per_chip_results": results,
        }

        results_path = CONFIG["results_dir"] / "ensemble_v12_results.json"
        CONFIG["results_dir"].mkdir(parents=True, exist_ok=True)

        with open(results_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nResults saved to {results_path}")

    logger.info("=" * 70)
    logger.info("ENSEMBLE v12 TEST COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
