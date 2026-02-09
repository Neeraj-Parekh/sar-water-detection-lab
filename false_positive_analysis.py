#!/usr/bin/env python3
"""
================================================================================
SHADOW AND WET-SOIL FALSE POSITIVE ANALYSIS
================================================================================
Detailed analysis of false positives to understand and reduce errors.

Categories analyzed:
1. Terrain shadows (steep slopes + low backscatter)
2. Wet soil (low HAND + moderate backscatter)
3. Urban areas (double-bounce + geometric patterns)
4. Vegetation (high volume scattering)

Author: SAR Water Detection Project
Date: January 2026
================================================================================
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import rasterio
from scipy import ndimage
from scipy.ndimage import (
    uniform_filter,
    sobel,
    binary_erosion,
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    label,
    minimum_filter,
    maximum_filter,
)
from sklearn.model_selection import train_test_split

import lightgbm as lgb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "random_seed": 42,
    "chip_dirs": [
        Path("/home/mit-aoe/sar_water_detection/chips"),
        Path("/home/mit-aoe/sar_water_detection/chips_expanded"),
    ],
    "model_dir": Path("/home/mit-aoe/sar_water_detection/models"),
    "results_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    "viz_dir": Path("/home/mit-aoe/sar_water_detection/visualizations"),
    "bands": {"VV": 0, "VH": 1, "DEM": 3, "HAND": 4, "SLOPE": 5, "TWI": 6, "TRUTH": 7},
}

CONFIG["viz_dir"].mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(CONFIG["results_dir"] / "false_positive_analysis.log"),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def db_to_linear(db: np.ndarray) -> np.ndarray:
    return np.power(10, db / 10)


def linear_to_db(linear: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.clip(linear, 1e-10, None))


def load_chips() -> Tuple[List[np.ndarray], List[str]]:
    """Load all chips."""
    chips = []
    names = []
    for chip_dir in CONFIG["chip_dirs"]:
        if not chip_dir.exists():
            continue
        for f in chip_dir.glob("*.npy"):
            try:
                chip = np.load(f).astype(np.float32)
                if chip.shape[0] >= 8 and np.nansum(chip[7]) > 0:
                    chips.append(chip)
                    names.append(f.stem)
            except:
                pass
        for f in chip_dir.glob("*.tif"):
            try:
                with rasterio.open(f) as src:
                    chip = src.read().astype(np.float32)
                if chip.shape[0] >= 8 and np.nansum(chip[7]) > 0:
                    chips.append(chip)
                    names.append(f.stem)
            except:
                pass
    return chips, names


# =============================================================================
# FALSE POSITIVE CLASSIFICATION
# =============================================================================


def classify_false_positives(
    fp_mask: np.ndarray,
    vv: np.ndarray,
    vh: np.ndarray,
    hand: np.ndarray,
    slope: np.ndarray,
    dem: np.ndarray,
    twi: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Classify false positive pixels into categories.

    Categories:
    1. Shadow: Steep slope (>15°) + Low VH (<-20 dB) + Aspect away from radar
    2. Wet Soil: Low HAND (<5m) + Moderate VH (-20 to -15 dB)
    3. Urban Double-Bounce: High VV (>-10 dB) + High texture
    4. Dark Vegetation: Low VH + High RVI
    5. Unknown: Doesn't fit other categories
    """
    categories = {}

    # Initialize all as unknown
    remaining = fp_mask.copy()

    # 1. SHADOW-LIKE
    # Steep terrain with low backscatter
    shadow_mask = (
        (slope > 15)  # Steep slope
        & (vh < -20)  # Very low VH (could be shadow)
        & fp_mask
    )
    # Also check if slope faces away from assumed radar look direction
    # For ascending pass, radar looks right (~90° azimuth)
    dem_dy = sobel(dem, axis=0)
    dem_dx = sobel(dem, axis=1)
    aspect = np.arctan2(dem_dy, dem_dx)
    aspect_deg = np.degrees(aspect)
    # Shadow more likely when facing away (aspect ~90° for right-looking)
    shadow_aspect = np.abs(np.abs(aspect_deg) - 90) < 60
    shadow_mask = shadow_mask & shadow_aspect

    categories["shadow"] = shadow_mask.astype(np.float32)
    remaining = remaining & ~shadow_mask

    # 2. WET SOIL
    # Low elevation + moderate backscatter (brighter than water but in lowland)
    wet_soil_mask = (
        (hand < 5)  # Low HAND (near drainage)
        & (vh > -22)  # Brighter than typical water
        & (vh < -14)  # But not too bright
        & (slope < 10)  # Relatively flat
        & remaining
    )
    categories["wet_soil"] = wet_soil_mask.astype(np.float32)
    remaining = remaining & ~wet_soil_mask

    # 3. URBAN DOUBLE-BOUNCE
    # High VV (double-bounce), high texture variability
    vv_local_std = np.sqrt(
        np.maximum(uniform_filter(vv**2, size=5) - uniform_filter(vv, size=5) ** 2, 0)
    )
    high_texture = vv_local_std > 3  # High local variability

    urban_mask = (
        (vv > -12)  # High VV (double-bounce)
        & high_texture  # Textured (buildings)
        & (slope < 5)  # Flat (urban areas)
        & remaining
    )
    categories["urban"] = urban_mask.astype(np.float32)
    remaining = remaining & ~urban_mask

    # 4. DARK VEGETATION
    # Low backscatter but high RVI (volume scattering)
    vv_lin = db_to_linear(vv)
    vh_lin = db_to_linear(vh)
    rvi = 4 * vh_lin / (vv_lin + vh_lin + 1e-10)

    dark_veg_mask = (
        (vh < -18)  # Low VH
        & (rvi > 0.5)  # High RVI (vegetation-like)
        & (slope < 15)  # Not too steep
        & remaining
    )
    categories["dark_vegetation"] = dark_veg_mask.astype(np.float32)
    remaining = remaining & ~dark_veg_mask

    # 5. UNKNOWN
    categories["unknown"] = remaining.astype(np.float32)

    return categories


def classify_false_negatives(
    fn_mask: np.ndarray,
    vv: np.ndarray,
    vh: np.ndarray,
    hand: np.ndarray,
    slope: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Classify false negative pixels (missed water).

    Categories:
    1. Bright Water: High VH (>-15 dB) - wind-roughened water
    2. Narrow Water: Small connected components
    3. Boundary Water: Near edges of water bodies
    4. High HAND Water: Water at unexpected elevation
    """
    categories = {}
    remaining = fn_mask.copy()

    # 1. BRIGHT WATER (wind-roughened)
    bright_water = (
        (vh > -16)  # Unusually bright for water
        & fn_mask
    )
    categories["bright_water"] = bright_water.astype(np.float32)
    remaining = remaining & ~bright_water

    # 2. HIGH HAND WATER (unexpected elevation)
    high_hand_water = (
        (hand > 10)  # Higher than typical water
        & remaining
    )
    categories["high_hand_water"] = high_hand_water.astype(np.float32)
    remaining = remaining & ~high_hand_water

    # 3. STEEP WATER (near slopes - could be misclassified)
    steep_water = (
        (slope > 10)  # Near steep terrain
        & remaining
    )
    categories["steep_water"] = steep_water.astype(np.float32)
    remaining = remaining & ~steep_water

    # 4. UNKNOWN
    categories["unknown"] = remaining.astype(np.float32)

    return categories


# =============================================================================
# FEATURE EXTRACTION (matching training)
# =============================================================================


def extract_features_simple(chip: np.ndarray) -> np.ndarray:
    """Extract 69 features matching training."""
    from skimage.morphology import disk, opening, closing, white_tophat, black_tophat
    from skimage.filters import threshold_otsu

    vv = chip[0]
    vh = chip[1]
    dem = chip[3]
    hand = chip[4]
    slope = chip[5]
    twi = chip[6]

    features = []

    # Basic SAR (4)
    features.extend([vv, vh, vv - vh])
    vv_lin = db_to_linear(vv)
    vh_lin = db_to_linear(vh)
    features.append(np.log10(vv_lin / (vh_lin + 1e-10) + 1e-10))

    # Topographic (6)
    features.extend([hand, slope, twi, dem])
    dem_dx = sobel(dem, axis=1)
    dem_dy = sobel(dem, axis=0)
    features.append(np.sqrt(dem_dx**2 + dem_dy**2))
    features.append(np.arctan2(dem_dy, dem_dx))

    # Polarimetric (4)
    vv_local = uniform_filter(vv_lin, size=5)
    vh_local = uniform_filter(vh_lin, size=5)
    total = vv_local + vh_local + 1e-10
    p = np.clip(vv_local / total, 1e-10, 1 - 1e-10)
    features.append(-p * np.log2(p) - (1 - p) * np.log2(1 - p))
    features.append(np.degrees(np.arctan2(np.sqrt(vh_local), np.sqrt(vv_local))))
    features.append(4 * vh_local / (vv_local + vh_local + 1e-10))
    features.append(linear_to_db(vv_lin + 2 * vh_lin))

    # Multi-scale (26)
    for scale in [3, 5, 9, 15, 21]:
        vv_mean = uniform_filter(vv, size=scale)
        vv_sq = uniform_filter(vv**2, size=scale)
        vv_std = np.sqrt(np.maximum(vv_sq - vv_mean**2, 0))
        features.extend([vv_mean, vv_std])
        vh_mean = uniform_filter(vh, size=scale)
        vh_sq = uniform_filter(vh**2, size=scale)
        vh_std = np.sqrt(np.maximum(vh_sq - vh_mean**2, 0))
        features.extend([vh_mean, vh_std])
        if scale <= 9:
            features.extend(
                [minimum_filter(vv, size=scale), minimum_filter(vh, size=scale)]
            )

    # Speckle (4)
    for band in [vv, vh]:
        band_lin = db_to_linear(band)
        local_mean = uniform_filter(band_lin, size=9)
        local_sq = uniform_filter(band_lin**2, size=9)
        local_var = local_sq - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 1e-10))
        features.append(np.clip((local_mean / local_std) ** 2, 0, 100))
        features.append(np.clip(local_std / (local_mean + 1e-10), 0, 5))

    # Texture (6)
    for band in [vv, vh]:
        local_mean = uniform_filter(band.astype(np.float64), size=11)
        local_sq = uniform_filter(band.astype(np.float64) ** 2, size=11)
        contrast = np.sqrt(np.maximum(local_sq - local_mean**2, 0))
        local_max = maximum_filter(band, size=11)
        local_min = minimum_filter(band, size=11)
        homogeneity = 1.0 / (1.0 + local_max - local_min + 1e-10)
        energy = 1.0 / (1.0 + contrast)
        features.extend([contrast, homogeneity, energy])

    # Morphological (4)
    vh_norm = (
        (vh - np.nanmin(vh)) / (np.nanmax(vh) - np.nanmin(vh) + 1e-10) * 255
    ).astype(np.uint8)
    selem = disk(3)
    features.append(opening(vh_norm, selem).astype(np.float32) / 255)
    features.append(closing(vh_norm, selem).astype(np.float32) / 255)
    features.append(white_tophat(vh_norm, selem).astype(np.float32) / 255)
    features.append(black_tophat(vh_norm, selem).astype(np.float32) / 255)

    # Line detection (1)
    responses = []
    for angle in np.linspace(0, np.pi, 8, endpoint=False):
        kernel = np.zeros((15, 15))
        center = 7
        for t in range(-center, center + 1):
            x = int(center + t * np.cos(angle))
            y = int(center + t * np.sin(angle))
            if 0 <= x < 15 and 0 <= y < 15:
                kernel[y, x] = 1
        kernel = kernel / (kernel.sum() + 1e-10)
        responses.append(ndimage.convolve(vh, kernel))
    features.append(np.maximum.reduce(responses))

    # Adaptive thresholds (6)
    vv_clean = vv[~np.isnan(vv)].flatten()
    vh_clean = vh[~np.isnan(vh)].flatten()
    try:
        vv_otsu = threshold_otsu(
            np.clip(vv_clean, np.percentile(vv_clean, 1), np.percentile(vv_clean, 99))
        )
        vh_otsu = threshold_otsu(
            np.clip(vh_clean, np.percentile(vh_clean, 1), np.percentile(vh_clean, 99))
        )
    except:
        vv_otsu = np.median(vv_clean)
        vh_otsu = np.median(vh_clean)
    features.extend([vv - vv_otsu, vh - vh_otsu])
    features.extend(
        [(vv < vv_otsu).astype(np.float32), (vh < vh_otsu).astype(np.float32)]
    )
    features.extend(
        [(vv < vv_otsu).astype(np.float32), (vh < vh_otsu).astype(np.float32)]
    )

    # Physics composite (5)
    hand_score = np.clip(1 - hand / 10.0, 0, 1)
    slope_score = np.clip(1 - slope / 15.0, 0, 1)
    vh_score = np.clip((-18.0 - vh) / 10, 0, 1)
    twi_score = np.clip((twi - 5) / 10, 0, 1)
    physics = (hand_score * slope_score * vh_score * twi_score) ** 0.25
    features.extend([physics, hand_score, slope_score, vh_score, twi_score])

    # Gradients (3)
    vh_dx = sobel(vh, axis=1)
    vh_dy = sobel(vh, axis=0)
    features.append(np.sqrt(vh_dx**2 + vh_dy**2))
    vv_dx = sobel(vv, axis=1)
    vv_dy = sobel(vv, axis=0)
    features.append(np.sqrt(vv_dx**2 + vv_dy**2))
    features.append(ndimage.laplace(vh))

    feature_stack = np.stack(features, axis=0).astype(np.float32)
    feature_stack = np.nan_to_num(feature_stack, nan=0.0, posinf=0.0, neginf=0.0)
    return feature_stack


# =============================================================================
# VISUALIZATION
# =============================================================================


def plot_fp_fn_analysis(fp_categories: Dict, fn_categories: Dict, save_path: Path):
    """Plot false positive/negative breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # FP breakdown
    fp_counts = {k: int(v.sum()) for k, v in fp_categories.items() if v.sum() > 0}
    if fp_counts:
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(fp_counts)))
        axes[0].pie(
            fp_counts.values(),
            labels=fp_counts.keys(),
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        axes[0].set_title(
            f"False Positive Categories\n(Total: {sum(fp_counts.values())} pixels)"
        )

    # FN breakdown
    fn_counts = {k: int(v.sum()) for k, v in fn_categories.items() if v.sum() > 0}
    if fn_counts:
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(fn_counts)))
        axes[1].pie(
            fn_counts.values(),
            labels=fn_counts.keys(),
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        axes[1].set_title(
            f"False Negative Categories\n(Total: {sum(fn_counts.values())} pixels)"
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_fp_characteristics(
    fp_vh: np.ndarray,
    fp_hand: np.ndarray,
    fp_slope: np.ndarray,
    fp_categories: Dict[str, np.ndarray],
    save_path: Path,
):
    """Plot characteristics of false positive pixels."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # VH distribution by category
    ax = axes[0, 0]
    for cat_name, cat_mask in fp_categories.items():
        if cat_mask.sum() > 10:
            cat_vh = fp_vh[cat_mask.astype(bool)]
            ax.hist(cat_vh, bins=50, alpha=0.5, label=cat_name, density=True)
    ax.set_xlabel("VH (dB)")
    ax.set_ylabel("Density")
    ax.set_title("VH Distribution by FP Category")
    ax.legend()
    ax.axvline(x=-18, color="r", linestyle="--", label="Water threshold")

    # HAND distribution
    ax = axes[0, 1]
    for cat_name, cat_mask in fp_categories.items():
        if cat_mask.sum() > 10:
            cat_hand = fp_hand[cat_mask.astype(bool)]
            ax.hist(cat_hand, bins=50, alpha=0.5, label=cat_name, density=True)
    ax.set_xlabel("HAND (m)")
    ax.set_ylabel("Density")
    ax.set_title("HAND Distribution by FP Category")
    ax.legend()

    # Slope distribution
    ax = axes[1, 0]
    for cat_name, cat_mask in fp_categories.items():
        if cat_mask.sum() > 10:
            cat_slope = fp_slope[cat_mask.astype(bool)]
            ax.hist(cat_slope, bins=50, alpha=0.5, label=cat_name, density=True)
    ax.set_xlabel("Slope (degrees)")
    ax.set_ylabel("Density")
    ax.set_title("Slope Distribution by FP Category")
    ax.legend()

    # VH vs HAND scatter
    ax = axes[1, 1]
    sample_idx = np.random.choice(len(fp_vh), size=min(5000, len(fp_vh)), replace=False)
    scatter = ax.scatter(
        fp_vh[sample_idx],
        fp_hand[sample_idx],
        c=fp_slope[sample_idx],
        cmap="viridis",
        alpha=0.3,
        s=1,
    )
    ax.set_xlabel("VH (dB)")
    ax.set_ylabel("HAND (m)")
    ax.set_title("FP Pixels: VH vs HAND (colored by Slope)")
    plt.colorbar(scatter, ax=ax, label="Slope (deg)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_spatial_distribution(
    fp_mask: np.ndarray,
    fn_mask: np.ndarray,
    chip: np.ndarray,
    chip_name: str,
    save_path: Path,
):
    """Plot spatial distribution of errors on a chip."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    vv = chip[0]
    vh = chip[1]
    truth = chip[7]

    # VV
    axes[0, 0].imshow(vv, cmap="gray", vmin=-30, vmax=0)
    axes[0, 0].set_title("VV (dB)")
    axes[0, 0].axis("off")

    # VH
    axes[0, 1].imshow(vh, cmap="gray", vmin=-35, vmax=-5)
    axes[0, 1].set_title("VH (dB)")
    axes[0, 1].axis("off")

    # Truth
    axes[0, 2].imshow(truth, cmap="Blues", vmin=0, vmax=1)
    axes[0, 2].set_title("Ground Truth")
    axes[0, 2].axis("off")

    # FP map
    axes[1, 0].imshow(fp_mask, cmap="Reds", vmin=0, vmax=1)
    axes[1, 0].set_title(f"False Positives ({int(fp_mask.sum())} pixels)")
    axes[1, 0].axis("off")

    # FN map
    axes[1, 1].imshow(fn_mask, cmap="Blues", vmin=0, vmax=1)
    axes[1, 1].set_title(f"False Negatives ({int(fn_mask.sum())} pixels)")
    axes[1, 1].axis("off")

    # Combined error overlay
    error_rgb = np.zeros((*truth.shape, 3))
    error_rgb[fp_mask > 0] = [1, 0, 0]  # FP = red
    error_rgb[fn_mask > 0] = [0, 0, 1]  # FN = blue
    axes[1, 2].imshow(error_rgb)
    axes[1, 2].set_title("Errors (Red=FP, Blue=FN)")
    axes[1, 2].axis("off")

    plt.suptitle(f"Error Analysis: {chip_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# MAIN ANALYSIS
# =============================================================================


def main():
    logger.info("=" * 80)
    logger.info("SHADOW AND WET-SOIL FALSE POSITIVE ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Started: {datetime.now().isoformat()}")

    np.random.seed(CONFIG["random_seed"])

    # Load data
    chips, names = load_chips()
    logger.info(f"Loaded {len(chips)} chips")

    # Split
    _, test_data = train_test_split(
        list(zip(chips, names)),
        test_size=0.15,
        random_state=CONFIG["random_seed"],
    )
    test_chips = [x[0] for x in test_data]
    test_names = [x[1] for x in test_data]

    # Load LightGBM model
    logger.info("\nLoading LightGBM model...")
    lgb_model = lgb.Booster(
        model_file=str(CONFIG["model_dir"] / "lightgbm_v4_comprehensive.txt")
    )

    # Aggregate results
    all_fp_vh = []
    all_fp_hand = []
    all_fp_slope = []
    all_fn_vh = []
    all_fn_hand = []

    total_fp_categories = {}
    total_fn_categories = {}

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_test_chips": len(test_chips),
        "per_chip": [],
    }

    logger.info(f"\nAnalyzing {len(test_chips)} test chips...")

    for i, (chip, name) in enumerate(zip(test_chips, test_names)):
        if (i + 1) % 10 == 0:
            logger.info(f"  Processing {i + 1}/{len(test_chips)}")

        vv = chip[0]
        vh = chip[1]
        dem = chip[3]
        hand = chip[4]
        slope = chip[5]
        twi = chip[6]
        truth = chip[7]

        # Get LightGBM prediction
        features = extract_features_simple(chip)
        h, w = truth.shape
        X = features.reshape(features.shape[0], -1).T
        lgb_proba = lgb_model.predict(X).reshape(h, w)
        pred_binary = (lgb_proba > 0.5).astype(np.uint8)
        truth_binary = (truth > 0.5).astype(np.uint8)

        # Identify FP and FN
        fp_mask = pred_binary & (~truth_binary)
        fn_mask = (~pred_binary) & truth_binary

        # Classify FP
        fp_categories = classify_false_positives(fp_mask, vv, vh, hand, slope, dem, twi)
        fn_categories = classify_false_negatives(fn_mask, vv, vh, hand, slope)

        # Aggregate
        if fp_mask.sum() > 0:
            all_fp_vh.extend(vh[fp_mask].flatten().tolist())
            all_fp_hand.extend(hand[fp_mask].flatten().tolist())
            all_fp_slope.extend(slope[fp_mask].flatten().tolist())

        if fn_mask.sum() > 0:
            all_fn_vh.extend(vh[fn_mask].flatten().tolist())
            all_fn_hand.extend(hand[fn_mask].flatten().tolist())

        # Aggregate categories
        for cat, mask in fp_categories.items():
            if cat not in total_fp_categories:
                total_fp_categories[cat] = 0
            total_fp_categories[cat] += int(mask.sum())

        for cat, mask in fn_categories.items():
            if cat not in total_fn_categories:
                total_fn_categories[cat] = 0
            total_fn_categories[cat] += int(mask.sum())

        # Per-chip results
        chip_result = {
            "name": name,
            "n_fp": int(fp_mask.sum()),
            "n_fn": int(fn_mask.sum()),
            "fp_categories": {k: int(v.sum()) for k, v in fp_categories.items()},
            "fn_categories": {k: int(v.sum()) for k, v in fn_categories.items()},
        }
        results["per_chip"].append(chip_result)

        # Save visualization for first 5 chips with significant errors
        if i < 5 and (fp_mask.sum() > 100 or fn_mask.sum() > 100):
            plot_error_spatial_distribution(
                fp_mask,
                fn_mask,
                chip,
                name,
                CONFIG["viz_dir"] / f"error_analysis_{name}.png",
            )

    # Convert to arrays
    all_fp_vh = np.array(all_fp_vh)
    all_fp_hand = np.array(all_fp_hand)
    all_fp_slope = np.array(all_fp_slope)
    all_fn_vh = np.array(all_fn_vh)
    all_fn_hand = np.array(all_fn_hand)

    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("FALSE POSITIVE CATEGORY BREAKDOWN")
    logger.info("=" * 60)
    total_fp = sum(total_fp_categories.values())
    for cat, count in sorted(total_fp_categories.items(), key=lambda x: -x[1]):
        pct = count / total_fp * 100 if total_fp > 0 else 0
        logger.info(f"  {cat:20s}: {count:8d} ({pct:5.1f}%)")

    logger.info("\n" + "=" * 60)
    logger.info("FALSE NEGATIVE CATEGORY BREAKDOWN")
    logger.info("=" * 60)
    total_fn = sum(total_fn_categories.values())
    for cat, count in sorted(total_fn_categories.items(), key=lambda x: -x[1]):
        pct = count / total_fn * 100 if total_fn > 0 else 0
        logger.info(f"  {cat:20s}: {count:8d} ({pct:5.1f}%)")

    # Statistics
    logger.info("\n" + "=" * 60)
    logger.info("FALSE POSITIVE STATISTICS")
    logger.info("=" * 60)
    if len(all_fp_vh) > 0:
        logger.info(
            f"  VH:    mean={np.mean(all_fp_vh):.2f}, std={np.std(all_fp_vh):.2f}, "
            f"range=[{np.min(all_fp_vh):.2f}, {np.max(all_fp_vh):.2f}]"
        )
        logger.info(
            f"  HAND:  mean={np.mean(all_fp_hand):.2f}, std={np.std(all_fp_hand):.2f}"
        )
        logger.info(
            f"  Slope: mean={np.mean(all_fp_slope):.2f}, std={np.std(all_fp_slope):.2f}"
        )

    logger.info("\n" + "=" * 60)
    logger.info("FALSE NEGATIVE STATISTICS")
    logger.info("=" * 60)
    if len(all_fn_vh) > 0:
        logger.info(
            f"  VH:    mean={np.mean(all_fn_vh):.2f}, std={np.std(all_fn_vh):.2f}, "
            f"range=[{np.min(all_fn_vh):.2f}, {np.max(all_fn_vh):.2f}]"
        )
        logger.info(
            f"  HAND:  mean={np.mean(all_fn_hand):.2f}, std={np.std(all_fn_hand):.2f}"
        )

    # Save aggregate results
    results["fp_categories"] = total_fp_categories
    results["fn_categories"] = total_fn_categories
    results["fp_statistics"] = {
        "count": len(all_fp_vh),
        "vh_mean": float(np.mean(all_fp_vh)) if len(all_fp_vh) > 0 else None,
        "vh_std": float(np.std(all_fp_vh)) if len(all_fp_vh) > 0 else None,
        "hand_mean": float(np.mean(all_fp_hand)) if len(all_fp_hand) > 0 else None,
        "slope_mean": float(np.mean(all_fp_slope)) if len(all_fp_slope) > 0 else None,
    }
    results["fn_statistics"] = {
        "count": len(all_fn_vh),
        "vh_mean": float(np.mean(all_fn_vh)) if len(all_fn_vh) > 0 else None,
        "hand_mean": float(np.mean(all_fn_hand)) if len(all_fn_hand) > 0 else None,
    }

    # Generate plots
    logger.info("\nGenerating visualizations...")

    # Create dummy category arrays for aggregate plotting
    fp_cat_arrays = {}
    for cat, count in total_fp_categories.items():
        arr = np.zeros(len(all_fp_vh), dtype=bool)
        if count > 0:
            # Assign proportionally
            n_assign = min(count, len(all_fp_vh))
            arr[:n_assign] = True
        fp_cat_arrays[cat] = arr

    if len(all_fp_vh) > 0:
        plot_fp_characteristics(
            all_fp_vh,
            all_fp_hand,
            all_fp_slope,
            fp_cat_arrays,
            CONFIG["viz_dir"] / "fp_characteristics.png",
        )
        logger.info("  Saved fp_characteristics.png")

    # Save results
    with open(CONFIG["results_dir"] / "false_positive_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(
        f"\nResults saved to: {CONFIG['results_dir'] / 'false_positive_analysis_results.json'}"
    )
    logger.info(f"Completed: {datetime.now().isoformat()}")

    # Recommendations
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATIONS FOR IMPROVEMENT")
    logger.info("=" * 60)

    if total_fp_categories.get("shadow", 0) > total_fp * 0.1:
        logger.info(
            "1. SHADOW REDUCTION: Add terrain shadow mask using DEM aspect and radar geometry"
        )
        logger.info("   - Implement layover/shadow detection based on incidence angle")
        logger.info("   - Add aspect-based feature to discriminate shadows from water")

    if total_fp_categories.get("wet_soil", 0) > total_fp * 0.1:
        logger.info("2. WET SOIL DISCRIMINATION: Use temporal stability")
        logger.info("   - Water is temporally stable, wet soil varies")
        logger.info("   - Add multi-temporal SAR features if available")
        logger.info("   - Use stricter VH threshold in low-HAND areas")

    if total_fn_categories.get("bright_water", 0) > total_fn * 0.1:
        logger.info(
            "3. BRIGHT WATER RECOVERY: Adjust thresholds for wind-roughened water"
        )
        logger.info("   - Use texture-based features to identify rough water")
        logger.info("   - Consider wind speed data if available")

    return results


if __name__ == "__main__":
    main()
