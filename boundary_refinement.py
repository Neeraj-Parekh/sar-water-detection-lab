#!/usr/bin/env python3
"""
================================================================================
BOUNDARY REFINEMENT WITH CRF AND ACTIVE CONTOURS
================================================================================
Improves water boundary detection using:
1. Conditional Random Fields (CRF) post-processing
2. Active Contour / Level Set methods
3. Morphological boundary refinement

Author: SAR Water Detection Project
Date: January 2026
================================================================================
"""

import os
import sys
import json
import time
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
)
from sklearn.model_selection import train_test_split

import lightgbm as lgb

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
    "scales": [3, 5, 9, 15, 21],
}

CONFIG["viz_dir"].mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(CONFIG["results_dir"] / "boundary_refinement.log"),
    ],
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING & FEATURE EXTRACTION (copied from comprehensive_evaluation)
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
# CRF POST-PROCESSING
# =============================================================================


def apply_crf_refinement(
    sar_image: np.ndarray,
    probabilities: np.ndarray,
    sxy_bilateral: int = 80,
    srgb_bilateral: int = 13,
    compat_bilateral: int = 10,
    sxy_gaussian: int = 3,
    compat_gaussian: int = 3,
    n_iterations: int = 5,
) -> np.ndarray:
    """
    Apply CRF refinement adapted for SAR imagery.
    Uses log-transformed SAR for better pairwise potentials.
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError:
        logger.warning("pydensecrf not installed, using simple CRF approximation")
        return simple_crf_approximation(sar_image, probabilities)

    h, w = sar_image.shape
    n_classes = 2

    # Prepare probabilities (2 classes: water, non-water)
    probs = np.stack([1 - probabilities, probabilities], axis=0)
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    probs = probs / probs.sum(axis=0, keepdims=True)

    # Log-transform SAR for CRF
    sar_log = np.log10(np.abs(sar_image) + 1e-10)
    sar_normalized = (
        (sar_log - sar_log.min()) / (sar_log.max() - sar_log.min() + 1e-10) * 255
    ).astype(np.uint8)

    # Create pseudo-RGB from SAR
    sar_rgb = np.stack([sar_normalized] * 3, axis=-1).astype(np.uint8)

    d = dcrf.DenseCRF2D(w, h, n_classes)

    # Unary potentials
    U = unary_from_softmax(probs)
    d.setUnaryEnergy(U)

    # Pairwise Gaussian (spatial smoothness)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian)

    # Pairwise Bilateral (appearance-based)
    d.addPairwiseBilateral(
        sxy=sxy_bilateral,
        srgb=srgb_bilateral,
        rgbim=sar_rgb,
        compat=compat_bilateral,
    )

    Q = d.inference(n_iterations)
    result = np.argmax(np.array(Q).reshape((n_classes, h, w)), axis=0)

    return result.astype(np.float32)


def simple_crf_approximation(
    sar_image: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    """
    Simple CRF-like approximation using iterative bilateral filtering.
    Used when pydensecrf is not available.
    """
    from scipy.ndimage import gaussian_filter

    refined = probabilities.copy()

    for _ in range(5):
        # Spatial smoothing
        smoothed = gaussian_filter(refined, sigma=2)

        # Edge-aware component
        sar_edges = np.abs(sobel(sar_image, axis=0)) + np.abs(sobel(sar_image, axis=1))
        edge_weight = 1 / (1 + sar_edges / sar_edges.max())

        # Combine
        refined = 0.7 * refined + 0.3 * smoothed * edge_weight

    return (refined > 0.5).astype(np.float32)


# =============================================================================
# ACTIVE CONTOUR / LEVEL SET
# =============================================================================


def active_contour_levelset(
    sar_image: np.ndarray,
    initial_mask: np.ndarray,
    n_iterations: int = 100,
    dt: float = 0.5,
    mu: float = 0.2,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
) -> np.ndarray:
    """
    Level set evolution for SAR water body delineation.
    Uses Chan-Vese model adapted for multiplicative SAR noise.

    Args:
        sar_image: SAR backscatter image (linear scale)
        initial_mask: Initial segmentation (from LightGBM)
        n_iterations: Number of evolution steps
        dt: Time step
        mu: Curvature regularization weight
        lambda1, lambda2: Region fitting weights
    """
    # Initialize level set function from mask
    phi = distance_transform_edt(initial_mask) - distance_transform_edt(
        1 - initial_mask
    )
    phi = phi.astype(np.float64)

    # Convert to linear if in dB
    if sar_image.min() < 0:
        sar_lin = db_to_linear(sar_image)
    else:
        sar_lin = sar_image.copy()

    sar_lin = sar_lin.astype(np.float64)

    for i in range(n_iterations):
        # Heaviside function (smooth approximation)
        epsilon = 1.0
        H = 0.5 * (1 + (2 / np.pi) * np.arctan(phi / epsilon))

        # Compute region means (inside and outside)
        c1 = np.sum(sar_lin * H) / (np.sum(H) + 1e-10)
        c2 = np.sum(sar_lin * (1 - H)) / (np.sum(1 - H) + 1e-10)

        # For SAR (multiplicative noise), use I-divergence based term
        # F = log(c2/c1) + I*(1/c1 - 1/c2) for Gamma noise model
        # Simplified: use squared difference for stability
        F = lambda1 * (sar_lin - c1) ** 2 - lambda2 * (sar_lin - c2) ** 2

        # Curvature term
        phi_x = np.gradient(phi, axis=1)
        phi_y = np.gradient(phi, axis=0)
        norm_grad = np.sqrt(phi_x**2 + phi_y**2 + 1e-8)

        nx = phi_x / norm_grad
        ny = phi_y / norm_grad

        curvature = np.gradient(nx, axis=1) + np.gradient(ny, axis=0)

        # Level set evolution
        phi = phi + dt * (mu * curvature - F)

        # Re-initialize periodically for stability
        if (i + 1) % 20 == 0:
            phi = distance_transform_edt(phi > 0) - distance_transform_edt(phi <= 0)

    return (phi > 0).astype(np.float32)


def geodesic_active_contour(
    sar_image: np.ndarray,
    initial_mask: np.ndarray,
    n_iterations: int = 50,
    alpha: float = 1.0,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Geodesic Active Contour using edge-stopping function.
    Better for SAR as it uses edge information directly.
    """
    # Compute edge-stopping function
    sar_smooth = gaussian_filter(sar_image, sigma=sigma)
    grad_x = sobel(sar_smooth, axis=1)
    grad_y = sobel(sar_smooth, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Edge stopping function: g = 1 / (1 + |grad|^2)
    g = 1.0 / (1.0 + alpha * grad_mag**2)

    # Initialize level set
    phi = distance_transform_edt(initial_mask) - distance_transform_edt(
        1 - initial_mask
    )
    phi = phi.astype(np.float64)

    dt = 0.5

    for _ in range(n_iterations):
        # Gradient of phi
        phi_x = np.gradient(phi, axis=1)
        phi_y = np.gradient(phi, axis=0)
        norm_grad = np.sqrt(phi_x**2 + phi_y**2 + 1e-8)

        # Curvature
        nx = phi_x / norm_grad
        ny = phi_y / norm_grad
        curvature = np.gradient(nx, axis=1) + np.gradient(ny, axis=0)

        # Gradient of g
        g_x = np.gradient(g, axis=1)
        g_y = np.gradient(g, axis=0)

        # Evolution: phi_t = g * kappa * |grad_phi| + grad_g . grad_phi
        term1 = g * curvature * norm_grad
        term2 = g_x * phi_x + g_y * phi_y

        phi = phi + dt * (term1 + term2)

    return (phi > 0).astype(np.float32)


# =============================================================================
# MORPHOLOGICAL BOUNDARY REFINEMENT
# =============================================================================


def morphological_boundary_refinement(
    pred_mask: np.ndarray,
    sar_image: np.ndarray,
    min_area: int = 50,
) -> np.ndarray:
    """
    Morphological operations to clean up boundaries.
    """
    # Remove small holes
    filled = ndimage.binary_fill_holes(pred_mask)

    # Remove small objects
    labeled, n_features = label(filled)
    sizes = ndimage.sum(filled, labeled, range(1, n_features + 1))
    mask_sizes = sizes > min_area
    remove_small = mask_sizes[labeled - 1]
    remove_small[labeled == 0] = False
    cleaned = remove_small.astype(np.float32)

    # Smooth boundaries with opening-closing
    from scipy.ndimage import binary_opening, binary_closing

    smoothed = binary_closing(binary_opening(cleaned, iterations=1), iterations=1)

    return smoothed.astype(np.float32)


def edge_aware_refinement(
    pred_mask: np.ndarray,
    sar_image: np.ndarray,
    edge_threshold: float = 0.3,
) -> np.ndarray:
    """
    Refine boundaries using SAR edge information.
    """
    # Detect edges in SAR
    sar_smooth = gaussian_filter(sar_image, sigma=1)
    edges = np.abs(sobel(sar_smooth, axis=0)) + np.abs(sobel(sar_smooth, axis=1))
    edges = edges / (edges.max() + 1e-10)

    # Get prediction boundary
    pred_boundary = binary_dilation(pred_mask, iterations=1) & ~binary_erosion(
        pred_mask, iterations=1
    )

    # For boundary pixels, check if there's a strong SAR edge nearby
    # If not, the boundary might be incorrect
    edge_support = gaussian_filter(edges, sigma=2)

    # Adjust boundary based on edge support
    refined = pred_mask.copy()

    # Where boundary lacks edge support, erode slightly
    weak_boundary = pred_boundary & (edge_support < edge_threshold)
    refined[weak_boundary] = 0

    return refined.astype(np.float32)


# =============================================================================
# COMBINED REFINEMENT PIPELINE
# =============================================================================


def refine_prediction(
    pred_proba: np.ndarray,
    chip: np.ndarray,
    methods: List[str] = ["crf", "levelset", "morphological"],
) -> Dict[str, np.ndarray]:
    """
    Apply multiple refinement methods to a prediction.
    """
    vv = chip[0]
    vh = chip[1]

    results = {"original": (pred_proba > 0.5).astype(np.float32)}

    # CRF refinement
    if "crf" in methods:
        try:
            crf_result = apply_crf_refinement(vh, pred_proba)
            results["crf"] = crf_result
        except Exception as e:
            logger.warning(f"CRF failed: {e}")
            results["crf"] = results["original"]

    # Level set refinement
    if "levelset" in methods:
        try:
            ls_result = active_contour_levelset(
                vh, results["original"], n_iterations=50
            )
            results["levelset"] = ls_result
        except Exception as e:
            logger.warning(f"Level set failed: {e}")
            results["levelset"] = results["original"]

    # Geodesic active contour
    if "geodesic" in methods:
        try:
            gac_result = geodesic_active_contour(
                vh, results["original"], n_iterations=30
            )
            results["geodesic"] = gac_result
        except Exception as e:
            logger.warning(f"GAC failed: {e}")
            results["geodesic"] = results["original"]

    # Morphological refinement
    if "morphological" in methods:
        try:
            morph_result = morphological_boundary_refinement(results["original"], vh)
            results["morphological"] = morph_result
        except Exception as e:
            logger.warning(f"Morphological failed: {e}")
            results["morphological"] = results["original"]

    # Edge-aware refinement
    if "edge_aware" in methods:
        try:
            edge_result = edge_aware_refinement(results["original"], vh)
            results["edge_aware"] = edge_result
        except Exception as e:
            logger.warning(f"Edge-aware failed: {e}")
            results["edge_aware"] = results["original"]

    # Combined: apply morphological after level set
    if "combined" in methods:
        try:
            combined = active_contour_levelset(vh, results["original"], n_iterations=30)
            combined = morphological_boundary_refinement(combined, vh)
            results["combined"] = combined
        except Exception as e:
            results["combined"] = results["original"]

    return results


# =============================================================================
# EVALUATION
# =============================================================================


def compute_boundary_metrics(pred: np.ndarray, truth: np.ndarray) -> Dict:
    """Compute boundary-specific metrics."""
    pred_binary = (pred > 0.5).astype(np.uint8)
    truth_binary = (truth > 0.5).astype(np.uint8)

    # Overall IoU
    tp = np.sum(pred_binary & truth_binary)
    fp = np.sum(pred_binary & ~truth_binary)
    fn = np.sum(~pred_binary & truth_binary)
    iou = tp / (tp + fp + fn + 1e-10)

    # Boundary IoU
    truth_boundary = binary_dilation(truth_binary) & ~binary_erosion(truth_binary)
    pred_boundary = binary_dilation(pred_binary) & ~binary_erosion(pred_binary)

    boundary_tp = np.sum(pred_boundary & truth_boundary)
    boundary_fp = np.sum(pred_boundary & ~truth_boundary)
    boundary_fn = np.sum(~pred_boundary & truth_boundary)
    boundary_iou = boundary_tp / (boundary_tp + boundary_fp + boundary_fn + 1e-10)

    # Hausdorff-like distance (average boundary error)
    if truth_boundary.sum() > 0 and pred_boundary.sum() > 0:
        truth_dist = distance_transform_edt(~truth_boundary)
        pred_dist = distance_transform_edt(~pred_boundary)

        avg_boundary_error = (
            np.mean(truth_dist[pred_boundary]) + np.mean(pred_dist[truth_boundary])
        ) / 2
    else:
        avg_boundary_error = float("inf")

    return {
        "iou": float(iou),
        "boundary_iou": float(boundary_iou),
        "avg_boundary_error_px": float(avg_boundary_error),
    }


# =============================================================================
# FEATURE EXTRACTION (simplified for LightGBM prediction)
# =============================================================================


def extract_features_simple(chip: np.ndarray) -> np.ndarray:
    """Extract features matching the 69-feature training pipeline."""
    from scipy.ndimage import minimum_filter, maximum_filter
    from skimage.morphology import disk, opening, closing, white_tophat, black_tophat

    vv = chip[0]
    vh = chip[1]
    dem = chip[3]
    hand = chip[4]
    slope = chip[5]
    twi = chip[6]

    features = []

    # 1. Basic SAR (4)
    features.extend([vv, vh, vv - vh])
    vv_lin = db_to_linear(vv)
    vh_lin = db_to_linear(vh)
    features.append(np.log10(vv_lin / (vh_lin + 1e-10) + 1e-10))

    # 2. Topographic (6)
    features.extend([hand, slope, twi, dem])
    dem_dx = sobel(dem, axis=1)
    dem_dy = sobel(dem, axis=0)
    features.append(np.sqrt(dem_dx**2 + dem_dy**2))
    features.append(np.arctan2(dem_dy, dem_dx))

    # 3. Polarimetric (4)
    vv_local = uniform_filter(vv_lin, size=5)
    vh_local = uniform_filter(vh_lin, size=5)
    total = vv_local + vh_local + 1e-10
    p = np.clip(vv_local / total, 1e-10, 1 - 1e-10)
    features.append(-p * np.log2(p) - (1 - p) * np.log2(1 - p))  # entropy
    features.append(
        np.degrees(np.arctan2(np.sqrt(vh_local), np.sqrt(vv_local)))
    )  # alpha
    features.append(4 * vh_local / (vv_local + vh_local + 1e-10))  # RVI
    features.append(linear_to_db(vv_lin + 2 * vh_lin))  # span

    # 4. Multi-scale (26)
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

    # 5. Speckle (4)
    for band in [vv, vh]:
        band_lin = db_to_linear(band)
        local_mean = uniform_filter(band_lin, size=9)
        local_sq = uniform_filter(band_lin**2, size=9)
        local_var = local_sq - local_mean**2
        local_std = np.sqrt(np.maximum(local_var, 1e-10))
        features.append(np.clip((local_mean / local_std) ** 2, 0, 100))  # ENL
        features.append(np.clip(local_std / (local_mean + 1e-10), 0, 5))  # CV

    # 6. Texture (6)
    for band in [vv, vh]:
        local_mean = uniform_filter(band.astype(np.float64), size=11)
        local_sq = uniform_filter(band.astype(np.float64) ** 2, size=11)
        contrast = np.sqrt(np.maximum(local_sq - local_mean**2, 0))
        local_max = maximum_filter(band, size=11)
        local_min = minimum_filter(band, size=11)
        homogeneity = 1.0 / (1.0 + local_max - local_min + 1e-10)
        energy = 1.0 / (1.0 + contrast)
        features.extend([contrast, homogeneity, energy])

    # 7. Morphological (4)
    vh_norm = (
        (vh - np.nanmin(vh)) / (np.nanmax(vh) - np.nanmin(vh) + 1e-10) * 255
    ).astype(np.uint8)
    selem = disk(3)
    features.append(opening(vh_norm, selem).astype(np.float32) / 255)
    features.append(closing(vh_norm, selem).astype(np.float32) / 255)
    features.append(white_tophat(vh_norm, selem).astype(np.float32) / 255)
    features.append(black_tophat(vh_norm, selem).astype(np.float32) / 255)

    # 8. Line detection (1)
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

    # 9. Adaptive thresholds (6)
    from skimage.filters import threshold_otsu

    vv_clean = vv[~np.isnan(vv)].flatten()
    vh_clean = vh[~np.isnan(vh)].flatten()
    vv_otsu = threshold_otsu(
        np.clip(vv_clean, np.percentile(vv_clean, 1), np.percentile(vv_clean, 99))
    )
    vh_otsu = threshold_otsu(
        np.clip(vh_clean, np.percentile(vh_clean, 1), np.percentile(vh_clean, 99))
    )
    features.extend([vv - vv_otsu, vh - vh_otsu])
    features.extend(
        [(vv < vv_otsu).astype(np.float32), (vh < vh_otsu).astype(np.float32)]
    )
    features.extend(
        [(vv < vv_otsu).astype(np.float32), (vh < vh_otsu).astype(np.float32)]
    )  # kapur approx

    # 10. Physics composite (5)
    hand_score = np.clip(1 - hand / 10.0, 0, 1)
    slope_score = np.clip(1 - slope / 15.0, 0, 1)
    vh_score = np.clip((-18.0 - vh) / 10, 0, 1)
    twi_score = np.clip((twi - 5) / 10, 0, 1)
    physics = (hand_score * slope_score * vh_score * twi_score) ** 0.25
    features.extend([physics, hand_score, slope_score, vh_score, twi_score])

    # 11. Gradients (3)
    vh_dx = sobel(vh, axis=1)
    vh_dy = sobel(vh, axis=0)
    features.append(np.sqrt(vh_dx**2 + vh_dy**2))
    vv_dx = sobel(vv, axis=1)
    vv_dy = sobel(vv, axis=0)
    features.append(np.sqrt(vv_dx**2 + vv_dy**2))
    features.append(ndimage.laplace(vh))

    # Stack
    feature_stack = np.stack(features, axis=0).astype(np.float32)
    feature_stack = np.nan_to_num(feature_stack, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_stack


# =============================================================================
# MAIN
# =============================================================================


def main():
    logger.info("=" * 80)
    logger.info("BOUNDARY REFINEMENT WITH CRF AND ACTIVE CONTOURS")
    logger.info("=" * 80)
    logger.info(f"Started: {datetime.now().isoformat()}")

    np.random.seed(CONFIG["random_seed"])

    # Load data
    chips, names = load_chips()
    logger.info(f"Loaded {len(chips)} chips")

    # Split
    train_chips, test_chips, train_names, test_names = train_test_split(
        list(zip(chips, names)),
        list(range(len(chips))),
        test_size=0.15,
        random_state=CONFIG["random_seed"],
    )
    test_data = [x[0] for x in train_chips[-31:]]  # Use last 31 as test
    test_names = [x[1] for x in train_chips[-31:]]

    # Load LightGBM model
    logger.info("\nLoading LightGBM model...")
    lgb_model = lgb.Booster(
        model_file=str(CONFIG["model_dir"] / "lightgbm_v4_comprehensive.txt")
    )

    # Evaluate refinement methods
    results = {
        "timestamp": datetime.now().isoformat(),
        "methods": {},
        "per_chip": [],
    }

    methods = ["crf", "levelset", "morphological", "edge_aware", "combined"]

    for method in methods:
        results["methods"][method] = {
            "iou": [],
            "boundary_iou": [],
            "boundary_error": [],
        }

    results["methods"]["original"] = {
        "iou": [],
        "boundary_iou": [],
        "boundary_error": [],
    }

    logger.info(f"\nEvaluating {len(test_data)} test chips...")

    for i, (chip, name) in enumerate(zip(test_data, test_names)):
        if (i + 1) % 10 == 0:
            logger.info(f"  Processing {i + 1}/{len(test_data)}")

        truth = chip[7]

        # Get LightGBM prediction
        features = extract_features_simple(chip)
        h, w = truth.shape
        X = features.reshape(features.shape[0], -1).T
        lgb_proba = lgb_model.predict(X).reshape(h, w)

        # Apply refinements
        refined = refine_prediction(lgb_proba, chip, methods)

        chip_result = {"name": name, "methods": {}}

        for method_name, pred in refined.items():
            metrics = compute_boundary_metrics(pred, truth)
            chip_result["methods"][method_name] = metrics

            if method_name in results["methods"]:
                results["methods"][method_name]["iou"].append(metrics["iou"])
                results["methods"][method_name]["boundary_iou"].append(
                    metrics["boundary_iou"]
                )
                results["methods"][method_name]["boundary_error"].append(
                    metrics["avg_boundary_error_px"]
                )

        results["per_chip"].append(chip_result)

    # Aggregate results
    logger.info("\n" + "=" * 60)
    logger.info("BOUNDARY REFINEMENT RESULTS")
    logger.info("=" * 60)

    for method_name, method_results in results["methods"].items():
        if method_results["iou"]:
            mean_iou = np.mean(method_results["iou"])
            mean_boundary_iou = np.mean(method_results["boundary_iou"])
            mean_error = np.mean(
                [e for e in method_results["boundary_error"] if e < 1000]
            )

            results["methods"][method_name]["mean_iou"] = float(mean_iou)
            results["methods"][method_name]["mean_boundary_iou"] = float(
                mean_boundary_iou
            )
            results["methods"][method_name]["mean_boundary_error"] = float(mean_error)

            logger.info(
                f"{method_name:15s}: IoU={mean_iou:.4f}, Boundary IoU={mean_boundary_iou:.4f}, "
                f"Boundary Error={mean_error:.2f}px"
            )

    # Save results
    with open(CONFIG["results_dir"] / "boundary_refinement_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(
        f"\nResults saved to: {CONFIG['results_dir'] / 'boundary_refinement_results.json'}"
    )
    logger.info(f"Completed: {datetime.now().isoformat()}")

    return results


if __name__ == "__main__":
    main()
