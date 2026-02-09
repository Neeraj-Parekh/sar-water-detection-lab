#!/usr/bin/env python3
"""
================================================================================
TTA EVALUATION SCRIPT - Test-Time Augmentation Benchmark
================================================================================

This script evaluates models with and without Test-Time Augmentation (TTA).
Measures the IoU improvement from TTA on the validation set.

Usage:
    python evaluate_tta.py --model_path path/to/model.pth
    python evaluate_tta.py --model_type lightgbm --model_path path/to/model.txt

Author: SAR Water Detection Project
Date: 2026-01-26
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
from scipy.ndimage import uniform_filter
from skimage.filters import frangi, hessian

warnings_import_failed = False
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings_import_failed = True

try:
    import lightgbm as lgb

    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import TTA module
try:
    from tta_module import TestTimeAugmentation, TTAConfig, apply_tta, apply_tta_numpy

    TTA_AVAILABLE = True
    logger.info("TTA module loaded successfully")
except ImportError as e:
    logger.warning(f"TTA module not available: {e}")
    TTA_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "chip_dir": Path("/home/mit-aoe/sar_water_detection/chips_expanded_npy"),
    "model_dir": Path("/home/mit-aoe/sar_water_detection/models"),
    "results_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    "target_size": 512,
    "val_split": 0.2,
    "random_seed": 42,
}


# =============================================================================
# METRICS
# =============================================================================


def compute_metrics(
    pred: np.ndarray, target: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute segmentation metrics."""
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = (target > threshold).astype(np.float32)

    # IoU
    intersection = np.sum(pred_binary * target_binary)
    union = np.sum(pred_binary) + np.sum(target_binary) - intersection
    iou = (intersection + 1e-8) / (union + 1e-8)

    # Dice
    dice = (2 * intersection + 1e-8) / (
        np.sum(pred_binary) + np.sum(target_binary) + 1e-8
    )

    # Precision/Recall
    tp = intersection
    fp = np.sum(pred_binary) - tp
    fn = np.sum(target_binary) - tp

    precision = (tp + 1e-8) / (tp + fp + 1e-8)
    recall = (tp + 1e-8) / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# =============================================================================
# PREPROCESSING
# =============================================================================


def compute_frangi_vesselness(
    vh: np.ndarray, scales: List[int] = [1, 2, 3]
) -> np.ndarray:
    """Compute Frangi Vesselness filter."""
    vh_norm = (vh - vh.min()) / (vh.max() - vh.min() + 1e-8)
    vh_inv = 1.0 - vh_norm

    try:
        vesselness = frangi(
            vh_inv.astype(np.float64),
            sigmas=scales,
            black_ridges=False,
            mode="reflect",
        )
    except Exception:
        vesselness = np.zeros_like(vh)

    vesselness = np.nan_to_num(vesselness, nan=0.0)
    if vesselness.max() > 0:
        vesselness = vesselness / vesselness.max()

    return vesselness.astype(np.float32)


def preprocess_chip(chip: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess chip to features + label."""
    vv = chip[:, :, 0]
    vh = chip[:, :, 1]
    dem = chip[:, :, 2]
    slope = chip[:, :, 3]
    hand = chip[:, :, 4]
    twi = chip[:, :, 5]
    label = chip[:, :, 6]
    mndwi = chip[:, :, 7] if chip.shape[2] > 7 else np.zeros_like(vv)

    # Derived features
    vh_texture = uniform_filter(vh**2, size=5) - uniform_filter(vh, size=5) ** 2
    vh_texture = np.sqrt(np.maximum(vh_texture, 0))
    frangi_feat = compute_frangi_vesselness(vh)

    def normalize(x, vmin=None, vmax=None):
        if vmin is None:
            vmin = np.nanpercentile(x, 1)
        if vmax is None:
            vmax = np.nanpercentile(x, 99)
        return np.clip((x - vmin) / (vmax - vmin + 1e-8), 0, 1)

    features = np.stack(
        [
            normalize(vv, -30, 0),
            normalize(vh, -35, -5),
            normalize(dem, 0, 2000),
            normalize(slope, 0, 45),
            normalize(hand, 0, 100),
            normalize(twi, 0, 20),
            normalize(mndwi, -1, 1),
            normalize(vh_texture),
            frangi_feat,
        ],
        axis=-1,
    )

    features = np.nan_to_num(features, nan=0.0)
    label = np.nan_to_num(label, nan=0.0)

    return features.astype(np.float32), label.astype(np.float32)


# =============================================================================
# MODEL LOADING
# =============================================================================


def load_pytorch_model(model_path: Path, device: str = "cuda"):
    """Load a PyTorch model."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available")

    # Import the model architecture
    try:
        from train_with_cldice_v10 import AttentionUNetV10
    except ImportError:
        from attention_unet_v9_sota import AttentionUNet as AttentionUNetV10

    checkpoint = torch.load(model_path, map_location=device)

    config = checkpoint.get("config", {})
    model = AttentionUNetV10(
        in_channels=config.get("in_channels", 9),
        out_channels=1,
        base_filters=config.get("base_filters", 32),
        dropout=config.get("dropout", 0.15),
        deep_supervision=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def load_lightgbm_model(model_path: Path):
    """Load a LightGBM model."""
    if not LGB_AVAILABLE:
        raise ImportError("LightGBM not available")

    model = lgb.Booster(model_file=str(model_path))
    return model


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================


def evaluate_pytorch_model(
    model,
    val_chips: List[Path],
    device: str = "cuda",
    use_tta: bool = False,
    tta_config: Optional[dict] = None,
) -> Dict[str, float]:
    """Evaluate PyTorch model with optional TTA."""
    all_metrics = []

    for chip_path in val_chips:
        chip = np.load(chip_path)
        features, label = preprocess_chip(chip)

        # Convert to tensor
        x = torch.from_numpy(features.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            if use_tta and TTA_AVAILABLE:
                # Apply TTA
                tta = TestTimeAugmentation(
                    model,
                    TTAConfig(
                        use_rotations=tta_config.get("use_rotations", True),
                        use_flips=tta_config.get("use_flips", True),
                        use_multi_scale=tta_config.get("use_multi_scale", False),
                    ),
                )
                pred = tta(x)
            else:
                pred = torch.sigmoid(model(x))

            pred = pred.squeeze().cpu().numpy()

        metrics = compute_metrics(pred, label)
        all_metrics.append(metrics)

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_metrics


def evaluate_lightgbm_model(
    model,
    val_chips: List[Path],
    use_tta: bool = False,
    feature_extractor: Optional[Callable] = None,
) -> Dict[str, float]:
    """Evaluate LightGBM model with optional TTA."""
    all_metrics = []

    def lgb_predict(features: np.ndarray) -> np.ndarray:
        """LightGBM prediction function for TTA."""
        h, w = features.shape[:2]

        # Flatten for LightGBM
        if feature_extractor is not None:
            X = feature_extractor(features)
        else:
            X = features.reshape(-1, features.shape[-1])

        pred_flat = model.predict(X)
        return pred_flat.reshape(h, w)

    for chip_path in val_chips:
        chip = np.load(chip_path)
        features, label = preprocess_chip(chip)

        if use_tta and TTA_AVAILABLE:
            pred = apply_tta_numpy(
                lgb_predict, features, use_rotations=True, use_flips=True
            )
        else:
            pred = lgb_predict(features)

        metrics = compute_metrics(pred, label)
        all_metrics.append(metrics)

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    return avg_metrics


# =============================================================================
# MAIN EVALUATION
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Evaluate model with/without TTA")
    parser.add_argument("--model_path", type=str, help="Path to model file")
    parser.add_argument(
        "--model_type", type=str, default="pytorch", choices=["pytorch", "lightgbm"]
    )
    parser.add_argument("--chip_dir", type=str, default=None)
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of validation samples to evaluate",
    )
    parser.add_argument(
        "--use_multi_scale",
        action="store_true",
        help="Use multi-scale TTA (slower but potentially better)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TTA EVALUATION BENCHMARK")
    logger.info("=" * 60)

    # Setup paths
    chip_dir = Path(args.chip_dir) if args.chip_dir else CONFIG["chip_dir"]

    if not chip_dir.exists():
        logger.error(f"Chip directory not found: {chip_dir}")
        return

    # Load validation chips
    all_chips = sorted(chip_dir.glob("*.npy"))
    np.random.seed(CONFIG["random_seed"])
    np.random.shuffle(all_chips)

    n_val = max(1, int(len(all_chips) * CONFIG["val_split"]))
    val_chips = all_chips[:n_val]

    if args.n_samples:
        val_chips = val_chips[: args.n_samples]

    logger.info(f"Evaluating on {len(val_chips)} validation chips")

    # Load model
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        # Find latest model
        if args.model_type == "pytorch":
            model_path = CONFIG["model_dir"] / "attention_unet_v10_cldice_best.pth"
        else:
            model_path = CONFIG["model_dir"] / "lightgbm_v9.txt"

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    logger.info(f"Loading model: {model_path}")

    # TTA configuration
    tta_config = {
        "use_rotations": True,
        "use_flips": True,
        "use_multi_scale": args.use_multi_scale,
    }

    device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"

    # Evaluate
    results = {
        "model_path": str(model_path),
        "model_type": args.model_type,
        "n_val_samples": len(val_chips),
        "tta_config": tta_config,
    }

    if args.model_type == "pytorch":
        model = load_pytorch_model(model_path, device)

        # Without TTA
        logger.info("Evaluating without TTA...")
        t0 = time.time()
        metrics_no_tta = evaluate_pytorch_model(model, val_chips, device, use_tta=False)
        time_no_tta = time.time() - t0

        # With TTA
        logger.info("Evaluating with TTA...")
        t0 = time.time()
        metrics_with_tta = evaluate_pytorch_model(
            model, val_chips, device, use_tta=True, tta_config=tta_config
        )
        time_with_tta = time.time() - t0

    else:  # lightgbm
        model = load_lightgbm_model(model_path)

        # Without TTA
        logger.info("Evaluating without TTA...")
        t0 = time.time()
        metrics_no_tta = evaluate_lightgbm_model(model, val_chips, use_tta=False)
        time_no_tta = time.time() - t0

        # With TTA
        logger.info("Evaluating with TTA...")
        t0 = time.time()
        metrics_with_tta = evaluate_lightgbm_model(model, val_chips, use_tta=True)
        time_with_tta = time.time() - t0

    # Calculate improvements
    improvements = {}
    for key in metrics_no_tta.keys():
        improvements[key] = metrics_with_tta[key] - metrics_no_tta[key]

    results["without_tta"] = {
        "metrics": metrics_no_tta,
        "time_seconds": time_no_tta,
    }
    results["with_tta"] = {
        "metrics": metrics_with_tta,
        "time_seconds": time_with_tta,
    }
    results["improvement"] = improvements
    results["slowdown_factor"] = time_with_tta / time_no_tta

    # Print results
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    logger.info("\nWithout TTA:")
    for k, v in metrics_no_tta.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info(f"  Time: {time_no_tta:.2f}s")

    logger.info("\nWith TTA:")
    for k, v in metrics_with_tta.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info(f"  Time: {time_with_tta:.2f}s")

    logger.info("\nImprovement from TTA:")
    for k, v in improvements.items():
        sign = "+" if v >= 0 else ""
        logger.info(f"  {k}: {sign}{v:.4f} ({sign}{v * 100:.2f}%)")

    logger.info(f"\nSlowdown factor: {results['slowdown_factor']:.2f}x")

    # Save results
    CONFIG["results_dir"].mkdir(parents=True, exist_ok=True)
    results_path = (
        CONFIG["results_dir"] / f"tta_evaluation_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {results_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"IoU without TTA: {metrics_no_tta['iou']:.4f}")
    logger.info(f"IoU with TTA:    {metrics_with_tta['iou']:.4f}")
    logger.info(
        f"IoU improvement: +{improvements['iou']:.4f} (+{improvements['iou'] * 100:.2f}%)"
    )
    logger.info(f"Time overhead:   {results['slowdown_factor']:.2f}x slower")


if __name__ == "__main__":
    main()
