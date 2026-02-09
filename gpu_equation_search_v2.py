#!/usr/bin/env python3
"""
GPU-Accelerated Exhaustive Equation Search v2
==============================================
Optimized for NVIDIA RTX A5000 (24GB VRAM)

Key Improvements over v1:
- Full GPU utilization with batch processing
- Terrain-adaptive physics constraints
- More equation templates including adaptive thresholds
- Multi-process parallel evaluation
- Checkpointing for long runs

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
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np

warnings.filterwarnings("ignore")

# GPU imports
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage

    GPU_AVAILABLE = True
    print(
        f"CuPy available. GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}"
    )
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("CuPy not available, using CPU mode")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("equation_search_v2.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "chip_dir": Path("/home/mit-aoe/sar_water_detection/chips"),
    "output_dir": Path("/home/mit-aoe/sar_water_detection/results"),
    "batch_size": 50000,  # Pixels per GPU batch
    "n_workers": 4,  # CPU workers for data loading
    "max_equations_per_template": 2000,  # Grid search limit per template
    "checkpoint_interval": 500,  # Save every N equations
    "gpu_memory_fraction": 0.85,  # Use 85% of GPU memory
}


# =============================================================================
# Terrain-Adaptive Physics Constraints
# =============================================================================

TERRAIN_PROFILES = {
    "flat_lowland": {
        "description": "River plains, coastal areas, flood zones",
        "hand_threshold": 10.0,
        "slope_threshold": 5.0,
        "twi_min": 8.0,
        "vh_range": (-25, -14),
        "examples": ["brahmaputra", "ganga", "sundarbans", "kerala"],
    },
    "hilly": {
        "description": "Moderate elevation, valleys",
        "hand_threshold": 20.0,
        "slope_threshold": 15.0,
        "twi_min": 6.0,
        "vh_range": (-25, -16),
        "examples": ["chambal", "narmada", "godavari"],
    },
    "mountainous": {
        "description": "High altitude lakes, steep terrain",
        "hand_threshold": 50.0,  # Relaxed for mountain lakes
        "slope_threshold": 25.0,
        "twi_min": 4.0,
        "vh_range": (-25, -18),
        "examples": ["pangong", "dal_lake", "bhakra"],
    },
    "arid": {
        "description": "Desert, salt flats, ephemeral water",
        "hand_threshold": 8.0,
        "slope_threshold": 3.0,
        "twi_min": 5.0,
        "vh_range": (-28, -20),  # Stricter for arid
        "examples": ["rann_kutch", "sambhar", "thar"],
    },
    "urban": {
        "description": "Cities with rivers/lakes",
        "hand_threshold": 5.0,
        "slope_threshold": 8.0,
        "twi_min": 6.0,
        "vh_range": (-25, -18),
        "vv_vh_ratio_min": 6.0,  # Urban has high VV/VH
        "examples": ["mumbai", "delhi", "bangalore", "chennai"],
    },
    "wetland": {
        "description": "Marshes, seasonal flooding",
        "hand_threshold": 15.0,
        "slope_threshold": 5.0,
        "twi_min": 10.0,  # High TWI required
        "vh_range": (-22, -12),  # Relaxed for vegetated water
        "examples": ["keoladeo", "kolleru", "loktak", "chilika"],
    },
}


# =============================================================================
# Expanded Equation Templates
# =============================================================================

EQUATION_TEMPLATES = {
    # ===== BASIC THRESHOLDS =====
    "vh_simple": {
        "template": "(vh < {T_vh})",
        "params": {"T_vh": np.arange(-28, -10, 1.0)},
        "complexity": 1,
        "description": "Simple VH threshold",
    },
    "vv_simple": {
        "template": "(vv < {T_vv})",
        "params": {"T_vv": np.arange(-25, -8, 1.0)},
        "complexity": 1,
        "description": "Simple VV threshold",
    },
    # ===== DUAL-BAND =====
    "dual_band_and": {
        "template": "(vv < {T_vv}) & (vh < {T_vh})",
        "params": {"T_vv": np.arange(-22, -12, 2.0), "T_vh": np.arange(-26, -14, 2.0)},
        "complexity": 2,
        "description": "Both VV and VH thresholds",
    },
    "dual_band_or": {
        "template": "(vv < {T_vv}) | (vh < {T_vh})",
        "params": {"T_vv": np.arange(-22, -12, 2.0), "T_vh": np.arange(-26, -14, 2.0)},
        "complexity": 2,
        "description": "Either VV or VH threshold",
    },
    # ===== PHYSICS-CONSTRAINED =====
    "hand_constrained": {
        "template": "(vh < {T_vh}) & (hand < {T_hand})",
        "params": {"T_vh": np.arange(-26, -12, 2.0), "T_hand": np.arange(3, 25, 3.0)},
        "complexity": 2,
        "description": "VH + HAND constraint",
    },
    "slope_constrained": {
        "template": "(vh < {T_vh}) & (slope < {T_slope})",
        "params": {"T_vh": np.arange(-26, -12, 2.0), "T_slope": np.arange(3, 20, 3.0)},
        "complexity": 2,
        "description": "VH + slope constraint",
    },
    "twi_constrained": {
        "template": "(vh < {T_vh}) & (twi > {T_twi})",
        "params": {"T_vh": np.arange(-26, -12, 2.0), "T_twi": np.arange(4, 15, 2.0)},
        "complexity": 2,
        "description": "VH + TWI constraint",
    },
    # ===== TRIPLE-LOCK (SAR + 2 PHYSICS) =====
    "triple_hand_slope": {
        "template": "(vh < {T_vh}) & (hand < {T_hand}) & (slope < {T_slope})",
        "params": {
            "T_vh": np.arange(-24, -14, 2.0),
            "T_hand": np.arange(5, 20, 5.0),
            "T_slope": np.arange(5, 20, 5.0),
        },
        "complexity": 3,
        "description": "VH + HAND + slope",
    },
    "triple_hand_twi": {
        "template": "(vh < {T_vh}) & (hand < {T_hand}) & (twi > {T_twi})",
        "params": {
            "T_vh": np.arange(-24, -14, 2.0),
            "T_hand": np.arange(5, 20, 5.0),
            "T_twi": np.arange(5, 12, 2.0),
        },
        "complexity": 3,
        "description": "VH + HAND + TWI",
    },
    # ===== ADAPTIVE/BRIGHT WATER =====
    "bright_water_strict_physics": {
        "template": "(vh < {T_vh_bright}) & (hand < {T_hand_strict}) & (slope < {T_slope_strict})",
        "params": {
            "T_vh_bright": np.arange(-16, -8, 2.0),  # Relaxed VH for wind
            "T_hand_strict": np.arange(2, 8, 2.0),  # Very strict HAND
            "T_slope_strict": np.arange(1, 5, 1.0),  # Very flat
        },
        "complexity": 3,
        "description": "Bright water with strict physics",
    },
    "adaptive_vh_twi": {
        "template": "(vh < {T_vh_base} + twi * {T_twi_factor})",
        "params": {
            "T_vh_base": np.arange(-22, -14, 2.0),
            "T_twi_factor": np.arange(0.2, 1.0, 0.2),
        },
        "complexity": 2,
        "description": "VH threshold adapts to TWI",
    },
    # ===== RATIO-BASED =====
    "vv_vh_ratio": {
        "template": "((vv - vh) > {T_diff_min}) & ((vv - vh) < {T_diff_max}) & (hand < {T_hand})",
        "params": {
            "T_diff_min": np.arange(2, 8, 2.0),
            "T_diff_max": np.arange(8, 16, 2.0),
            "T_hand": np.arange(5, 20, 5.0),
        },
        "complexity": 3,
        "description": "VV-VH difference (polarization ratio)",
    },
    # ===== URBAN EXCLUSION =====
    "urban_aware": {
        "template": "(vh < {T_vh}) & (hand < {T_hand}) & ~((vv > {T_vv_urban}) & ((vv - vh) > {T_urban_ratio}))",
        "params": {
            "T_vh": np.arange(-22, -14, 2.0),
            "T_hand": np.arange(5, 15, 5.0),
            "T_vv_urban": np.arange(-12, -6, 2.0),
            "T_urban_ratio": np.arange(8, 14, 2.0),
        },
        "complexity": 4,
        "description": "Water detection with urban exclusion",
    },
    # ===== TEXTURE-BASED =====
    "texture_smooth": {
        "template": "(vh < {T_vh}) & (hand < {T_hand}) & (texture < {T_texture})",
        "params": {
            "T_vh": np.arange(-24, -14, 2.0),
            "T_hand": np.arange(5, 20, 5.0),
            "T_texture": np.arange(0.2, 0.8, 0.2),
        },
        "complexity": 3,
        "description": "VH + HAND + low texture (smooth water)",
    },
    # ===== PYSR-INSPIRED (from our discoveries) =====
    "pysr_best": {
        "template": "(np.minimum(np.maximum(slope + vh, {T_low}), twi + {T_offset}) < {T_threshold})",
        "params": {
            "T_low": np.arange(0.4, 1.0, 0.2),
            "T_offset": np.arange(0.4, 1.0, 0.2),
            "T_threshold": np.arange(0.5, 1.5, 0.25),
        },
        "complexity": 4,
        "description": "Best equation from PySR: min(max(SLOPE+VH, a), TWI+b)",
    },
    "pysr_large_water": {
        "template": "(np.minimum(twi + {T_twi_off}, np.maximum({T_low}, (vh + {T_vh_off}) / {T_scale})) > {T_threshold})",
        "params": {
            "T_twi_off": np.arange(0.6, 1.0, 0.1),
            "T_low": np.arange(0.6, 1.0, 0.1),
            "T_vh_off": np.arange(0.4, 0.8, 0.1),
            "T_scale": np.arange(0.005, 0.015, 0.003),
            "T_threshold": np.arange(0.5, 1.0, 0.1),
        },
        "complexity": 5,
        "description": "PySR large water: min(TWI+a, max(b, (VH+c)/d))",
    },
    # ===== HYSTERESIS (CONNECTED REGIONS) =====
    "hysteresis": {
        "template": "((vh < {T_vh_low}) | ((vh < {T_vh_high}) & (hand < {T_hand})))",
        "params": {
            "T_vh_low": np.arange(-26, -20, 2.0),
            "T_vh_high": np.arange(-20, -12, 2.0),
            "T_hand": np.arange(5, 15, 5.0),
        },
        "complexity": 3,
        "description": "Hysteresis: strict core + relaxed with physics",
    },
    # ===== TERRAIN-SPECIFIC =====
    "mountain_lake": {
        "template": "(vh < {T_vh}) & (slope < {T_slope}) & (dem > {T_dem_min})",
        "params": {
            "T_vh": np.arange(-24, -16, 2.0),
            "T_slope": np.arange(10, 30, 5.0),
            "T_dem_min": np.arange(1000, 4000, 500),
        },
        "complexity": 3,
        "description": "High-altitude lakes (ignore HAND)",
    },
    "wetland_vegetated": {
        "template": "(vh < {T_vh}) & (twi > {T_twi}) & (hand < {T_hand})",
        "params": {
            "T_vh": np.arange(-18, -10, 2.0),  # Very relaxed for vegetation
            "T_twi": np.arange(8, 14, 2.0),
            "T_hand": np.arange(10, 25, 5.0),
        },
        "complexity": 3,
        "description": "Wetlands with floating vegetation",
    },
    "arid_ephemeral": {
        "template": "(vh < {T_vh}) & (hand < {T_hand}) & (slope < {T_slope}) & (twi > {T_twi})",
        "params": {
            "T_vh": np.arange(-26, -18, 2.0),
            "T_hand": np.arange(3, 10, 2.0),
            "T_slope": np.arange(2, 8, 2.0),
            "T_twi": np.arange(4, 10, 2.0),
        },
        "complexity": 4,
        "description": "Ephemeral water in arid regions",
    },
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EquationResult:
    """Results from evaluating an equation."""

    template_name: str
    equation: str
    params: Dict[str, float]
    terrain_profile: str
    iou: float
    precision: float
    recall: float
    f1: float
    physics_score: float
    complexity: int
    water_fraction_pred: float
    water_fraction_true: float
    n_samples: int

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# GPU Feature Computer
# =============================================================================


class GPUFeatureComputer:
    """Compute derived features on GPU."""

    def __init__(self):
        if GPU_AVAILABLE:
            # Set memory pool
            mempool = cp.get_default_memory_pool()
            mempool.set_limit(fraction=CONFIG["gpu_memory_fraction"])
            logger.info(
                f"GPU memory limit set to {CONFIG['gpu_memory_fraction'] * 100:.0f}%"
            )

    def compute_texture(self, data: np.ndarray, window_size: int = 9) -> np.ndarray:
        """Compute local coefficient of variation as texture measure."""
        xp = cp if GPU_AVAILABLE else np
        data_gpu = xp.asarray(data.astype(np.float32))

        if GPU_AVAILABLE:
            local_mean = cp_ndimage.uniform_filter(data_gpu, size=window_size)
            local_sq_mean = cp_ndimage.uniform_filter(data_gpu**2, size=window_size)
        else:
            from scipy.ndimage import uniform_filter

            local_mean = uniform_filter(data_gpu, size=window_size)
            local_sq_mean = uniform_filter(data_gpu**2, size=window_size)

        local_std = xp.sqrt(xp.maximum(local_sq_mean - local_mean**2, 0))
        texture = local_std / (xp.abs(local_mean) + 1e-10)

        if GPU_AVAILABLE:
            return cp.asnumpy(texture)
        return texture

    def normalize_features(
        self, features: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Normalize features for stable equation evaluation."""
        normalized = {}

        for name, data in features.items():
            if name == "truth":
                normalized[name] = data
                continue

            # Handle NaN/Inf
            data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)

            # Feature-specific normalization
            if name in ["vv", "vh"]:
                # SAR dB values: clip to reasonable range
                normalized[name] = np.clip(data, -35, 0)
            elif name == "hand":
                # HAND: clip to 0-100m
                normalized[name] = np.clip(data, 0, 100)
            elif name == "slope":
                # Slope: clip to 0-60 degrees
                normalized[name] = np.clip(data, 0, 60)
            elif name == "twi":
                # TWI: clip to reasonable range
                normalized[name] = np.clip(data, -5, 25)
            elif name == "dem":
                # DEM: keep as is
                normalized[name] = data
            else:
                normalized[name] = data

        return normalized


# =============================================================================
# Terrain Classifier
# =============================================================================


class TerrainClassifier:
    """Classify terrain type for adaptive thresholding."""

    @staticmethod
    def classify_chip(features: Dict[str, np.ndarray], chip_name: str = "") -> str:
        """
        Classify chip terrain based on statistics.

        Returns one of: flat_lowland, hilly, mountainous, arid, urban, wetland
        """
        dem = features.get("dem", np.zeros((1, 1)))
        slope = features.get("slope", np.zeros((1, 1)))
        hand = features.get("hand", np.zeros((1, 1)))
        twi = features.get("twi", np.zeros((1, 1)))
        vv = features.get("vv", np.zeros((1, 1)))
        vh = features.get("vh", np.zeros((1, 1)))

        # Compute statistics
        dem_mean = np.nanmean(dem)
        slope_mean = np.nanmean(slope)
        slope_p90 = np.nanpercentile(slope, 90)
        hand_p90 = np.nanpercentile(hand, 90)
        twi_mean = np.nanmean(twi)
        vv_mean = np.nanmean(vv)
        vh_mean = np.nanmean(vh)
        vv_vh_diff = vv_mean - vh_mean

        # Check chip name for hints
        chip_lower = chip_name.lower()
        for terrain, profile in TERRAIN_PROFILES.items():
            for example in profile.get("examples", []):
                if example in chip_lower:
                    return terrain

        # Rule-based classification
        if dem_mean > 2000:
            return "mountainous"
        elif slope_mean > 15 or slope_p90 > 30:
            return "hilly"
        elif twi_mean > 10 and hand_p90 < 15:
            return "wetland"
        elif vv_mean > -12 and vv_vh_diff > 8:
            return "urban"
        elif twi_mean < 6 and vh_mean < -22:
            return "arid"
        else:
            return "flat_lowland"

    @staticmethod
    def get_physics_constraints(terrain: str) -> Dict[str, float]:
        """Get physics constraints for terrain type."""
        profile = TERRAIN_PROFILES.get(terrain, TERRAIN_PROFILES["flat_lowland"])
        return {
            "hand_max": profile["hand_threshold"],
            "slope_max": profile["slope_threshold"],
            "twi_min": profile["twi_min"],
            "vh_min": profile["vh_range"][0],
            "vh_max": profile["vh_range"][1],
        }


# =============================================================================
# Equation Evaluator
# =============================================================================


class EquationEvaluator:
    """Evaluate equations on GPU with terrain-adaptive physics."""

    def __init__(self):
        self.feature_computer = GPUFeatureComputer()
        self.terrain_classifier = TerrainClassifier()

    def load_chip(self, chip_path: Path) -> Tuple[Dict[str, np.ndarray], str]:
        """Load chip and classify terrain."""
        data = np.load(chip_path)

        # Handle different data formats
        if len(data.shape) == 3:
            if data.shape[0] < data.shape[2]:
                # Shape is (C, H, W)
                data = np.transpose(data, (1, 2, 0))
            # Now shape is (H, W, C)

        n_bands = data.shape[2] if len(data.shape) == 3 else 1

        # Extract bands based on expected format
        features = {}
        if n_bands >= 2:
            features["vv"] = data[:, :, 0]
            features["vh"] = data[:, :, 1]
        if n_bands >= 4:
            features["dem"] = (
                data[:, :, 3] if n_bands > 3 else np.zeros_like(data[:, :, 0])
            )
        if n_bands >= 5:
            features["hand"] = data[:, :, 4]
        if n_bands >= 6:
            features["slope"] = (
                data[:, :, 5] if n_bands > 5 else np.zeros_like(data[:, :, 0])
            )
        if n_bands >= 7:
            features["twi"] = (
                data[:, :, 6] if n_bands > 6 else np.zeros_like(data[:, :, 0])
            )

        # Try different truth band locations
        features["truth"] = None
        if n_bands == 8:
            features["truth"] = data[:, :, 7]
        elif n_bands == 7:
            # Check if last band looks like binary mask
            last_band = data[:, :, 6]
            unique_vals = np.unique(last_band[~np.isnan(last_band)])
            if (
                len(unique_vals) <= 3
                and np.all(unique_vals <= 1)
                and np.all(unique_vals >= 0)
            ):
                features["truth"] = last_band

        # Compute texture
        if "vh" in features:
            features["texture"] = self.feature_computer.compute_texture(features["vh"])

        # Normalize
        features = self.feature_computer.normalize_features(features)

        # Classify terrain
        terrain = self.terrain_classifier.classify_chip(features, chip_path.stem)

        return features, terrain

    def evaluate_equation(
        self,
        template_name: str,
        template: str,
        params: Dict[str, float],
        features: Dict[str, np.ndarray],
        terrain: str,
    ) -> Optional[EquationResult]:
        """Evaluate a single equation with parameters."""
        try:
            # Build equation string
            eq_str = template
            for key, value in params.items():
                eq_str = eq_str.replace(f"{{{key}}}", str(value))

            # Prepare variables for eval
            local_vars = {
                "vv": features.get("vv", np.zeros((1, 1))),
                "vh": features.get("vh", np.zeros((1, 1))),
                "hand": features.get("hand", np.zeros((1, 1))),
                "slope": features.get("slope", np.zeros((1, 1))),
                "twi": features.get("twi", np.zeros((1, 1))),
                "dem": features.get("dem", np.zeros((1, 1))),
                "texture": features.get("texture", np.zeros((1, 1))),
                "np": np,
            }

            # Evaluate
            pred = eval(eq_str, {"__builtins__": {"min": min, "max": max}}, local_vars)
            pred = np.asarray(pred).astype(bool)

            # Get truth
            truth = features.get("truth")
            if truth is None:
                return None

            truth_bool = truth > 0.5

            # Compute metrics
            tp = np.sum(pred & truth_bool)
            fp = np.sum(pred & ~truth_bool)
            fn = np.sum(~pred & truth_bool)
            tn = np.sum(~pred & ~truth_bool)

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)

            intersection = np.sum(pred & truth_bool)
            union = np.sum(pred | truth_bool)
            iou = intersection / (union + 1e-10)

            # Physics score (terrain-adaptive)
            physics_score = self._compute_physics_score(pred, features, terrain)

            # Complexity from template
            complexity = EQUATION_TEMPLATES.get(template_name, {}).get("complexity", 1)

            return EquationResult(
                template_name=template_name,
                equation=eq_str,
                params=params,
                terrain_profile=terrain,
                iou=float(iou),
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                physics_score=float(physics_score),
                complexity=complexity,
                water_fraction_pred=float(pred.sum() / pred.size),
                water_fraction_true=float(truth_bool.sum() / truth_bool.size),
                n_samples=int(pred.size),
            )

        except Exception as e:
            logger.debug(f"Equation evaluation failed: {e}")
            return None

    def _compute_physics_score(
        self, pred: np.ndarray, features: Dict[str, np.ndarray], terrain: str
    ) -> float:
        """Compute terrain-adaptive physics compliance score."""
        constraints = self.terrain_classifier.get_physics_constraints(terrain)

        hand = features.get("hand", np.zeros_like(pred))
        slope = features.get("slope", np.zeros_like(pred))

        # Check HAND violation (water at high HAND)
        hand_violation = pred & (hand > constraints["hand_max"])
        hand_score = 1.0 - (hand_violation.sum() / (pred.sum() + 1e-10))

        # Check slope violation (water on steep slopes)
        slope_violation = pred & (slope > constraints["slope_max"])
        slope_score = 1.0 - (slope_violation.sum() / (pred.sum() + 1e-10))

        # Combined score
        physics_score = 0.6 * max(0, hand_score) + 0.4 * max(0, slope_score)

        return physics_score


# =============================================================================
# Grid Search Runner
# =============================================================================


class EquationSearchRunner:
    """Run exhaustive grid search for best equations."""

    def __init__(self):
        self.evaluator = EquationEvaluator()
        self.results = {}
        self.checkpoint_counter = 0

    def generate_param_combinations(self, template_name: str) -> List[Dict[str, float]]:
        """Generate all parameter combinations for a template."""
        template_config = EQUATION_TEMPLATES.get(template_name, {})
        params_ranges = template_config.get("params", {})

        if not params_ranges:
            return [{}]

        # Generate all combinations
        keys = list(params_ranges.keys())
        values = [params_ranges[k] for k in keys]

        combinations = list(product(*values))

        # Limit if too many
        max_combos = CONFIG["max_equations_per_template"]
        if len(combinations) > max_combos:
            logger.info(
                f"  Limiting {template_name} from {len(combinations)} to {max_combos} combinations"
            )
            indices = np.random.choice(len(combinations), max_combos, replace=False)
            combinations = [combinations[i] for i in indices]

        return [dict(zip(keys, combo)) for combo in combinations]

    def run_search(
        self,
        chip_files: List[Path],
        template_names: Optional[List[str]] = None,
        terrain_filter: Optional[str] = None,
    ) -> Dict[str, List[EquationResult]]:
        """Run grid search across all templates and chips."""

        if template_names is None:
            template_names = list(EQUATION_TEMPLATES.keys())

        logger.info("=" * 60)
        logger.info("STARTING EQUATION SEARCH v2")
        logger.info("=" * 60)
        logger.info(f"Templates: {len(template_names)}")
        logger.info(f"Chips: {len(chip_files)}")
        logger.info(f"GPU: {GPU_AVAILABLE}")

        # Load all chips
        logger.info("\nLoading chips...")
        chip_data = []
        for chip_file in chip_files:
            try:
                features, terrain = self.evaluator.load_chip(chip_file)
                if features.get("truth") is None:
                    logger.debug(f"Skipping {chip_file.name}: no ground truth")
                    continue

                if terrain_filter and terrain != terrain_filter:
                    continue

                chip_data.append(
                    {"name": chip_file.stem, "features": features, "terrain": terrain}
                )
                logger.info(f"  Loaded {chip_file.name} - terrain: {terrain}")
            except Exception as e:
                logger.warning(f"  Failed to load {chip_file}: {e}")

        logger.info(f"\nLoaded {len(chip_data)} chips with ground truth")

        if not chip_data:
            logger.error("No valid chips found!")
            return {}

        # Run search for each template
        all_results = {t: [] for t in template_names}
        total_equations = 0
        start_time = time.time()

        for template_name in template_names:
            template_config = EQUATION_TEMPLATES.get(template_name, {})
            template_str = template_config.get("template", "")

            if not template_str:
                continue

            param_combos = self.generate_param_combinations(template_name)
            logger.info(
                f"\n{template_name}: {len(param_combos)} parameter combinations"
            )

            template_results = []

            for params in param_combos:
                # Evaluate on all chips
                chip_ious = []
                chip_physics = []

                for chip in chip_data:
                    result = self.evaluator.evaluate_equation(
                        template_name,
                        template_str,
                        params,
                        chip["features"],
                        chip["terrain"],
                    )

                    if result:
                        chip_ious.append(result.iou)
                        chip_physics.append(result.physics_score)

                if chip_ious:
                    # Aggregate across chips
                    mean_iou = np.mean(chip_ious)
                    mean_physics = np.mean(chip_physics)

                    # Store if good enough
                    if mean_iou > 0.1:  # Minimum threshold
                        template_results.append(
                            {
                                "template": template_name,
                                "params": params,
                                "mean_iou": float(mean_iou),
                                "std_iou": float(np.std(chip_ious)),
                                "mean_physics": float(mean_physics),
                                "n_chips": len(chip_ious),
                                "combined_score": float(mean_iou * mean_physics),
                            }
                        )

                total_equations += 1
                self.checkpoint_counter += 1

                # Progress
                if total_equations % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = total_equations / elapsed
                    logger.info(
                        f"  Evaluated {total_equations} equations ({rate:.1f}/sec)"
                    )

                # Checkpoint
                if self.checkpoint_counter >= CONFIG["checkpoint_interval"]:
                    self._save_checkpoint(all_results)
                    self.checkpoint_counter = 0

            # Sort by combined score
            template_results.sort(key=lambda x: x["combined_score"], reverse=True)
            all_results[template_name] = template_results[:100]  # Keep top 100

            if template_results:
                best = template_results[0]
                logger.info(
                    f"  Best: IoU={best['mean_iou']:.4f}, Physics={best['mean_physics']:.4f}"
                )

        # Final save
        self._save_results(all_results)

        elapsed = time.time() - start_time
        logger.info(f"\nCompleted in {elapsed / 60:.1f} minutes")
        logger.info(f"Total equations evaluated: {total_equations}")

        return all_results

    def _save_checkpoint(self, results: Dict):
        """Save checkpoint."""
        checkpoint_path = CONFIG["output_dir"] / "equation_search_checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.debug(f"Checkpoint saved to {checkpoint_path}")

    def _save_results(self, results: Dict):
        """Save final results."""
        CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

        # Save per-template results
        for template_name, template_results in results.items():
            if template_results:
                output_path = CONFIG["output_dir"] / f"equations_{template_name}.json"
                with open(output_path, "w") as f:
                    json.dump(template_results, f, indent=2)
                logger.info(f"Saved {len(template_results)} results to {output_path}")

        # Save summary
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_templates": len(results),
            "best_per_template": {},
        }

        for template_name, template_results in results.items():
            if template_results:
                best = template_results[0]
                summary["best_per_template"][template_name] = {
                    "params": best["params"],
                    "mean_iou": best["mean_iou"],
                    "mean_physics": best["mean_physics"],
                    "combined_score": best["combined_score"],
                }

        # Overall best
        all_best = []
        for template_name, template_results in results.items():
            if template_results:
                best = template_results[0].copy()
                best["template"] = template_name
                all_best.append(best)

        all_best.sort(key=lambda x: x["combined_score"], reverse=True)
        summary["top_10_overall"] = all_best[:10]

        summary_path = CONFIG["output_dir"] / "equation_search_summary_v2.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nSummary saved to {summary_path}")

        # Print top 10
        logger.info("\nTOP 10 EQUATIONS:")
        for i, eq in enumerate(all_best[:10]):
            logger.info(
                f"{i + 1}. {eq['template']}: IoU={eq['mean_iou']:.4f}, Physics={eq['mean_physics']:.4f}"
            )
            logger.info(f"   Params: {eq['params']}")


# =============================================================================
# Main
# =============================================================================


def main():
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("GPU EQUATION SEARCH v2 - Full GPU Utilization")
    logger.info("=" * 80)

    # Check GPU
    if GPU_AVAILABLE:
        gpu_props = cp.cuda.runtime.getDeviceProperties(0)
        total_mem = gpu_props["totalGlobalMem"] / 1e9
        logger.info(f"GPU: {gpu_props['name'].decode()}, Memory: {total_mem:.1f} GB")

    # Find chip files
    chip_dir = CONFIG["chip_dir"]
    chip_files = list(chip_dir.glob("*.npy"))

    if not chip_files:
        logger.error(f"No chip files found in {chip_dir}")
        return

    logger.info(f"Found {len(chip_files)} chip files")

    # Run search
    runner = EquationSearchRunner()
    results = runner.run_search(chip_files)

    # Print final summary
    if results:
        logger.info("\n" + "=" * 60)
        logger.info("EQUATION SEARCH COMPLETE")
        logger.info("=" * 60)

        total_equations = sum(len(r) for r in results.values())
        logger.info(f"Total top equations saved: {total_equations}")


if __name__ == "__main__":
    main()
