"""
Leave-One-Basin-Out (LOBO) Validator with Block Bootstrap
==========================================================

This module implements spatially-aware cross-validation for water detection
equations, including:
- LOBO cross-validation (spatial independence)
- Block bootstrap for confidence intervals (preserves spatial correlation)
- Multi-metric evaluation (IoU, PR-AUC, physics score, topology metrics)

Author: SAR Water Detection Lab
Date: 2026-01-14

References:
- Block bootstrap for spatial data: Politis & Romano (1994)
- Spatial CV: Roberts et al. (2017) "Cross-validation strategies"
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
import json
import logging
from collections import defaultdict

from scipy.stats import spearmanr
from sklearn.model_selection import LeaveOneGroupOut
from gpu_equation_search import GPUFeatureComputer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ValidationResult:
    """Results from LOBO validation of one equation."""
    equation: str
    regime: str
    
    # Per-fold metrics (lists)
    fold_ious: List[float]
    fold_f1s: List[float]
    fold_precisions: List[float]
    fold_recalls: List[float]
    fold_physics: List[float]
    fold_basins: List[str]
    
    # Aggregated metrics with CIs
    mean_iou: float
    ci_iou: Tuple[float, float]
    mean_f1: float
    ci_f1: Tuple[float, float]
    mean_physics: float
    ci_physics: Tuple[float, float]
    
    # Combined ranking score
    combined_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'equation': self.equation,
            'regime': self.regime,
            'fold_ious': [float(x) for x in self.fold_ious],
            'fold_f1s': [float(x) for x in self.fold_f1s],
            'fold_basins': self.fold_basins,
            'mean_iou': float(self.mean_iou),
            'ci_iou_lower': float(self.ci_iou[0]),
            'ci_iou_upper': float(self.ci_iou[1]),
            'mean_f1': float(self.mean_f1),
            'ci_f1_lower': float(self.ci_f1[0]),
            'ci_f1_upper': float(self.ci_f1[1]),
            'mean_physics': float(self.mean_physics),
            'ci_physics_lower': float(self.ci_physics[0]),
            'ci_physics_upper': float(self.ci_physics[1]),
            'combined_score': float(self.combined_score),
        }


@dataclass
class ChipMetadata:
    """Metadata for a single chip."""
    chip_id: str
    basin_id: str
    regime: str
    path: Path
    
    # Optional fields
    area_km2: float = 0.0
    water_fraction: float = 0.0


# =============================================================================
# Block Bootstrap (Spatially Correct)
# =============================================================================

class BlockBootstrap:
    """Block bootstrap for spatially correlated metrics.
    
    Unlike naive bootstrap which resamples individual values, block bootstrap
    resamples entire spatial blocks to preserve correlation structure.
    
    For LOBO, each "block" is a fold (basin).
    """
    
    def __init__(self, n_resamples: int = 1000, random_state: int = 42):
        """Initialize block bootstrap.
        
        Args:
            n_resamples: Number of bootstrap iterations
            random_state: Random seed for reproducibility
        """
        self.n_resamples = n_resamples
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def compute_ci(self, fold_values: List[float], 
                   confidence: float = 0.95) -> Tuple[float, float, float]:
        """Compute confidence interval using block bootstrap.
        
        Args:
            fold_values: Metric values from each fold (one per basin)
            confidence: Confidence level (default 0.95 = 95%)
            
        Returns:
            Tuple of (mean, lower_ci, upper_ci)
        """
        fold_values = np.array(fold_values)
        n_folds = len(fold_values)
        
        if n_folds < 3:
            # Insufficient folds for bootstrap
            return np.mean(fold_values), np.min(fold_values), np.max(fold_values)
        
        # Generate bootstrap samples
        bootstrap_means = []
        
        for _ in range(self.n_resamples):
            # Resample fold indices WITH replacement
            indices = self.rng.choice(n_folds, size=n_folds, replace=True)
            resampled = fold_values[indices]
            bootstrap_means.append(np.mean(resampled))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Compute percentiles
        alpha = 1 - confidence
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_ci = np.percentile(bootstrap_means, lower_percentile)
        upper_ci = np.percentile(bootstrap_means, upper_percentile)
        
        return np.mean(fold_values), lower_ci, upper_ci
    
    def compute_multi_metric_ci(self, fold_metrics: Dict[str, List[float]],
                                 confidence: float = 0.95) -> Dict[str, Tuple[float, float, float]]:
        """Compute CIs for multiple metrics simultaneously.
        
        Args:
            fold_metrics: Dictionary mapping metric names to fold values
            confidence: Confidence level
            
        Returns:
            Dictionary mapping metric names to (mean, lower_ci, upper_ci)
        """
        results = {}
        for metric_name, values in fold_metrics.items():
            mean, lower, upper = self.compute_ci(values, confidence)
            results[metric_name] = (mean, lower, upper)
        return results


# =============================================================================
# Metrics Computation
# =============================================================================

class MetricsComputer:
    """Compute evaluation metrics for water detection."""
    
    @staticmethod
    def compute_iou(pred: np.ndarray, truth: np.ndarray) -> float:
        """Compute Intersection over Union.
        
        IoU = TP / (TP + FP + FN)
        
        Args:
            pred: Binary prediction mask
            truth: Binary ground truth mask
            
        Returns:
            IoU score [0, 1]
        """
        pred_bool = pred > 0.5
        truth_bool = truth > 0.5
        
        intersection = np.logical_and(pred_bool, truth_bool).sum()
        union = np.logical_or(pred_bool, truth_bool).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return float(intersection / union)
    
    @staticmethod
    def compute_precision_recall_f1(pred: np.ndarray, 
                                     truth: np.ndarray) -> Tuple[float, float, float]:
        """Compute precision, recall, and F1 score.
        
        Args:
            pred: Binary prediction mask
            truth: Binary ground truth mask
            
        Returns:
            Tuple of (precision, recall, F1)
        """
        pred_bool = pred > 0.5
        truth_bool = truth > 0.5
        
        tp = np.logical_and(pred_bool, truth_bool).sum()
        fp = np.logical_and(pred_bool, ~truth_bool).sum()
        fn = np.logical_and(~pred_bool, truth_bool).sum()
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        return float(precision), float(recall), float(f1)
    
    @staticmethod
    def compute_pr_auc(pred_prob: np.ndarray, truth: np.ndarray, 
                       n_thresholds: int = 20) -> float:
        """Compute Precision-Recall Area Under Curve.
        
        Args:
            pred_prob: Probability prediction (soft mask)
            truth: Binary ground truth
            n_thresholds: Number of thresholds to sample
            
        Returns:
            PR-AUC score
        """
        thresholds = np.linspace(0, 1, n_thresholds + 1)
        precisions = []
        recalls = []
        
        truth_bool = truth > 0.5
        
        for thresh in thresholds:
            pred_binary = pred_prob >= thresh
            tp = np.logical_and(pred_binary, truth_bool).sum()
            fp = np.logical_and(pred_binary, ~truth_bool).sum()
            fn = np.logical_and(~pred_binary, truth_bool).sum()
            
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Sort by recall and compute AUC using trapezoidal rule
        sorted_indices = np.argsort(recalls)
        recalls_sorted = np.array(recalls)[sorted_indices]
        precisions_sorted = np.array(precisions)[sorted_indices]
        
        auc = np.trapz(precisions_sorted, recalls_sorted)
        
        return float(max(0.0, min(1.0, auc)))
    
    @staticmethod
    def compute_thin_river_recall(pred: np.ndarray, truth: np.ndarray,
                                   width_threshold: int = 5) -> float:
        """Compute recall specifically for thin river pixels.
        
        Uses morphological erosion to identify thin structures.
        
        Args:
            pred: Binary prediction
            truth: Binary ground truth
            width_threshold: Width in pixels to consider "thin"
            
        Returns:
            Recall on thin river pixels
        """
        from scipy.ndimage import binary_erosion, distance_transform_edt
        
        truth_bool = truth > 0.5
        
        # Compute distance from non-water
        dist = distance_transform_edt(truth_bool)
        
        # Thin pixels are those close to edge
        thin_mask = truth_bool & (dist <= width_threshold / 2)
        
        if thin_mask.sum() == 0:
            return 1.0  # No thin rivers
        
        pred_bool = pred > 0.5
        thin_recalled = np.logical_and(pred_bool, thin_mask).sum()
        
        return float(thin_recalled / thin_mask.sum())
    
    @staticmethod
    def compute_area_bias(pred: np.ndarray, truth: np.ndarray) -> float:
        """Compute area bias (predicted/actual - 1).
        
        Positive = over-estimation, Negative = under-estimation
        Target = 0.0
        
        Args:
            pred: Binary prediction
            truth: Binary ground truth
            
        Returns:
            Area bias
        """
        pred_area = (pred > 0.5).sum()
        truth_area = (truth > 0.5).sum()
        
        if truth_area == 0:
            return 0.0 if pred_area == 0 else 999.0
        
        return float(pred_area / truth_area - 1.0)
    
    @staticmethod
    def compute_physics_score(pred: np.ndarray, hand: np.ndarray, 
                               slope: np.ndarray) -> float:
        """Compute physics compliance score.
        
        Combines:
        - HAND monotonicity (water at low HAND)
        - Slope exclusion (no water on steep slopes)
        
        Args:
            pred: Binary prediction
            hand: HAND array (meters)
            slope: Slope array (degrees)
            
        Returns:
            Physics score [0, 1]
        """
        pred_bool = pred > 0.5
        
        # HAND check
        pred_flat = pred_bool.flatten().astype(float)
        hand_flat = hand.flatten()
        
        valid = ~(np.isnan(pred_flat) | np.isnan(hand_flat))
        
        if valid.sum() > 100:
            corr, _ = spearmanr(pred_flat[valid], hand_flat[valid])
            hand_score = 1.0 if corr < 0 else max(0.0, 1.0 - corr)
        else:
            hand_score = 0.5
        
        # Slope check
        steep_mask = slope > 15.0
        water_on_steep = np.logical_and(pred_bool, steep_mask).sum()
        total_water = pred_bool.sum()
        
        if total_water > 0:
            slope_violation = water_on_steep / total_water
            slope_score = 1.0 - slope_violation
        else:
            slope_score = 1.0
        
        # Combine
        return 0.7 * hand_score + 0.3 * slope_score


# =============================================================================
# LOBO Validator
# =============================================================================

class LOBOValidator:
    """Leave-One-Basin-Out cross-validation for equation candidates."""
    
    def __init__(self, chip_metadata: List[ChipMetadata],
                 n_bootstrap: int = 1000):
        """Initialize LOBO validator.
        
        Args:
            chip_metadata: List of chip metadata with basin assignments
            n_bootstrap: Bootstrap iterations for CIs
        """
        self.chip_metadata = chip_metadata
        self.bootstrap = BlockBootstrap(n_resamples=n_bootstrap)
        self.metrics = MetricsComputer()
        self.feature_computer = GPUFeatureComputer() # For computing derived features
        
        # Group chips by basin
        self.basin_chips = defaultdict(list)
        for chip in chip_metadata:
            self.basin_chips[chip.basin_id].append(chip)
        
        self.basins = list(self.basin_chips.keys())
        logger.info(f"LOBO initialized with {len(self.basins)} basins, "
                   f"{len(chip_metadata)} chips")
    
    def load_chip_data(self, chip: ChipMetadata) -> Dict[str, np.ndarray]:
        """Load chip data from NPY file.
        
        Args:
            chip: Chip metadata
            
        Returns:
            Dictionary with bands
        """
        data = np.load(chip.path)
        

        
        # Extract bands
        vv = data[:, :, 0]
        vh = data[:, :, 1]
        dem = data[:, :, 2] if data.shape[2] > 2 else np.zeros_like(vv)
        slope = data[:, :, 3] if data.shape[2] > 3 else np.zeros_like(vv)
        hand = data[:, :, 4] if data.shape[2] > 4 else np.zeros_like(vv)
        twi = data[:, :, 5] if data.shape[2] > 5 else np.zeros_like(vv)
        
        # Handle truth (index 7 for 8-band, index 6 for 7-band heuristic)
        truth = None
        if data.shape[2] == 8:
            truth = data[:, :, 7]
        elif data.shape[2] == 7:
             uniq_vals = np.unique(data[:, :, 6])
             if len(uniq_vals) <= 2 and np.all(np.isin(uniq_vals, [0, 1])):
                 truth = data[:, :, 6]
        elif data.shape[2] > 6:
            truth = data[:, :, 6]

        # Convert to linear
        vv_linear = 10 ** (vv / 10)
        vh_linear = 10 ** (vh / 10)

        # Compute features
        features = {
            'vv': vv,
            'vh': vh,
            'dem': dem,
            'slope': slope,
            'hand': hand,
            'twi': twi,
            'truth': truth,
            'cov': self.feature_computer.compute_cov(vv, window_size=9),
            'entropy': self.feature_computer.compute_glcm_entropy(vv, window_size=9),
            'cpr': self.feature_computer.compute_cpr(vv, vh),
            'sdwi': self.feature_computer.compute_sdwi(vv_linear, vh_linear),
            'swi': self.feature_computer.compute_swi(vv, vh),
            'frangi': self.feature_computer.compute_frangi(-vv),
            'shadow_mask': self.feature_computer.compute_shadow_mask(dem, slope),
            'layover_mask': self.feature_computer.compute_layover_mask(dem, slope),
        }
        
        return features
    
    def evaluate_equation_on_chip(self, equation: str, 
                                   chip_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate equation on a single chip.
        
        Args:
            equation: Executable equation string
            chip_data: Loaded chip data
            
        Returns:
            Dictionary of metrics
        """
        # Build evaluation context
        local_vars = {
            'vv': chip_data['vv'],
            'vh': chip_data['vh'],
            'hand': chip_data.get('hand', np.zeros_like(chip_data['vv'])),
            'slope': chip_data.get('slope', np.zeros_like(chip_data['vv'])),
            'twi': chip_data.get('twi', np.zeros_like(chip_data['vv'])),
            'entropy': chip_data.get('entropy', np.zeros_like(chip_data['vv'])),
            'cov': chip_data.get('cov', np.zeros_like(chip_data['vv'])),
            'cpr': chip_data.get('cpr', np.zeros_like(chip_data['vv'])),
            'sdwi': chip_data.get('sdwi', np.zeros_like(chip_data['vv'])),
            'swi': chip_data.get('swi', chip_data.get('sdwi')),
            'frangi': chip_data.get('frangi', np.zeros_like(chip_data['vv'])),
            'shadow_mask': chip_data.get('shadow_mask', np.zeros_like(chip_data['vv'], dtype=bool)),
            'layover_mask': chip_data.get('layover_mask', np.zeros_like(chip_data['vv'], dtype=bool)),
            'np': np,
        }
        
        try:
            pred = eval(equation, {"__builtins__": {}}, local_vars)
            pred = pred.astype(bool)
        except Exception as e:
            logger.warning(f"Equation eval failed: {e}")
            return {
                'iou': 0.0, 'f1': 0.0, 'precision': 0.0, 
                'recall': 0.0, 'physics': 0.0
            }
        
        truth = chip_data.get('truth')
        hand = chip_data.get('hand')
        slope = chip_data.get('slope')
        
        if truth is not None:
            iou = self.metrics.compute_iou(pred, truth)
            precision, recall, f1 = self.metrics.compute_precision_recall_f1(pred, truth)
        else:
            iou = f1 = precision = recall = 0.5
        
        if hand is not None and slope is not None:
            physics = self.metrics.compute_physics_score(pred, hand, slope)
        else:
            physics = 0.5
        
        return {
            'iou': iou,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'physics': physics,
        }
    
    def run_lobo(self, equation: str, regime: str = 'unknown') -> ValidationResult:
        """Run LOBO cross-validation for an equation.
        
        Args:
            equation: Executable equation string
            regime: Water body regime
            
        Returns:
            ValidationResult with per-fold and aggregate metrics
        """
        fold_ious = []
        fold_f1s = []
        fold_precisions = []
        fold_recalls = []
        fold_physics = []
        fold_basins = []
        
        for test_basin in self.basins:
            # Test on held-out basin
            test_chips = self.basin_chips[test_basin]
            
            # Aggregate metrics across chips in test basin
            basin_ious = []
            basin_f1s = []
            basin_precisions = []
            basin_recalls = []
            basin_physics = []
            
            for chip in test_chips:
                try:
                    chip_data = self.load_chip_data(chip)
                    metrics = self.evaluate_equation_on_chip(equation, chip_data)
                    
                    basin_ious.append(metrics['iou'])
                    basin_f1s.append(metrics['f1'])
                    basin_precisions.append(metrics['precision'])
                    basin_recalls.append(metrics['recall'])
                    basin_physics.append(metrics['physics'])
                except Exception as e:
                    logger.warning(f"Error on chip {chip.chip_id}: {e}")
                    continue
            
            if len(basin_ious) > 0:
                fold_ious.append(np.mean(basin_ious))
                fold_f1s.append(np.mean(basin_f1s))
                fold_precisions.append(np.mean(basin_precisions))
                fold_recalls.append(np.mean(basin_recalls))
                fold_physics.append(np.mean(basin_physics))
                fold_basins.append(test_basin)
        
        # Compute bootstrap CIs
        mean_iou, lower_iou, upper_iou = self.bootstrap.compute_ci(fold_ious)
        mean_f1, lower_f1, upper_f1 = self.bootstrap.compute_ci(fold_f1s)
        mean_physics, lower_physics, upper_physics = self.bootstrap.compute_ci(fold_physics)
        
        # Combined score
        combined = mean_iou * mean_physics
        
        return ValidationResult(
            equation=equation,
            regime=regime,
            fold_ious=fold_ious,
            fold_f1s=fold_f1s,
            fold_precisions=fold_precisions,
            fold_recalls=fold_recalls,
            fold_physics=fold_physics,
            fold_basins=fold_basins,
            mean_iou=mean_iou,
            ci_iou=(lower_iou, upper_iou),
            mean_f1=mean_f1,
            ci_f1=(lower_f1, upper_f1),
            mean_physics=mean_physics,
            ci_physics=(lower_physics, upper_physics),
            combined_score=combined,
        )
    
    def validate_candidates(self, candidates: List[Dict[str, Any]],
                            output_path: Optional[Path] = None) -> List[ValidationResult]:
        """Validate multiple equation candidates.
        
        Args:
            candidates: List of candidate dicts with 'equation' and 'regime' keys
            output_path: Optional path to save results
            
        Returns:
            List of ValidationResults, sorted by combined score
        """
        results = []
        
        for i, candidate in enumerate(candidates):
            equation = candidate.get('equation')
            regime = candidate.get('regime', 'unknown')
            
            logger.info(f"Validating candidate {i+1}/{len(candidates)}...")
            
            result = self.run_lobo(equation, regime)
            results.append(result)
            
            logger.info(f"  IoU: {result.mean_iou:.3f} [{result.ci_iou[0]:.3f}, {result.ci_iou[1]:.3f}]")
        
        # Sort by combined score
        results.sort(key=lambda r: r.combined_score, reverse=True)
        
        # Save if output path provided
        if output_path:
            output_data = [r.to_dict() for r in results]
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Saved {len(results)} validation results to {output_path}")
        
        return results


# =============================================================================
# Utility Functions
# =============================================================================

def load_chip_metadata_from_dir(chip_dir: Path, 
                                 basin_mapping: Optional[Dict[str, str]] = None) -> List[ChipMetadata]:
    """Load chip metadata from a directory.
    
    Args:
        chip_dir: Directory containing NPY files
        basin_mapping: Optional dict mapping chip_id to basin_id
                      If not provided, infers from directory structure
        
    Returns:
        List of ChipMetadata
    """
    metadata = []
    chip_files = list(chip_dir.glob("*.npy"))
    
    for chip_file in chip_files:
        chip_id = chip_file.stem
        
        # Infer basin from chip ID or mapping
        if basin_mapping and chip_id in basin_mapping:
            basin_id = basin_mapping[chip_id]
        else:
            # Default: use first part of filename as basin
            parts = chip_id.split('_')
            basin_id = parts[0] if len(parts) > 1 else chip_id
        
        metadata.append(ChipMetadata(
            chip_id=chip_id,
            basin_id=basin_id,
            regime='unknown',  # Set by user
            path=chip_file,
        ))
    
    return metadata


def create_basin_mapping_from_csv(csv_path: Path) -> Dict[str, str]:
    """Load basin mapping from CSV file.
    
    Expected format:
    chip_id,basin_id,regime
    chip_001,basin_A,wide_river
    chip_002,basin_A,wide_river
    ...
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Dictionary mapping chip_id to basin_id
    """
    import csv
    
    mapping = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row['chip_id']] = row['basin_id']
    
    return mapping


# =============================================================================
# Main Entry Point
# =============================================================================

def run_validation(chip_dir: Path, candidates_file: Path,
                   output_file: Path, basin_csv: Optional[Path] = None):
    """Run LOBO validation on equation candidates.
    
    Args:
        chip_dir: Directory containing NPY chip files
        candidates_file: JSON file with equation candidates
        output_file: Path for output JSON
        basin_csv: Optional CSV with basin assignments
    """
    # Load basin mapping
    basin_mapping = None
    if basin_csv and basin_csv.exists():
        basin_mapping = create_basin_mapping_from_csv(basin_csv)
    
    # Load chip metadata
    chip_metadata = load_chip_metadata_from_dir(chip_dir, basin_mapping)
    
    if len(chip_metadata) == 0:
        logger.error(f"No chips found in {chip_dir}")
        return
    
    # Load candidates
    with open(candidates_file, 'r') as f:
        candidates = json.load(f)
    
    logger.info(f"Loaded {len(candidates)} candidates to validate")
    
    # Run validation
    validator = LOBOValidator(chip_metadata)
    results = validator.validate_candidates(candidates, output_file)
    
    # Print top 5
    print("\n=== TOP 5 EQUATIONS ===")
    for i, result in enumerate(results[:5]):
        print(f"\n{i+1}. {result.equation}")
        print(f"   IoU: {result.mean_iou:.3f} [{result.ci_iou[0]:.3f}, {result.ci_iou[1]:.3f}]")
        print(f"   F1:  {result.mean_f1:.3f} [{result.ci_f1[0]:.3f}, {result.ci_f1[1]:.3f}]")
        print(f"   Physics: {result.mean_physics:.3f}")
        print(f"   Combined: {result.combined_score:.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LOBO Validation with Block Bootstrap")
    parser.add_argument("--chip-dir", type=Path, required=True,
                       help="Directory containing NPY chip files")
    parser.add_argument("--candidates", type=Path, required=True,
                       help="JSON file with equation candidates")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output JSON file")
    parser.add_argument("--basin-csv", type=Path, default=None,
                       help="Optional CSV with basin assignments")
    
    args = parser.parse_args()
    
    run_validation(
        chip_dir=args.chip_dir,
        candidates_file=args.candidates,
        output_file=args.output,
        basin_csv=args.basin_csv
    )
