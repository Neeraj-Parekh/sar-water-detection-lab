"""
GPU Pipeline Orchestration Script for SAR Water Detection
===========================================================

Main entry point to run the complete GPU equation search pipeline.
Designed to be executed on the RTX A5000 GPU server via SSH.

Usage:
    python run_gpu_pipeline.py --chip-dir /path/to/chips --output-dir /path/to/results

Author: SAR Water Detection Lab
Date: 2026-01-14
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np

# Local imports
from gpu_equation_search import (
    GPUEquationEvaluator, 
    EQUATION_TEMPLATES, 
    PARAM_RANGES, 
    REGIME_GRAMMAR,
    run_search as run_exhaustive_search,
    GPU_AVAILABLE
)
from lobo_validator import (
    LOBOValidator, 
    load_chip_metadata_from_dir,
    create_basin_mapping_from_csv,
    ValidationResult
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# System Diagnostics
# =============================================================================

def check_system():
    """Run system diagnostics and print GPU info."""
    print("\n" + "="*60)
    print("       SAR WATER DETECTION - GPU PIPELINE")
    print("="*60)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Python: {sys.version}")
    
    # NumPy
    print(f"NumPy: {np.__version__}")
    
    # CuPy / GPU
    if GPU_AVAILABLE:
        import cupy as cp
        print(f"CuPy: {cp.__version__}")
        print(f"CUDA Version: {cp.cuda.runtime.runtimeGetVersion()}")
        
        # GPU info
        for i in range(cp.cuda.runtime.getDeviceCount()):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
            mem_gb = props['totalGlobalMem'] / (1024**3)
            print(f"GPU {i}: {name} ({mem_gb:.1f} GB)")
    else:
        print("CuPy: NOT AVAILABLE (running on CPU)")
    
    # PyTorch (if available)
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available (PyTorch): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch: NOT INSTALLED")
    
    # SciPy
    try:
        import scipy
        print(f"SciPy: {scipy.__version__}")
    except ImportError:
        print("SciPy: NOT INSTALLED (required)")
    
    print("="*60 + "\n")
    
    return GPU_AVAILABLE


def count_chips(chip_dir: Path) -> Dict[str, Any]:
    """Count and categorize chips."""
    chip_files = list(chip_dir.glob("*.npy"))
    
    stats = {
        'total_chips': len(chip_files),
        'total_size_mb': sum(f.stat().st_size for f in chip_files) / (1024**2),
        'sample_shape': None,
    }
    
    if chip_files:
        sample = np.load(chip_files[0])
        stats['sample_shape'] = sample.shape
    
    return stats


# =============================================================================
# Pipeline Stages
# =============================================================================

class PipelineStage:
    """Base class for pipeline stages."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.status = "pending"
    
    def run(self, *args, **kwargs) -> Any:
        raise NotImplementedError
    
    def __enter__(self):
        self.start_time = time.time()
        self.status = "running"
        logger.info(f"▶ Starting stage: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if exc_type:
            self.status = "failed"
            logger.error(f"✗ Stage failed: {self.name} ({duration:.1f}s)")
        else:
            self.status = "completed"
            logger.info(f"✓ Completed stage: {self.name} ({duration:.1f}s)")


class ExhaustiveSearchStage(PipelineStage):
    """Stage 1: Exhaustive grid search for equations."""
    
    def __init__(self, chip_dir: Path, output_dir: Path, 
                 regimes: List[str], max_candidates: int):
        super().__init__("Exhaustive Grid Search")
        self.chip_dir = chip_dir
        self.output_dir = output_dir
        self.regimes = regimes
        self.max_candidates = max_candidates
    
    def run(self) -> Dict[str, List[Dict]]:
        """Run exhaustive search."""
        results = run_exhaustive_search(
            chip_dir=self.chip_dir,
            output_dir=self.output_dir,
            regimes=self.regimes,
            max_candidates_per_regime=self.max_candidates
        )
        return results


class LOBOValidationStage(PipelineStage):
    """Stage 2: LOBO cross-validation with bootstrap CIs."""
    
    def __init__(self, chip_dir: Path, candidates_file: Path,
                 output_file: Path, basin_csv: Optional[Path] = None):
        super().__init__("LOBO Cross-Validation")
        self.chip_dir = chip_dir
        self.candidates_file = candidates_file
        self.output_file = output_file
        self.basin_csv = basin_csv
    
    def run(self) -> List[ValidationResult]:
        """Run LOBO validation."""
        # Load basin mapping
        basin_mapping = None
        if self.basin_csv and self.basin_csv.exists():
            basin_mapping = create_basin_mapping_from_csv(self.basin_csv)
        
        # Load chip metadata
        chip_metadata = load_chip_metadata_from_dir(self.chip_dir, basin_mapping)
        
        if len(chip_metadata) == 0:
            raise ValueError(f"No chips found in {self.chip_dir}")
        
        # Load candidates
        with open(self.candidates_file, 'r') as f:
            candidates = json.load(f)
        
        logger.info(f"Loaded {len(candidates)} candidates to validate")
        
        # Run validation
        validator = LOBOValidator(chip_metadata, n_bootstrap=1000)
        results = validator.validate_candidates(candidates, self.output_file)
        
        return results


class FinalReportStage(PipelineStage):
    """Stage 3: Generate final report with locked equations."""
    
    def __init__(self, results_dir: Path, output_file: Path):
        super().__init__("Final Report Generation")
        self.results_dir = results_dir
        self.output_file = output_file
    
    def run(self) -> Dict[str, Any]:
        """Generate final report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'locked_equations': {},
            'validation_summary': {},
        }
        
        # Load validation results for each regime
        for file in self.results_dir.glob("validated_*.json"):
            regime = file.stem.replace("validated_", "")
            
            with open(file, 'r') as f:
                results = json.load(f)
            
            if results:
                top = results[0]
                report['locked_equations'][regime] = {
                    'equation': top['equation'],
                    'iou': top['mean_iou'],
                    'ci_iou': [top.get('ci_iou_lower', 0), top.get('ci_iou_upper', 1)],
                    'physics_score': top['mean_physics'],
                    'combined_score': top['combined_score'],
                }
                report['validation_summary'][regime] = {
                    'candidates_evaluated': len(results),
                    'top5_iou_range': [results[min(4, len(results)-1)]['mean_iou'], 
                                       results[0]['mean_iou']],
                }
        
        # Save report
        with open(self.output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Final report saved to {self.output_file}")
        
        return report


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(chip_dir: Path, output_dir: Path,
                 regimes: Optional[List[str]] = None,
                 max_candidates: int = 5000,
                 basin_csv: Optional[Path] = None,
                 skip_search: bool = False):
    """Run complete GPU pipeline.
    
    Args:
        chip_dir: Directory containing NPY chip files
        output_dir: Directory for all outputs
        regimes: List of regimes to process (default: all)
        max_candidates: Max candidates per regime for grid search
        basin_csv: Optional CSV mapping chips to basins
        skip_search: If True, skip exhaustive search (use existing results)
    """
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    search_dir = output_dir / "search_results"
    validation_dir = output_dir / "validation_results"
    search_dir.mkdir(exist_ok=True)
    validation_dir.mkdir(exist_ok=True)
    
    # Default regimes
    if regimes is None:
        regimes = list(REGIME_GRAMMAR.keys())
    
    # Check system
    gpu_ok = check_system()
    if not gpu_ok:
        logger.warning("GPU not available - running in CPU mode (SLOW)")
    
    # Count chips
    chip_stats = count_chips(chip_dir)
    logger.info(f"Found {chip_stats['total_chips']} chips "
               f"({chip_stats['total_size_mb']:.1f} MB)")
    logger.info(f"Sample shape: {chip_stats['sample_shape']}")
    
    pipeline_start = time.time()
    
    # Stage 1: Exhaustive Search
    if not skip_search:
        with ExhaustiveSearchStage(chip_dir, search_dir, regimes, max_candidates) as stage:
            search_results = stage.run()
    else:
        logger.info("Skipping exhaustive search (using existing results)")
        search_results = {}
        for regime in regimes:
            results_file = search_dir / f"top_equations_{regime}.json"
            if results_file.exists():
                with open(results_file) as f:
                    search_results[regime] = json.load(f)
    
    # Stage 2: LOBO Validation (for each regime)
    all_validation_results = {}
    
    for regime in regimes:
        candidates_file = search_dir / f"top_equations_{regime}.json"
        
        if not candidates_file.exists():
            logger.warning(f"No candidates found for regime '{regime}'")
            continue
        
        output_file = validation_dir / f"validated_{regime}.json"
        
        with LOBOValidationStage(chip_dir, candidates_file, output_file, basin_csv) as stage:
            results = stage.run()
            all_validation_results[regime] = results
    
    # Stage 3: Final Report
    report_file = output_dir / "final_report.json"
    
    with FinalReportStage(validation_dir, report_file) as stage:
        report = stage.run()
    
    # Summary
    pipeline_duration = time.time() - pipeline_start
    
    print("\n" + "="*60)
    print("                 PIPELINE COMPLETE")
    print("="*60)
    print(f"\nTotal duration: {pipeline_duration:.1f} seconds")
    print(f"Output directory: {output_dir}")
    
    print("\n📋 LOCKED EQUATIONS:")
    for regime, eq_data in report.get('locked_equations', {}).items():
        iou = eq_data['iou']
        ci = eq_data['ci_iou']
        print(f"\n  [{regime.upper()}]")
        print(f"  IoU: {iou:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        print(f"  Physics: {eq_data['physics_score']:.3f}")
        print(f"  Equation: {eq_data['equation'][:60]}...")
    
    print("\n" + "="*60)
    
    return report


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GPU Exhaustive Equation Search Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_gpu_pipeline.py --chip-dir ./chips --output-dir ./results

  # Skip search, just validate existing candidates
  python run_gpu_pipeline.py --chip-dir ./chips --output-dir ./results --skip-search

  # Run for specific regimes only
  python run_gpu_pipeline.py --chip-dir ./chips --output-dir ./results --regimes wide_river narrow_river
        """
    )
    
    parser.add_argument("--chip-dir", type=Path, required=True,
                       help="Directory containing NPY chip files")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for all results")
    parser.add_argument("--regimes", nargs="+", default=None,
                       help="Regimes to process (default: all)")
    parser.add_argument("--max-candidates", type=int, default=5000,
                       help="Max candidates per regime (default: 5000)")
    parser.add_argument("--basin-csv", type=Path, default=None,
                       help="CSV mapping chips to basins (for LOBO)")
    parser.add_argument("--skip-search", action="store_true",
                       help="Skip exhaustive search, use existing results")
    
    args = parser.parse_args()
    
    # Validate paths
    if not args.chip_dir.exists():
        print(f"ERROR: Chip directory not found: {args.chip_dir}")
        sys.exit(1)
    
    # Run pipeline
    try:
        report = run_pipeline(
            chip_dir=args.chip_dir,
            output_dir=args.output_dir,
            regimes=args.regimes,
            max_candidates=args.max_candidates,
            basin_csv=args.basin_csv,
            skip_search=args.skip_search
        )
        
        # Exit code based on results
        if report.get('locked_equations'):
            sys.exit(0)
        else:
            print("WARNING: No equations found")
            sys.exit(1)
            
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
