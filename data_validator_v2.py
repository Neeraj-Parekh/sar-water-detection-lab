#!/usr/bin/env python3
"""
================================================================================
RESEARCH-GRADE DATA VALIDATOR v2
================================================================================

Enhanced validation with:
1. CO-REGISTRATION CHECK - Detect SAR/DEM misalignment
2. LABEL SANITY SCORE - Detect suspicious "Water" labels that are too bright
3. STRIPING/ARTIFACT DETECTION - Find burst boundaries and black stripes
4. GMM-BASED QUALITY METRICS - Statistical health of each chip

Author: SAR Water Detection Project
Date: 2026-01-25
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage, stats
from scipy.ndimage import uniform_filter, sobel
from sklearn.mixture import GaussianMixture

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ChipValidationResult:
    """Validation result for a single chip."""

    chip_name: str
    is_valid: bool
    quality_score: float  # 0-1, higher = better
    issues: List[str]

    # Individual checks
    slope_corruption: bool
    sar_dem_misalignment: bool
    label_sanity_pass: bool
    striping_detected: bool

    # Metrics
    vh_median_water: Optional[float]
    vh_median_land: Optional[float]
    gmm_threshold: Optional[float]
    sar_slope_correlation: Optional[float]
    water_fraction: float
    nan_fraction: float


def check_slope_corruption(slope: np.ndarray) -> Tuple[bool, str]:
    """
    Check if SLOPE is corrupted (0-1 range instead of 0-90 degrees).
    """
    slope_valid = slope[~np.isnan(slope)]
    if len(slope_valid) == 0:
        return True, "SLOPE all NaN"

    slope_max = np.nanmax(slope_valid)
    slope_p99 = np.nanpercentile(slope_valid, 99)

    if slope_max <= 1.5:
        return True, f"SLOPE in 0-1 range (max={slope_max:.3f}, needs 0-90)"
    elif slope_p99 > 90:
        return True, f"SLOPE exceeds 90 degrees (p99={slope_p99:.1f})"

    return False, ""


def check_sar_dem_coregistration(
    vh: np.ndarray, slope: np.ndarray, aspect: Optional[np.ndarray] = None
) -> Tuple[bool, float, str]:
    """
    Check co-registration between SAR and DEM.

    In mountainous areas, bright SAR slopes should align with DEM slopes
    facing the satellite (radar-facing slopes). If correlation is low,
    the chip is MISALIGNED.

    Uses correlation between SAR intensity gradient and terrain slope.
    """
    # Compute SAR intensity gradient
    vh_gx = sobel(vh, axis=1)
    vh_gy = sobel(vh, axis=0)
    vh_gradient = np.sqrt(vh_gx**2 + vh_gy**2)

    # Flatten and remove NaN
    vh_flat = vh_gradient.flatten()
    slope_flat = slope.flatten()

    valid_mask = (~np.isnan(vh_flat)) & (~np.isnan(slope_flat))
    vh_valid = vh_flat[valid_mask]
    slope_valid = slope_flat[valid_mask]

    if len(vh_valid) < 1000:
        return False, 0.0, "Insufficient valid pixels for co-registration check"

    # Compute correlation
    try:
        correlation, p_value = stats.pearsonr(vh_valid, slope_valid)
    except Exception:
        return False, 0.0, "Correlation computation failed"

    # In mountainous terrain (high slope variance), we expect some correlation
    slope_std = np.std(slope_valid)

    if slope_std > 5:  # Mountainous terrain
        if abs(correlation) < 0.05:
            return (
                True,
                correlation,
                f"Poor SAR-DEM alignment in mountains (corr={correlation:.3f})",
            )

    return False, correlation, ""


def check_label_sanity(
    vh: np.ndarray, label: np.ndarray, bright_threshold: float = -12.0
) -> Tuple[bool, float, str]:
    """
    Check if water labels make physical sense.

    Water should be DARK in VH (typically < -18 dB).
    If median VH of "water" pixels is > -12 dB, the label is suspicious.
    """
    water_mask = label > 0.5
    water_count = water_mask.sum()

    if water_count < 100:
        return True, np.nan, "Insufficient water pixels"

    vh_water = vh[water_mask]
    vh_water_valid = vh_water[~np.isnan(vh_water)]

    if len(vh_water_valid) < 50:
        return True, np.nan, "Insufficient valid water VH values"

    median_vh_water = np.median(vh_water_valid)

    if median_vh_water > bright_threshold:
        return (
            False,
            median_vh_water,
            f"Water labels too bright: median VH = {median_vh_water:.1f} dB (expected < {bright_threshold})",
        )

    return True, median_vh_water, ""


def check_striping_artifacts(
    data: np.ndarray, channel_name: str = "VH"
) -> Tuple[bool, str]:
    """
    Detect SAR striping/burst boundary artifacts.

    Look for rows/columns where mean value drops to noise floor instantly.
    """
    # Check for horizontal stripes (rows with abnormal values)
    row_means = np.nanmean(data, axis=1)
    row_stds = np.nanstd(data, axis=1)

    # Detect sudden drops
    row_diffs = np.abs(np.diff(row_means))
    overall_std = np.nanstd(row_means)

    if overall_std > 0:
        spike_threshold = 4 * overall_std
        spikes = np.where(row_diffs > spike_threshold)[0]

        if len(spikes) > 2:
            return (
                True,
                f"{channel_name}: {len(spikes)} horizontal stripe boundaries detected",
            )

    # Check for vertical stripes (columns)
    col_means = np.nanmean(data, axis=0)
    col_diffs = np.abs(np.diff(col_means))
    col_std = np.nanstd(col_means)

    if col_std > 0:
        spike_threshold = 4 * col_std
        spikes = np.where(col_diffs > spike_threshold)[0]

        if len(spikes) > 2:
            return (
                True,
                f"{channel_name}: {len(spikes)} vertical stripe boundaries detected",
            )

    # Check for completely zero rows/columns
    zero_rows = np.sum(np.all(np.abs(data) < 1e-6, axis=1))
    zero_cols = np.sum(np.all(np.abs(data) < 1e-6, axis=0))

    if zero_rows > 5 or zero_cols > 5:
        return True, f"{channel_name}: {zero_rows} zero rows, {zero_cols} zero cols"

    return False, ""


def compute_gmm_quality(vh: np.ndarray, n_components: int = 2) -> Dict:
    """
    Use GMM to assess chip quality and find optimal threshold.
    """
    vh_flat = vh.flatten()
    vh_valid = vh_flat[~np.isnan(vh_flat)]

    if len(vh_valid) < 500:
        return {
            "gmm_threshold": None,
            "water_mode": None,
            "land_mode": None,
            "separation": None,
            "gmm_quality": "insufficient_data",
        }

    try:
        gmm = GaussianMixture(n_components=n_components, random_state=42, max_iter=100)
        gmm.fit(vh_valid.reshape(-1, 1))

        means = gmm.means_.flatten()
        stds = np.sqrt(gmm.covariances_.flatten())

        water_idx = np.argmin(means)
        land_idx = np.argmax(means)

        water_mode = float(means[water_idx])
        land_mode = float(means[land_idx])
        water_std = float(stds[water_idx])
        land_std = float(stds[land_idx])

        # Threshold = midpoint
        threshold = (water_mode + land_mode) / 2

        # Separation quality: how well separated are the modes?
        separation = (land_mode - water_mode) / (water_std + land_std + 1e-6)

        quality = (
            "good" if separation > 1.5 else "marginal" if separation > 1.0 else "poor"
        )

        return {
            "gmm_threshold": threshold,
            "water_mode": water_mode,
            "land_mode": land_mode,
            "separation": separation,
            "gmm_quality": quality,
        }
    except Exception as e:
        return {
            "gmm_threshold": None,
            "water_mode": None,
            "land_mode": None,
            "separation": None,
            "gmm_quality": f"error: {str(e)}",
        }


def validate_chip(chip_path: Path) -> ChipValidationResult:
    """
    Comprehensive validation of a single chip.
    """
    chip_name = chip_path.stem
    issues = []

    try:
        chip = np.load(chip_path)
    except Exception as e:
        return ChipValidationResult(
            chip_name=chip_name,
            is_valid=False,
            quality_score=0.0,
            issues=[f"Failed to load: {e}"],
            slope_corruption=False,
            sar_dem_misalignment=False,
            label_sanity_pass=False,
            striping_detected=False,
            vh_median_water=None,
            vh_median_land=None,
            gmm_threshold=None,
            sar_slope_correlation=None,
            water_fraction=0.0,
            nan_fraction=1.0,
        )

    # Extract channels
    vv = chip[:, :, 0].astype(np.float32)
    vh = chip[:, :, 1].astype(np.float32)
    dem = chip[:, :, 2].astype(np.float32)
    slope = chip[:, :, 3].astype(np.float32)
    hand = chip[:, :, 4].astype(np.float32)
    twi = chip[:, :, 5].astype(np.float32)
    label = chip[:, :, 6].astype(np.float32) if chip.shape[2] > 6 else np.zeros_like(vv)

    # Basic stats
    total_pixels = vv.size
    nan_count = np.sum(np.isnan(vh))
    nan_fraction = nan_count / total_pixels
    water_fraction = np.sum(label > 0.5) / total_pixels

    if nan_fraction > 0.5:
        issues.append(f"High NaN fraction: {nan_fraction:.1%}")

    # 1. SLOPE CORRUPTION CHECK
    slope_corrupt, slope_msg = check_slope_corruption(slope)
    if slope_corrupt:
        issues.append(slope_msg)

    # 2. SAR-DEM CO-REGISTRATION CHECK
    sar_dem_misalign, sar_slope_corr, coreg_msg = check_sar_dem_coregistration(
        vh, slope
    )
    if sar_dem_misalign:
        issues.append(coreg_msg)

    # 3. LABEL SANITY CHECK
    label_pass, vh_median_water, label_msg = check_label_sanity(vh, label)
    if not label_pass and label_msg:
        issues.append(label_msg)

    # VH median for land pixels
    land_mask = label < 0.5
    vh_land = vh[land_mask]
    vh_median_land = (
        float(np.median(vh_land[~np.isnan(vh_land)]))
        if np.sum(land_mask) > 100
        else None
    )

    # 4. STRIPING CHECK
    striping_vh, stripe_msg_vh = check_striping_artifacts(vh, "VH")
    striping_vv, stripe_msg_vv = check_striping_artifacts(vv, "VV")
    striping = striping_vh or striping_vv
    if striping_vh:
        issues.append(stripe_msg_vh)
    if striping_vv:
        issues.append(stripe_msg_vv)

    # 5. GMM QUALITY
    gmm_result = compute_gmm_quality(vh)
    gmm_threshold = gmm_result.get("gmm_threshold")
    if gmm_result.get("gmm_quality") == "poor":
        issues.append(f"Poor GMM separation: modes not well separated")

    # Compute quality score
    quality_score = 1.0
    if slope_corrupt:
        quality_score -= 0.4
    if sar_dem_misalign:
        quality_score -= 0.3
    if not label_pass:
        quality_score -= 0.3
    if striping:
        quality_score -= 0.2
    if nan_fraction > 0.1:
        quality_score -= 0.1
    if gmm_result.get("gmm_quality") == "poor":
        quality_score -= 0.1

    quality_score = max(0.0, min(1.0, quality_score))

    # Determine validity
    is_valid = (
        not slope_corrupt
        and not sar_dem_misalign
        and label_pass
        and not striping
        and nan_fraction < 0.3
    )

    return ChipValidationResult(
        chip_name=chip_name,
        is_valid=bool(is_valid),  # Fix: numpy bool to Python bool
        quality_score=float(quality_score),
        issues=issues,
        slope_corruption=bool(slope_corrupt),  # Fix: numpy bool to Python bool
        sar_dem_misalignment=bool(sar_dem_misalign),  # Fix: numpy bool to Python bool
        label_sanity_pass=bool(label_pass),  # Fix: numpy bool to Python bool
        striping_detected=bool(striping),  # Fix: numpy bool to Python bool
        vh_median_water=float(vh_median_water)
        if vh_median_water is not None and not np.isnan(vh_median_water)
        else None,
        vh_median_land=float(vh_median_land) if vh_median_land is not None else None,
        gmm_threshold=float(gmm_threshold) if gmm_threshold is not None else None,
        sar_slope_correlation=float(sar_slope_corr) if sar_slope_corr else None,
        water_fraction=float(water_fraction),
        nan_fraction=float(nan_fraction),
    )


def validate_dataset(chip_dir: Path, output_report: Path = None) -> Dict:
    """
    Validate all chips in a directory.
    """
    chip_files = sorted(chip_dir.glob("*_with_truth.npy"))

    if not chip_files:
        # Try other patterns
        chip_files = sorted(chip_dir.glob("*.npy"))

    logger.info(f"Found {len(chip_files)} chips to validate")

    results = []
    valid_chips = []
    invalid_chips = []

    for chip_path in chip_files:
        result = validate_chip(chip_path)
        results.append(result)

        if result.is_valid:
            valid_chips.append(result.chip_name)
        else:
            invalid_chips.append(result.chip_name)

        status = "VALID" if result.is_valid else "INVALID"
        logger.info(
            f"  {result.chip_name}: {status} (score={result.quality_score:.2f})"
        )
        if result.issues:
            for issue in result.issues:
                logger.info(f"    - {issue}")

    # Summary
    summary = {
        "total_chips": len(chip_files),
        "valid_chips": len(valid_chips),
        "invalid_chips": len(invalid_chips),
        "validation_rate": len(valid_chips) / len(chip_files) if chip_files else 0,
        "valid_chip_names": valid_chips,
        "invalid_chip_names": invalid_chips,
        "issue_breakdown": {
            "slope_corruption": sum(1 for r in results if r.slope_corruption),
            "sar_dem_misalignment": sum(1 for r in results if r.sar_dem_misalignment),
            "label_sanity_fail": sum(1 for r in results if not r.label_sanity_pass),
            "striping_detected": sum(1 for r in results if r.striping_detected),
        },
        "detailed_results": [asdict(r) for r in results],
    }

    # Save report
    if output_report:
        with open(output_report, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Report saved to: {output_report}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total chips: {summary['total_chips']}")
    logger.info(
        f"Valid chips: {summary['valid_chips']} ({summary['validation_rate']:.1%})"
    )
    logger.info(f"Invalid chips: {summary['invalid_chips']}")
    logger.info("\nIssue breakdown:")
    for issue, count in summary["issue_breakdown"].items():
        logger.info(f"  {issue}: {count}")

    if invalid_chips:
        logger.info("\nInvalid chips (exclude from training):")
        for name in invalid_chips:
            logger.info(f"  - {name}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Research-grade SAR chip validator")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/mit-aoe/sar_water_detection/chips_expanded_npy",
        help="Directory containing chip .npy files",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="validation_report_v2.json",
        help="Output JSON report path",
    )

    args = parser.parse_args()

    chip_dir = Path(args.input_dir)
    output_report = Path(args.output_report)

    if not chip_dir.exists():
        logger.error(f"Chip directory not found: {chip_dir}")
        return

    validate_dataset(chip_dir, output_report)


if __name__ == "__main__":
    main()
