#!/usr/bin/env python3
"""
================================================================================
DATA VALIDATION SCRIPT - Gatekeeper for New Chip Data
================================================================================

Prevents training on corrupted data by validating:
1. SLOPE values (must be 0-90 degrees, not 0-1 normalized)
2. VH/VV values (must be in dB range, not flat zeros)
3. DEM/HAND/TWI physical ranges
4. Ground truth mask alignment and coverage
5. NaN/Inf contamination

Lessons learned from v8 disaster: 75/86 chips had SLOPE in 0-1 range instead of 0-90!

Usage:
    python data_validator.py --input_dir /path/to/chips --output_report validation_report.json

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
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION RULES (Based on physical constraints)
# =============================================================================

VALIDATION_RULES = {
    "vv": {
        "min": -35.0,  # Minimum VV in dB (very smooth water)
        "max": 5.0,  # Maximum VV in dB (very rough surfaces)
        "typical_mean": (-20.0, -5.0),  # Typical mean range
        "flat_threshold": 0.01,  # If std < this, data is suspiciously flat
        "description": "VV backscatter (dB)",
    },
    "vh": {
        "min": -40.0,  # Minimum VH in dB
        "max": 0.0,  # Maximum VH in dB
        "typical_mean": (-28.0, -12.0),
        "flat_threshold": 0.01,
        "description": "VH backscatter (dB)",
    },
    "dem": {
        "min": -500.0,  # Below sea level (Dead Sea, etc.)
        "max": 9000.0,  # Highest mountains
        "typical_mean": (0.0, 2000.0),
        "flat_threshold": 0.1,
        "description": "Digital Elevation Model (m)",
    },
    "slope": {
        "min": 0.0,  # Flat terrain
        "max": 90.0,  # Vertical cliff
        "typical_mean": (0.0, 30.0),
        "corruption_check": {
            # If max < 1.5, data is likely in 0-1 range (CORRUPTED!)
            "max_threshold": 1.5,
            "error_msg": "SLOPE appears to be in 0-1 range instead of 0-90 degrees!",
        },
        "flat_threshold": 0.001,
        "description": "Terrain slope (degrees)",
    },
    "hand": {
        "min": 0.0,  # At drainage level
        "max": 500.0,  # Very high above drainage
        "typical_mean": (0.0, 100.0),
        "flat_threshold": 0.1,
        "description": "Height Above Nearest Drainage (m)",
    },
    "twi": {
        "min": 0.0,  # Ridge tops
        "max": 30.0,  # Valley bottoms
        "typical_mean": (5.0, 15.0),
        "flat_threshold": 0.01,
        "description": "Topographic Wetness Index",
    },
    "label": {
        "min": 0.0,
        "max": 1.0,
        "binary_check": True,  # Should only contain 0 and 1
        "min_water_fraction": 0.001,  # At least 0.1% water
        "max_water_fraction": 0.99,  # At most 99% water
        "description": "Ground truth water mask",
    },
    "mndwi": {
        "min": -1.0,
        "max": 1.0,
        "typical_mean": (-0.5, 0.5),
        "flat_threshold": 0.001,
        "description": "Modified Normalized Difference Water Index",
    },
}

# Channel order in .npy files
CHANNEL_ORDER = ["vv", "vh", "dem", "slope", "hand", "twi", "label", "mndwi"]


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_range(data: np.ndarray, rule: Dict) -> Tuple[bool, str]:
    """Check if data falls within valid physical range."""
    valid_data = data[~np.isnan(data) & ~np.isinf(data)]

    if len(valid_data) == 0:
        return False, "All values are NaN or Inf"

    data_min = float(valid_data.min())
    data_max = float(valid_data.max())

    if data_min < rule["min"] - 1e-6:
        return False, f"Min value {data_min:.4f} below allowed minimum {rule['min']}"

    if data_max > rule["max"] + 1e-6:
        return False, f"Max value {data_max:.4f} above allowed maximum {rule['max']}"

    return True, f"Range OK: [{data_min:.2f}, {data_max:.2f}]"


def validate_typical_mean(data: np.ndarray, rule: Dict) -> Tuple[bool, str]:
    """Check if mean is within typical range (warning, not error)."""
    if "typical_mean" not in rule:
        return True, "No typical mean check"

    valid_data = data[~np.isnan(data) & ~np.isinf(data)]
    if len(valid_data) == 0:
        return False, "No valid data for mean check"

    data_mean = float(valid_data.mean())
    low, high = rule["typical_mean"]

    if data_mean < low or data_mean > high:
        return False, f"Mean {data_mean:.2f} outside typical range [{low}, {high}]"

    return True, f"Mean OK: {data_mean:.2f}"


def validate_not_flat(data: np.ndarray, rule: Dict) -> Tuple[bool, str]:
    """Check if data has variance (not suspiciously constant)."""
    if "flat_threshold" not in rule:
        return True, "No flatness check"

    valid_data = data[~np.isnan(data) & ~np.isinf(data)]
    if len(valid_data) == 0:
        return False, "No valid data for flatness check"

    data_std = float(valid_data.std())

    if data_std < rule["flat_threshold"]:
        return (
            False,
            f"Data appears flat (std={data_std:.6f} < {rule['flat_threshold']})",
        )

    return True, f"Variance OK (std={data_std:.4f})"


def validate_slope_corruption(data: np.ndarray, rule: Dict) -> Tuple[bool, str]:
    """Critical check: SLOPE must be in degrees (0-90), not normalized (0-1)."""
    if "corruption_check" not in rule:
        return True, "No corruption check"

    valid_data = data[~np.isnan(data) & ~np.isinf(data)]
    if len(valid_data) == 0:
        return False, "No valid data for corruption check"

    data_max = float(valid_data.max())
    threshold = rule["corruption_check"]["max_threshold"]

    if data_max < threshold:
        return False, rule["corruption_check"]["error_msg"]

    return True, f"SLOPE range OK (max={data_max:.2f} degrees)"


def validate_nan_inf(data: np.ndarray) -> Tuple[bool, str, Dict]:
    """Check for NaN and Inf contamination."""
    total_pixels = data.size
    nan_count = int(np.isnan(data).sum())
    inf_count = int(np.isinf(data).sum())

    nan_fraction = nan_count / total_pixels
    inf_fraction = inf_count / total_pixels

    stats = {
        "nan_count": nan_count,
        "nan_fraction": nan_fraction,
        "inf_count": inf_count,
        "inf_fraction": inf_fraction,
    }

    if nan_fraction > 0.1:  # More than 10% NaN
        return False, f"Too many NaN values: {nan_fraction * 100:.1f}%", stats

    if inf_fraction > 0.01:  # More than 1% Inf
        return False, f"Too many Inf values: {inf_fraction * 100:.1f}%", stats

    return (
        True,
        f"NaN/Inf OK (NaN: {nan_fraction * 100:.2f}%, Inf: {inf_fraction * 100:.2f}%)",
        stats,
    )


def validate_label(data: np.ndarray, rule: Dict) -> Tuple[bool, str, Dict]:
    """Validate ground truth mask."""
    valid_data = data[~np.isnan(data)]

    # Check binary
    unique_values = np.unique(valid_data)
    is_binary = len(unique_values) <= 2 and all(v in [0, 1] for v in unique_values)

    # Check water fraction
    water_fraction = float((valid_data > 0.5).mean())

    stats = {
        "unique_values": len(unique_values),
        "is_binary": is_binary,
        "water_fraction": water_fraction,
        "water_pixels": int((valid_data > 0.5).sum()),
        "total_pixels": int(valid_data.size),
    }

    issues = []

    if not is_binary:
        issues.append(f"Not binary: {len(unique_values)} unique values")

    if water_fraction < rule.get("min_water_fraction", 0):
        issues.append(f"Very low water fraction: {water_fraction * 100:.2f}%")

    if water_fraction > rule.get("max_water_fraction", 1):
        issues.append(f"Very high water fraction: {water_fraction * 100:.2f}%")

    if issues:
        return False, "; ".join(issues), stats

    return True, f"Label OK (water: {water_fraction * 100:.1f}%)", stats


def validate_alignment(data: np.ndarray) -> Tuple[bool, str]:
    """Check if all channels have same shape."""
    if len(data.shape) != 3:
        return False, f"Expected 3D array, got shape {data.shape}"

    h, w, c = data.shape

    if h < 100 or w < 100:
        return False, f"Chip too small: {h}x{w}"

    if h != w:
        return False, f"Non-square chip: {h}x{w}"

    if c < 7:
        return False, f"Missing channels: expected >=7, got {c}"

    return True, f"Shape OK: {h}x{w}x{c}"


# =============================================================================
# MAIN VALIDATOR
# =============================================================================


class ChipValidator:
    """Validates SAR water detection chips for data quality."""

    def __init__(self, strict_mode: bool = True):
        """
        Args:
            strict_mode: If True, treat warnings as errors
        """
        self.strict_mode = strict_mode
        self.rules = VALIDATION_RULES
        self.channel_order = CHANNEL_ORDER

    def validate_chip(self, chip_path: Path) -> Dict:
        """
        Validate a single chip file.

        Returns:
            Dict with validation results
        """
        result = {
            "file": str(chip_path),
            "name": chip_path.stem,
            "valid": True,
            "errors": [],
            "warnings": [],
            "channel_stats": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Load chip
        try:
            data = np.load(chip_path)
        except Exception as e:
            result["valid"] = False
            result["errors"].append(f"Failed to load: {e}")
            return result

        # Check alignment
        ok, msg = validate_alignment(data)
        if not ok:
            result["valid"] = False
            result["errors"].append(msg)
            return result

        h, w, c = data.shape
        result["shape"] = {"height": h, "width": w, "channels": c}

        # Validate each channel
        for i, channel_name in enumerate(self.channel_order):
            if i >= c:
                result["warnings"].append(f"Missing channel: {channel_name}")
                continue

            channel_data = data[:, :, i]
            rule = self.rules.get(channel_name, {})

            channel_result = {
                "index": i,
                "min": float(np.nanmin(channel_data)),
                "max": float(np.nanmax(channel_data)),
                "mean": float(np.nanmean(channel_data)),
                "std": float(np.nanstd(channel_data)),
                "checks": {},
            }

            # Run all checks
            # 1. NaN/Inf check
            ok, msg, stats = validate_nan_inf(channel_data)
            channel_result["checks"]["nan_inf"] = {"passed": ok, "message": msg}
            channel_result.update(stats)
            if not ok:
                result["errors"].append(f"{channel_name}: {msg}")
                result["valid"] = False

            # 2. Range check
            if "min" in rule:
                ok, msg = validate_range(channel_data, rule)
                channel_result["checks"]["range"] = {"passed": ok, "message": msg}
                if not ok:
                    result["errors"].append(f"{channel_name}: {msg}")
                    result["valid"] = False

            # 3. Typical mean check (warning only)
            ok, msg = validate_typical_mean(channel_data, rule)
            channel_result["checks"]["typical_mean"] = {"passed": ok, "message": msg}
            if not ok:
                result["warnings"].append(f"{channel_name}: {msg}")
                if self.strict_mode:
                    result["valid"] = False

            # 4. Flatness check
            ok, msg = validate_not_flat(channel_data, rule)
            channel_result["checks"]["flatness"] = {"passed": ok, "message": msg}
            if not ok:
                result["warnings"].append(f"{channel_name}: {msg}")
                if self.strict_mode:
                    result["valid"] = False

            # 5. SLOPE corruption check (critical!)
            if channel_name == "slope":
                ok, msg = validate_slope_corruption(channel_data, rule)
                channel_result["checks"]["corruption"] = {"passed": ok, "message": msg}
                if not ok:
                    result["errors"].append(f"CRITICAL - {msg}")
                    result["valid"] = False

            # 6. Label validation
            if channel_name == "label":
                ok, msg, stats = validate_label(channel_data, rule)
                channel_result["checks"]["label"] = {"passed": ok, "message": msg}
                channel_result.update(stats)
                if not ok:
                    result["warnings"].append(f"{channel_name}: {msg}")

            result["channel_stats"][channel_name] = channel_result

        return result

    def validate_directory(
        self, input_dir: Path, pattern: str = "*_with_truth.npy"
    ) -> Dict:
        """
        Validate all chips in a directory.

        Returns:
            Dict with summary and per-chip results
        """
        chip_files = sorted(input_dir.glob(pattern))

        summary = {
            "input_dir": str(input_dir),
            "pattern": pattern,
            "total_chips": len(chip_files),
            "valid_chips": 0,
            "invalid_chips": 0,
            "chips_with_warnings": 0,
            "common_errors": defaultdict(int),
            "common_warnings": defaultdict(int),
            "timestamp": datetime.now().isoformat(),
            "strict_mode": self.strict_mode,
        }

        results = []

        logger.info(f"Validating {len(chip_files)} chips in {input_dir}...")

        for chip_path in chip_files:
            result = self.validate_chip(chip_path)
            results.append(result)

            if result["valid"]:
                summary["valid_chips"] += 1
                status = "VALID"
            else:
                summary["invalid_chips"] += 1
                status = "INVALID"

            if result["warnings"]:
                summary["chips_with_warnings"] += 1

            # Track common issues
            for error in result["errors"]:
                # Simplify error message for grouping
                key = error.split(":")[0] if ":" in error else error[:50]
                summary["common_errors"][key] += 1

            for warning in result["warnings"]:
                key = warning.split(":")[0] if ":" in warning else warning[:50]
                summary["common_warnings"][key] += 1

            # Log progress
            error_count = len(result["errors"])
            warn_count = len(result["warnings"])
            logger.info(
                f"  {result['name']}: {status} (errors={error_count}, warnings={warn_count})"
            )

        # Convert defaultdicts to regular dicts for JSON
        summary["common_errors"] = dict(summary["common_errors"])
        summary["common_warnings"] = dict(summary["common_warnings"])

        return {"summary": summary, "results": results}


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Validate SAR water detection chips")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing chip .npy files",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="validation_report.json",
        help="Output JSON report file",
    )
    parser.add_argument(
        "--pattern", type=str, default="*_with_truth.npy", help="File pattern to match"
    )
    parser.add_argument(
        "--strict", action="store_true", help="Treat warnings as errors"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("SAR WATER DETECTION - DATA VALIDATOR")
    logger.info("=" * 70)
    logger.info(f"Input: {input_dir}")
    logger.info(f"Pattern: {args.pattern}")
    logger.info(f"Strict mode: {args.strict}")
    logger.info("=" * 70)

    # Run validation
    validator = ChipValidator(strict_mode=args.strict)
    report = validator.validate_directory(input_dir, args.pattern)

    # Print summary
    summary = report["summary"]
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total chips:        {summary['total_chips']}")
    logger.info(
        f"Valid chips:        {summary['valid_chips']} ({100 * summary['valid_chips'] / max(1, summary['total_chips']):.1f}%)"
    )
    logger.info(f"Invalid chips:      {summary['invalid_chips']}")
    logger.info(f"Chips with warnings: {summary['chips_with_warnings']}")

    if summary["common_errors"]:
        logger.info("\nCommon Errors:")
        for error, count in sorted(
            summary["common_errors"].items(), key=lambda x: -x[1]
        ):
            logger.info(f"  {count:3d}x: {error}")

    if summary["common_warnings"]:
        logger.info("\nCommon Warnings:")
        for warning, count in sorted(
            summary["common_warnings"].items(), key=lambda x: -x[1]
        ):
            logger.info(f"  {count:3d}x: {warning}")

    # Save report
    output_path = Path(args.output_report)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\nReport saved to: {output_path}")
    logger.info("=" * 70)

    # Exit code based on validity
    if summary["invalid_chips"] > 0:
        logger.warning(
            f"\nWARNING: {summary['invalid_chips']} chips failed validation!"
        )
        sys.exit(1)
    else:
        logger.info("\nSUCCESS: All chips passed validation!")
        sys.exit(0)


if __name__ == "__main__":
    main()
