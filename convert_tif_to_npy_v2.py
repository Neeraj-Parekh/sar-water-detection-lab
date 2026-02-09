#!/usr/bin/env python3
"""
================================================================================
Convert TIF to NPY - chips_expanded conversion
================================================================================
Converts .tif files from chips_expanded/ to .npy format matching chips/ structure.

TIF Structure (8 bands):
  Band 1: VV (dB)
  Band 2: VH (dB)
  Band 3: MNDWI
  Band 4: DEM
  Band 5: HAND
  Band 6: SLOPE
  Band 7: TWI
  Band 8: JRC_water (label)

NPY Structure (8 channels):
  Channel 0: VV
  Channel 1: VH
  Channel 2: DEM
  Channel 3: SLOPE
  Channel 4: HAND
  Channel 5: TWI
  Channel 6: Label (JRC_water > 0)
  Channel 7: MNDWI (optional)
"""

import os
import numpy as np
import rasterio
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def convert_tif_to_npy(input_dir: Path, output_dir: Path, target_size: int = 513):
    """Convert all TIF files to NPY format."""

    output_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(input_dir.glob("*.tif"))
    logger.info(f"Found {len(tif_files)} TIF files in {input_dir}")

    converted = 0
    skipped = 0

    for tif_path in tif_files:
        try:
            with rasterio.open(tif_path) as src:
                if src.count < 8:
                    logger.warning(f"Skipping {tif_path.name}: only {src.count} bands")
                    skipped += 1
                    continue

                # Read all bands
                data = src.read()  # Shape: (bands, height, width)

                # Extract bands (1-indexed in description, 0-indexed in array)
                # Band order: VV, VH, MNDWI, DEM, HAND, SLOPE, TWI, JRC_water
                vv = data[0]  # Band 1: VV
                vh = data[1]  # Band 2: VH
                mndwi = data[2]  # Band 3: MNDWI
                dem = data[3]  # Band 4: DEM
                hand = data[4]  # Band 5: HAND
                slope = data[5]  # Band 6: SLOPE
                twi = data[6]  # Band 7: TWI
                jrc = data[7]  # Band 8: JRC_water (label)

                # Create label (water = 1, non-water = 0)
                label = (jrc > 0).astype(np.float32)

                # Handle NaN values
                vv = np.nan_to_num(vv, nan=-20.0)
                vh = np.nan_to_num(vh, nan=-25.0)
                dem = np.nan_to_num(dem, nan=0.0)
                slope = np.nan_to_num(slope, nan=0.0)
                hand = np.nan_to_num(hand, nan=100.0)  # High HAND = unlikely water
                twi = np.nan_to_num(twi, nan=5.0)
                mndwi = np.nan_to_num(mndwi, nan=0.0)
                label = np.nan_to_num(label, nan=0.0)

                # Clip SLOPE to valid range (0-90)
                slope = np.clip(slope, 0, 90)

                # Clip HAND to reasonable range
                hand = np.clip(hand, 0, 500)

                # Stack in NPY format: (H, W, C)
                # Match chips/ format: VV, VH, DEM, SLOPE, HAND, TWI, Label, MNDWI
                h, w = vv.shape

                # Crop/pad to target size
                if h > target_size:
                    vv = vv[:target_size, :]
                    vh = vh[:target_size, :]
                    dem = dem[:target_size, :]
                    slope = slope[:target_size, :]
                    hand = hand[:target_size, :]
                    twi = twi[:target_size, :]
                    label = label[:target_size, :]
                    mndwi = mndwi[:target_size, :]
                    h = target_size

                if w > target_size:
                    vv = vv[:, :target_size]
                    vh = vh[:, :target_size]
                    dem = dem[:, :target_size]
                    slope = slope[:, :target_size]
                    hand = hand[:, :target_size]
                    twi = twi[:, :target_size]
                    label = label[:, :target_size]
                    mndwi = mndwi[:, :target_size]
                    w = target_size

                # Stack channels
                npy_data = np.stack(
                    [
                        vv.astype(np.float32),
                        vh.astype(np.float32),
                        dem.astype(np.float32),
                        slope.astype(np.float32),
                        hand.astype(np.float32),
                        twi.astype(np.float32),
                        label.astype(np.float32),
                        mndwi.astype(np.float32),
                    ],
                    axis=-1,
                )

                # Output filename
                output_name = tif_path.stem + "_with_truth.npy"
                output_path = output_dir / output_name

                np.save(output_path, npy_data)

                water_frac = label.mean()
                logger.info(
                    f"Converted: {tif_path.name} -> {output_name} ({h}x{w}, water: {water_frac:.1%})"
                )
                converted += 1

        except Exception as e:
            logger.error(f"Error converting {tif_path.name}: {e}")
            skipped += 1

    logger.info(f"\nConversion complete: {converted} converted, {skipped} skipped")
    return converted, skipped


def validate_converted(output_dir: Path):
    """Validate converted NPY files."""
    npy_files = sorted(output_dir.glob("*.npy"))
    logger.info(f"\nValidating {len(npy_files)} NPY files...")

    valid = 0
    issues = []

    for npy_path in npy_files:
        try:
            data = np.load(npy_path)

            if data.ndim != 3 or data.shape[2] < 7:
                issues.append((npy_path.name, f"Bad shape: {data.shape}"))
                continue

            vv, vh = data[:, :, 0], data[:, :, 1]
            slope, hand = data[:, :, 3], data[:, :, 4]
            label = data[:, :, 6]

            chip_issues = []

            if slope.max() > 90:
                chip_issues.append(f"SLOPE={slope.max():.0f}")
            if vv.max() > 10:
                chip_issues.append(f"VV_MAX={vv.max():.1f}")
            if np.isnan(data).any():
                chip_issues.append("HAS_NAN")

            if chip_issues:
                issues.append((npy_path.name, ", ".join(chip_issues)))
            else:
                valid += 1

        except Exception as e:
            issues.append((npy_path.name, str(e)))

    logger.info(f"Valid: {valid}, Issues: {len(issues)}")
    for name, issue in issues[:10]:
        logger.warning(f"  {name}: {issue}")

    return valid, issues


if __name__ == "__main__":
    input_dir = Path("/home/mit-aoe/sar_water_detection/chips_expanded")
    output_dir = Path("/home/mit-aoe/sar_water_detection/chips_expanded_npy")

    convert_tif_to_npy(input_dir, output_dir)
    validate_converted(output_dir)
