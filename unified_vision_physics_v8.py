#!/usr/bin/env python3
"""
================================================================================
UNIFIED VISION-PHYSICS WATER DETECTOR V8
================================================================================

Combines three components:
1. VISION (U-Net): Spatial pattern recognition, edge detection
2. ML (LightGBM): Feature-based classification with texture/statistics
3. PHYSICS: Hard constraints that are physically impossible to violate

Key Design Principles:
- Physics as VETO, not just penalty
- Vision for edge cases where SAR signature is ambiguous
- ML for bulk classification
- Automatic data quality validation
- Signature-adaptive thresholds

Author: SAR Water Detection Project
Date: 2026-01-25
"""

import numpy as np
from scipy.ndimage import (
    uniform_filter,
    minimum_filter,
    maximum_filter,
    label as scipy_label,
)
from typing import Tuple, Optional, Dict, List
import logging
import json
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WaterType(Enum):
    """Classification of water body types based on SAR signature."""

    DARK_CALM = "dark_calm"  # Typical water, VV < -18dB
    DARK_MODERATE = "dark_moderate"  # -18 to -15 dB
    BRIGHT_WIND = "bright_wind"  # Wind roughened, -15 to -12 dB
    BRIGHT_URBAN = "bright_urban"  # Urban double-bounce, > -12 dB
    UNKNOWN = "unknown"


@dataclass
class DataQualityReport:
    """Report on data quality issues."""

    is_valid: bool
    issues: List[str]
    corrections_applied: Dict[str, str]
    confidence: float  # 0-1, how confident we are in the data


class DataValidator:
    """
    Validates and corrects common data quality issues.
    Prevents training/inference on corrupted data.
    """

    SLOPE_MAX = 90.0  # Physical maximum slope in degrees
    SLOPE_MIN = 0.0
    HAND_MAX = 500.0  # Maximum reasonable HAND value
    HAND_MIN = 0.0
    VV_MAX = 10.0  # Maximum reasonable VV in dB (urban)
    VV_MIN = -40.0  # Minimum reasonable VV in dB
    VH_MAX = 5.0
    VH_MIN = -45.0
    TWI_MAX = 30.0
    TWI_MIN = 0.0

    @classmethod
    def validate_and_correct(
        cls, data: Dict[str, np.ndarray], auto_correct: bool = True
    ) -> Tuple[Dict[str, np.ndarray], DataQualityReport]:
        """
        Validate data and optionally apply corrections.

        Args:
            data: Dict with keys 'vv', 'vh', 'dem', 'slope', 'hand', 'twi'
            auto_correct: Whether to apply corrections automatically

        Returns:
            corrected_data, quality_report
        """
        issues = []
        corrections = {}
        corrected = data.copy()

        # Check SLOPE
        if "slope" in data:
            slope = data["slope"]
            if np.any(slope > cls.SLOPE_MAX):
                issues.append(
                    f"SLOPE_OVERFLOW: max={slope.max():.1f} (limit={cls.SLOPE_MAX})"
                )
                if auto_correct:
                    # Check if it's a units issue (degrees * 10 or radians)
                    if slope.max() > 360:
                        factor = slope.max() / 45  # Assume 45 degrees is typical max
                        corrected["slope"] = np.clip(
                            slope / factor, cls.SLOPE_MIN, cls.SLOPE_MAX
                        )
                        corrections["slope"] = f"divided by {factor:.1f} and clipped"
                    else:
                        corrected["slope"] = np.clip(
                            slope, cls.SLOPE_MIN, cls.SLOPE_MAX
                        )
                        corrections["slope"] = "clipped to 0-90"
            if np.any(slope < cls.SLOPE_MIN):
                issues.append(f"SLOPE_UNDERFLOW: min={slope.min():.1f}")
                if auto_correct:
                    corrected["slope"] = np.maximum(corrected["slope"], cls.SLOPE_MIN)

        # Check HAND
        if "hand" in data:
            hand = data["hand"]
            if np.any(hand > cls.HAND_MAX):
                issues.append(f"HAND_OVERFLOW: max={hand.max():.1f}")
                if auto_correct:
                    corrected["hand"] = np.clip(hand, cls.HAND_MIN, cls.HAND_MAX)
                    corrections["hand"] = "clipped to 0-500"
            if np.any(np.isnan(hand)):
                issues.append("HAND_NAN")
                if auto_correct:
                    corrected["hand"] = np.nan_to_num(
                        hand, nan=100.0
                    )  # High HAND = unlikely water
                    corrections["hand"] = "NaN replaced with 100"

        # Check VV
        if "vv" in data:
            vv = data["vv"]
            if np.any(vv > cls.VV_MAX):
                issues.append(f"VV_HIGH: max={vv.max():.1f}dB (urban/bright)")
                # Don't auto-correct VV as high values might be valid (urban)
            if np.any(vv < cls.VV_MIN):
                issues.append(f"VV_LOW: min={vv.min():.1f}dB")
            if np.any(np.isnan(vv)):
                issues.append("VV_NAN")
                if auto_correct:
                    corrected["vv"] = np.nan_to_num(vv, nan=-20.0)
                    corrections["vv"] = "NaN replaced with -20"

        # Check VH
        if "vh" in data:
            vh = data["vh"]
            if np.any(np.isnan(vh)):
                issues.append("VH_NAN")
                if auto_correct:
                    corrected["vh"] = np.nan_to_num(vh, nan=-25.0)
                    corrections["vh"] = "NaN replaced with -25"

        # Check TWI
        if "twi" in data:
            twi = data["twi"]
            if np.any(np.isnan(twi)):
                issues.append("TWI_NAN")
                if auto_correct:
                    corrected["twi"] = np.nan_to_num(twi, nan=5.0)
                    corrections["twi"] = "NaN replaced with 5"

        # Calculate confidence based on issues
        confidence = 1.0 - min(len(issues) * 0.1, 0.5)  # Max 50% reduction
        if any("OVERFLOW" in i for i in issues):
            confidence -= 0.2

        is_valid = len(issues) == 0 or all(
            k in corrections
            for k in [
                i.split(":")[0].lower().split("_")[0] for i in issues if "NAN" not in i
            ]
        )

        report = DataQualityReport(
            is_valid=is_valid,
            issues=issues,
            corrections_applied=corrections,
            confidence=max(0.3, confidence),
        )

        return corrected, report


class PhysicsConstraints:
    """
    Hard physics constraints that cannot be violated.
    These are VETO rules, not soft penalties.
    """

    @staticmethod
    def compute_hard_constraints(
        hand: np.ndarray, slope: np.ndarray, twi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute physics feasibility mask and confidence.

        Returns:
            impossible_mask: True where water is PHYSICALLY IMPOSSIBLE
            physics_confidence: How confident physics is (0=uncertain, 1=certain)
        """
        # Clip to valid ranges first
        slope = np.clip(slope, 0, 90)
        hand = np.clip(hand, 0, 500)

        # VETO CONDITIONS (water is impossible here)
        impossible = np.zeros_like(hand, dtype=bool)
        confidence = (
            np.ones_like(hand, dtype=np.float32) * 0.5
        )  # Default medium confidence

        # 1. Very high HAND (> 100m above nearest drainage) - water cannot exist
        impossible |= hand > 100
        confidence = np.where(hand > 100, 0.95, confidence)

        # 2. Very steep slopes (> 45°) - water cannot pool
        impossible |= slope > 45
        confidence = np.where(slope > 45, 0.90, confidence)

        # 3. High HAND + steep slope combined (> 30m AND > 20°)
        impossible |= (hand > 30) & (slope > 20)
        confidence = np.where((hand > 30) & (slope > 20), 0.85, confidence)

        # HIGH CONFIDENCE WATER CONDITIONS
        # Low HAND + low slope + high TWI = definitely water-friendly
        water_favorable = (hand < 5) & (slope < 5) & (twi > 10)
        confidence = np.where(water_favorable, 0.90, confidence)

        # Very low HAND = stream/river channel
        confidence = np.where(hand < 2, 0.95, confidence)

        return impossible, confidence

    @staticmethod
    def compute_soft_physics(
        hand: np.ndarray, slope: np.ndarray, twi: np.ndarray
    ) -> np.ndarray:
        """
        Compute soft physics score (0-1) for combining with ML.
        Unlike hard constraints, this allows some water in marginal areas.
        """
        # Clip to valid ranges
        slope = np.clip(slope, 0, 90)
        hand = np.clip(hand, 0, 500)

        # Physics score: 1 = favorable for water, 0 = unfavorable
        hand_score = 1.0 / (1.0 + np.exp((hand - 20) / 10.0))  # 50% at HAND=20m
        slope_score = 1.0 / (1.0 + np.exp((slope - 15) / 5.0))  # 50% at slope=15°
        twi_score = 1.0 / (1.0 + np.exp((6 - twi) / 2.0))  # 50% at TWI=6

        # Weighted combination (HAND most important)
        physics_score = 0.5 * hand_score + 0.3 * slope_score + 0.2 * twi_score

        return physics_score.astype(np.float32)


class SignatureClassifier:
    """
    Classifies SAR signature to select appropriate detection strategy.
    """

    @staticmethod
    def classify(vv: np.ndarray, vh: np.ndarray) -> Tuple[WaterType, Dict]:
        """
        Classify the dominant water signature in the scene.
        """
        vv_mean = np.nanmean(vv)
        vv_std = np.nanstd(vv)
        vh_mean = np.nanmean(vh)

        # Check for bimodality (water + land separation)
        vv_p10 = np.nanpercentile(vv, 10)
        vv_p90 = np.nanpercentile(vv, 90)
        dynamic_range = vv_p90 - vv_p10

        metadata = {
            "vv_mean": float(vv_mean),
            "vv_std": float(vv_std),
            "vh_mean": float(vh_mean),
            "dynamic_range": float(dynamic_range),
        }

        # Classification logic
        if vv_mean < -20:
            water_type = WaterType.DARK_CALM
        elif vv_mean < -15:
            water_type = WaterType.DARK_MODERATE
        elif vv_mean < -10:
            water_type = WaterType.BRIGHT_WIND
        else:
            water_type = WaterType.BRIGHT_URBAN

        metadata["water_type"] = water_type.value

        return water_type, metadata

    @staticmethod
    def get_adaptive_threshold(vv: np.ndarray, water_type: WaterType) -> float:
        """Get adaptive VV threshold based on scene type."""
        if water_type == WaterType.DARK_CALM:
            # Standard dark water
            return -18.0
        elif water_type == WaterType.DARK_MODERATE:
            # Use Otsu or percentile
            dark_pixels = vv[vv < np.percentile(vv, 50)]
            return (
                float(np.percentile(dark_pixels, 80))
                if len(dark_pixels) > 100
                else -16.0
            )
        elif water_type == WaterType.BRIGHT_WIND:
            # Wind roughened - look for relatively dark
            return float(np.percentile(vv, 30))
        else:  # BRIGHT_URBAN
            # Urban - very permissive, let physics constrain
            return float(np.percentile(vv, 40))


class UnifiedVisionPhysicsDetector:
    """
    Main detector combining Vision (U-Net), ML (LightGBM), and Physics.

    Detection Pipeline:
    1. Validate and correct data
    2. Classify SAR signature
    3. Apply physics VETO (hard constraints)
    4. Run ML prediction
    5. (Optional) Refine with U-Net for edges
    6. Apply soft physics weighting
    7. Post-process and output
    """

    def __init__(
        self,
        lgb_model_path: Optional[str] = None,
        unet_model_path: Optional[str] = None,
    ):
        self.lgb_model = None
        self.unet_model = None
        self.validator = DataValidator()
        self.physics = PhysicsConstraints()
        self.classifier = SignatureClassifier()

        if lgb_model_path:
            self._load_lgb(lgb_model_path)
        if unet_model_path:
            self._load_unet(unet_model_path)

    def _load_lgb(self, path: str):
        try:
            import lightgbm as lgb

            self.lgb_model = lgb.Booster(model_file=path)
            logger.info(f"Loaded LightGBM from {path}")
        except Exception as e:
            logger.warning(f"Could not load LightGBM: {e}")

    def _load_unet(self, path: str):
        try:
            import torch

            # Load state dict to get architecture
            checkpoint = torch.load(path, map_location="cpu")
            # Assume CBAM U-Net architecture
            from unet_v6_fixed import CBAMUNet

            self.unet_model = CBAMUNet(in_channels=6, out_channels=1)
            self.unet_model.load_state_dict(checkpoint)
            self.unet_model.eval()
            logger.info(f"Loaded U-Net from {path}")
        except Exception as e:
            logger.warning(f"Could not load U-Net: {e}")

    def detect(
        self,
        vv: np.ndarray,
        vh: np.ndarray,
        dem: np.ndarray,
        slope: np.ndarray,
        hand: np.ndarray,
        twi: np.ndarray,
        use_unet: bool = True,
        auto_correct_data: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Full detection pipeline.

        Args:
            vv, vh: SAR polarizations (dB)
            dem, slope, hand, twi: Terrain features
            use_unet: Whether to use U-Net refinement
            auto_correct_data: Whether to auto-correct data issues

        Returns:
            water_mask: Binary water mask (0-1)
            metadata: Detection statistics and quality info
        """
        metadata = {}

        # Step 1: Validate and correct data
        data = {
            "vv": vv,
            "vh": vh,
            "dem": dem,
            "slope": slope,
            "hand": hand,
            "twi": twi,
        }
        corrected, quality_report = self.validator.validate_and_correct(
            data, auto_correct_data
        )

        metadata["data_quality"] = {
            "is_valid": quality_report.is_valid,
            "issues": quality_report.issues,
            "corrections": quality_report.corrections_applied,
            "confidence": quality_report.confidence,
        }

        if not quality_report.is_valid and not auto_correct_data:
            logger.warning(f"Data quality issues: {quality_report.issues}")
            # Return empty mask if data is invalid and not corrected
            return np.zeros_like(vv), metadata

        # Use corrected data
        vv = corrected["vv"]
        vh = corrected["vh"]
        slope = corrected["slope"]
        hand = corrected["hand"]
        twi = corrected["twi"]

        # Step 2: Classify signature
        water_type, sig_meta = self.classifier.classify(vv, vh)
        metadata["signature"] = sig_meta

        # Step 3: Apply physics VETO
        impossible, physics_conf = self.physics.compute_hard_constraints(
            hand, slope, twi
        )
        metadata["physics_veto_fraction"] = float(impossible.mean())

        # Step 4: ML prediction
        if self.lgb_model is not None:
            ml_proba = self._run_lgb(vv, vh, dem, slope, hand, twi)
            metadata["method"] = "lgb"
        else:
            # Fallback to adaptive thresholding
            threshold = self.classifier.get_adaptive_threshold(vv, water_type)
            ml_proba = 1.0 / (1.0 + np.exp((vv - threshold) / 2.0))
            metadata["method"] = "threshold"
            metadata["threshold"] = threshold

        # Step 5: U-Net refinement (optional)
        if use_unet and self.unet_model is not None:
            unet_proba = self._run_unet(vv, vh, dem, slope, hand, twi)
            # Ensemble: trust U-Net more for edges, ML more for bulk
            edge_mask = self._detect_edges(ml_proba)
            combined_proba = np.where(
                edge_mask,
                0.7 * unet_proba + 0.3 * ml_proba,
                0.3 * unet_proba + 0.7 * ml_proba,
            )
            metadata["method"] += "+unet"
        else:
            combined_proba = ml_proba

        # Step 6: Apply soft physics
        soft_physics = self.physics.compute_soft_physics(hand, slope, twi)
        # Weight: 70% ML/UNet, 30% physics
        final_proba = combined_proba * (0.7 + 0.3 * soft_physics)

        # Apply hard physics VETO
        final_proba = np.where(impossible, 0.0, final_proba)

        # Step 7: Threshold and post-process
        water_mask = final_proba > 0.5
        water_mask = self._remove_small_regions(water_mask, min_size=50)

        metadata["water_fraction"] = float(water_mask.mean())
        metadata["mean_probability"] = float(final_proba.mean())

        return water_mask.astype(np.float32), metadata

    def _run_lgb(self, vv, vh, dem, slope, hand, twi) -> np.ndarray:
        """Run LightGBM prediction."""
        # Feature extraction (must match training)
        from retrain_v7 import extract_features

        features = extract_features(vv, vh, dem, slope, hand, twi)
        h, w, n = features.shape
        X = features.reshape(-1, n)
        proba = self.lgb_model.predict(X).reshape(h, w)
        return proba

    def _run_unet(self, vv, vh, dem, slope, hand, twi) -> np.ndarray:
        """Run U-Net prediction."""
        import torch

        # Prepare input (6 channels: VV, VH, DEM, SLOPE, HAND, TWI)
        h, w = vv.shape

        # Pad to multiple of 16
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16

        inputs = np.stack([vv, vh, dem, slope, hand, twi], axis=0)
        inputs = np.pad(inputs, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")

        # Normalize
        for i in range(6):
            inputs[i] = (inputs[i] - inputs[i].mean()) / (inputs[i].std() + 1e-6)

        # Run inference
        with torch.no_grad():
            x = torch.from_numpy(inputs[np.newaxis]).float()
            out = self.unet_model(x)
            proba = torch.sigmoid(out).numpy()[0, 0]

        # Remove padding
        proba = proba[:h, :w]

        return proba

    def _detect_edges(self, proba: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """Detect edges where U-Net should be trusted more."""
        from scipy.ndimage import sobel

        grad_x = sobel(proba, axis=0)
        grad_y = sobel(proba, axis=1)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        return gradient_mag > threshold

    def _remove_small_regions(self, mask: np.ndarray, min_size: int = 50) -> np.ndarray:
        """Remove small connected components."""
        labeled, num = scipy_label(mask)
        if num == 0:
            return mask

        cleaned = np.zeros_like(mask)
        for i in range(1, num + 1):
            region = labeled == i
            if region.sum() >= min_size:
                cleaned |= region

        return cleaned


def create_full_documentation():
    """Create comprehensive documentation of all versions."""

    doc = """
# SAR Water Detection - Complete Version History

## Version Timeline

| Version | Date | Model | IoU | Key Changes |
|---------|------|-------|-----|-------------|
| v1 | 2026-01-24 | LightGBM baseline | ~0.40 | Simple features |
| v2 | 2026-01-24 | LightGBM pixel | ~0.50 | Per-pixel features |
| v3 | 2026-01-24 | LightGBM SAR-only | ~0.60 | SAR texture features |
| v4 | 2026-01-25 | LightGBM comprehensive | 0.88* | All features, CBAM U-Net |
| v5 | 2026-01-25 | Retrain attempt | - | Blocked by numpy issue |
| v6 | 2026-01-25 | LightGBM retrained | 0.50 | chips/ (corrupted SLOPE) |
| v7 | 2026-01-25 | LightGBM clean | **0.79** | chips_expanded_npy (clean) |
| v8 | 2026-01-25 | Vision+Physics | TBD | Unified hybrid system |

*v4 evaluated on training data

## Key Discoveries

### Data Quality Issues
- 75/86 chips in `chips/` have corrupted SLOPE (values up to 2365°)
- SLOPE should be 0-90° (degrees)
- chips_expanded_npy/ has proper SLOPE (0-90°)

### SAR Signature Variability
- Calm water: VV < -18 dB (dark)
- Wind roughened: -18 to -12 dB
- Urban water: > -12 dB (bright due to double-bounce)

### Physics Constraints
- HARD VETO: HAND > 100m OR slope > 45° = no water
- SOFT: HAND, slope, TWI combined for likelihood

## Files by Version

### v1_baseline/
- master_training_pipeline.py
- lightgbm_baseline.txt

### v2_pixel/
- master_training_pipeline_v2.py
- lightgbm_pixel.txt

### v3_sar_only/
- master_training_pipeline_v3.py
- lightgbm_v3_sar_only.txt

### v4_comprehensive/
- master_training_pipeline_v4.py
- lightgbm_v4_comprehensive.txt
- unet_v4_standalone.py
- ensemble_water_detector.py

### v5_retrain/
- retrain_v5.py
- ensemble_v2.py
- unet_v5_improved.py

### v6_corrupted/
- retrain_v6.py
- lightgbm_v6_retrained.txt
- ensemble_v3.py
- unet_v6_fixed.py

### v7_clean/
- retrain_v7.py
- lightgbm_v7_clean.txt
- All equation search results
- All evaluation results

## Equations Found

### Best Physics-Based Equations

1. **VH Simple**: `water = VH < -23`
   - IoU: ~0.65 on clean data

2. **HAND Constrained**: `water = (VH < -22) AND (HAND < 10)`
   - IoU: ~0.70

3. **Full Physics**: `water = (VH < -21) AND (HAND < 15) AND (slope < 10) AND (TWI > 6)`
   - IoU: ~0.72

4. **Adaptive VH+TWI**: Learned by PySR
   - IoU: ~0.75

### Best ML Models

1. **LightGBM v7**: IoU 0.79 (70 features)
2. **U-Net v4**: IoU ~0.80 (CBAM architecture)

## Data Validation Rules

```python
SLOPE: 0-90° (clip if > 90)
HAND: 0-500m (clip if > 500)
VV: -40 to +10 dB (warn if out of range)
VH: -45 to +5 dB (warn if out of range)
TWI: 0-30 (fill NaN with 5)
```

## Future Improvements

1. Train on combined chips/ + chips_expanded_npy/
2. Add MNDWI as feature (optical index)
3. Separate models for urban vs rural
4. Temporal consistency for flood monitoring
"""

    return doc


if __name__ == "__main__":
    # Create documentation
    doc = create_full_documentation()
    print(doc)
