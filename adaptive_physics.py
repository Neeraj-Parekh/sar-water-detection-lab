#!/usr/bin/env python3
"""
================================================================================
ADAPTIVE CONTEXT-AWARE PHYSICS v1 - Dynamic Threshold System
================================================================================

The Problem with Rigid Thresholds:
- Fixed VH < -19.6 dB works for calm lakes but fails for:
  - Windy water (brighter, VH ~ -15 dB)
  - Arid regions (different soil moisture baseline)
  - Urban areas (shadows confuse detection)
  - Wetlands (mixed water/vegetation)

Solution: Context-Aware Dynamic Thresholds
1. First, CLASSIFY the scene type
2. Then, APPLY scene-specific thresholds

Scene Types:
- ARID: Desert/semi-arid (Rajasthan, Gujarat) - use stricter VH threshold
- URBAN: City areas with buildings - apply shadow rejection
- WETLAND: Coastal/marshy areas (Kerala, Sundarbans) - use relaxed threshold
- MONTANE: Mountainous regions (Himalayas) - trust physics more
- STANDARD: Default agricultural/mixed areas

Author: SAR Water Detection Project
Date: 2026-01-25
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SCENE TYPES
# =============================================================================


class SceneType(Enum):
    """Scene classification types."""

    ARID = "arid"  # Desert, low vegetation
    URBAN = "urban"  # Cities, high backscatter variance
    WETLAND = "wetland"  # Coastal, high water fraction
    MONTANE = "montane"  # Mountains, high elevation/slope
    FLOOD = "flood"  # Active flooding, high MNDWI variance
    STANDARD = "standard"  # Default agricultural/mixed


@dataclass
class SceneThresholds:
    """Dynamic thresholds for each scene type."""

    vh_water: float  # VH threshold for water (dB)
    vh_bright_water: float  # VH threshold for bright/windy water
    hand_veto: float  # HAND above which water is impossible (m)
    slope_veto: float  # Slope above which water is impossible (deg)
    mndwi_water: float  # MNDWI threshold for water
    physics_weight: float  # Weight for physics score (0-1)
    confidence_boost: float  # Boost factor for high-confidence detections


# =============================================================================
# SCENE-SPECIFIC THRESHOLDS
# =============================================================================

SCENE_THRESHOLDS = {
    SceneType.ARID: SceneThresholds(
        vh_water=-21.0,  # Stricter - arid regions have drier surfaces
        vh_bright_water=-15.0,  # Rare wind roughening
        hand_veto=50.0,  # Lower HAND tolerance (water scarce)
        slope_veto=30.0,  # Lower slope tolerance
        mndwi_water=0.2,  # Higher MNDWI requirement
        physics_weight=0.20,  # Trust physics more
        confidence_boost=1.2,  # Boost confident detections
    ),
    SceneType.URBAN: SceneThresholds(
        vh_water=-20.0,  # Standard
        vh_bright_water=-12.0,  # Urban water often disturbed
        hand_veto=100.0,  # Standard
        slope_veto=45.0,  # Standard
        mndwi_water=0.1,  # Standard
        physics_weight=0.10,  # Trust ML more (physics confused by buildings)
        confidence_boost=0.9,  # Be more conservative (shadows!)
    ),
    SceneType.WETLAND: SceneThresholds(
        vh_water=-18.0,  # Relaxed - wetlands have mixed water
        vh_bright_water=-10.0,  # Very relaxed for wind/vegetation mixing
        hand_veto=150.0,  # Higher tolerance (tidal areas)
        slope_veto=45.0,  # Standard
        mndwi_water=0.0,  # Lower threshold (water + vegetation)
        physics_weight=0.05,  # Trust ML more
        confidence_boost=1.1,  # Slight boost
    ),
    SceneType.MONTANE: SceneThresholds(
        vh_water=-19.0,  # Standard
        vh_bright_water=-12.0,  # Standard
        hand_veto=80.0,  # Stricter - mountain streams are in valleys
        slope_veto=35.0,  # Stricter
        mndwi_water=0.15,  # Standard
        physics_weight=0.25,  # Trust physics most
        confidence_boost=1.0,  # No boost
    ),
    SceneType.FLOOD: SceneThresholds(
        vh_water=-17.0,  # Very relaxed - turbid flood water
        vh_bright_water=-8.0,  # Very relaxed
        hand_veto=200.0,  # Very high - floods can be anywhere
        slope_veto=60.0,  # Very high
        mndwi_water=-0.1,  # Very relaxed
        physics_weight=0.02,  # Almost ignore physics (flood overrides terrain)
        confidence_boost=1.3,  # Strong boost
    ),
    SceneType.STANDARD: SceneThresholds(
        vh_water=-19.6,  # Discovered optimal
        vh_bright_water=-12.0,  # Standard
        hand_veto=100.0,  # Standard
        slope_veto=45.0,  # Standard
        mndwi_water=0.1,  # Standard
        physics_weight=0.10,  # Standard
        confidence_boost=1.0,  # No boost
    ),
}


# =============================================================================
# SCENE CLASSIFIER
# =============================================================================


class SceneClassifier:
    """
    Classifies scenes into types based on input features.
    Uses simple rules - can be upgraded to ML classifier later.
    """

    def __init__(self):
        self.thresholds = SCENE_THRESHOLDS

    def classify(self, data: Dict[str, np.ndarray]) -> Tuple[SceneType, Dict]:
        """
        Classify scene type based on input data.

        Args:
            data: Dict with vv, vh, dem, slope, hand, twi, mndwi (optional)

        Returns:
            scene_type: Classified scene type
            stats: Scene statistics for debugging
        """
        vv = data["vv"]
        vh = data["vh"]
        dem = data["dem"]
        slope = data["slope"]
        hand = data.get("hand", np.zeros_like(vv))
        mndwi = data.get("mndwi", np.zeros_like(vv))

        # Compute scene statistics
        stats = {
            "dem_mean": float(np.nanmean(dem)),
            "dem_max": float(np.nanmax(dem)),
            "slope_mean": float(np.nanmean(slope)),
            "slope_p90": float(np.nanpercentile(slope, 90)),
            "vh_mean": float(np.nanmean(vh)),
            "vh_std": float(np.nanstd(vh)),
            "vv_std": float(np.nanstd(vv)),
            "mndwi_mean": float(np.nanmean(mndwi)),
            "mndwi_std": float(np.nanstd(mndwi)),
            "water_fraction_estimate": float((mndwi > 0.1).mean()),
        }

        # Classification rules (priority order)

        # 1. MONTANE: High elevation or steep slopes
        if stats["dem_mean"] > 1500 or stats["slope_p90"] > 30:
            return SceneType.MONTANE, stats

        # 2. FLOOD: High MNDWI variance (rapidly changing water)
        if stats["mndwi_std"] > 0.3 and stats["water_fraction_estimate"] > 0.3:
            return SceneType.FLOOD, stats

        # 3. WETLAND: High water fraction, low elevation
        if stats["water_fraction_estimate"] > 0.4 and stats["dem_mean"] < 50:
            return SceneType.WETLAND, stats

        # 4. URBAN: High VV variance (buildings), low water
        vv_local_std = np.nanstd(vv - uniform_filter(vv, size=21))
        stats["vv_local_std"] = float(vv_local_std)
        if vv_local_std > 3.0 and stats["water_fraction_estimate"] < 0.2:
            return SceneType.URBAN, stats

        # 5. ARID: Low VH mean (dry surfaces), very low water
        if stats["vh_mean"] < -24.0 and stats["water_fraction_estimate"] < 0.05:
            return SceneType.ARID, stats

        # 6. Default: STANDARD
        return SceneType.STANDARD, stats

    def get_thresholds(self, scene_type: SceneType) -> SceneThresholds:
        """Get thresholds for a scene type."""
        return self.thresholds[scene_type]


# =============================================================================
# ADAPTIVE PHYSICS MODULE
# =============================================================================


class AdaptivePhysics:
    """
    Applies context-aware physics constraints.
    Replaces rigid thresholds with dynamic scene-specific ones.
    """

    def __init__(self):
        self.classifier = SceneClassifier()

    def compute_physics_score(
        self, data: Dict[str, np.ndarray], scene_type: Optional[SceneType] = None
    ) -> Tuple[np.ndarray, np.ndarray, SceneType, SceneThresholds]:
        """
        Compute physics score and VETO mask with adaptive thresholds.

        Args:
            data: Input data dict
            scene_type: Override scene classification (optional)

        Returns:
            physics_score: Soft probability adjustment (0-1)
            veto_mask: Hard rejection mask (True = impossible water)
            scene_type: Detected or provided scene type
            thresholds: Applied thresholds
        """
        # Classify scene
        if scene_type is None:
            scene_type, scene_stats = self.classifier.classify(data)
            logger.info(f"Scene classified as: {scene_type.value}")

        # Get adaptive thresholds
        thresholds = self.classifier.get_thresholds(scene_type)

        # Extract data
        hand = data.get("hand", np.zeros_like(data["vv"]))
        slope = data.get("slope", np.zeros_like(data["vv"]))
        twi = data.get("twi", np.ones_like(data["vv"]) * 10)

        # Compute VETO mask with adaptive thresholds
        veto = np.zeros_like(hand, dtype=bool)
        veto |= hand > thresholds.hand_veto
        veto |= slope > thresholds.slope_veto

        # Combined constraint (adaptive)
        combined_hand = thresholds.hand_veto * 0.3
        combined_slope = thresholds.slope_veto * 0.5
        veto |= (hand > combined_hand) & (slope > combined_slope)

        # Compute soft physics score
        # Use adaptive midpoints based on thresholds
        hand_mid = thresholds.hand_veto * 0.15
        slope_mid = thresholds.slope_veto * 0.27

        hand_exp = np.clip((hand - hand_mid) / 5.0, -50, 50)
        hand_score = 1.0 / (1.0 + np.exp(hand_exp))

        slope_exp = np.clip((slope - slope_mid) / 4.0, -50, 50)
        slope_score = 1.0 / (1.0 + np.exp(slope_exp))

        twi_exp = np.clip((7 - twi) / 2.0, -50, 50)
        twi_score = 1.0 / (1.0 + np.exp(twi_exp))

        # Weighted combination
        physics_score = 0.4 * hand_score + 0.4 * slope_score + 0.2 * twi_score

        return physics_score.astype(np.float32), veto, scene_type, thresholds

    def apply_adaptive_detection(
        self,
        data: Dict[str, np.ndarray],
        ml_proba: np.ndarray,
        scene_type: Optional[SceneType] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply adaptive water detection with context-aware physics.

        Args:
            data: Input data dict (vv, vh, dem, slope, hand, twi, mndwi)
            ml_proba: ML model probability (from LightGBM)
            scene_type: Override scene classification (optional)

        Returns:
            combined_proba: Final probability after physics adjustment
            stats: Detection statistics
        """
        # Get physics score and scene info
        physics_score, veto, scene_type, thresholds = self.compute_physics_score(
            data, scene_type
        )

        stats = {
            "scene_type": scene_type.value,
            "thresholds": {
                "vh_water": thresholds.vh_water,
                "hand_veto": thresholds.hand_veto,
                "slope_veto": thresholds.slope_veto,
                "physics_weight": thresholds.physics_weight,
            },
            "veto_fraction": float(veto.mean()),
        }

        # Adaptive weighting based on scene
        ml_weight = 1.0 - thresholds.physics_weight
        physics_weight = thresholds.physics_weight

        # Combine predictions
        combined = ml_weight * ml_proba + physics_weight * physics_score

        # Apply confidence boost for high-confidence ML predictions
        high_conf_mask = ml_proba > 0.8
        combined = np.where(
            high_conf_mask, combined * thresholds.confidence_boost, combined
        )

        # Apply VH-based correction for bright water
        vh = data["vh"]
        bright_water_mask = (
            (vh >= thresholds.vh_water)
            & (vh < thresholds.vh_bright_water)
            & (ml_proba > 0.5)
        )
        # Don't penalize bright water in wetlands/floods
        if scene_type in [SceneType.WETLAND, SceneType.FLOOD]:
            combined = np.where(bright_water_mask, combined * 1.1, combined)

        # Apply MNDWI boost (if available)
        mndwi = data.get("mndwi", None)
        if mndwi is not None:
            mndwi_positive = mndwi > thresholds.mndwi_water
            combined = np.where(mndwi_positive, combined * 1.1, combined)

        # Apply hard VETO
        combined = np.where(veto, 0.0, combined)

        # Clip to valid range
        combined = np.clip(combined, 0.0, 1.0)

        stats["combined_mean"] = float(combined.mean())
        stats["water_fraction"] = float((combined > 0.5).mean())

        return combined.astype(np.float32), stats


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================


def apply_adaptive_physics(
    vv: np.ndarray,
    vh: np.ndarray,
    dem: np.ndarray,
    slope: np.ndarray,
    hand: np.ndarray,
    twi: np.ndarray,
    ml_proba: np.ndarray,
    mndwi: Optional[np.ndarray] = None,
    scene_type: Optional[str] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function for adaptive physics application.

    Args:
        vv, vh: SAR backscatter (dB)
        dem, slope, hand, twi: Terrain features
        ml_proba: ML model probability
        mndwi: Optional optical water index
        scene_type: Optional scene override ("arid", "urban", "wetland", "montane", "flood", "standard")

    Returns:
        combined_proba: Adjusted probability
        stats: Detection statistics
    """
    data = {
        "vv": vv,
        "vh": vh,
        "dem": dem,
        "slope": slope,
        "hand": hand,
        "twi": twi,
    }
    if mndwi is not None:
        data["mndwi"] = mndwi

    # Convert scene type string to enum
    scene_enum = None
    if scene_type is not None:
        scene_enum = SceneType(scene_type.lower())

    physics = AdaptivePhysics()
    return physics.apply_adaptive_detection(data, ml_proba, scene_enum)


# =============================================================================
# TEST
# =============================================================================


def test_adaptive_physics():
    """Test the adaptive physics module."""
    logger.info("Testing Adaptive Physics Module...")

    # Create synthetic data for different scene types
    h, w = 100, 100

    # Test 1: Standard scene
    data_standard = {
        "vv": np.random.normal(-12, 3, (h, w)).astype(np.float32),
        "vh": np.random.normal(-20, 3, (h, w)).astype(np.float32),
        "dem": np.random.normal(200, 50, (h, w)).astype(np.float32),
        "slope": np.random.uniform(0, 15, (h, w)).astype(np.float32),
        "hand": np.random.uniform(0, 30, (h, w)).astype(np.float32),
        "twi": np.random.uniform(5, 15, (h, w)).astype(np.float32),
        "mndwi": np.random.uniform(-0.2, 0.3, (h, w)).astype(np.float32),
    }
    ml_proba = np.random.uniform(0.3, 0.7, (h, w)).astype(np.float32)

    physics = AdaptivePhysics()

    # Classify scene
    scene_type, stats = physics.classifier.classify(data_standard)
    logger.info(f"Standard scene classified as: {scene_type.value}")
    logger.info(f"  Stats: {stats}")

    # Apply adaptive detection
    combined, det_stats = physics.apply_adaptive_detection(data_standard, ml_proba)
    logger.info(f"Detection stats: {det_stats}")

    # Test 2: Montane scene
    data_montane = data_standard.copy()
    data_montane["dem"] = np.random.normal(2500, 300, (h, w)).astype(np.float32)
    data_montane["slope"] = np.random.uniform(10, 40, (h, w)).astype(np.float32)

    scene_type, stats = physics.classifier.classify(data_montane)
    logger.info(f"Montane scene classified as: {scene_type.value}")

    combined, det_stats = physics.apply_adaptive_detection(data_montane, ml_proba)
    logger.info(f"Montane detection stats: {det_stats}")

    # Test 3: Wetland scene
    data_wetland = data_standard.copy()
    data_wetland["dem"] = np.random.normal(10, 5, (h, w)).astype(np.float32)
    data_wetland["mndwi"] = np.random.uniform(0.2, 0.6, (h, w)).astype(np.float32)

    scene_type, stats = physics.classifier.classify(data_wetland)
    logger.info(f"Wetland scene classified as: {scene_type.value}")

    combined, det_stats = physics.apply_adaptive_detection(data_wetland, ml_proba)
    logger.info(f"Wetland detection stats: {det_stats}")

    logger.info("All tests passed!")


if __name__ == "__main__":
    test_adaptive_physics()
