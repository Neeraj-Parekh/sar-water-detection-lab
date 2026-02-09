#!/usr/bin/env python3
"""
Adaptive Water Detection System
================================
Physics-guided water detection with terrain-adaptive thresholds.

Features:
1. Context Detection - Classify scene type
2. Bright Water Handler - Handle wind-roughened water
3. Urban Mask - Exclude urban false positives
4. Terrain-Adaptive Physics Safety Net
5. Ensemble-ready architecture

Author: SAR Water Detection Lab
Date: 2026-01-25
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from scipy.ndimage import uniform_filter, gaussian_filter, label
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# Terrain Profiles - Adaptive Physics Constraints
# =============================================================================


@dataclass
class TerrainProfile:
    """Terrain-specific thresholds and constraints."""

    name: str
    description: str

    # SAR thresholds
    vh_threshold: float
    vh_bright: float  # Relaxed for windy water
    vv_threshold: float

    # Physics constraints
    hand_max: float
    hand_strict: float  # For bright water
    slope_max: float
    slope_strict: float  # For bright water
    twi_min: float

    # Urban detection
    vv_urban_min: float
    vv_vh_ratio_urban: float

    # Confidence adjustment
    confidence_base: float = 0.9


# Terrain-specific profiles
TERRAIN_PROFILES = {
    "flat_lowland": TerrainProfile(
        name="flat_lowland",
        description="River plains, coastal areas, flood zones",
        vh_threshold=-18.0,
        vh_bright=-12.0,
        vv_threshold=-14.0,
        hand_max=15.0,
        hand_strict=5.0,
        slope_max=10.0,
        slope_strict=3.0,
        twi_min=7.0,
        vv_urban_min=-10.0,
        vv_vh_ratio_urban=8.0,
        confidence_base=0.95,
    ),
    "hilly": TerrainProfile(
        name="hilly",
        description="Moderate elevation, valleys, gorges",
        vh_threshold=-17.0,
        vh_bright=-13.0,
        vv_threshold=-13.0,
        hand_max=25.0,
        hand_strict=8.0,
        slope_max=20.0,
        slope_strict=8.0,
        twi_min=6.0,
        vv_urban_min=-10.0,
        vv_vh_ratio_urban=8.0,
        confidence_base=0.90,
    ),
    "mountainous": TerrainProfile(
        name="mountainous",
        description="High altitude lakes, steep terrain",
        vh_threshold=-16.0,
        vh_bright=-14.0,
        vv_threshold=-12.0,
        hand_max=100.0,  # Relaxed - mountain lakes have high HAND
        hand_strict=50.0,
        slope_max=30.0,
        slope_strict=15.0,
        twi_min=4.0,
        vv_urban_min=-8.0,
        vv_vh_ratio_urban=10.0,
        confidence_base=0.85,
    ),
    "arid": TerrainProfile(
        name="arid",
        description="Desert, salt flats, ephemeral water",
        vh_threshold=-20.0,  # Stricter
        vh_bright=-16.0,
        vv_threshold=-16.0,
        hand_max=8.0,
        hand_strict=3.0,
        slope_max=5.0,
        slope_strict=2.0,
        twi_min=5.0,
        vv_urban_min=-12.0,
        vv_vh_ratio_urban=10.0,
        confidence_base=0.85,
    ),
    "urban": TerrainProfile(
        name="urban",
        description="Cities with rivers/lakes",
        vh_threshold=-20.0,  # Stricter to avoid urban FP
        vh_bright=-16.0,
        vv_threshold=-16.0,
        hand_max=8.0,
        hand_strict=3.0,
        slope_max=10.0,
        slope_strict=5.0,
        twi_min=6.0,
        vv_urban_min=-12.0,
        vv_vh_ratio_urban=6.0,  # Lower threshold
        confidence_base=0.85,
    ),
    "wetland": TerrainProfile(
        name="wetland",
        description="Marshes, seasonal flooding, vegetated water",
        vh_threshold=-14.0,  # Very relaxed for vegetation
        vh_bright=-10.0,
        vv_threshold=-10.0,
        hand_max=20.0,
        hand_strict=10.0,
        slope_max=8.0,
        slope_strict=4.0,
        twi_min=10.0,  # High TWI required
        vv_urban_min=-10.0,
        vv_vh_ratio_urban=8.0,
        confidence_base=0.90,
    ),
    "coastal": TerrainProfile(
        name="coastal",
        description="Mangroves, estuaries, tidal zones",
        vh_threshold=-16.0,
        vh_bright=-12.0,
        vv_threshold=-12.0,
        hand_max=10.0,
        hand_strict=5.0,
        slope_max=8.0,
        slope_strict=3.0,
        twi_min=8.0,
        vv_urban_min=-10.0,
        vv_vh_ratio_urban=8.0,
        confidence_base=0.90,
    ),
}


# =============================================================================
# Context Detection
# =============================================================================


class ContextDetector:
    """Detect scene context for adaptive thresholding."""

    def __init__(self):
        self.known_locations = {
            # Urban
            "mumbai": "urban",
            "delhi": "urban",
            "bangalore": "urban",
            "chennai": "urban",
            "kolkata": "urban",
            "hyderabad": "urban",
            "ahmedabad": "urban",
            "pune": "urban",
            "jaipur": "urban",
            # Coastal
            "sundarbans": "coastal",
            "bhitarkanika": "coastal",
            "pulicat": "coastal",
            "pichavaram": "coastal",
            "coringa": "coastal",
            "mangrove": "coastal",
            # Wetland
            "keoladeo": "wetland",
            "kolleru": "wetland",
            "harike": "wetland",
            "loktak": "wetland",
            "deepor": "wetland",
            "chilika": "wetland",
            "vembanad": "wetland",
            "kuttanad": "wetland",
            # Mountain
            "pangong": "mountainous",
            "tso_moriri": "mountainous",
            "dal_lake": "mountainous",
            "wular": "mountainous",
            "bhakra": "mountainous",
            "nainital": "mountainous",
            # Arid
            "rann": "arid",
            "kutch": "arid",
            "sambhar": "arid",
            "thar": "arid",
            # Rivers (default to flat_lowland)
            "brahmaputra": "flat_lowland",
            "ganga": "flat_lowland",
            "godavari": "flat_lowland",
            "yamuna": "hilly",
            "narmada": "hilly",
            "chambal": "hilly",
        }

    def detect_from_name(self, chip_name: str) -> Optional[str]:
        """Try to detect terrain from chip name."""
        chip_lower = chip_name.lower()
        for location, terrain in self.known_locations.items():
            if location in chip_lower:
                return terrain
        return None

    def detect_from_features(self, features: Dict[str, np.ndarray]) -> str:
        """Classify terrain based on feature statistics."""
        vv = features.get("vv", np.zeros((1, 1)))
        vh = features.get("vh", np.zeros((1, 1)))
        dem = features.get("dem", np.zeros((1, 1)))
        slope = features.get("slope", np.zeros((1, 1)))
        hand = features.get("hand", np.zeros((1, 1)))
        twi = features.get("twi", np.zeros((1, 1)))

        # Compute statistics (handling NaN)
        dem_mean = np.nanmean(dem)
        slope_mean = np.nanmean(slope)
        slope_p90 = (
            np.nanpercentile(slope[~np.isnan(slope)], 90)
            if np.any(~np.isnan(slope))
            else 0
        )
        hand_p90 = (
            np.nanpercentile(hand[~np.isnan(hand)], 90)
            if np.any(~np.isnan(hand))
            else 0
        )
        twi_mean = np.nanmean(twi)
        vv_mean = np.nanmean(vv)
        vh_mean = np.nanmean(vh)
        vv_vh_diff = vv_mean - vh_mean

        # Decision tree classification
        if dem_mean > 2000:
            return "mountainous"
        elif slope_mean > 12 or slope_p90 > 25:
            if dem_mean > 500:
                return "mountainous"
            else:
                return "hilly"
        elif vv_mean > -12 and vv_vh_diff > 7:
            return "urban"
        elif twi_mean > 10 and hand_p90 < 15:
            return "wetland"
        elif twi_mean < 5 and vh_mean < -22:
            return "arid"
        elif hand_p90 < 8 and slope_mean < 3:
            return "coastal"
        else:
            return "flat_lowland"

    def detect(
        self, features: Dict[str, np.ndarray], chip_name: str = ""
    ) -> Tuple[str, TerrainProfile]:
        """Detect terrain and return profile."""
        # Try name-based first
        terrain = self.detect_from_name(chip_name)

        # Fall back to feature-based
        if terrain is None:
            terrain = self.detect_from_features(features)

        profile = TERRAIN_PROFILES.get(terrain, TERRAIN_PROFILES["flat_lowland"])
        return terrain, profile


# =============================================================================
# Bright Water Handler
# =============================================================================


class BrightWaterHandler:
    """
    Handle wind-roughened water with VH > -16 dB.

    Strategy: Use relaxed VH thresholds with strict physics constraints.

    UPDATED with equation search v3 findings:
    - Hysteresis approach (IoU=0.475): VH_low=-24, VH_high=-18, HAND=5m
    - This two-tier approach is the best pure physics equation
    """

    def __init__(self):
        self.texture_window = 9

        # Hysteresis thresholds from equation search v3 (best IoU=0.4745)
        self.vh_core = -24.0  # Core water threshold (very confident)
        self.vh_extended = -18.0  # Extended water threshold (less confident)
        self.hand_extended = 5.0  # Strict HAND for extended detection

    def compute_texture(self, vh: np.ndarray) -> np.ndarray:
        """Compute local texture (coefficient of variation)."""
        vh_safe = np.nan_to_num(vh, nan=-25)
        local_mean = uniform_filter(vh_safe, size=self.texture_window)
        local_sq_mean = uniform_filter(vh_safe**2, size=self.texture_window)
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
        texture = local_std / (np.abs(local_mean) + 1e-10)
        return texture

    def detect_hysteresis_water(
        self,
        vh: np.ndarray,
        hand: np.ndarray,
        slope: np.ndarray,
        profile: TerrainProfile,
    ) -> np.ndarray:
        """
        Hysteresis approach: Two-tier VH thresholds.

        Core water (VH < -24): High confidence, relaxed HAND
        Extended water (VH < -18): Lower confidence, strict HAND

        This was the best physics equation (IoU=0.475, Physics=0.976).
        """
        # Core water: Very dark, relaxed physics
        core_water = (
            (vh < self.vh_core)
            & (hand < profile.hand_max)
            & (slope < profile.slope_max)
        )

        # Extended water: Moderately dark, strict physics
        extended_water = (
            (vh >= self.vh_core)
            & (vh < self.vh_extended)
            & (hand < self.hand_extended)  # Strict HAND
            & (slope < profile.slope_strict)  # Strict slope
        )

        # Assign confidence levels
        prob = np.zeros_like(vh, dtype=np.float32)
        prob = np.where(core_water, profile.confidence_base, prob)
        prob = np.where(extended_water, profile.confidence_base * 0.8, prob)

        return prob

    def detect_bright_water(
        self,
        vv: np.ndarray,
        vh: np.ndarray,
        hand: np.ndarray,
        slope: np.ndarray,
        twi: np.ndarray,
        profile: TerrainProfile,
    ) -> np.ndarray:
        """
        Detect bright water using compensating physics constraints.

        Returns probability map [0, 1].
        """
        # Use hysteresis approach as base (best performing)
        prob = self.detect_hysteresis_water(vh, hand, slope, profile)

        # Bright water (very relaxed VH, very strict physics)
        # From equation search: VH=-14, HAND=4, slope=4 (IoU=0.336)
        bright_water = (
            (vh >= self.vh_extended)
            & (vh < profile.vh_bright)  # Not too bright (-14 to -18)
            & (hand < 4.0)  # Very strict HAND from search results
            & (slope < 4.0)  # Very strict slope
            & (twi > profile.twi_min)  # High wetness
        )
        bright_confidence = profile.confidence_base * 0.7

        # Compute texture for additional filtering
        texture = self.compute_texture(vh)
        texture_filter = texture < 0.5  # Water should be smooth

        # Add bright water (lower confidence)
        prob = np.where(
            bright_water & texture_filter, np.maximum(prob, bright_confidence), prob
        )

        return prob

    def get_bright_water_mask(
        self, vh: np.ndarray, profile: TerrainProfile
    ) -> np.ndarray:
        """Get mask of potentially bright water regions."""
        return (vh >= profile.vh_threshold) & (vh < profile.vh_bright)


# =============================================================================
# Urban Mask
# =============================================================================


class UrbanMaskDetector:
    """
    Detect and mask urban areas to prevent false positives.

    Urban signature: High VV (double-bounce), Low VH, High VV/VH ratio.
    """

    def __init__(self):
        self.window_size = 15  # For local statistics

    def compute_urban_probability(
        self, vv: np.ndarray, vh: np.ndarray, profile: TerrainProfile
    ) -> np.ndarray:
        """
        Compute probability of urban area.

        Returns probability [0, 1] where 1 = definitely urban.
        """
        # VV/VH difference (in dB, this is log ratio)
        vv_vh_diff = vv - vh

        # Urban indicators
        high_vv = vv > profile.vv_urban_min
        high_ratio = vv_vh_diff > profile.vv_vh_ratio_urban
        low_vh = vh < -18  # Urban has low VH

        # Compute local texture (urban has moderate texture)
        vv_safe = np.nan_to_num(vv, nan=-20)
        local_mean = uniform_filter(vv_safe, size=self.window_size)
        local_sq_mean = uniform_filter(vv_safe**2, size=self.window_size)
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean**2, 0))
        texture = local_std / (np.abs(local_mean) + 1e-10)
        has_texture = texture > 0.15

        # Combine: Urban needs high VV, high ratio, and some texture
        urban_score = (
            high_vv.astype(float) * 0.3
            + high_ratio.astype(float) * 0.4
            + low_vh.astype(float) * 0.15
            + has_texture.astype(float) * 0.15
        )

        return urban_score

    def get_urban_mask(
        self,
        vv: np.ndarray,
        vh: np.ndarray,
        profile: TerrainProfile,
        threshold: float = 0.6,
    ) -> np.ndarray:
        """Get binary urban mask."""
        urban_prob = self.compute_urban_probability(vv, vh, profile)
        return urban_prob > threshold

    def apply_urban_exclusion(
        self,
        water_prob: np.ndarray,
        vv: np.ndarray,
        vh: np.ndarray,
        profile: TerrainProfile,
    ) -> np.ndarray:
        """Remove urban false positives from water predictions."""
        urban_mask = self.get_urban_mask(vv, vh, profile)

        # Zero out water probability in urban areas
        water_prob = np.where(urban_mask, 0.0, water_prob)

        return water_prob


# =============================================================================
# Terrain-Adaptive Physics Safety Net
# =============================================================================


class PhysicsSafetyNet:
    """
    Apply terrain-adaptive physics constraints.

    Key insight: Constraints should vary by terrain type.
    - Mountain lakes can have high HAND
    - Wetlands can have relaxed VH
    - Arid regions need strict constraints
    """

    def __init__(self, soft_constraints: bool = True):
        self.soft_constraints = soft_constraints

    def apply_constraints(
        self,
        water_prob: np.ndarray,
        features: Dict[str, np.ndarray],
        profile: TerrainProfile,
    ) -> np.ndarray:
        """Apply terrain-adaptive physics constraints."""
        hand = features.get("hand", np.zeros_like(water_prob))
        slope = features.get("slope", np.zeros_like(water_prob))
        dem = features.get("dem", np.zeros_like(water_prob))

        if self.soft_constraints:
            return self._apply_soft_constraints(water_prob, hand, slope, dem, profile)
        else:
            return self._apply_hard_constraints(water_prob, hand, slope, dem, profile)

    def _apply_soft_constraints(
        self,
        water_prob: np.ndarray,
        hand: np.ndarray,
        slope: np.ndarray,
        dem: np.ndarray,
        profile: TerrainProfile,
    ) -> np.ndarray:
        """Apply soft (probabilistic) constraints."""
        result = water_prob.copy()

        # HAND penalty (sigmoid transition)
        hand_penalty = 1.0 / (1.0 + np.exp((hand - profile.hand_max) / 3.0))
        result = result * hand_penalty

        # Slope penalty (linear transition)
        slope_penalty = np.clip(1.0 - (slope - profile.slope_max) / 10.0, 0.2, 1.0)
        result = result * slope_penalty

        # Extreme slope hard cutoff (water cannot exist on cliffs)
        result = np.where(slope > 35, 0.0, result)

        # Mountain lake exception: if terrain is mountainous and DEM > 2000m,
        # relax HAND constraint
        if profile.name == "mountainous":
            high_altitude = dem > 2000
            result = np.where(
                high_altitude & (slope < 15) & (water_prob > 0.5),
                water_prob * 0.9,
                result,
            )

        return result

    def _apply_hard_constraints(
        self,
        water_prob: np.ndarray,
        hand: np.ndarray,
        slope: np.ndarray,
        dem: np.ndarray,
        profile: TerrainProfile,
    ) -> np.ndarray:
        """Apply hard (binary) constraints."""
        result = water_prob.copy()

        # HAND constraint
        result = np.where(hand > profile.hand_max * 1.5, 0.0, result)

        # Slope constraint
        result = np.where(slope > profile.slope_max * 1.5, 0.0, result)

        return result


# =============================================================================
# Wet Soil Discriminator
# =============================================================================


class WetSoilDiscriminator:
    """
    Distinguish wet soil from water.

    Without temporal data: Use conservative approach in ambiguous zones.
    With temporal data: Check persistence.
    """

    def __init__(self):
        self.vh_wet_soil_range = (-22, -14)  # Ambiguous range

    def get_ambiguous_zone(
        self, vh: np.ndarray, hand: np.ndarray, slope: np.ndarray
    ) -> np.ndarray:
        """Identify zones likely to be wet soil (ambiguous)."""
        vh_ambiguous = (vh > self.vh_wet_soil_range[0]) & (
            vh < self.vh_wet_soil_range[1]
        )
        low_hand = hand < 15
        moderate_slope = (slope > 1) & (slope < 12)

        return vh_ambiguous & low_hand & moderate_slope

    def adjust_confidence(
        self,
        water_prob: np.ndarray,
        vh: np.ndarray,
        hand: np.ndarray,
        slope: np.ndarray,
    ) -> np.ndarray:
        """Reduce confidence in ambiguous wet soil zones."""
        ambiguous = self.get_ambiguous_zone(vh, hand, slope)

        # Reduce confidence by 30% in ambiguous zones
        result = np.where(ambiguous, water_prob * 0.7, water_prob)

        return result

    def temporal_discrimination(
        self, vh_t1: np.ndarray, vh_t2: np.ndarray, days_apart: int = 7
    ) -> np.ndarray:
        """
        Use temporal change to distinguish water from wet soil.

        Wet soil drying: VH increases by ~3-5 dB over a week
        Water: VH stable or varies with wind (both directions)
        """
        vh_change = vh_t2 - vh_t1

        # Wet soil signature: VH increases (getting brighter as it dries)
        wet_soil_prob = np.clip((vh_change - 2) / 5, 0, 1)

        return wet_soil_prob


# =============================================================================
# Main Adaptive Water Detector
# =============================================================================


class AdaptiveWaterDetector:
    """
    Main class combining all components.

    Pipeline:
    1. Detect terrain context
    2. Apply context-specific thresholds
    3. Handle bright water
    4. Apply urban mask
    5. Apply physics safety net
    6. Adjust for wet soil ambiguity
    """

    def __init__(self, use_soft_physics: bool = True):
        self.context_detector = ContextDetector()
        self.bright_water_handler = BrightWaterHandler()
        self.urban_mask_detector = UrbanMaskDetector()
        self.physics_safety_net = PhysicsSafetyNet(soft_constraints=use_soft_physics)
        self.wet_soil_discriminator = WetSoilDiscriminator()

    def detect(
        self,
        features: Dict[str, np.ndarray],
        chip_name: str = "",
        return_debug: bool = False,
    ) -> np.ndarray:
        """
        Full adaptive water detection pipeline.

        Args:
            features: Dictionary with 'vv', 'vh', 'hand', 'slope', 'twi', 'dem'
            chip_name: Optional chip name for location-based context
            return_debug: If True, return debug information

        Returns:
            Water probability map [0, 1]
        """
        # Get required features
        vv = features.get("vv", np.zeros((256, 256)))
        vh = features.get("vh", np.zeros((256, 256)))
        hand = features.get("hand", np.zeros_like(vv))
        slope = features.get("slope", np.zeros_like(vv))
        twi = features.get("twi", np.zeros_like(vv))
        dem = features.get("dem", np.zeros_like(vv))

        # Handle NaN
        vv = np.nan_to_num(vv, nan=-20)
        vh = np.nan_to_num(vh, nan=-25)
        hand = np.nan_to_num(hand, nan=50)
        slope = np.nan_to_num(slope, nan=30)
        twi = np.nan_to_num(twi, nan=5)
        dem = np.nan_to_num(dem, nan=100)

        # Step 1: Detect terrain context
        terrain, profile = self.context_detector.detect(features, chip_name)

        # Step 2: Bright water detection (includes dark water)
        water_prob = self.bright_water_handler.detect_bright_water(
            vv, vh, hand, slope, twi, profile
        )

        # Step 3: Apply urban mask
        water_prob = self.urban_mask_detector.apply_urban_exclusion(
            water_prob, vv, vh, profile
        )

        # Step 4: Apply physics safety net
        water_prob = self.physics_safety_net.apply_constraints(
            water_prob, features, profile
        )

        # Step 5: Adjust for wet soil ambiguity
        water_prob = self.wet_soil_discriminator.adjust_confidence(
            water_prob, vh, hand, slope
        )

        if return_debug:
            debug_info = {
                "terrain": terrain,
                "profile": profile,
                "bright_water_mask": self.bright_water_handler.get_bright_water_mask(
                    vh, profile
                ),
                "urban_mask": self.urban_mask_detector.get_urban_mask(vv, vh, profile),
                "ambiguous_zone": self.wet_soil_discriminator.get_ambiguous_zone(
                    vh, hand, slope
                ),
            }
            return water_prob, debug_info

        return water_prob

    def detect_binary(
        self,
        features: Dict[str, np.ndarray],
        chip_name: str = "",
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Get binary water mask."""
        prob = self.detect(features, chip_name)
        return prob > threshold


# =============================================================================
# Ensemble-Ready Interface
# =============================================================================


class EnsembleWaterDetector:
    """
    Wrapper for ensemble predictions.

    Combines:
    - Adaptive rule-based detection
    - LightGBM predictions
    - U-Net predictions
    """

    def __init__(self, weights: Dict[str, float] = None):
        self.adaptive_detector = AdaptiveWaterDetector()

        # Default weights
        self.weights = weights or {"adaptive": 0.2, "lightgbm": 0.5, "unet": 0.3}

    def predict(
        self,
        features: Dict[str, np.ndarray],
        lightgbm_pred: Optional[np.ndarray] = None,
        unet_pred: Optional[np.ndarray] = None,
        chip_name: str = "",
    ) -> np.ndarray:
        """
        Ensemble prediction.

        Args:
            features: Input features
            lightgbm_pred: LightGBM probability prediction (optional)
            unet_pred: U-Net probability prediction (optional)
            chip_name: Chip name for context

        Returns:
            Combined water probability
        """
        # Get adaptive prediction
        adaptive_prob = self.adaptive_detector.detect(features, chip_name)

        # Normalize weights based on available predictions
        active_weights = {"adaptive": self.weights["adaptive"]}
        if lightgbm_pred is not None:
            active_weights["lightgbm"] = self.weights["lightgbm"]
        if unet_pred is not None:
            active_weights["unet"] = self.weights["unet"]

        # Normalize
        total_weight = sum(active_weights.values())
        for k in active_weights:
            active_weights[k] /= total_weight

        # Combine
        combined = adaptive_prob * active_weights["adaptive"]

        if lightgbm_pred is not None:
            combined += lightgbm_pred * active_weights["lightgbm"]

        if unet_pred is not None:
            combined += unet_pred * active_weights["unet"]

        # Apply physics safety net to combined result
        terrain, profile = self.adaptive_detector.context_detector.detect(
            features, chip_name
        )
        combined = self.adaptive_detector.physics_safety_net.apply_constraints(
            combined, features, profile
        )

        return combined


# =============================================================================
# Testing
# =============================================================================


def test_adaptive_detector():
    """Test the adaptive water detector."""
    print("=" * 60)
    print("Testing Adaptive Water Detector")
    print("=" * 60)

    # Create synthetic test data
    np.random.seed(42)
    h, w = 256, 256

    # Create terrain (low in center, high at edges)
    y, x = np.mgrid[0:h, 0:w]
    dem = 100 + 50 * ((x - w / 2) ** 2 + (y - h / 2) ** 2) / (w * h / 4)
    dem = dem.astype(np.float32)

    # Create water body in center
    water_mask = ((x - w / 2) ** 2 + (y - h / 2) ** 2) < (w / 4) ** 2

    # SAR values
    vv = np.where(
        water_mask, -20 + np.random.randn(h, w) * 2, -10 + np.random.randn(h, w) * 3
    )
    vh = np.where(
        water_mask, -25 + np.random.randn(h, w) * 2, -18 + np.random.randn(h, w) * 3
    )

    # Terrain features
    slope = np.gradient(dem)[0] ** 2 + np.gradient(dem)[1] ** 2
    slope = np.sqrt(slope) * 10
    hand = np.where(water_mask, 2, 20)
    twi = np.where(water_mask, 12, 6)

    features = {
        "vv": vv.astype(np.float32),
        "vh": vh.astype(np.float32),
        "dem": dem.astype(np.float32),
        "slope": slope.astype(np.float32),
        "hand": hand.astype(np.float32),
        "twi": twi.astype(np.float32),
    }

    # Test detector
    detector = AdaptiveWaterDetector()
    water_prob, debug = detector.detect(features, "test_lake", return_debug=True)

    print(f"Terrain detected: {debug['terrain']}")
    print(f"Water prob range: [{water_prob.min():.3f}, {water_prob.max():.3f}]")
    print(f"Water prob mean: {water_prob.mean():.3f}")
    print(f"True water fraction: {water_mask.mean():.3f}")
    print(f"Predicted water fraction (>0.5): {(water_prob > 0.5).mean():.3f}")

    # Compute IoU
    pred_binary = water_prob > 0.5
    intersection = (pred_binary & water_mask).sum()
    union = (pred_binary | water_mask).sum()
    iou = intersection / (union + 1e-10)
    print(f"IoU: {iou:.4f}")

    print("\nTest completed successfully!")
    return True


if __name__ == "__main__":
    test_adaptive_detector()
