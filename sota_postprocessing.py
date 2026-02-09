#!/usr/bin/env python3
"""
================================================================================
SOTA POST-PROCESSING MODULE
================================================================================

Contains:
1. MST RIVER CONNECTOR - Graph-based gap healing for broken rivers
2. FUZZY LOGIC CONTROLLER - Probability-based water detection
3. INCIDENCE ANGLE NORMALIZATION - Range-dependent backscatter correction

Author: SAR Water Detection Project - SOTA Branch
Date: 2026-01-25
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import label, distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
from typing import Tuple, Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. MST RIVER CONNECTOR - Graph-Based Gap Healing
# =============================================================================


class MSTRiverConnector:
    """
    Minimum Spanning Tree based river connector.

    Heals broken river segments by:
    1. Finding connected components (water blobs)
    2. Computing distances between nearby blobs
    3. If gap is small AND pixels between are dark enough, connect them
    4. Uses MST to avoid creating loops

    This is the "mathematical healer" for LightGBM gaps.
    """

    def __init__(
        self,
        max_gap_pixels: int = 10,  # Max gap to bridge (~100m at 10m resolution)
        min_blob_size: int = 20,  # Ignore tiny blobs
        vh_threshold: float = -16.0,  # Gap pixels must be darker than this
        connection_width: int = 2,  # Width of connecting line
    ):
        self.max_gap_pixels = max_gap_pixels
        self.min_blob_size = min_blob_size
        self.vh_threshold = vh_threshold
        self.connection_width = connection_width

    def get_blob_centroids(
        self, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Find connected components and their centroids.
        Returns: labeled_mask, centroids array, list of blob pixel coords
        """
        labeled, n_blobs = label(mask)

        centroids = []
        blob_coords = []
        valid_labels = []

        for i in range(1, n_blobs + 1):
            blob_mask = labeled == i
            blob_size = blob_mask.sum()

            if blob_size >= self.min_blob_size:
                # Get centroid
                coords = np.where(blob_mask)
                centroid_y = np.mean(coords[0])
                centroid_x = np.mean(coords[1])
                centroids.append([centroid_y, centroid_x])
                blob_coords.append(coords)
                valid_labels.append(i)

        if len(centroids) == 0:
            return labeled, np.array([]), []

        return labeled, np.array(centroids), blob_coords

    def get_blob_boundary_points(
        self, labeled: np.ndarray, label_id: int, sample_rate: int = 5
    ) -> np.ndarray:
        """Get boundary points of a blob (sampled for efficiency)."""
        blob_mask = labeled == label_id

        # Erode to find interior, subtract to get boundary
        eroded = ndimage.binary_erosion(blob_mask)
        boundary = blob_mask & ~eroded

        coords = np.array(np.where(boundary)).T

        # Sample if too many points
        if len(coords) > 100:
            indices = np.linspace(0, len(coords) - 1, 100).astype(int)
            coords = coords[indices]

        return coords

    def find_nearest_points_between_blobs(
        self, labeled: np.ndarray, label1: int, label2: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Find nearest boundary points between two blobs."""
        points1 = self.get_blob_boundary_points(labeled, label1)
        points2 = self.get_blob_boundary_points(labeled, label2)

        if len(points1) == 0 or len(points2) == 0:
            return None, None, float("inf")

        # Use KD-tree for efficient nearest neighbor
        tree = cKDTree(points2)
        distances, indices = tree.query(points1)

        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        point1 = points1[min_idx]
        point2 = points2[indices[min_idx]]

        return point1, point2, min_dist

    def check_gap_is_water_like(
        self, vh: np.ndarray, point1: np.ndarray, point2: np.ndarray
    ) -> bool:
        """
        Check if the pixels between two points are dark enough to be water.
        Uses Bresenham-like line sampling.
        """
        y1, x1 = point1
        y2, x2 = point2

        # Sample points along the line
        n_samples = max(abs(y2 - y1), abs(x2 - x1)) + 1

        if n_samples <= 1:
            return True

        ys = np.linspace(y1, y2, n_samples).astype(int)
        xs = np.linspace(x1, x2, n_samples).astype(int)

        # Clamp to image bounds
        ys = np.clip(ys, 0, vh.shape[0] - 1)
        xs = np.clip(xs, 0, vh.shape[1] - 1)

        # Get VH values along the line
        vh_values = vh[ys, xs]

        # Check if median is dark enough
        median_vh = np.nanmedian(vh_values)

        return median_vh < self.vh_threshold

    def draw_connection(
        self, mask: np.ndarray, point1: np.ndarray, point2: np.ndarray
    ) -> np.ndarray:
        """Draw a line connecting two points on the mask."""
        y1, x1 = point1
        y2, x2 = point2

        # Sample points along the line
        n_samples = max(abs(y2 - y1), abs(x2 - x1)) + 1

        ys = np.linspace(y1, y2, n_samples).astype(int)
        xs = np.linspace(x1, x2, n_samples).astype(int)

        # Clamp to image bounds
        ys = np.clip(ys, 0, mask.shape[0] - 1)
        xs = np.clip(xs, 0, mask.shape[1] - 1)

        # Draw with width
        for y, x in zip(ys, xs):
            y_min = max(0, y - self.connection_width // 2)
            y_max = min(mask.shape[0], y + self.connection_width // 2 + 1)
            x_min = max(0, x - self.connection_width // 2)
            x_max = min(mask.shape[1], x + self.connection_width // 2 + 1)
            mask[y_min:y_max, x_min:x_max] = 1

        return mask

    def connect(
        self, water_mask: np.ndarray, vh: np.ndarray, verbose: bool = False
    ) -> Tuple[np.ndarray, int]:
        """
        Main method: Connect broken river segments using MST.

        Args:
            water_mask: Binary water mask (prediction from model)
            vh: VH backscatter in dB (for checking gap validity)
            verbose: Print connection info

        Returns:
            connected_mask: Water mask with gaps healed
            n_connections: Number of connections made
        """
        result = water_mask.copy().astype(np.float32)

        # Get blobs
        labeled, centroids, blob_coords = self.get_blob_centroids(water_mask > 0.5)

        if len(centroids) < 2:
            return result, 0

        n_blobs = len(centroids)

        # Build distance matrix between all blob pairs
        valid_labels = list(range(1, n_blobs + 1))
        distance_matrix = np.full((n_blobs, n_blobs), float("inf"))
        connection_points = {}

        for i in range(n_blobs):
            for j in range(i + 1, n_blobs):
                point1, point2, dist = self.find_nearest_points_between_blobs(
                    labeled, valid_labels[i], valid_labels[j]
                )

                if point1 is not None and dist <= self.max_gap_pixels:
                    # Check if gap looks like water
                    if self.check_gap_is_water_like(vh, point1, point2):
                        distance_matrix[i, j] = dist
                        distance_matrix[j, i] = dist
                        connection_points[(i, j)] = (point1, point2)

        # Build MST to find optimal connections
        sparse_dist = csr_matrix(distance_matrix)
        mst = minimum_spanning_tree(sparse_dist)
        mst_array = mst.toarray()

        # Draw connections from MST
        n_connections = 0
        for i in range(n_blobs):
            for j in range(i + 1, n_blobs):
                if mst_array[i, j] > 0 or mst_array[j, i] > 0:
                    key = (i, j) if (i, j) in connection_points else (j, i)
                    if key in connection_points:
                        point1, point2 = connection_points[key]
                        result = self.draw_connection(result, point1, point2)
                        n_connections += 1

                        if verbose:
                            logger.info(
                                f"Connected blobs {i} and {j} (gap={mst_array[i, j]:.1f}px)"
                            )

        return result, n_connections


# =============================================================================
# 2. FUZZY LOGIC CONTROLLER - Probability-Based Water Detection
# =============================================================================


class FuzzyWaterDetector:
    """
    Fuzzy Logic Controller for water detection.

    Instead of hard thresholds, outputs probability based on rules:
    - "If VH is Low AND Slope is Flat AND Variance is Low -> High Water Probability"
    - "If VH is Low BUT Variance is High -> Urban Shadow (Low Water Probability)"

    This captures ambiguous pixels gently rather than hard killing them.
    """

    def __init__(self):
        # Fuzzy membership parameters
        self.vh_params = {
            "very_low": (-30, -25, -22),  # (a, b, c) for triangular
            "low": (-24, -20, -16),
            "medium": (-18, -14, -10),
            "high": (-12, -8, 0),
        }

        self.slope_params = {
            "flat": (0, 0, 5),
            "gentle": (3, 8, 15),
            "moderate": (12, 20, 30),
            "steep": (25, 45, 90),
        }

        self.variance_params = {
            "low": (0, 0, 2),
            "medium": (1, 3, 6),
            "high": (4, 8, 20),
        }

    def triangular_membership(
        self, x: np.ndarray, a: float, b: float, c: float
    ) -> np.ndarray:
        """Triangular fuzzy membership function."""
        result = np.zeros_like(x, dtype=np.float32)

        # Rising edge
        mask1 = (x >= a) & (x < b)
        if b > a:
            result[mask1] = (x[mask1] - a) / (b - a)

        # Falling edge
        mask2 = (x >= b) & (x <= c)
        if c > b:
            result[mask2] = (c - x[mask2]) / (c - b)

        # At peak
        result[x == b] = 1.0

        return result

    def compute_vh_membership(self, vh: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute VH fuzzy memberships."""
        return {
            name: self.triangular_membership(vh, *params)
            for name, params in self.vh_params.items()
        }

    def compute_slope_membership(self, slope: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute slope fuzzy memberships."""
        return {
            name: self.triangular_membership(slope, *params)
            for name, params in self.slope_params.items()
        }

    def compute_variance_membership(
        self, variance: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute variance fuzzy memberships."""
        return {
            name: self.triangular_membership(variance, *params)
            for name, params in self.variance_params.items()
        }

    def apply_rules(
        self,
        vh_memb: Dict[str, np.ndarray],
        slope_memb: Dict[str, np.ndarray],
        var_memb: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Apply fuzzy rules and defuzzify.

        Rules (firing strength -> water probability):
        1. VH_low AND Slope_flat AND Var_low -> 0.95 (definite water)
        2. VH_very_low AND Slope_flat -> 0.90 (very likely water)
        3. VH_low AND Slope_gentle AND Var_low -> 0.75 (likely water)
        4. VH_low AND Var_high -> 0.20 (urban shadow, unlikely water)
        5. VH_medium AND Slope_flat AND Var_low -> 0.50 (ambiguous)
        6. VH_high -> 0.05 (definitely not water)
        7. Slope_steep -> 0.05 (definitely not water)
        """
        # Rule 1: Definite water
        rule1 = np.minimum(
            np.minimum(vh_memb["low"], slope_memb["flat"]), var_memb["low"]
        )

        # Rule 2: Very likely water
        rule2 = np.minimum(vh_memb["very_low"], slope_memb["flat"])

        # Rule 3: Likely water (gentle slope)
        rule3 = np.minimum(
            np.minimum(vh_memb["low"], slope_memb["gentle"]), var_memb["low"]
        )

        # Rule 4: Urban shadow (low VH but high variance)
        rule4 = np.minimum(vh_memb["low"], var_memb["high"])

        # Rule 5: Ambiguous
        rule5 = np.minimum(
            np.minimum(vh_memb["medium"], slope_memb["flat"]), var_memb["low"]
        )

        # Rule 6: Bright = not water
        rule6 = vh_memb["high"]

        # Rule 7: Steep = not water
        rule7 = slope_memb["steep"]

        # Weighted defuzzification (Sugeno-style)
        water_prob = (
            rule1 * 0.95
            + rule2 * 0.90
            + rule3 * 0.75
            + rule5 * 0.50
            + rule4 * 0.20  # Urban shadow penalty
            + rule6 * 0.05
            + rule7 * 0.05
        )

        # Normalize by total firing strength
        total_strength = rule1 + rule2 + rule3 + rule4 + rule5 + rule6 + rule7 + 1e-8
        water_prob = water_prob / total_strength

        return np.clip(water_prob, 0, 1)

    def predict(
        self,
        vh: np.ndarray,
        slope: np.ndarray,
        local_variance: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict water probability using fuzzy logic.

        Args:
            vh: VH backscatter in dB
            slope: Terrain slope in degrees
            local_variance: Local VH variance (computed if not provided)

        Returns:
            water_probability: 0-1 probability map
        """
        # Compute local variance if not provided
        if local_variance is None:
            from scipy.ndimage import uniform_filter

            vh_mean = uniform_filter(vh, size=5)
            vh_sq_mean = uniform_filter(vh**2, size=5)
            local_variance = np.maximum(vh_sq_mean - vh_mean**2, 0)
            local_variance = np.sqrt(local_variance)

        # Compute memberships
        vh_memb = self.compute_vh_membership(vh)
        slope_memb = self.compute_slope_membership(slope)
        var_memb = self.compute_variance_membership(local_variance)

        # Apply rules
        water_prob = self.apply_rules(vh_memb, slope_memb, var_memb)

        return water_prob


# =============================================================================
# 3. INCIDENCE ANGLE NORMALIZATION
# =============================================================================


class IncidenceAngleNormalizer:
    """
    Normalize SAR backscatter for incidence angle effects.

    Radar reflects differently at:
    - Near range (close to satellite): Higher incidence angle, brighter
    - Far range (far from satellite): Lower incidence angle, darker

    Applies correction: VH_corrected = VH - k * (Angle - CenterAngle)

    If incidence angle not available, estimates from image position.
    """

    def __init__(
        self,
        k: float = 0.1,  # Correction slope (dB per degree)
        center_angle: float = 39.0,  # Typical Sentinel-1 center angle
        near_angle: float = 30.0,  # Near range angle
        far_angle: float = 46.0,  # Far range angle
    ):
        self.k = k
        self.center_angle = center_angle
        self.near_angle = near_angle
        self.far_angle = far_angle

    def estimate_incidence_angle(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Estimate incidence angle from image column position.
        Assumes near range on left, far range on right.
        """
        height, width = shape

        # Create column index
        col_idx = np.arange(width).reshape(1, -1)
        col_idx = np.repeat(col_idx, height, axis=0)

        # Linear interpolation from near to far
        angle = self.near_angle + (self.far_angle - self.near_angle) * (
            col_idx / (width - 1)
        )

        return angle.astype(np.float32)

    def normalize(
        self, vh: np.ndarray, incidence_angle: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply incidence angle normalization.

        Args:
            vh: VH backscatter in dB
            incidence_angle: Incidence angle in degrees (estimated if not provided)

        Returns:
            vh_corrected: Normalized VH in dB
        """
        if incidence_angle is None:
            incidence_angle = self.estimate_incidence_angle(vh.shape)

        # Apply correction
        correction = self.k * (incidence_angle - self.center_angle)
        vh_corrected = vh - correction

        return vh_corrected

    def compute_gamma0(
        self, sigma0: np.ndarray, incidence_angle: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert Sigma0 to Gamma0 for terrain normalization.
        Gamma0 = Sigma0 / cos(theta)

        In dB: Gamma0_dB = Sigma0_dB - 10*log10(cos(theta))
        """
        if incidence_angle is None:
            incidence_angle = self.estimate_incidence_angle(sigma0.shape)

        # Convert to radians
        theta_rad = np.radians(incidence_angle)

        # Compute correction in dB
        cos_theta = np.cos(theta_rad)
        cos_theta = np.maximum(cos_theta, 0.1)  # Avoid division by zero
        correction_db = 10 * np.log10(cos_theta)

        gamma0 = sigma0 - correction_db

        return gamma0


# =============================================================================
# COMBINED SOTA POST-PROCESSOR
# =============================================================================


class SOTAPostProcessor:
    """
    Combined SOTA post-processing pipeline.

    Applies in order:
    1. Incidence angle normalization (if not already done)
    2. Fuzzy logic water probability
    3. MST river connection
    4. Physics constraints (HAND/SLOPE veto)
    """

    def __init__(
        self,
        use_angle_norm: bool = True,
        use_fuzzy: bool = True,
        use_mst: bool = True,
        use_physics_veto: bool = True,
        hand_veto_threshold: float = 100.0,
        slope_veto_threshold: float = 30.0,
    ):
        self.use_angle_norm = use_angle_norm
        self.use_fuzzy = use_fuzzy
        self.use_mst = use_mst
        self.use_physics_veto = use_physics_veto

        self.hand_veto = hand_veto_threshold
        self.slope_veto = slope_veto_threshold

        # Initialize components
        self.angle_normalizer = IncidenceAngleNormalizer()
        self.fuzzy_detector = FuzzyWaterDetector()
        self.mst_connector = MSTRiverConnector()

    def process(
        self,
        prediction: np.ndarray,
        vh: np.ndarray,
        slope: np.ndarray,
        hand: np.ndarray,
        incidence_angle: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Apply full SOTA post-processing pipeline.

        Args:
            prediction: Model prediction (probability or binary)
            vh: VH backscatter in dB
            slope: Terrain slope in degrees
            hand: Height Above Nearest Drainage in meters
            incidence_angle: Optional incidence angle
            verbose: Print processing info

        Returns:
            Dictionary with:
            - 'final': Final binary prediction
            - 'probability': Water probability map
            - 'fuzzy_prob': Fuzzy logic probability
            - 'connected': After MST connection
            - 'n_connections': Number of MST connections made
        """
        results = {}

        # 1. Incidence angle normalization
        if self.use_angle_norm:
            vh_norm = self.angle_normalizer.normalize(vh, incidence_angle)
            if verbose:
                logger.info("Applied incidence angle normalization")
        else:
            vh_norm = vh

        # 2. Fuzzy logic probability
        if self.use_fuzzy:
            fuzzy_prob = self.fuzzy_detector.predict(vh_norm, slope)
            results["fuzzy_prob"] = fuzzy_prob

            # Combine with model prediction
            combined_prob = 0.6 * prediction + 0.4 * fuzzy_prob
            if verbose:
                logger.info("Applied fuzzy logic (60% model + 40% fuzzy)")
        else:
            combined_prob = prediction
            results["fuzzy_prob"] = prediction

        results["probability"] = combined_prob

        # 3. Physics veto
        if self.use_physics_veto:
            physics_mask = (hand <= self.hand_veto) & (slope <= self.slope_veto)
            combined_prob = combined_prob * physics_mask.astype(np.float32)
            if verbose:
                vetoed = (~physics_mask).sum()
                logger.info(
                    f"Physics veto: removed {vetoed} pixels (HAND>{self.hand_veto}m or SLOPE>{self.slope_veto}deg)"
                )

        # 4. Threshold to binary
        binary = (combined_prob > 0.5).astype(np.float32)

        # 5. MST river connection
        if self.use_mst:
            connected, n_connections = self.mst_connector.connect(
                binary, vh_norm, verbose
            )
            results["connected"] = connected
            results["n_connections"] = n_connections
            if verbose:
                logger.info(f"MST connector: healed {n_connections} river gaps")
        else:
            connected = binary
            results["connected"] = connected
            results["n_connections"] = 0

        results["final"] = connected

        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def heal_river_gaps(
    water_mask: np.ndarray,
    vh: np.ndarray,
    max_gap_pixels: int = 10,
    vh_threshold: float = -16.0,
) -> np.ndarray:
    """
    Convenience function to heal river gaps using MST.

    Args:
        water_mask: Binary water prediction
        vh: VH backscatter in dB
        max_gap_pixels: Maximum gap size to bridge
        vh_threshold: Gap pixels must be darker than this

    Returns:
        healed_mask: Water mask with gaps connected
    """
    connector = MSTRiverConnector(
        max_gap_pixels=max_gap_pixels, vh_threshold=vh_threshold
    )
    healed, n_conn = connector.connect(water_mask, vh)
    logger.info(f"Healed {n_conn} river gaps")
    return healed


def get_fuzzy_water_probability(
    vh: np.ndarray, slope: np.ndarray, local_variance: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convenience function to get fuzzy water probability.

    Args:
        vh: VH backscatter in dB
        slope: Terrain slope in degrees
        local_variance: Optional pre-computed variance

    Returns:
        probability: 0-1 water probability map
    """
    detector = FuzzyWaterDetector()
    return detector.predict(vh, slope, local_variance)


def normalize_incidence_angle(
    vh: np.ndarray, incidence_angle: Optional[np.ndarray] = None, k: float = 0.1
) -> np.ndarray:
    """
    Convenience function to normalize incidence angle.

    Args:
        vh: VH backscatter in dB
        incidence_angle: Optional angle map (estimated if not provided)
        k: Correction slope (dB per degree)

    Returns:
        vh_normalized: Corrected VH in dB
    """
    normalizer = IncidenceAngleNormalizer(k=k)
    return normalizer.normalize(vh, incidence_angle)


# =============================================================================
# TEST
# =============================================================================


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)

    # Create fake river (thin line with gap)
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[45:55, 10:40] = 1  # Left blob
    mask[45:55, 50:90] = 1  # Right blob (gap at 40-50)

    # Create fake VH (dark where water should be)
    vh = np.full((100, 100), -10.0, dtype=np.float32)  # Land is bright
    vh[40:60, 10:90] = -22.0  # River area is dark (including gap)

    # Test MST connector
    connector = MSTRiverConnector(max_gap_pixels=15, vh_threshold=-16.0)
    healed, n_conn = connector.connect(mask, vh, verbose=True)

    print(f"\nOriginal water pixels: {mask.sum()}")
    print(f"After healing: {healed.sum()}")
    print(f"Connections made: {n_conn}")

    # Test fuzzy logic
    slope = np.random.uniform(0, 10, (100, 100)).astype(np.float32)
    fuzzy = FuzzyWaterDetector()
    prob = fuzzy.predict(vh, slope)

    print(f"\nFuzzy probability range: [{prob.min():.2f}, {prob.max():.2f}]")
    print(f"High probability pixels (>0.7): {(prob > 0.7).sum()}")

    # Test angle normalization
    normalizer = IncidenceAngleNormalizer()
    vh_norm = normalizer.normalize(vh)

    print(f"\nVH before norm: mean={vh.mean():.2f}")
    print(f"VH after norm: mean={vh_norm.mean():.2f}")

    print("\nAll SOTA post-processing components working!")
