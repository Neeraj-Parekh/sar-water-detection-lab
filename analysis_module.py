"""
SAR Water Detection - Advanced Analysis Module
===============================================

Additional analysis features:
- Stability Heatmap (threshold perturbation)
- Confidence Map (definite/probable/possible)
- HAND Depth Contours (h1/h2/h3/h4)
- Pixel Statistics
- Best Band Selector (Fisher Distance)
"""

import numpy as np
from scipy.ndimage import uniform_filter, binary_dilation

from filter_engine_complete import (
    numpy_otsu
)
import re

# =============================================================================
# SMART TEXT PARSER (NLP)
# =============================================================================

def parse_smart_text(text):
    """
    Parse natural language text to extract filter configurations.
    
    Supports formats like:
    - "refined_lee(window=7)"
    - "T_low = -23.48"
    - "k_cfar(pfa=1e-4)"
    - "HAND_thresh = 4.0 m"
    - Variable assignments: Speckle_Filter = "refined_lee(window=7)"
    - Comments (ignored via #)
    """
    configs = []
    
    # Strip comments first (anything after #)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove inline comments
        if '#' in line:
            line = line[:line.index('#')]
        cleaned_lines.append(line.strip())
    
    text = ' '.join(cleaned_lines).lower()
    
    # 1. Speckle Filters
    # Refined Lee
    if 'refined_lee' in text:
        win_match = re.search(r'refined_lee.*?window\s*=\s*(\d+)', text)
        win = int(win_match.group(1)) if win_match else 7
        configs.append({
            'filter': 'Refined Lee',
            'params': {'window_size': win, 'threshold': -18.0},
            'reason': f'Extracted: refined_lee(window={win})'
        })
        
    # Gamma MAP
    if 'gamma_map' in text:
        win_match = re.search(r'gamma_map.*?window\s*=\s*(\d+)', text)
        win = int(win_match.group(1)) if win_match else 5
        configs.append({
            'filter': 'Gamma MAP',
            'params': {'window_size': win, 'threshold': -18.0},
            'reason': f'Extracted: gamma_map(window={win})'
        })

    # SRAD
    if 'srad' in text:
        iter_match = re.search(r'srad.*?(?:iter|n_iter)\s*=\s*(\d+)', text)
        iters = int(iter_match.group(1)) if iter_match else 15
        configs.append({
            'filter': 'SRAD',
            'params': {'num_iter': iters},
            'reason': f'Extracted: srad(iter={iters})'
        })
        
    # BayesShrink
    if 'bayesshrink' in text:
        configs.append({
            'filter': 'BayesShrink',
            'params': {'levels': 3},
            'reason': 'Extracted: bayesshrink'
        })
        
    # 2. CFAR
    if 'k_cfar' in text or 'cfar' in text:
        pfa_match = re.search(r'pfa\s*=\s*([\deE.-]+)', text)
        pfa = float(pfa_match.group(1)) if pfa_match else 1e-4
        configs.append({
            'filter': 'K-Dist CFAR',
            'params': {'pfa': pfa},
            'reason': f'Extracted: CFAR(pfa={pfa})'
        })
        
    # 3. Thresholds (Hysteresis & Simple)
    # T_low / T_high pattern
    t_low = re.search(r't_low\s*=\s*([-\d.]+)', text)
    t_high = re.search(r't_high[_\w]*\s*=\s*([-\d.]+)', text)
    
    if t_low and t_high:
        low_val = float(t_low.group(1))
        high_val = float(t_high.group(1))
        configs.append({
            'filter': 'Hysteresis',
            'params': {'band': 'vh', 'low': low_val, 'high': high_val},
            'reason': f'Extracted: Hysteresis (low={low_val}, high={high_val})'
        })
    elif t_low:
        val = float(t_low.group(1))
        configs.append({
            'filter': 'Simple Threshold',
            'params': {'band': 'vh', 'threshold': val},
            'reason': f'Extracted: T_low={val}'
        })

    # 4. HAND
    hand_match = re.search(r'hand_thresh\s*=\s*([\d.]+)', text)
    if hand_match:
        val = float(hand_match.group(1))
        configs.append({
            'filter': 'HAND Definite',
            'params': {'hand_thresh': val},
            'reason': f'Extracted: HAND={val}m'
        })
        
    # 5. Texture / Edge
    if 'glcm' in text:
        configs.append({
            'filter': 'GLCM Entropy',
            'params': {'threshold': 1.5},
            'reason': 'Extracted: GLCM'
        })
        
    if 'touzi' in text:
        # Parse window parameter if present
        win_match = re.search(r'touzi.*?(?:win|window)\s*=\s*(\d+)', text)
        win = int(win_match.group(1)) if win_match else 7
        configs.append({
            'filter': 'Touzi Ratio',
            'params': {'window_size': win, 'ratio_thresh': 0.4},
            'reason': f'Extracted: Touzi(win={win})'
        })
        
    # 6. Morphology
    if 'morphology' in text or 'min_area' in text:
        min_area_match = re.search(r'min_area\s*=\s*(\d+)', text)
        fill_holes_match = re.search(r'fill_holes\s*=\s*(\d+)', text)
        
        min_area = int(min_area_match.group(1)) if min_area_match else 50
        fill_holes = int(fill_holes_match.group(1)) if fill_holes_match else 200
        
        configs.append({
            'filter': 'Morphology Clean',
            'params': {'min_area': min_area, 'fill_holes': fill_holes},
            'reason': f'Extracted: Morphology (min_area={min_area}, fill_holes={fill_holes})'
        })

    return configs


# Define local hand functions since filter_engine_complete doesn't export them
def hand_definite(hand, mndwi, vh, hand_thresh=3.0):
    return (hand < hand_thresh) & (vh < -15.0)

def hand_probable(hand, mndwi, vh, hand_thresh=5.0):
    return (hand < hand_thresh) & (vh < -15.0)

def hand_possible(hand, mndwi, hand_thresh=10.0):
    return hand < hand_thresh

def threshold_simple(data, threshold=-17.0):
    return data < threshold


# =============================================================================
# STABILITY HEATMAP
# =============================================================================

def stability_heatmap(data, threshold, perturbation=0.5, num_levels=5):
    """
    Perturb threshold and map pixels that flip state.
    
    Args:
        data: Input band data
        threshold: Base threshold value
        perturbation: Amount to perturb (+/-)
        num_levels: Number of perturbation levels
    
    Returns:
        stability_map: Float array (0 = unstable, 1 = stable)
        flip_count: Array counting number of times each pixel flipped
    """
    thresholds = np.linspace(threshold - perturbation, threshold + perturbation, num_levels)
    
    masks = []
    for t in thresholds:
        masks.append(data < t)
    
    # Stack and count consistency
    stack = np.stack(masks, axis=0)
    agreement = stack.sum(axis=0)
    
    # Stability = fraction of agreement (all water or all land)
    stability = np.maximum(agreement, num_levels - agreement) / num_levels
    
    # Flip count = number of transitions
    flip_count = num_levels - np.maximum(agreement, num_levels - agreement)
    
    return stability, flip_count


def generate_stability_overlay(data, threshold=-17.0, perturbation=1.0):
    """
    Generate color-coded stability overlay.
    
    Blue: Stable (always water or land)
    Yellow: Unstable (flips between)
    """
    stability, _ = stability_heatmap(data, threshold, perturbation)
    
    h, w = data.shape
    overlay = np.zeros((h, w, 3))
    
    # Blue for stable, yellow for unstable
    overlay[:, :, 0] = 1 - stability  # R: high when unstable
    overlay[:, :, 1] = 1 - stability  # G: high when unstable (yellow)
    overlay[:, :, 2] = stability       # B: high when stable
    
    return overlay, stability


# =============================================================================
# CONFIDENCE MAP
# =============================================================================

def confidence_map(hand, mndwi, vh, masks_list=None):
    """
    Generate confidence map based on filter agreement.
    
    Returns:
        confidence: 0 (low), 1 (medium), 2 (high)
        labels: String labels for each level
    """
    # Use HAND-based confidence levels
    definite = hand_definite(hand, mndwi, vh, hand_thresh=3.0)
    probable = hand_probable(hand, mndwi, vh, hand_thresh=5.0)
    possible = hand_possible(hand, mndwi, hand_thresh=10.0)
    
    # Build confidence map
    confidence = np.zeros_like(hand, dtype=int)
    confidence[possible] = 1  # Low confidence
    confidence[probable] = 2  # Medium confidence
    confidence[definite] = 3  # High confidence
    
    return confidence


def confidence_from_voting(masks_list):
    """
    Generate confidence map from multiple filter masks.
    
    Definite: 5+ filters agree
    Probable: 3-4 filters agree
    Possible: 1-2 filters agree
    """
    if not masks_list:
        return None
    
    stack = np.stack([m.astype(int) for m in masks_list], axis=0)
    votes = stack.sum(axis=0)
    
    n = len(masks_list)
    
    confidence = np.zeros_like(masks_list[0], dtype=int)
    
    # Scale thresholds based on number of masks
    high_thresh = max(1, int(n * 0.6))    # 60% agreement
    med_thresh = max(1, int(n * 0.3))     # 30% agreement
    
    confidence[votes >= 1] = 1             # Possible
    confidence[votes >= med_thresh] = 2    # Probable
    confidence[votes >= high_thresh] = 3   # Definite
    
    return confidence


# =============================================================================
# HAND DEPTH CONTOURS
# =============================================================================

def hand_depth_contours(hand, levels=[1, 3, 5, 10]):
    """
    Generate HAND depth contour classes.
    
    h1: < 1m (Deep channel / definite water)
    h2: 1-3m (Floodplain)
    h3: 3-5m (Risk zone)
    h4: 5-10m (Possible flood extent)
    
    Returns:
        contour_map: Integer class map (0-4)
        class_info: Dict with class names and colors
    """
    hand_clean = np.nan_to_num(hand, nan=999)
    
    contour_map = np.zeros_like(hand_clean, dtype=int)
    
    # Assign classes (higher = deeper in drainage)
    contour_map[hand_clean < levels[3]] = 1  # h4: Possible
    contour_map[hand_clean < levels[2]] = 2  # h3: Risk
    contour_map[hand_clean < levels[1]] = 3  # h2: Floodplain
    contour_map[hand_clean < levels[0]] = 4  # h1: Deep channel
    
    class_info = {
        0: {'name': 'Safe (>10m)', 'color': '#FFFFFF'},
        1: {'name': 'h4: Possible (5-10m)', 'color': '#E3F2FD'},
        2: {'name': 'h3: Risk (3-5m)', 'color': '#90CAF9'},
        3: {'name': 'h2: Floodplain (1-3m)', 'color': '#42A5F5'},
        4: {'name': 'h1: Deep (<1m)', 'color': '#1565C0'}
    }
    
    return contour_map, class_info


# =============================================================================
# PIXEL STATISTICS
# =============================================================================

def get_pixel_stats(chip_data, row, col):
    """
    Get all band values for a specific pixel.
    
    Returns dict with all band values and derived metrics.
    """
    stats = {
        'location': f'({row}, {col})',
        'vv_db': float(chip_data['vv'][row, col]),
        'vh_db': float(chip_data['vh'][row, col]),
        'mndwi': float(chip_data['mndwi'][row, col]),
        'dem_m': float(chip_data['dem'][row, col]),
        'hand_m': float(chip_data['hand'][row, col]),
        'slope_deg': float(chip_data['slope'][row, col]),
        'twi': float(chip_data['twi'][row, col])
    }
    
    # Derived assessments
    stats['vh_water'] = 'Yes' if stats['vh_db'] < -17.0 else 'No'
    stats['hand_water'] = 'Definite' if stats['hand_m'] < 3 else (
        'Probable' if stats['hand_m'] < 5 else (
            'Possible' if stats['hand_m'] < 10 else 'Unlikely'
        )
    )
    stats['flat'] = 'Yes' if stats['slope_deg'] < 5 else 'No'
    
    return stats


def pixel_probe_text(chip_data, row, col):
    """Generate formatted text for pixel probe tooltip."""
    stats = get_pixel_stats(chip_data, row, col)
    
    text = f"""
    📍 Pixel ({row}, {col})
    ─────────────────
    VV:    {stats['vv_db']:.1f} dB
    VH:    {stats['vh_db']:.1f} dB ({stats['vh_water']})
    MNDWI: {stats['mndwi']:.3f}
    ─────────────────
    HAND:  {stats['hand_m']:.1f}m ({stats['hand_water']})
    Slope: {stats['slope_deg']:.1f}° ({stats['flat']})
    TWI:   {stats['twi']:.1f}
    DEM:   {stats['dem_m']:.0f}m
    """
    
    return text.strip()


# =============================================================================
# BEST BAND SELECTOR (Fisher Distance)
# =============================================================================

def fisher_distance(data, mask):
    """
    Calculate Fisher Distance between water and land classes.
    
    Higher = better class separability.
    """
    water = data[mask]
    land = data[~mask]
    
    if len(water) < 10 or len(land) < 10:
        return 0.0
    
    mu_w = np.mean(water)
    mu_l = np.mean(land)
    var_w = np.var(water)
    var_l = np.var(land)
    
    fisher = (mu_w - mu_l)**2 / (var_w + var_l + 1e-8)
    
    return fisher


def best_band_selector(chip_data, reference_mask=None):
    """
    Select the band with highest Fisher Distance (best separability).
    
    Returns:
        best_band: Name of best band
        scores: Dict of Fisher distances for each band
    """
    if reference_mask is None:
        # Use simple VH threshold as reference
        reference_mask = chip_data['vh'] < -17.0
    
    bands = {
        'VV': chip_data['vv'],
        'VH': chip_data['vh'],
        'MNDWI': chip_data['mndwi']
    }
    
    scores = {}
    for name, data in bands.items():
        scores[name] = fisher_distance(data, reference_mask)
    
    best_band = max(scores, key=scores.get)
    
    return best_band, scores


# =============================================================================
# AUTO-NOISE TUNER
# =============================================================================

def estimate_noise_floor(data, percentile=5):
    """
    Estimate noise floor from data distribution.
    Uses dark regions as noise reference.
    """
    flat = data[~np.isnan(data)].flatten()
    noise_level = np.percentile(flat, percentile)
    return noise_level


def auto_noise_tuner(data, hand):
    """
    Auto-calculate optimal threshold based on noise analysis.
    
    Uses shadow regions (low HAND, very dark) as noise reference.
    """
    # Find candidate shadow regions
    hand_clean = np.nan_to_num(hand, nan=999)
    
    # Very dark pixels in low HAND areas
    dark_mask = data < -25
    low_hand = hand_clean < 10
    
    candidates = dark_mask & low_hand
    
    if candidates.sum() < 100:
        # Not enough candidates, use percentile
        return estimate_noise_floor(data)
    
    # Noise floor = peak of dark region histogram
    dark_values = data[candidates]
    noise_floor = np.median(dark_values)
    
    # Recommended threshold = noise floor + margin
    recommended_thresh = noise_floor + 3.0  # 3 dB above noise
    
    return recommended_thresh


# =============================================================================
# VARIETY GENERATOR
# =============================================================================

def generate_variety_configs(chip_data, num_configs=15):
    """
    Generate intelligent, physics-based filter configurations.
    
    Analyzes the chip data characteristics to select appropriate filters:
    1. Histogram bimodality test → Otsu or Triangle
    2. Texture homogeneity → GLCM-based if heterogeneous
    3. HAND availability → Hydro-geomorphic constraints
    4. Data-driven thresholds (percentiles, auto-computed Otsu)
    
    Returns configs sorted by expected reliability for this specific chip.
    """
    configs = []
    
    # =========================================================================
    # STEP 1: Analyze Chip Characteristics
    # =========================================================================
    vh_data = chip_data.get('vh')
    vv_data = chip_data.get('vv')
    hand_data = chip_data.get('hand')
    mndwi_data = chip_data.get('mndwi')
    twi_data = chip_data.get('twi')
    
    if vh_data is None:
        return [{'filter': 'None', 'params': {}, 'reason': 'No VH data'}]
    
    # Compute key statistics
    vh_valid = vh_data[~np.isnan(vh_data)]
    vh_mean = np.mean(vh_valid)
    vh_std = np.std(vh_valid)
    vh_p5 = np.percentile(vh_valid, 5)
    vh_p10 = np.percentile(vh_valid, 10)
    vh_p25 = np.percentile(vh_valid, 25)
    vh_median = np.median(vh_valid)
    
    # Bimodality test (Ashman's D)
    # If histogram is clearly bimodal, Otsu is reliable
    try:
        from scipy.stats import kurtosis
        vh_kurtosis = kurtosis(vh_valid)
        is_bimodal = vh_kurtosis < -0.5  # Platykurtic = likely bimodal
    except:
        is_bimodal = False
    
    # Texture homogeneity (local variance)
    try:
        from scipy.ndimage import uniform_filter
        local_var = uniform_filter(vh_data**2, size=5) - uniform_filter(vh_data, size=5)**2
        avg_local_var = np.nanmean(local_var)
        is_heterogeneous = avg_local_var > 10  # High local variance = heterogeneous
    except:
        is_heterogeneous = False
    
    # HAND availability check
    has_hand = hand_data is not None and not np.all(np.isnan(hand_data))
    has_twi = twi_data is not None and not np.all(np.isnan(twi_data))
    
    # =========================================================================
    # STEP 2: Physics-Based Filter Selection
    # =========================================================================
    
    # --- A. Automatic Thresholding Methods ---
    if is_bimodal:
        # Otsu is reliable for bimodal histograms
        configs.append({
            'filter': 'Otsu (Global)',
            'params': {'band': 'vh'},
            'reason': 'Bimodal histogram detected - Otsu optimal',
            'confidence': 0.9
        })
    else:
        # Triangle method for unimodal/skewed histograms
        configs.append({
            'filter': 'Triangle',
            'params': {'band': 'vh'},
            'reason': 'Non-bimodal histogram - Triangle robust',
            'confidence': 0.75
        })
    
    # --- B. Percentile-Based Thresholds (Data-Driven) ---
    # Water typically in bottom 5-15% of backscatter values
    configs.append({
        'filter': 'Simple Threshold',
        'params': {'band': 'vh', 'threshold': float(vh_p10)},
        'reason': f'P10 threshold ({vh_p10:.1f} dB) - conservative water estimate',
        'confidence': 0.7
    })
    
    configs.append({
        'filter': 'Simple Threshold',
        'params': {'band': 'vh', 'threshold': float(vh_p25)},
        'reason': f'P25 threshold ({vh_p25:.1f} dB) - liberal water estimate',
        'confidence': 0.6
    })
    
    # --- C. Hysteresis (Two-Stage) ---
    # Seeds at P5, grow to P15
    low_thresh = float(np.percentile(vh_valid, 5))
    high_thresh = float(np.percentile(vh_valid, 15))
    configs.append({
        'filter': 'Hysteresis',
        'params': {'band': 'vh', 'low': low_thresh, 'high': high_thresh},
        'reason': f'Hysteresis [{low_thresh:.1f}, {high_thresh:.1f}] - confident cores + edges',
        'confidence': 0.8
    })
    
    # --- D. Texture-Based (if heterogeneous scene) ---
    if is_heterogeneous:
        configs.append({
            'filter': 'GLCM Entropy',
            'params': {'band': 'vh', 'window_size': 5, 'threshold': 1.5},
            'reason': 'Heterogeneous scene - texture filtering recommended',
            'confidence': 0.65
        })
        configs.append({
            'filter': 'Coeff of Variation',
            'params': {'threshold': 0.4},
            'reason': 'CoV for homogeneity-based segmentation',
            'confidence': 0.6
        })
    
    # --- E. Hydro-Geomorphic Constraints ---
    if has_hand:
        hand_valid = hand_data[~np.isnan(hand_data)]
        hand_median = np.median(hand_valid)
        
        # Adaptive HAND threshold based on terrain
        if hand_median < 10:  # Flat terrain
            hand_thresh = 5.0
        else:  # Hilly terrain
            hand_thresh = 10.0
        
        configs.append({
            'filter': 'HAND Definite',
            'params': {'hand_thresh': hand_thresh},
            'reason': f'HAND < {hand_thresh}m + SAR - terrain-constrained',
            'confidence': 0.85
        })
        configs.append({
            'filter': 'HAND Probable',
            'params': {'hand_thresh': hand_thresh * 1.5},
            'reason': f'HAND < {hand_thresh*1.5}m - extended flood zone',
            'confidence': 0.7
        })
    
    if has_twi:
        twi_valid = twi_data[~np.isnan(twi_data)]
        twi_p75 = np.percentile(twi_valid, 75)
        configs.append({
            'filter': 'TWI High',
            'params': {'min_twi': float(twi_p75)},
            'reason': f'TWI > {twi_p75:.1f} - high water accumulation potential',
            'confidence': 0.6
        })
    
    # --- F. VV-based (if available) ---
    if vv_data is not None:
        vv_valid = vv_data[~np.isnan(vv_data)]
        vv_p10 = np.percentile(vv_valid, 10)
        configs.append({
            'filter': 'Simple Threshold',
            'params': {'band': 'vv', 'threshold': float(vv_p10)},
            'reason': f'VV P10 ({vv_p10:.1f} dB) - calm water detection',
            'confidence': 0.65
        })
        
        # Cross-Pol Ratio
        configs.append({
            'filter': 'Cross-Pol Ratio',
            'params': {'threshold': 0.3},
            'reason': 'CPR < 0.3 - specular surfaces (water)',
            'confidence': 0.7
        })
        
        # SDWI Index
        configs.append({
            'filter': 'SDWI',
            'params': {'threshold': 0.0},
            'reason': 'SDWI > 0 - dual-pol water index',
            'confidence': 0.75
        })
    
    # --- G. Pre-processing + Threshold combo ---
    configs.append({
        'filter': 'Refined Lee',
        'params': {'window_size': 7, 'threshold': float(vh_p10)},
        'reason': 'Refined Lee speckle filter + P10 threshold',
        'confidence': 0.7
    })
    
    # =========================================================================
    # STEP 3: Sort by Confidence and Return
    # =========================================================================
    configs.sort(key=lambda x: x.get('confidence', 0.5), reverse=True)
    
    # Limit to requested number
    return configs[:num_configs]


# =============================================================================
# NEW FEATURES (Flow & Eco)
# =============================================================================

def flow_direction_overlay(dem, num_arrows=20):
    """
    Generate flow direction arrows based on DEM gradient.
    Returns quiver plot data points (x, y, u, v).
    """
    # Calculate gradients
    dy, dx = np.gradient(dem)
    
    # Downsample for visualization (arrows shouldn't be every pixel)
    h, w = dem.shape
    step_y = max(1, h // num_arrows)
    step_x = max(1, w // num_arrows)
    
    y, x = np.mgrid[step_y//2:h:step_y, step_x//2:w:step_x]
    
    # Negative gradient = downhill flow
    u = -dx[y, x]
    v = -dy[y, x]
    
    # Normalize
    norm = np.sqrt(u**2 + v**2)
    u = np.divide(u, norm, where=norm > 0)
    v = np.divide(v, norm, where=norm > 0)
    
    return x, y, u, v


def eco_classifier(chip_data, water_mask):
    """
    Classify water into 'Open Lake' vs 'Vegetated Wetland'.
    
    Lake: Low texture variance, very dark VH.
    Wetland: High texture variance, brighter VH.
    """
    vh = chip_data['vh']
    
    # Texture measure (Coef of Variation)
    mean = uniform_filter(vh, size=5)
    var = uniform_filter((vh - mean)**2, size=5)
    std = np.sqrt(np.maximum(var, 0))
    cov = std / np.abs(mean + 1e-8)
    
    # Classification
    # High CoV + Water Mask = Wetland
    # Low CoV + Water Mask = Lake
    
    classification = np.zeros_like(vh, dtype=int)
    
    # 1 = Lake, 2 = Wetland
    classification[(water_mask) & (cov < 0.3)] = 1
    classification[(water_mask) & (cov >= 0.3)] = 2
    
    return classification, {1: 'Lake (Open)', 2: 'Wetland (Vegetated)'}

