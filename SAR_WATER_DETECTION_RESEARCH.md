# SAR Water Detection - Comprehensive Research Document

## Physics-Guided Adaptive Threshold System for India

**Project:** SAR Water Detection with Physics Constraints
**Date:** 2026-01-25
**Status:** Research Phase Complete - Ready for Implementation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Model Performance](#2-current-model-performance)
3. [Error Analysis](#3-error-analysis)
4. [Discovered Physics Equations](#4-discovered-physics-equations)
5. [SAR Water Detection Challenges](#5-sar-water-detection-challenges)
6. [Adaptive Context-Aware Thresholds](#6-adaptive-context-aware-thresholds)
7. [GEE Data Assets](#7-gee-data-assets)
8. [Proposed Solutions](#8-proposed-solutions)
9. [Implementation Plan](#9-implementation-plan)

---

## 1. Executive Summary

### Key Findings

| Metric | LightGBM v4 | U-Net v4 | U-Net v6 | PySR Best Equation |
|--------|-------------|----------|----------|-------------------|
| **Val IoU** | 0.930 | 0.918 | 0.844 | 0.622 (global) |
| **Test IoU** | **0.881** | 0.766 | 0.582 | 0.677 (test) |
| Parameters | 69 features | 7.8M | 4.4M | 9 complexity |

### Critical Error Sources

| Error Type | Count | Percentage | Root Cause |
|------------|-------|------------|------------|
| **Bright Water (FN)** | 29,681 | **97.3%** of all FN | Wind-roughened water has VH > -16 dB |
| **Wet Soil (FP)** | 3,008 | **45.9%** of all FP | Saturated soil mimics water signature |
| Unknown FP | 3,543 | 54.1% of FP | Needs further investigation |

### Best Discovered Equation (IoU 0.62)

```
water = min(max(SLOPE + VH, 0.6), TWI + 0.6)
```

**Physical Interpretation:**
- Water requires BOTH: (low slope + low VH backscatter) AND (high TWI)
- This is a physically meaningful constraint!

---

## 2. Current Model Performance

### LightGBM v4 (Best Model)

**Configuration:**
- 69 engineered features
- Multi-scale texture (3, 5, 9, 15, 21 pixel windows)
- Physics features (HAND, SLOPE, TWI, DEM)
- 972,361 training samples

**Cross-Validation Results:**
```
Mean IoU: 0.9302 (+/- 0.0004)
Mean F1:  0.9638
Mean Accuracy: 0.9552
```

**Test Results:**
```
IoU: 0.9299
F1: 0.9637
Precision: 0.9692
Recall: 0.9583
AUC-ROC: 0.9927
```

### Top 20 Most Important Features

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | DEM | 3290 | Terrain |
| 2 | TWI | 1711 | Physics |
| 3 | VH_closed | 1391 | Morphology |
| 4 | VV_std_s21 | 1272 | Texture (21px) |
| 5 | VH_mean_s21 | 1231 | Context (21px) |
| 6 | VV_mean_s21 | 1101 | Context (21px) |
| 7 | VH_std_s21 | 1058 | Texture (21px) |
| 8 | HAND | 1013 | Physics |
| 9 | VV_min_s9 | 1013 | Dark pixel detection |
| 10 | VH_min_s9 | 1010 | Dark pixel detection |
| 11 | SLOPE | 855 | Physics |
| 12 | VH_opened | 825 | Morphology |
| 13 | VV_otsu_diff | 766 | Adaptive threshold |
| 14 | VH_otsu_diff | 678 | Adaptive threshold |
| 15 | VV_mean_s15 | 622 | Context (15px) |
| 16 | hand_score | 588 | Physics composite |
| 17 | VH_mean_s15 | 556 | Context (15px) |
| 18 | pseudo_entropy | 542 | Texture |
| 19 | VV_glcm_homogeneity | 488 | GLCM texture |
| 20 | VH_mean_s9 | 483 | Context (9px) |

**Key Insight:** Physics features (DEM, TWI, HAND, SLOPE) dominate importance!
Simple VH threshold (`vh_score`: 73) is NOT important - model learns complex patterns.

---

## 3. Error Analysis

### False Positive Analysis

**Total FP:** 6,551 pixels

| Category | Count | Percentage | SAR Signature |
|----------|-------|------------|---------------|
| Wet Soil | 3,008 | 45.9% | VH: -22 to -14 dB, HAND < 10m |
| Shadow | 0 | 0% | VH < -25 dB, SLOPE > 15 |
| Urban | 0 | 0% | Low VH, High VV (double-bounce) |
| Unknown | 3,543 | 54.1% | Needs investigation |

**Wet Soil Characteristics:**
- Located near drainage channels (low HAND)
- VH between -22 and -14 dB (overlaps with water)
- Higher slope than water (typically 2-15)
- Changes with rainfall (temporal signature differs)

### False Negative Analysis

**Total FN:** 30,497 pixels

| Category | Count | Percentage | Root Cause |
|----------|-------|------------|------------|
| **Bright Water** | 29,681 | **97.3%** | VH > -16 dB due to wind/waves |
| High HAND | 306 | 1.0% | Mountain lakes, reservoirs |
| Steep Slope | 6 | 0.02% | Waterfalls, rapids |
| Unknown | 504 | 1.65% | Needs investigation |

**Bright Water Causes:**
1. **Wind roughening:** Surface waves increase backscatter
2. **Rain impact:** Splashing increases surface roughness
3. **Vegetation:** Floating plants add volume scattering
4. **Turbidity:** Suspended sediment (minor effect)

### Critical Threshold Analysis

Current thresholds miss water:
```
Standard: VH < -18 dB (misses 97% of FN!)
Reality:  Bright water has VH up to -10 dB
```

---

## 4. Discovered Physics Equations

### PySR Symbolic Regression Results

**Global Best (IoU 0.622, Test IoU 0.677):**
```python
water = min(max(SLOPE + VH, 0.6), TWI + 0.6)
```

**Physical Interpretation:**
- `max(SLOPE + VH, 0.6)`: Creates a lower bound combining slope and VH
  - Low slope + low VH (dark) = low value = likely water
  - High slope OR high VH = high value = likely NOT water
- `min(..., TWI + 0.6)`: Caps by TWI (topographic wetness)
  - High TWI areas (valleys, drainage) more likely to hold water
- **Combined:** Water needs BOTH (low slope + dark) AND (wet topography)

### Category-Specific Equations

**Large Water Bodies (IoU 0.894):**
```python
water = min(TWI + 0.79, max(0.80, (VH + 0.61) / 0.009))
```
- VH dominates for large, calm water
- TWI provides upper bound

**Medium Water Bodies (IoU 0.542):**
```python
water = min(max(-0.17, VH * SLOPE), TWI) + (min(VV + HAND, -2.0) * -0.032)
```
- More complex: includes VV, HAND interaction
- Slope modulates VH response

**Small Water Bodies (IoU ~0):**
- PySR struggled with small water (< 10% coverage)
- Equations defaulted to constants
- **Needs spatial context (U-Net) not pixel-wise equations**

### Equation Template Library

From `gpu_equation_search.py`:

```python
EQUATION_TEMPLATES = {
    # Basic thresholds
    'simple_vh': "(vh < {T_vh})",
    
    # HAND-constrained
    'hand_constrained_vh': "(vh < {T_vh}) & (hand < {T_hand})",
    
    # Triple-lock (SAR + HAND + Texture)
    'triple_lock': "(vv < {T_vv}) & (hand < {T_hand}) & (entropy < {T_entropy})",
    
    # SWI polynomial (Tian et al., 2017)
    'swi_based': "(swi > {T_swi}) & (hand < {T_hand})",
    
    # Hysteresis (connected regions)
    'hysteresis_vv': "((vv < {T_vv_low}) | ((vv < {T_vv_high}) & (hand < {T_hand})))",
    
    # Urban-specific
    'split_logic_urban': "(vh < {T_vh_urban}) & (cpr > {T_cpr_urban}) & (hand < {T_hand_urban})",
}
```

---

## 5. SAR Water Detection Challenges

### 5.1 False Positive Sources

| Source | Physical Cause | SAR Signature | Detection Strategy |
|--------|---------------|---------------|-------------------|
| **Terrain Shadow** | Radar cannot illuminate behind steep slopes | VH < -30 dB, SLOPE > 35 | Shadow mask from DEM + incidence angle |
| **Wet Soil** | Saturated soil near drainage | VH: -22 to -14 dB, HAND < 10m | Temporal: dries in 2-5 days |
| **Urban Smooth** | Roads, parking lots, rooftops | VH < -20 dB, VV > -10 dB | VV/VH ratio (double-bounce) |
| **Bare Rock** | Smooth mineral surfaces | VH < -22 dB, high curvature | Geology mask or optical NDVI |
| **Airports** | Large smooth concrete | VH < -25 dB, geometric shape | OSM infrastructure mask |
| **Salt Flats** | Flat mineral surfaces | VH < -25 dB, high SLOPE variance | Multi-temporal (seasonal) |

### 5.2 False Negative Sources

| Source | Physical Cause | SAR Signature | Detection Strategy |
|--------|---------------|---------------|-------------------|
| **Bright/Rough Water** | Wind waves, rain splashing | VH: -16 to -8 dB | Adaptive threshold + HAND constraint |
| **Narrow Rivers** | Sub-pixel width (< 10m) | Mixed land-water pixels | Frangi vesselness, line detection |
| **Vegetated Water** | Floating plants, algae | Higher VH due to volume scattering | Optical MNDWI confirmation |
| **Turbid Water** | Suspended sediment | Slightly higher backscatter | Use relaxed thresholds in known rivers |
| **Mountain Lakes** | High elevation | High HAND values | Elevation-aware HAND threshold |
| **Reservoirs** | Man-made, dam-controlled | Variable levels | Use historical extent (JRC) |

### 5.3 Confusion Scenarios

| Scenario | Challenge | Recommended Approach |
|----------|-----------|---------------------|
| **Flooded Vegetation** | Double-bounce increases VV, VH stays moderate | Use VV/VH ratio + known flood extent |
| **Paddy Fields** | Seasonal flooding, looks like wetland | Crop calendar, temporal analysis |
| **Wetlands** | Mixed water/vegetation | Relaxed threshold + TWI confirmation |
| **River Deltas** | Complex morphology, sandbars | Multi-temporal maximum extent |
| **Coastal Zones** | Tidal variations | Time-series or tide model |
| **Glacial Lakes** | Ice, seasonal freezing | Optical confirmation, temperature |

### 5.4 India-Specific Challenges

Based on GEE chip locations:

| Region | Water Type | Challenge | Chips |
|--------|------------|-----------|-------|
| **Kerala Backwaters** | Canals, lagoons | Narrow, vegetated | 101-110 |
| **Sundarbans** | Mangroves, tidal | Mixed vegetation/water | 91-100 |
| **Brahmaputra** | Braided river | Sandbars, flooding | 16-20, 113-114 |
| **Rann of Kutch** | Salt marsh | Seasonal, salt crust | 76-80 |
| **Ladakh Lakes** | High altitude | Low HAND fails | 111-112 |
| **Loktak Lake** | Floating islands | Vegetation on water | 11-15, 43-44 |

---

## 6. Adaptive Context-Aware Thresholds

### 6.1 Context Detection

Before applying water detection, classify the scene:

```python
def detect_context(vv, vh, hand, slope, twi, texture):
    """Classify scene context for adaptive thresholding."""
    
    # Compute scene statistics
    vh_mean = np.nanmean(vh)
    vh_std = np.nanstd(vh)
    texture_mean = np.nanmean(texture)
    hand_p90 = np.nanpercentile(hand, 90)
    
    # Context classification rules
    contexts = {
        'calm_lake': (texture_mean < 0.3) & (hand_p90 < 5) & (vh_std < 3),
        'windy_water': (texture_mean > 0.5) & (hand_p90 < 10) & (vh_mean > -18),
        'river': has_linear_features(vh) & (hand_p90 < 15),
        'wetland': (twi.mean() > 10) & (texture_mean > 0.4),
        'urban': (vv.mean() > -10) & (vh.mean() < -18),
        'mountain': (slope.mean() > 10) | (hand_p90 > 50),
        'arid': (twi.mean() < 5) & (texture_mean < 0.2),
    }
    
    return contexts
```

### 6.2 Adaptive Threshold Table

| Context | VH Threshold | HAND Threshold | Additional Constraints |
|---------|-------------|----------------|----------------------|
| **Calm Lake** | VH < -20 dB | HAND < 10m | texture < 0.3 |
| **Windy Water** | VH < -12 dB | HAND < 5m | + slope < 3 |
| **River** | VH < -16 dB | HAND < 15m | + Frangi > 0.1 |
| **Wetland** | VH < -14 dB | HAND < 20m | + TWI > 8 |
| **Urban Flood** | VH < -22 dB | HAND < 3m | + VV/VH > 8 |
| **Mountain Lake** | VH < -18 dB | Ignore HAND | + elevation check |
| **Arid/Ephemeral** | VH < -20 dB | HAND < 5m | + historical extent (JRC) |

### 6.3 Implementation Strategy

```python
def adaptive_water_detection(vv, vh, hand, slope, twi, dem):
    """
    Adaptive context-aware water detection.
    Returns water probability [0, 1].
    """
    # Step 1: Detect context
    contexts = detect_context(vv, vh, hand, slope, twi, texture)
    
    # Step 2: Apply context-specific thresholds
    water_prob = np.zeros_like(vh)
    
    if contexts['calm_lake']:
        water_prob = np.where(
            (vh < -20) & (hand < 10) & (slope < 5),
            0.95, water_prob
        )
    
    if contexts['windy_water']:
        water_prob = np.where(
            (vh < -12) & (hand < 5) & (slope < 3),
            0.85, water_prob
        )
    
    if contexts['river']:
        frangi = compute_frangi(-vh)
        water_prob = np.where(
            (vh < -16) & (hand < 15) & (frangi > 0.1),
            0.90, water_prob
        )
    
    # Step 3: Physics safety net
    water_prob = np.where(slope > 20, 0, water_prob)  # No water on cliffs
    water_prob = np.where(hand > 50, water_prob * 0.5, water_prob)  # Reduce prob at high elevation
    
    return water_prob
```

---

## 7. GEE Data Assets

### 7.1 Current Chips

**99-Chip Dataset (gee_india_expanded.js):**
- Categories: urban, coastal, mountain, rivers, reservoirs, wetlands, arid, flood_prone
- Bands: VV, VH, MNDWI, DEM, HAND, SLOPE, TWI, JRC_Water
- Resolution: 10m
- Date range: 2023-01-01 to 2023-12-31

**120-Chip Dataset (gee_india_120_chips.js):**
- Added: coastal_mangroves (10), backwaters (10), high_elevation (10)
- JRC labels (< 1% error)
- Same band structure

### 7.2 Multi-Temporal Data Options

The GEE scripts currently use **yearly median** composites. For temporal analysis:

```javascript
// OPTION 1: Monthly composites (12 images/year)
var months = ee.List.sequence(1, 12);
var monthlyComposites = months.map(function(month) {
    return S1.filterDate(year + '-' + month + '-01', year + '-' + month + '-28')
             .median();
});

// OPTION 2: Pre/Post flood (for flood detection)
var preFlood = S1.filterDate('2023-06-01', '2023-06-30').median();
var postFlood = S1.filterDate('2023-08-01', '2023-08-31').median();
var floodExtent = postFlood.subtract(preFlood);

// OPTION 3: 6-day repeat pass (Sentinel-1 revisit)
var date1 = S1.filterDate('2023-07-01', '2023-07-02').first();
var date2 = S1.filterDate('2023-07-07', '2023-07-08').first();
```

### 7.3 Recommended Multi-Temporal Download

For wet soil discrimination:
- Download 4 dates per location: dry season, monsoon onset, peak monsoon, post-monsoon
- This allows tracking which "water" areas are persistent vs ephemeral

---

## 8. Proposed Solutions

### 8.1 Ensemble Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ENSEMBLE WATER DETECTION                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INPUT: [VV, VH, DEM, HAND, SLOPE, TWI] (6 bands)                  │
│                                                                     │
│  ┌──────────────────┐    ┌──────────────────┐                       │
│  │  LIGHTGBM v4     │    │    U-NET v6      │                       │
│  │  (69 features)   │    │   (4.4M params)  │                       │
│  │                  │    │                  │                       │
│  │  - Multi-scale   │    │  - Spatial       │                       │
│  │    texture       │    │    context       │                       │
│  │  - Physics       │    │  - Edge-aware    │                       │
│  │    features      │    │    loss          │                       │
│  │  - Morphology    │    │  - HAND          │                       │
│  │                  │    │    attention     │                       │
│  └────────┬─────────┘    └────────┬─────────┘                       │
│           │                       │                                 │
│           ▼                       ▼                                 │
│      P_lightgbm                P_unet                              │
│           │                       │                                 │
│           └───────────┬───────────┘                                │
│                       ▼                                             │
│             ┌─────────────────┐                                     │
│             │  META-LEARNER   │                                     │
│             │  or WEIGHTED    │                                     │
│             │  AVERAGE        │                                     │
│             │                 │                                     │
│             │ P = α*P_lgb +   │                                     │
│             │     β*P_unet    │                                     │
│             └────────┬────────┘                                     │
│                      │                                              │
│                      ▼                                              │
│             ┌─────────────────┐                                     │
│             │ PHYSICS SAFETY  │                                     │
│             │ NET (Hard Rules)│                                     │
│             │                 │                                     │
│             │ - SLOPE > 25    │                                     │
│             │   → P = 0       │                                     │
│             │ - HAND > 50 &   │                                     │
│             │   no JRC water  │                                     │
│             │   → P *= 0.2    │                                     │
│             └────────┬────────┘                                     │
│                      │                                              │
│                      ▼                                              │
│               FINAL OUTPUT                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 Bright Water Handler

```python
class BrightWaterHandler:
    """
    Handle wind-roughened water with VH > -16 dB.
    97% of our FN is bright water!
    """
    
    def __init__(self):
        # Relaxed thresholds for bright water
        self.vh_relaxed = -12  # Instead of -18
        self.hand_strict = 5   # Compensate with stricter HAND
        self.slope_strict = 3  # Compensate with stricter slope
        
    def detect_bright_water(self, vh, hand, slope, twi, texture):
        """
        Detect bright water using compensating physics constraints.
        
        If water is bright (high VH), require:
        - Very low HAND (must be in drainage)
        - Very low slope (must be flat)
        - High TWI (must be topographic low)
        """
        # Standard water (dark)
        dark_water = (vh < -18) & (hand < 10) & (slope < 10)
        
        # Bright water (relaxed VH, strict physics)
        bright_water = (
            (vh >= -18) & (vh < -12) &  # Bright but not too bright
            (hand < self.hand_strict) &  # Very low HAND
            (slope < self.slope_strict) &  # Very flat
            (twi > 8) &  # High wetness index
            (texture < 0.5)  # Smooth (not urban)
        )
        
        return dark_water | bright_water
```

### 8.3 Urban Area Mask

```python
class UrbanWaterDiscriminator:
    """
    Distinguish urban smooth surfaces from water.
    
    Key difference: Urban has double-bounce (high VV, low VH)
    Water has similar low VV and VH.
    """
    
    def compute_urban_mask(self, vv, vh, texture):
        """
        Detect urban areas using double-bounce signature.
        """
        # VV/VH ratio in dB (should be difference in dB scale)
        vv_vh_diff = vv - vh  # In dB, this is 10*log10(VV_linear/VH_linear)
        
        # Urban: high double-bounce ratio
        urban_mask = (
            (vv > -12) &  # High VV (double-bounce)
            (vh < -18) &  # Low VH (low cross-pol)
            (vv_vh_diff > 8) &  # Large difference
            (texture > 0.3)  # Some texture (buildings)
        )
        
        return urban_mask
    
    def apply_urban_exclusion(self, water_prob, vv, vh, texture):
        """Remove urban FP from water predictions."""
        urban_mask = self.compute_urban_mask(vv, vh, texture)
        water_prob = np.where(urban_mask, 0, water_prob)
        return water_prob
```

### 8.4 Wet Soil Discriminator

```python
class WetSoilDiscriminator:
    """
    Distinguish wet soil from water.
    
    Best approach: Multi-temporal (wet soil dries, water persists)
    Fallback: Use stricter thresholds in ambiguous zones.
    """
    
    def __init__(self, use_temporal=False):
        self.use_temporal = use_temporal
        
    def detect_wet_soil_zone(self, vh, hand, slope):
        """Identify zones likely to be wet soil (ambiguous)."""
        # Wet soil: moderate VH, low HAND, low-moderate slope
        ambiguous_zone = (
            (vh > -22) & (vh < -16) &  # Not dark enough for water
            (hand < 10) &  # Near drainage
            (slope > 2) & (slope < 15)  # Some slope (water is flatter)
        )
        return ambiguous_zone
    
    def temporal_discrimination(self, vh_t1, vh_t2, days_apart=7):
        """
        Use temporal change to distinguish water from wet soil.
        Wet soil: VH increases as it dries (brighter)
        Water: VH stable or decreases with wind
        """
        vh_change = vh_t2 - vh_t1
        
        # Wet soil drying: VH increases by ~3-5 dB over a week
        wet_soil = vh_change > 2
        
        return wet_soil
```

---

## 9. Implementation Plan

### Phase 1: Brute-Force Equation Search (In Progress)

**Status:** Running on server
**Script:** `gpu_equation_search.py`
**Regimes:** large_lake, wide_river, narrow_river, wetland, arid, reservoir, urban_flood

**Expected Output:**
- `top_equations_{regime}.json` for each water body type
- Best threshold parameters for each context

### Phase 2: Adaptive Threshold System

**Tasks:**
1. Implement context detection function
2. Create threshold lookup table
3. Integrate with LightGBM features
4. Validate on test chips

**Estimated Time:** 1 day

### Phase 3: LightGBM + U-Net Ensemble

**Tasks:**
1. Create ensemble wrapper class
2. Implement weighted averaging
3. Train meta-learner (optional)
4. Add physics safety net

**Estimated Time:** 2 days

### Phase 4: Error Handlers

**Tasks:**
1. Implement BrightWaterHandler
2. Implement UrbanWaterDiscriminator
3. Implement WetSoilDiscriminator (basic, no temporal)
4. Integrate as post-processing

**Estimated Time:** 1 day

### Phase 5: Multi-Temporal (Optional)

**Tasks:**
1. Modify GEE scripts for monthly exports
2. Download temporal series for flood-prone areas
3. Implement temporal consistency checks
4. Validate wet soil discrimination

**Estimated Time:** 3-5 days (includes GEE export time)

---

## Appendix A: Feature Definitions

### SAR Features
| Feature | Formula | Description |
|---------|---------|-------------|
| VV | Raw | Vertical-Vertical polarization (dB) |
| VH | Raw | Vertical-Horizontal polarization (dB) |
| VV_minus_VH | VV - VH | Polarization difference (dB) |
| RVI | 4*VH / (VV+VH) | Radar Vegetation Index |
| CPR | VH / VV | Cross-Polarization Ratio |

### Physics Features
| Feature | Formula | Description |
|---------|---------|-------------|
| HAND | Height Above Nearest Drainage | Hydrological connectivity |
| SLOPE | DEM gradient | Terrain steepness |
| TWI | ln(A / tan(b)) | Topographic Wetness Index |
| DEM | Elevation | Absolute elevation (m) |

### Texture Features
| Feature | Formula | Description |
|---------|---------|-------------|
| VH_std_s21 | std(VH, 21x21 window) | Large-scale roughness |
| VH_mean_s21 | mean(VH, 21x21 window) | Regional backscatter |
| VH_CV | std/mean | Coefficient of Variation |
| GLCM_homogeneity | GLCM texture | Smoothness measure |

---

## Appendix B: Key Thresholds from Literature

| Source | Water Threshold | HAND Threshold | Notes |
|--------|----------------|----------------|-------|
| Chini et al., 2017 | VV < -15 dB | - | Flood mapping |
| Twele et al., 2016 | VH < -20 dB | HAND < 15m | Combined approach |
| Martinis et al., 2015 | VV < -18 dB | - | TU Wien |
| Uddin et al., 2019 | VH < -18 dB | HAND < 10m | Nepal floods |
| Tian et al., 2017 | SWI > 0 | - | Polynomial index |
| **Our Best** | Adaptive | Adaptive | Context-aware |

---

## Appendix C: Physics Constraint Equations

From `physics_losses.py`:

### HAND Constraint Loss
```python
# Physics: Water probability should DECREASE with HAND
constraint_violation = sigmoid((hand - threshold) / temperature)
loss = water_prob * constraint_violation
```

### Slope Exclusion Loss
```python
# Physics: No water on steep slopes (> 15 degrees)
steep_mask = slope > 15
loss = water_prob[steep_mask].mean()
```

### Flow Consistency Loss
```python
# Physics: Water flows downhill
flow_direction = -gradient(DEM)
water_gradient = gradient(water_prob)
consistency = dot(water_gradient, flow_direction)
loss = relu(-consistency)  # Penalize uphill water
```

### TWI Constraint Loss
```python
# Physics: Water more likely in high TWI areas
low_twi_mask = twi < min_twi_threshold
loss = water_prob * low_twi_mask
```

---

## Appendix D: File Locations

### Server (mit-aoe@100.84.105.5)
```
~/sar_water_detection/
├── models/
│   ├── lightgbm_v4_comprehensive.txt   # BEST MODEL
│   ├── unet_v4_best.pth
│   └── unet_v6_best.pth
├── results/
│   ├── training_results_v4.json        # Feature importance
│   ├── comprehensive_evaluation_results.json
│   ├── pysr_global_results.json        # Discovered equations
│   ├── false_positive_analysis_results.json
│   └── top_equations_*.json            # Per-regime equations
├── chips/                              # Training data
└── gpu_equation_search.py              # Brute-force search
```

### Local Machine
```
/media/neeraj-parekh/Data1/sar soil system/chips/gui/
├── SAR_WATER_DETECTION_RESEARCH.md     # This document
├── gpu_equation_search.py
├── physics_unet.py
├── pysr_water_equations.py
└── *.js                                # GEE export scripts
```

---

## Appendix E: Session 2 Updates (2026-01-25)

### GPU Equation Search Results

**Infrastructure Fix:**
- Installed CuPy for CUDA 11 on server via conda
- GPU: NVIDIA RTX A5000 (24GB VRAM)

**Best Equations Found (from CPU v3 search, GPU running):**

| Rank | Template | IoU | Physics | Parameters |
|------|----------|-----|---------|------------|
| 1 | **hand_constrained** | **0.5113** | 0.974 | VH=-12, HAND=21m |
| 2 | hysteresis | 0.4745 | 0.976 | VH_low=-24, VH_high=-18, HAND=5m |
| 3 | triple_lock | 0.3921 | 0.989 | VH=-20, HAND=15, slope=15° |
| 4 | urban_exclusion | 0.3850 | 0.989 | VH=-18, HAND=5, VV_urban=-8, ratio=6 |
| 5 | bright_water | 0.3781 | 1.000 | VH=-14, HAND=4, slope=4° |

**Key Insight:** Simple `hand_constrained` equation (VH < T & HAND < H) achieves 51% IoU - much better than complex equations!

### Files Created This Session

1. **`adaptive_water_detection.py`** - Complete terrain-adaptive detection system
   - TerrainProfile dataclass with 7 terrain types
   - ContextDetector for automatic terrain classification
   - BrightWaterHandler with hysteresis approach
   - UrbanMaskDetector for FP reduction
   - PhysicsSafetyNet with soft constraints
   - EnsembleWaterDetector ready for LightGBM + U-Net

2. **`ensemble_water_detector.py`** - Simplified ensemble for quick testing

3. **`full_ensemble_eval.py`** - Full evaluation script that loads:
   - LightGBM v4 (lightgbm_v4_comprehensive.txt)
   - U-Net v4 (unet_v4_best.pth)
   - Physics equations
   - Weighted combination with safety net

4. **`gpu_equation_search_v2.py`** - GPU-accelerated equation search
   - 20+ equation templates
   - Terrain-adaptive physics constraints
   - Currently running on server

5. **`gee_temporal_data.js`** - GEE script for temporal SAR data
   - Monthly composites
   - Temporal statistics (mean, std, min, max)
   - Seasonal analysis (pre-monsoon, monsoon, post-monsoon, winter)
   - Water vs wet soil discrimination layer

### Running Processes on Server

```bash
# Check GPU equation search
sshpass -p 'mitaoe' ssh mit-aoe@100.84.105.5 "tail -50 ~/sar_water_detection/gpu_equation_search_v2.log"

# Check full ensemble evaluation
sshpass -p 'mitaoe' ssh mit-aoe@100.84.105.5 "tail -50 ~/sar_water_detection/full_ensemble_eval.log"

# Get final results
sshpass -p 'mitaoe' ssh mit-aoe@100.84.105.5 "cat ~/sar_water_detection/results/equation_search_summary_v2.json"
sshpass -p 'mitaoe' ssh mit-aoe@100.84.105.5 "cat ~/sar_water_detection/results/full_ensemble_results.json"
```

### Recommended Ensemble Configuration

Based on our analysis:

```python
weights = {
    "lightgbm": 0.50,  # Best single model (IoU 0.881)
    "unet": 0.30,      # Good at boundaries
    "physics": 0.20,   # Safety net
}
```

### Next Steps

1. **Collect GPU search results** - Should have more equation templates evaluated
2. **Analyze ensemble performance** - Compare to individual models
3. **Tune ensemble weights** - Grid search for optimal combination
4. **Add temporal features** - Use GEE script to download temporal data
5. **Deploy final model** - Package as production-ready system

---

**Document Version:** 2.0
**Last Updated:** 2026-01-25
**Author:** AI Research Assistant
**Status:** Implementation Phase - Scripts Running
