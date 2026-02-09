# SAR Water Detection System - Final Documentation

## Project: Physics-Guided SAR Water Detection for India

**Version:** 1.0  
**Date:** January 25, 2026  
**Status:** Production Ready  
**Best IoU:** 0.882 (88.2%)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Data Pipeline](#3-data-pipeline)
4. [Model Performance Comparison](#4-model-performance-comparison)
5. [Best Model: LightGBM + Physics](#5-best-model-lightgbm--physics)
6. [Interpretable Equation](#6-interpretable-equation)
7. [Edge Case Handlers](#7-edge-case-handlers)
8. [Known Limitations](#8-known-limitations)
9. [File Reference](#9-file-reference)
10. [Deployment Guide](#10-deployment-guide)
11. [Future Improvements](#11-future-improvements)

---

## 1. Executive Summary

### Problem Statement
Detect surface water bodies from Sentinel-1 SAR (Synthetic Aperture Radar) imagery across diverse Indian landscapes, including:
- Rivers (Ganga, Brahmaputra, Godavari, etc.)
- Lakes (Dal, Chilika, Vembanad, etc.)
- Reservoirs (Hirakud, Bhakra, Nagarjuna Sagar, etc.)
- Wetlands (Sundarbans, Kerala Backwaters, etc.)
- Seasonal water bodies

### Solution
A hybrid ML + Physics system that combines:
- **LightGBM** for pixel-level classification (74 features)
- **Physics constraints** for domain knowledge enforcement
- **Edge case handlers** for challenging scenarios

### Key Results

| Metric | Value |
|--------|-------|
| **IoU (Intersection over Union)** | **0.882** |
| **Precision** | 0.961 (96.1%) |
| **Recall** | 0.915 (91.5%) |
| **F1 Score** | 0.937 |
| **Processing Speed** | ~2.5 sec/chip (513x513 pixels) |

---

## 2. System Architecture

```
                    INPUT DATA
                        |
        +---------------+---------------+
        |               |               |
      SAR Data      DEM Data      Optical Data
    (VV, VH dB)   (DEM, SLOPE,    (MNDWI from
                  HAND, TWI)      Sentinel-2)
        |               |               |
        +-------+-------+-------+-------+
                |
        FEATURE EXTRACTION
        (74 features including
         multi-scale texture,
         gradients, morphology)
                |
        +-------+-------+
        |               |
   LightGBM v9     Physics Module
   (ML Expert)     (Domain Expert)
        |               |
        |    +----------+----------+
        |    |                     |
        | Soft Score          Hard VETO
        | (probability        (impossible
        |  adjustment)         locations)
        |    |                     |
        +----+---------------------+
                |
        COMBINED PREDICTION
        (0.9 * LGB + 0.1 * Physics)
                |
        EDGE CASE HANDLERS
        - Bright Water Correction
        - Urban Shadow Removal
                |
        MORPHOLOGICAL CLEANUP
        - Remove small regions
        - Fill holes
                |
            OUTPUT
        (Binary Water Mask)
```

---

## 3. Data Pipeline

### 3.1 Input Channels

| Channel | Source | Range | Description |
|---------|--------|-------|-------------|
| VV | Sentinel-1 | -30 to 0 dB | Co-polarization backscatter |
| VH | Sentinel-1 | -35 to -5 dB | Cross-polarization backscatter |
| DEM | SRTM/FABDEM | 0-5000 m | Digital Elevation Model |
| SLOPE | Derived | 0-90 degrees | Terrain slope |
| HAND | Derived | 0-500 m | Height Above Nearest Drainage |
| TWI | Derived | 0-30 | Topographic Wetness Index |
| MNDWI | Sentinel-2 | -1 to 1 | Modified Normalized Difference Water Index |

### 3.2 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Chips | 99 (clean dataset) |
| Chip Size | 513 x 513 pixels |
| Resolution | 10m |
| Total Pixels | ~26 million |
| Water Pixels | ~4.2 million (16%) |
| Geographic Coverage | Pan-India |

### 3.3 Critical Data Notes

```
USE THIS:     chips_expanded_npy/     (99 chips, CLEAN)
DO NOT USE:   chips/                   (86 chips, 75 CORRUPTED with bad SLOPE values)
```

The corruption in `chips/` was caused by incorrect SLOPE normalization (values in 0-1 instead of 0-90 degrees).

---

## 4. Model Performance Comparison

### 4.1 All Models Tested

| Model | IoU | Precision | Recall | Notes |
|-------|-----|-----------|--------|-------|
| **LightGBM v9 + Physics** | **0.882** | **0.961** | 0.915 | **BEST - Production Model** |
| Full Pipeline v10 (LGB only) | 0.819 | 0.963 | 0.845 | Without physics boost |
| LightGBM v9 (standalone) | 0.807 | 0.901 | 0.885 | SAR + MNDWI features |
| Attention U-Net v7 | 0.797 | - | - | Shape expert |
| LightGBM v7 (SAR only) | 0.794 | 0.896 | 0.876 | No MNDWI |
| Simple Equation | 0.761 | 0.910 | 0.823 | Interpretable rule |
| LightGBM v8 | 0.617 | - | - | Trained on corrupted data |

### 4.2 Physics Contribution

| Configuration | IoU | Improvement |
|---------------|-----|-------------|
| LGB v9 alone | 0.807 | baseline |
| LGB v9 + Physics | 0.882 | **+7.5%** |

The physics module provides a massive 7.5% IoU improvement by:
1. Eliminating false positives on mountains/cliffs
2. Boosting confidence in low-lying areas
3. Providing domain-consistent predictions

---

## 5. Best Model: LightGBM + Physics

### 5.1 LightGBM v9 Configuration

```python
params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'max_depth': 12,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'min_child_samples': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'scale_pos_weight': 3.0,  # Handle class imbalance
}
```

### 5.2 Feature Importance (Top 10)

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | VH_min_s5 | 6.9M | SAR Texture |
| 2 | VH_min_s9 | 5.6M | SAR Texture |
| 3 | MNDWI_mean_s5 | 3.1M | Optical |
| 4 | DEM | 2.2M | Terrain |
| 5 | VH_mean_s21 | 1.0M | SAR Texture |
| 6 | VH_std_s15 | 0.9M | SAR Texture |
| 7 | HAND | 0.8M | Terrain |
| 8 | VV_min_s5 | 0.7M | SAR Texture |
| 9 | TWI | 0.6M | Terrain |
| 10 | SLOPE | 0.5M | Terrain |

**Key Insight:** Multi-scale VH minimum is the strongest feature (water appears dark and smooth in VH).

### 5.3 Physics Constraints

#### Hard VETO Rules (Absolute)
```python
# Water is IMPOSSIBLE if:
veto = (HAND > 100m) OR (SLOPE > 45 degrees)
veto |= (HAND > 30m) AND (SLOPE > 20 degrees)

# Force prediction to 0
prediction = 0 if veto else prediction
```

#### Soft Scoring (Probability Adjustment)
```python
# Sigmoid-based scores
hand_score = 1 / (1 + exp((HAND - 15) / 5))
slope_score = 1 / (1 + exp((SLOPE - 12) / 4))
twi_score = 1 / (1 + exp((7 - TWI) / 2))

# Combined physics score
physics_score = 0.4 * hand_score + 0.4 * slope_score + 0.2 * twi_score

# Final prediction
final = 0.9 * lgb_prediction + 0.1 * physics_score
```

---

## 6. Interpretable Equation

For situations requiring explainability, use this simple rule:

```
WATER = (MNDWI > 0.129) OR (VH < -19.6 dB AND MNDWI > -0.3)
```

### Performance
- **IoU:** 0.761
- **Precision:** 91.0%
- **Recall:** 82.3%

### Interpretation
1. **Primary Rule:** If optical water index (MNDWI) is strongly positive, it's water
2. **Secondary Rule:** If SAR shows very dark backscatter (VH < -19.6 dB) AND optical doesn't strongly indicate land (MNDWI > -0.3), it's water

This equation can be used for:
- Documentation and regulatory compliance
- Quick approximate detection
- Understanding model behavior

---

## 7. Edge Case Handlers

### 7.1 Bright Water Handler

**Problem:** Wind-roughened water appears brighter in SAR, causing false negatives.

**Solution:** Adaptive region growing from confident water seeds.

```python
# Configuration
vh_calm_threshold = -18.0 dB      # Normal calm water
vh_bright_max = -10.0 dB          # Maximum for wind-roughened
texture_threshold = 1.5           # Low texture = smooth surface

# Logic
seeds = (water_mask) AND (lgb_proba > 0.7)
ambiguous = (vh >= -18) AND (vh < -10) AND (texture < 1.5)
# Region grow from seeds into ambiguous areas
```

### 7.2 Urban Shadow Handler

**Problem:** Building shadows appear dark like water, causing false positives.

**Solution:** VV/VH ratio + texture analysis.

```python
# Urban shadows have:
# 1. High VV-VH difference (double-bounce nearby)
# 2. High local variance (heterogeneous)
# 3. Bright neighbors (buildings)

urban_mask = (vv - vh > 6 dB) OR (variance > 8)
urban_mask &= (max_filter(vv, 9) > -8 dB)  # Bright nearby

# Remove from water prediction
water_mask = water_mask AND NOT urban_mask
```

### 7.3 Morphological Cleanup

```python
# Remove small regions (noise)
min_region_size = 50 pixels  # ~5000 sq meters at 10m resolution

# Fill small holes
dilate(1 iteration) -> erode(1 iteration)
```

---

## 8. Known Limitations

### 8.1 Challenging Chip Categories

| Category | Example Chips | IoU | Issue |
|----------|--------------|-----|-------|
| Seasonal/Dry | Chambal, Thar | 0.0 | No water present |
| Small Tanks | Deccan Tanks | 0.0-0.23 | Below detection limit |
| Flood Events | Kosi, Assam | 0.02-0.37 | Temporal mismatch |
| Braided Rivers | Brahmaputra | 0.01-0.96 | Complex morphology |
| Mangroves | Sundarbans | 0.53-0.72 | Mixed water/vegetation |

### 8.2 Failure Modes

1. **Very Small Water Bodies** (< 50 pixels / 5000 sq m)
   - Below minimum detectable area
   - Solution: Reduce `min_region_size` for specific use cases

2. **Seasonal Wetlands**
   - Ground truth may be from different season than SAR
   - Solution: Temporal consistency checking

3. **Turbid/Sediment-laden Water**
   - Higher backscatter than clear water
   - Solution: Lower VH threshold in known turbid areas

4. **Floating Vegetation**
   - Aquatic plants cause high backscatter
   - Solution: Requires vegetation index integration

### 8.3 Geographic Limitations

The model was trained on Indian data and may not generalize to:
- Different climate zones
- Different water body types (glacial lakes, etc.)
- Different SAR acquisition geometries

---

## 9. File Reference

### 9.1 Server Location
```
Server: mit-aoe@100.84.105.5
Password: mitaoe
Directory: ~/sar_water_detection/
```

### 9.2 Model Files

| File | Description | IoU |
|------|-------------|-----|
| `models/lightgbm_v9_clean_mndwi.txt` | **Production Model** | 0.882* |
| `models/attention_unet_v7_best.pth` | Attention U-Net | 0.797 |
| `models/lightgbm_v7_clean.txt` | SAR-only baseline | 0.794 |

*With physics constraints

### 9.3 Script Files

| File | Purpose |
|------|---------|
| `test_ensemble.py` | **Best results** - LGB + Physics ensemble |
| `full_pipeline_v10.py` | Complete detection pipeline |
| `retrain_v9.py` | Training script for LightGBM v9 |
| `attention_unet_v7.py` | Attention U-Net training |
| `equation_search_v3_clean.py` | GPU equation search |
| `edge_case_handlers.py` | Bright water + urban handlers |
| `final_ensemble_v11.py` | Three-expert ensemble (experimental) |

### 9.4 Data Directories

| Directory | Contents | Status |
|-----------|----------|--------|
| `chips_expanded_npy/` | 99 clean chips (.npy) | USE THIS |
| `chips_expanded/` | Original TIF files | Reference |
| `chips/` | 86 chips (corrupted) | DO NOT USE |
| `results/` | All experiment results | Reference |

### 9.5 Results Files

| File | Contents |
|------|----------|
| `results/ensemble_test_results.json` | Best ensemble results |
| `results/retrain_v9_results_mndwi.json` | LGB v9 training results |
| `results/attention_unet_v7_results.json` | U-Net training results |
| `results/equation_search_v3.log` | Equation search results |
| `results/full_pipeline/` | Per-chip results + difference maps |

---

## 10. Deployment Guide

### 10.1 Quick Start

```bash
# SSH to server
sshpass -p 'mitaoe' ssh mit-aoe@100.84.105.5

# Activate environment
source ~/anaconda3/bin/activate
cd ~/sar_water_detection

# Run detection on all chips
python test_ensemble.py

# Or use full pipeline with edge case handlers
python full_pipeline_v10.py --chip_dir chips_expanded_npy/
```

### 10.2 Single Chip Inference

```python
import numpy as np
import lightgbm as lgb

# Load model
model = lgb.Booster(model_file='models/lightgbm_v9_clean_mndwi.txt')

# Load chip (channels: VV, VH, DEM, SLOPE, HAND, TWI, LABEL, MNDWI)
chip = np.load('chips_expanded_npy/chip_001_mumbai_harbor_with_truth.npy')

# Extract features (use extract_features function from full_pipeline_v10.py)
features = extract_features({
    'vv': chip[:,:,0],
    'vh': chip[:,:,1],
    'dem': chip[:,:,2],
    'slope': chip[:,:,3],
    'hand': chip[:,:,4],
    'twi': chip[:,:,5],
    'mndwi': chip[:,:,7],
})

# Predict
X = features.reshape(-1, 74)
proba = model.predict(X).reshape(chip.shape[0], chip.shape[1])

# Apply physics constraints
veto = (chip[:,:,4] > 100) | (chip[:,:,3] > 45)
proba = np.where(veto, 0, proba)

# Threshold
water_mask = (proba > 0.5).astype(np.uint8)
```

### 10.3 Production Checklist

- [ ] Use `chips_expanded_npy/` data only
- [ ] Load `lightgbm_v9_clean_mndwi.txt` model
- [ ] Apply physics VETO (HAND > 100 OR SLOPE > 45)
- [ ] Apply soft physics scoring (0.9 LGB + 0.1 physics)
- [ ] Set threshold at 0.5
- [ ] Remove regions < 50 pixels
- [ ] Validate on known water bodies

---

## 11. Future Improvements

### 11.1 Short-term

1. **Retrain U-Net on Full Chips**
   - Current: Trained on 256x256 crops, tested on 513x513
   - Action: Train on full 513x513 with proper augmentation

2. **Temporal Consistency**
   - Add multi-temporal SAR for flood detection
   - Use time-series to detect seasonal changes

3. **Active Learning**
   - Identify and label failing chips (Chambal, Thar, etc.)
   - Retrain with expanded training set

### 11.2 Medium-term

1. **Ensemble with Learned Weights**
   - Train a meta-model to optimally combine LGB + U-Net
   - Per-chip or per-region weighting

2. **Uncertainty Quantification**
   - Output confidence maps alongside predictions
   - Flag low-confidence regions for human review

3. **Cloud-Native Deployment**
   - Package as Docker container
   - Deploy on AWS/GCP for scalable processing

### 11.3 Long-term

1. **Foundation Model Integration**
   - Fine-tune SAR foundation models (e.g., Prithvi)
   - Transfer learning from global water datasets

2. **Multi-Sensor Fusion**
   - Integrate Landsat, MODIS for temporal density
   - Add Sentinel-3 for coastal waters

3. **Real-Time Flood Monitoring**
   - Near-real-time processing pipeline
   - Alert system for flood events

---

## Appendix A: Feature List (74 Features)

| # | Feature | Description |
|---|---------|-------------|
| 1-2 | VV, VH | Raw backscatter |
| 3 | VV/VH | Ratio |
| 4 | VV-VH | Difference |
| 5 | NDWI_SAR | (VV-VH)/(VV+VH) |
| 6 | RVI | 4*VH/(VV+VH) |
| 7-46 | Multi-scale texture | Mean, std, min, max at scales 3,5,9,15,21 for VV & VH |
| 47-50 | Gradients | VV/VH gradient magnitude |
| 51-52 | Laplacian | VV/VH laplacian |
| 53-56 | Morphology | Opening/closing VV/VH |
| 57-58 | Otsu-like | VV/VH minus median |
| 59-60 | Local contrast | VV/VH minus local mean |
| 61-64 | GLCM-like | Contrast, homogeneity for VV/VH |
| 65 | Pseudo-entropy | VV entropy approximation |
| 66-69 | Terrain | DEM, SLOPE, HAND, TWI |
| 70-72 | Physics scores | Sigmoid(HAND), Sigmoid(SLOPE), Sigmoid(TWI) |
| 73-74 | MNDWI | Raw + binary + mean + std |

---

## Appendix B: Per-Chip Results (Top Performers)

| Chip | Location | IoU | Precision | Recall |
|------|----------|-----|-----------|--------|
| chip_080 | Chilika Wetland 2 | 0.992 | 1.000 | 0.992 |
| chip_083 | Rann Kutch 2 | 0.992 | 1.000 | 0.992 |
| chip_033 | Gujarat Gulf Kutch | 0.992 | 1.000 | 0.992 |
| chip_062 | Hirakud 2 | 0.991 | 1.000 | 0.991 |
| chip_017 | Kerala Vembanad 2 | 0.985 | 0.999 | 0.986 |
| chip_018 | Kerala Alleppey | 0.976 | 0.999 | 0.977 |
| chip_036 | Tso Moriri | 0.975 | 0.995 | 0.980 |
| chip_081 | Vembanad Kol | 0.973 | 0.992 | 0.980 |
| chip_079 | Chilika Wetland 1 | 0.970 | 1.000 | 0.970 |
| chip_051 | Brahmaputra Majuli 2 | 0.963 | 0.982 | 0.981 |

---

## Appendix C: Contact & Resources

**Project Repository:** Local at `/media/neeraj-parekh/Data1/sar soil system/`

**Server Access:**
```
Host: 100.84.105.5
User: mit-aoe
Password: mitaoe
GPU: NVIDIA RTX A5000 (24GB)
```

**Key Dependencies:**
- Python 3.8+
- LightGBM 3.3+
- PyTorch 2.0+
- NumPy, SciPy, Rasterio

---

*Document generated: January 25, 2026*
*Version: 1.0 - Production Ready*
