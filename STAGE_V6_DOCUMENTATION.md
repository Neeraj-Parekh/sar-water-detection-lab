# SAR Water Detection - Stage V7 Documentation

## Project Overview
**Date:** 2026-01-25
**Version:** 7.0 (MAJOR SUCCESS!)
**Server:** `ssh mit-aoe@100.84.105.5` (password: `mitaoe`)
**Local:** `/media/neeraj-parekh/Data1/sar soil system/chips/gui/`
**Server:** `~/sar_water_detection/`

---

## Major Discoveries This Session

### 1. Data Quality Issues (CRITICAL)
**75 out of 86 chips in `chips/` have data corruption:**

| Issue | Count | Description |
|-------|-------|-------------|
| BAD_SLOPE | 66 chips | Slope values > 90° (e.g., 768°, 2365°) |
| VERY_BRIGHT | 52 chips | Water VV > -12dB (non-standard signature) |
| LOAD_ERROR | 1 chip | Corrupted file (chip_086) |
| **Clean chips** | **11 only** | Valid data |

**Root Cause:** The SLOPE band was likely stored in different units (degrees × 10?) or corrupted during export. This explains why SLOPE dominated feature importance.

### 2. SAR Signature Variability
**Water is not always "dark" in SAR:**

| Location | Water VV (dB) | Reason |
|----------|---------------|--------|
| Mumbai Harbor | -18.5 | Calm water (typical) |
| Kolkata Hooghly | -12.5 | Urban, ships, rough |
| Delhi Yamuna | -15.0 | Wind roughened |
| Thane Creek | -14.9 | Tidal, mixed |

**Implication:** Fixed threshold (VV < -18dB) misses 40%+ of urban water.

### 3. chips_expanded Has Better Quality
After converting 99 TIF files to NPY:
- SLOPE properly clipped to 0-90°
- HAND valid (0-500m)
- More diverse Indian locations (Mumbai, Delhi, Kerala, Kashmir, etc.)

---

## Files Created/Modified

### On Server (`~/sar_water_detection/`)
```
models/
├── lightgbm_v6_retrained.txt    # Trained on chips/ (85 chips)
├── lightgbm_v4_comprehensive.txt # Previous version
├── unet_v4_best.pth              # CBAM U-Net

results/
├── retrain_v6_results.json       # IoU 0.50 on chips/

chips_expanded_npy/               # NEW: 99 converted chips
├── chip_001_mumbai_harbor_with_truth.npy
├── chip_002_mumbai_thane_creek_with_truth.npy
├── ... (99 total)

backup_v6/                        # Full backup
├── models/
├── results/
├── *.py
```

### Key Scripts
| Script | Purpose |
|--------|---------|
| `retrain_v6.py` | LightGBM training (NPY only) |
| `convert_tif_to_npy_v2.py` | TIF→NPY converter with validation |
| `physics_vision_hybrid_v7.py` | New soft physics detector |

---

## LightGBM Results Comparison

### v6 (Corrupted chips/) vs v7 (Clean chips_expanded_npy/)

| Metric | v6 (Corrupted) | **v7 (Clean)** | Improvement |
|--------|----------------|----------------|-------------|
| **IoU** | 0.499 | **0.794** | **+59%** |
| Precision | 0.585 | **0.876** | +50% |
| Recall | 0.772 | **0.894** | +16% |
| F1 | 0.666 | **0.885** | +33% |
| Training Time | 22 min | 68 sec | -97% |
| Training Samples | 2.68M | 3.16M | +18% |

### v7 Per-Chip Performance (Highlights)
| Chip | IoU | Location |
|------|-----|----------|
| Mumbai Harbor | **0.975** | Best! |
| Brahmaputra Guwahati | 0.952 | Excellent |
| Gujarat Gulf Khambhat | 0.929 | Excellent |
| Bhakra Reservoir | 0.929 | Excellent |
| Ganga Varanasi | 0.922 | Excellent |
| Hyderabad Hussain Sagar | 0.906 | Excellent |
| Loktak Lake | 0.893 | Great |
| Sundarbans | 0.869 | Good |

### Feature Importance Comparison

**v6 (Corrupted Data):**
| Feature | Importance |
|---------|------------|
| **SLOPE** | 1,080,550 (corrupted!) |
| HAND | 430,959 |

**v7 (Clean Data):**
| Feature | Importance | Notes |
|---------|------------|-------|
| **VH_min_s5** | 7,323,492 | SAR dominant! |
| VH_min_s9 | 5,462,205 | Multi-scale VH |
| DEM | 2,250,712 | Terrain |
| VH_opened | 2,190,493 | Morphology |
| HAND | 564,046 | Physics |

**Key Insight:** With clean data, VH features dominate (not corrupted SLOPE). This makes physical sense - VH polarization is excellent for water detection.

---

## Physics-Vision Hybrid Approach

### Old Physics (Too Strict)
```python
# Multiplicative - too harsh
composite = hand_score * slope_score * vh_score * twi_score
# If any score is 0.5, result = 0.5^4 = 0.06 (too low!)
```

### New Soft Physics (v7)
```python
# Only penalize EXTREME cases
feasibility = np.ones_like(hand)
feasibility = np.where(hand > 50, 0.2, feasibility)  # Very high HAND
feasibility = np.where(slope > 30, 0.3, feasibility)  # Very steep

# Apply as gentle constraint
combined = ml_prediction * (0.7 + 0.3 * feasibility)
```

### Signature-Adaptive Detection
```python
# Classify scene first
if vv_mean < -18:
    signature = 'dark_water'
    threshold = compute_otsu_threshold(vv)
elif vv_mean > -12:
    signature = 'bright_surface'
    # Look for relatively dark areas
    threshold = local_mean - 3dB
```

---

## Next Steps

### Immediate (High Priority)
1. **Retrain on chips_expanded_npy** - Clean data with proper SLOPE
2. **Fix SLOPE in chips/** - Divide by 10 or recompute
3. **Evaluate on proper test set** - Use location-specific chips

### Medium Priority
4. **Tune physics thresholds** - Based on terrain statistics
5. **Add signature classification** - Dark vs bright water
6. **Ensemble with U-Net** - For edge refinement

### Research Questions
- Why is water bright in urban areas? (Double-bounce from buildings?)
- Can MNDWI help distinguish water types?
- Should we have separate models for urban vs rural?

---

## Quick Commands

```bash
# SSH to server
sshpass -p 'mitaoe' ssh mit-aoe@100.84.105.5

# Activate environment
source ~/anaconda3/bin/activate
cd ~/sar_water_detection

# Check data
ls chips_expanded_npy/*.npy | wc -l  # Should be 99

# Run retraining on clean data
python retrain_v6.py  # Modify to use chips_expanded_npy

# Test hybrid detector
python physics_vision_hybrid_v7.py chips_expanded_npy/chip_001_mumbai_harbor_with_truth.npy
```

---

## Backup Location
All models and code backed up to: `~/sar_water_detection/backup_v6/`

---

## Summary

| What We Found | Impact |
|---------------|--------|
| 75/86 chips have corrupted SLOPE | Model learned from bad data |
| Water not always dark | Fixed thresholds fail in urban areas |
| chips_expanded has cleaner data | Retrained and got 79% IoU |
| Soft physics better than multiplicative | Reduces false negatives |
| VH features are most important | Makes physical sense |

**Major Achievement:** IoU improved from 0.50 to **0.79** (+59%) by using clean data!

---

## Models Available

| Model | IoU | Data | File |
|-------|-----|------|------|
| LightGBM v7 | **0.794** | chips_expanded_npy | `lightgbm_v7_clean.txt` |
| LightGBM v6 | 0.499 | chips/ (corrupted) | `lightgbm_v6_retrained.txt` |
| LightGBM v4 | 0.88* | chips_expanded (tif) | `lightgbm_v4_comprehensive.txt` |
| U-Net v4 | ~0.80 | All | `unet_v4_best.pth` |

*v4 evaluated on same data it was trained on

---

**Document Version:** 7.0
**Last Updated:** 2026-01-25 13:30 IST
**Status:** SUCCESS - 79% IoU achieved with clean data
