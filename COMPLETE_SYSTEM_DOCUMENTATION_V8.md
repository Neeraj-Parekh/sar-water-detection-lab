# SAR Water Detection - Complete System Documentation V8

## Project Summary
**Date:** 2026-01-25
**Location:** `~/sar_water_detection/` on server `mit-aoe@100.84.105.5`
**Best Model:** LightGBM v7 (IoU 0.79 on clean data)

---

## Executive Summary

### What We Achieved
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| IoU | 0.21 (buggy) | **0.79** | +276% |
| Data Quality | 75/86 corrupted | 99/99 clean | Fixed |
| Physics | Multiplicative (harsh) | Soft + VETO | Better recall |

### Key Discoveries
1. **Data corruption** in chips/ (SLOPE values up to 2365°)
2. **SAR signature variability** (water not always dark)
3. **VH polarization** most important feature (not SLOPE)
4. **Physics as VETO** better than multiplicative

---

## Version History

| Ver | Date | Model | IoU | Dataset | Issues |
|-----|------|-------|-----|---------|--------|
| v1 | Jan 24 | LightGBM baseline | ~0.40 | chips/ | Basic features |
| v2 | Jan 24 | LightGBM pixel | ~0.50 | chips/ | Per-pixel |
| v3 | Jan 24 | LightGBM SAR-only | ~0.60 | chips/ | Texture features |
| v4 | Jan 25 | LightGBM + U-Net | 0.88* | chips_expanded | Comprehensive |
| v5 | Jan 25 | Blocked | - | - | numpy conflict |
| v6 | Jan 25 | LightGBM retrained | 0.50 | chips/ | Corrupted SLOPE |
| v7 | Jan 25 | LightGBM clean | **0.79** | chips_expanded_npy | Clean data |
| v8 | Jan 25 | Vision+Physics | TBD | All | Unified system |

*Evaluated on training data

---

## File Archive Structure

```
~/sar_water_detection/
├── archive/
│   ├── v1_baseline/
│   │   ├── master_training_pipeline.py
│   │   └── lightgbm_baseline.txt
│   ├── v2_pixel/
│   │   ├── master_training_pipeline_v2.py
│   │   └── lightgbm_pixel.txt
│   ├── v3_sar_only/
│   │   ├── master_training_pipeline_v3.py
│   │   └── lightgbm_v3_sar_only.txt
│   ├── v4_comprehensive/
│   │   ├── master_training_pipeline_v4.py
│   │   ├── lightgbm_v4_comprehensive.txt
│   │   ├── unet_v4_standalone.py
│   │   └── ensemble_water_detector.py
│   ├── v5_retrain/
│   │   ├── retrain_v5.py
│   │   ├── ensemble_v2.py
│   │   └── unet_v5_improved.py
│   ├── v6_corrupted/
│   │   ├── retrain_v6.py
│   │   ├── lightgbm_v6_retrained.txt
│   │   ├── ensemble_v3.py
│   │   └── unet_v6_fixed.py
│   └── v7_clean/
│       ├── retrain_v7.py
│       ├── lightgbm_v7_clean.txt
│       └── *.json (all results)
├── models/
│   └── lightgbm_v7_clean.txt (BEST)
├── chips_expanded_npy/ (99 clean chips)
├── chips/ (86 chips, 75 corrupted)
└── unified_vision_physics_v8.py
```

---

## Failure Case Analysis

### v7 Test Results (19 chips)
| Chip | IoU | Issue |
|------|-----|-------|
| Mumbai Harbor | 0.975 | None |
| Brahmaputra | 0.952 | None |
| Gujarat Gulf | 0.929 | None |
| ... (15 more good) | >0.5 | None |
| **Sambhar Lake** | 0.449 | LOW_SEPARATION |
| **Bhitarkanika** | 0.468 | HIGH_HAND (23m) |
| **Deccan Tank** | 0.000 | BRIGHT_WATER, 0% water |
| **Assam Flood 2** | 0.000 | 0% water in GT |

### Root Causes of Failures
1. **No water in ground truth** (2 chips) - Not a model issue
2. **Low SAR separation** - Water similar to land brightness
3. **High HAND in water** - DEM mismatch or tidal areas
4. **Bright water** - Urban double-bounce

---

## Physics Constraints

### Old Approach (Too Harsh)
```python
# Multiplicative - destroys predictions
composite = hand_score * slope_score * vh_score * twi_score
# 0.9 × 0.9 × 0.9 × 0.9 = 0.66 (too low!)
```

### New Approach (VETO + Soft)

#### Hard Physics VETO (Water Impossible)
```python
# These conditions VETO water detection
impossible = (hand > 100) | (slope > 45) | ((hand > 30) & (slope > 20))
```

#### Soft Physics (Weighting)
```python
# Sigmoid-based scores with relaxed thresholds
hand_score = 1 / (1 + exp((hand - 20) / 10))   # 50% at 20m
slope_score = 1 / (1 + exp((slope - 15) / 5))  # 50% at 15°
twi_score = 1 / (1 + exp((6 - twi) / 2))       # 50% at TWI=6

# Weighted combination (not multiplicative!)
physics_score = 0.5 * hand_score + 0.3 * slope_score + 0.2 * twi_score

# Apply as gentle constraint
final = ml_prediction * (0.7 + 0.3 * physics_score)
```

---

## Data Validation System

### Automatic Checks
| Field | Valid Range | Action |
|-------|-------------|--------|
| SLOPE | 0-90° | Clip if > 90 |
| HAND | 0-500m | Clip if > 500 |
| VV | -40 to +10 dB | Warn only |
| VH | -45 to +5 dB | Warn only |
| TWI | 0-30 | Fill NaN with 5 |

### Quality Report
```python
DataQualityReport:
  is_valid: bool
  issues: List[str]  # ['SLOPE_OVERFLOW', 'VV_NAN', ...]
  corrections_applied: Dict[str, str]
  confidence: float  # 0-1
```

---

## Vision + Physics Integration

### Architecture
```
Input Data
    │
    ▼
┌─────────────────┐
│ Data Validator  │ ──→ Quality Report
└────────┬────────┘
         │
    ▼────┴────▼
┌────────┐  ┌────────┐
│ ML/LGB │  │ U-Net  │
└───┬────┘  └───┬────┘
    │           │
    ▼───────────▼
┌─────────────────┐
│  Ensemble       │ (0.7 ML + 0.3 UNet for edges)
└────────┬────────┘
         │
    ▼────┴────▼
┌────────┐  ┌────────────┐
│ Soft   │  │ Hard VETO  │
│Physics │  │ (impossible)│
└───┬────┘  └───┬────────┘
    │           │
    ▼───────────▼
┌─────────────────┐
│ Final Mask      │
└─────────────────┘
```

### When Vision vs Physics Wins

| Scenario | Winner | Why |
|----------|--------|-----|
| Normal water | ML (LightGBM) | Learned patterns |
| Edge pixels | U-Net | Spatial context |
| Cliff/mountain | Physics VETO | Impossible |
| Urban bright | Vision | Physics confused |
| Tidal flats | Vision | HAND varies |

---

## Best Equations Found

### Simple (Interpretable)
```python
# VH Simple (IoU ~0.65)
water = VH < -23

# HAND Constrained (IoU ~0.70)
water = (VH < -22) AND (HAND < 10)

# Full Physics (IoU ~0.72)
water = (VH < -21) AND (HAND < 15) AND (slope < 10) AND (TWI > 6)
```

### Learned (PySR)
```python
# Adaptive VH+TWI (IoU ~0.75)
water = sigmoid((-VH - 20) / 2 + (TWI - 8) / 3)
```

### ML Features (Top 10)
| Feature | Importance |
|---------|------------|
| VH_min_s5 | 7,323,492 |
| VH_min_s9 | 5,462,205 |
| DEM | 2,250,712 |
| VH_opened | 2,190,493 |
| VH_mean_s21 | 887,000 |
| VH_min_s21 | 754,528 |
| VV_mean_s9 | 610,923 |
| HAND | 564,046 |
| VV_max_s21 | 422,411 |
| VH_max_s21 | 382,183 |

---

## Future Improvements

### Immediate
1. **Ensemble LightGBM + U-Net** - Proper weighted combination
2. **Train on combined data** - chips/ + chips_expanded_npy/
3. **Fix chips/ SLOPE** - Divide by factor or recompute

### Medium Term
4. **Add MNDWI** - Optical water index from TIF
5. **Temporal consistency** - For flood monitoring
6. **Separate urban model** - Handle bright water

### Long Term
7. **Foundation model** - Pre-trained SAR encoder
8. **Active learning** - Flag uncertain predictions
9. **Real-time inference** - GPU optimization

---

## Quick Commands

```bash
# SSH to server
sshpass -p 'mitaoe' ssh mit-aoe@100.84.105.5

# Activate environment
source ~/anaconda3/bin/activate
cd ~/sar_water_detection

# Run v7 model (best)
python retrain_v7.py

# Test unified detector
python unified_vision_physics_v8.py

# Check data quality
python -c "from unified_vision_physics_v8 import DataValidator; print(DataValidator.validate_and_correct(...))"
```

---

## Lessons Learned

### Data Quality
1. **Always validate data ranges** before training
2. **SLOPE > 90° is impossible** - Flag immediately
3. **Compare distributions** between datasets

### Model Design
1. **Physics as VETO** not penalty
2. **VH more reliable** than VV for water
3. **Soft constraints** for marginal cases

### Process
1. **Archive every version** with models + code
2. **Document equations** found
3. **Track what dataset** each model was trained on

---

**Document Version:** 8.0
**Last Updated:** 2026-01-25 14:00 IST
**Status:** Complete system with 0.79 IoU, ready for ensemble
