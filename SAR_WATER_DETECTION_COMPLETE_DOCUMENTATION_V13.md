# SAR Water Detection - Complete Documentation v13

## Project Overview
**Project:** Physics-Guided SAR Water Detection System for India
**Local Directory:** `/media/neeraj-parekh/Data1/sar soil system/chips/gui/`
**Server:** `ssh mit-aoe@100.84.105.5` (password: `mitaoe`)
**Server Directory:** `~/sar_water_detection/`
**GPU:** NVIDIA RTX A5000 (24GB VRAM)
**Date:** 2026-01-26

---

## Executive Summary

After extensive experimentation, **LightGBM v9 remains the best model** at IoU 0.882. Key findings:

| Model/Method | IoU | Status |
|--------------|-----|--------|
| **LightGBM v9** | **0.882** | ✅ BEST - Production Ready |
| LightGBM + U-Net v9 Ensemble | 0.881 | Same as LGB alone |
| U-Net v9 SOTA (Attention Gates) | 0.685 | Trained |
| U-Net v10 Extended (CBAM) | 0.604 | Failed - wrong architecture |
| Simple Equation | 0.761 | Backup option |

---

## Session Timeline (Jan 25-26, 2026)

### Phase 1: SOTA Module Creation
Created `sota_research_module.py` with research-grade implementations:

| Component | Implementation | Reference |
|-----------|----------------|-----------|
| **Frangi Vesselness** | Multi-scale Hessian eigenvalue analysis | Frangi et al., MICCAI 1998 |
| **CenterlineDiceLoss** | Zhang-Suen skeletonization + skeleton Dice | Zhang & Suen 1984 |
| **MST River Connector** | Prim's algorithm with KD-tree acceleration | Graph theory |
| **GMM Auto-Threshold** | 2-component Gaussian Mixture for adaptive thresholds | Per-chip dynamic |
| **Mamdani Fuzzy Controller** | IF-THEN rules with triangular membership functions | Mamdani 1974 |
| **Gamma0 Normalization** | Incidence angle correction: γ0 = σ0/cos(θ) | ESA Sentinel-1 |

### Phase 2: U-Net v9 Training
**File:** `attention_unet_v9_sota.py`

**Architecture:**
- 9 input channels (VV, VH, DEM, Slope, HAND, TWI, MNDWI, Physics, Frangi)
- ResidualBlocks with skip connections
- AttentionGates for decoder
- Skeleton Loss for topology preservation

**Training Results:**
```
Best IoU: 0.6854
Training Time: 8.4 minutes
Epochs: 29 (early stopped)
GMM Auto-Threshold: -20.52 dB (water: -27.33, land: -13.72)
Model saved: models/attention_unet_v9_sota_best.pth
```

### Phase 3: U-Net v10 Training (FAILED)
**File:** `attention_unet_v10_extended.py`

**Problem:** Used CBAM architecture instead of AttentionGates
**Result:** IoU 0.604 (worse than v9's 0.685)
**Lesson:** v9's architecture (ResidualBlocks + AttentionGates) is superior

```
Best IoU: 0.6037
Epochs: 52 (early stopped at patience 25)
Training Time: 13.2 minutes
```

### Phase 4: Ensemble Testing
**File:** `test_ensemble_v9_correct.py`

Tested: LightGBM v9 (0.882) + U-Net v9 (0.685) + Physics

**Results:**
| Metric | LGB Alone | Ensemble |
|--------|-----------|----------|
| IoU | 0.8821 | 0.8813 |
| Precision | 0.9616 | 0.9570 |
| Recall | 0.9143 | 0.9177 |

**Conclusion:** Adding weaker U-Net doesn't improve LightGBM results.

### Phase 5: Problem Chip Analysis
**File:** `analyze_problem_chips.py`

Analyzed 37 problem chips (IoU < 0.7):

| Category | Count | Cause | Fix |
|----------|-------|-------|-----|
| MINOR_ISSUES | 11 | IoU 0.5-0.7 | MST gap healing |
| POOR_SEPARATION | 8 | Water/land similar | Texture features |
| FLOOD_DYNAMICS | 8 | Temporal mismatch | Exclude or temporal labels |
| SEASONAL_DRY | 4 | Dry riverbeds | Exclude |
| MINIMAL_WATER | 3 | <0.1% water | Exclude |
| MODERATE_FAILURE | 2 | Unknown | Investigate |
| MISALIGNED | 1 | SAR-DEM shift | SAR-only features |

### Phase 6: MST Post-Processing Test (In Progress)
**File:** `test_mst_postprocessing.py`

Testing different MST configurations:
- gap10_vh-16.0: Avg improvement +0.0003
- gap15_vh-16.0: Avg improvement +0.0005 (best so far)
- gap20_vh-16.0: Testing...

**Key Finding:** MST helps some chips (chip_046_ganga_varanasi_2: +0.019 IoU) but can hurt others with false connections.

---

## All Models on Server

### U-Net Models
```
models/attention_unet_v7_best.pth      # IoU: 0.797 (crop-trained, breaks connectivity)
models/attention_unet_v8_fullchip_best.pth  # IoU: 0.687
models/attention_unet_v9_sota_best.pth      # IoU: 0.685 (9 channels, best architecture)
models/unet_v4_best.pth
models/unet_v5_best.pth
models/unet_v6_best.pth                # IoU: ~0.65 (6 channels)
```

### LightGBM Models
```
models/lightgbm_v9_clean_mndwi.txt     # IoU: 0.882 ⭐ BEST
models/lightgbm_v8_ensemble_mndwi.txt  # IoU: ~0.85
models/lightgbm_v7_clean.txt
models/lightgbm_v6_retrained.txt
```

---

## All Scripts Created

### Training Scripts
| Script | Purpose | Status |
|--------|---------|--------|
| `attention_unet_v9_sota.py` | U-Net v9 with SOTA features | ✅ Trained |
| `attention_unet_v10_extended.py` | Extended training (100 epochs) | ❌ Wrong architecture |

### Testing Scripts
| Script | Purpose | Status |
|--------|---------|--------|
| `test_ensemble_memsafe.py` | LightGBM only test | ✅ Completed |
| `test_ensemble_v9_correct.py` | Full ensemble test | ✅ Completed |
| `test_mst_postprocessing.py` | MST gap healing test | 🔄 Running |

### Analysis Scripts
| Script | Purpose | Status |
|--------|---------|--------|
| `analyze_problem_chips.py` | Problem chip categorization | ✅ Completed |
| `data_validator_v2.py` | Data integrity checks | ✅ Available |
| `sota_research_module.py` | SOTA implementations | ✅ Available |

---

## SOTA Components Status

### Implemented in `sota_research_module.py`

1. **Frangi Vesselness Filter** ✅
   - Multi-scale Hessian eigenvalue analysis
   - Highlights tube-like structures (rivers)
   - Used as 9th input channel in U-Net v9

2. **CenterlineDiceLoss** ✅
   - Zhang-Suen skeletonization
   - Penalizes broken river continuity
   - Used in U-Net v9 training

3. **MST River Connector** ✅
   - Graph-based gap healing
   - Tests show marginal improvement (+0.0005 avg)
   - Best config: gap15, vh_thresh=-16.0

4. **GMM Auto-Threshold** ✅
   - Per-chip adaptive thresholds
   - Finds optimal valley between water/land modes
   - Used in problem chip analysis

5. **Mamdani Fuzzy Controller** ✅
   - IF-THEN rules for soft decisions
   - Handles edge cases
   - Not yet integrated into final pipeline

6. **Gamma0 Normalization** ✅
   - Incidence angle correction
   - Not tested yet (data may already be normalized)

### Data Integrity Checks (in `data_validator_v2.py`)

1. **Co-registration Check** ✅
   - SAR-DEM alignment verification
   - Found 1 misaligned chip (chip_002)

2. **Label Sanity Score** ✅
   - Detects mislabeled water (too bright)
   - Checks median VH of water pixels

3. **Striping Detection** ✅
   - Detects burst boundaries
   - Checks for sudden intensity drops

---

## Data Summary

### India Chips Dataset
- **Total chips:** 99
- **Good chips (IoU ≥ 0.7):** 62
- **Problem chips (IoU < 0.7):** 37
- **Recommended exclusions:** 15 (temporal/seasonal/minimal water)

### Known Corrupted Chips
```
chip_033_gujarat_gulf_kutch  - SLOPE corruption
chip_062_hirakud_2           - SLOPE corruption  
chip_080_chilika_wetland_2   - SLOPE corruption
chip_067_stanley_reservoir   - SAR-DEM misalignment
chip_095_assam_flood_1       - SAR-DEM misalignment
chip_096_assam_flood_2       - SAR-DEM misalignment
```

---

## Key Discoveries

### 1. LightGBM Dominates Deep Learning
- LightGBM v9: IoU 0.882
- Best U-Net: IoU 0.685
- **Gap:** 20% - LightGBM is far superior for this task

### 2. Ensemble Doesn't Help When Models Are Imbalanced
- Mixing 0.882 (LGB) with 0.685 (U-Net) = 0.881
- Weaker model dilutes stronger one
- Only ensemble if models have similar accuracy

### 3. Architecture Matters for U-Net
- AttentionGates (v9) > CBAM (v10)
- ResidualBlocks essential for gradient flow
- 9 channels with Frangi vesselness helps narrow rivers

### 4. MST Gap Healing Has Limited Impact
- Average improvement: +0.0005 IoU
- Helps some chips (rivers), hurts others (false connections)
- Best for narrow river chips specifically

### 5. Problem Chips Fall into Clear Categories
- 30% are MINOR_ISSUES (fixable)
- 40% are TEMPORAL/DATA issues (exclude from training)
- 20% are POOR_SEPARATION (need texture features)

---

## Recommended Final Pipeline

Based on all experiments:

```python
# PRODUCTION PIPELINE

# 1. Load LightGBM v9 (best model)
model = lgb.Booster(model_file="lightgbm_v9_clean_mndwi.txt")

# 2. Extract features (74 total)
features = extract_lgb_features(vv, vh, dem, slope, hand, twi, mndwi)

# 3. Predict
proba = model.predict(features)

# 4. Physics VETO
veto = (hand > 100) | (slope > 45)
proba = np.where(veto, 0.0, proba)

# 5. Threshold
mask = proba > 0.5

# 6. Optional: MST gap healing for river chips
# mst = MSTRiverConnector(max_gap=15, vh_thresh=-16.0)
# mask, connections = mst.connect(mask, vh)

# 7. Remove small regions
mask = remove_small_regions(mask, min_size=50)
```

---

## What Still Needs To Be Done

### High Priority
1. [ ] Train U-Net v11 with correct architecture (AttentionGates) + extended epochs
2. [ ] Complete MST post-processing test
3. [ ] Create final production pipeline

### Medium Priority
4. [ ] Integrate Fuzzy logic for edge cases
5. [ ] Test Gamma0 normalization effect
6. [ ] Exclude bad chips from training data

### Low Priority
7. [ ] Visualize MST connections on sample chips
8. [ ] Create deployment package

---

## Quick Commands

```bash
# SSH to server
sshpass -p 'mitaoe' ssh mit-aoe@100.84.105.5

# Activate environment
source ~/anaconda3/bin/activate
cd ~/sar_water_detection

# Check GPU
nvidia-smi

# List models
ls -la models/*.pth models/*.txt

# Check running processes
ps aux | grep python

# View logs
tail -f logs/unet_v10_training.log
tail -f logs/mst_postprocessing.log
```

---

## Files Summary

### On Server (`~/sar_water_detection/`)
```
├── models/
│   ├── lightgbm_v9_clean_mndwi.txt    # BEST (IoU 0.882)
│   ├── attention_unet_v9_sota_best.pth # Best U-Net (IoU 0.685)
│   └── ... (other models)
├── chips_expanded_npy/                 # 99 India chips
├── results/
│   ├── memsafe_test_results.json       # LGB baseline test
│   ├── ensemble_v9_correct_results.json # Ensemble test
│   ├── problem_chip_analysis.json      # Problem analysis
│   └── mst_postprocessing_results.json # MST test (pending)
├── logs/
│   ├── unet_v10_training.log
│   ├── problem_chip_analysis.log
│   └── mst_postprocessing.log
├── sota_research_module.py             # SOTA implementations
├── data_validator_v2.py                # Data integrity checks
├── attention_unet_v9_sota.py           # U-Net v9 training
├── attention_unet_v10_extended.py      # U-Net v10 training (failed)
├── test_ensemble_v9_correct.py         # Ensemble test
├── test_mst_postprocessing.py          # MST test
└── analyze_problem_chips.py            # Problem analysis
```

---

## Performance Metrics Reference

### Best Model: LightGBM v9
```
Overall IoU:       0.8821
Overall Precision: 0.9616
Overall Recall:    0.9143
Overall F1:        0.9375

Features: 74 (SAR + texture + physics + MNDWI)
Training: ~5 minutes on server
Inference: ~3 seconds per chip
```

### U-Net v9 (For Reference)
```
Overall IoU:       0.6854
Channels: 9 (VV, VH, DEM, Slope, HAND, TWI, MNDWI, Physics, Frangi)
Architecture: ResidualBlocks + AttentionGates
Training: 8.4 minutes (29 epochs)
```

---

**Document Version:** 13.0
**Last Updated:** 2026-01-26 00:05
**Status:** LightGBM v9 (0.882) is production-ready. U-Net and MST experiments ongoing.
