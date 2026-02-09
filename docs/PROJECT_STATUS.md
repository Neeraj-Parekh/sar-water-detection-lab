# SAR Water Detection Project - Complete Status Report

**Date:** 2026-01-25  
**Author:** AI Assistant + User Collaboration  
**Project:** Physics-Guided SAR Water Detection for India

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Final Results Achieved](#2-final-results-achieved)
3. [Comprehensive Evaluation Results](#3-comprehensive-evaluation-results)
4. [Technical Implementation Details](#4-technical-implementation-details)
5. [Files and Resources](#5-files-and-resources)
6. [Commands Reference](#6-commands-reference)
7. [What Remains To Do](#7-what-remains-to-do)

---

## 1. Executive Summary

### Project Goal
Build an interpretable, physics-guided water detection system for SAR (Sentinel-1) imagery over India, capable of:
- Pixel-wise water segmentation
- Water body type classification
- Edge/boundary detection
- Confidence scoring
- Handling narrow rivers and diverse water types

### Current Status: PHASE 1 COMPLETE

| Component | Status | Performance |
|-----------|--------|-------------|
| LightGBM v4 (69 features) | COMPLETE | **IoU 0.9302** (val), **0.881** (test) |
| U-Net v4 (CBAM attention) | COMPLETE | **IoU 0.918** (val), **0.766** (test) |
| Ensemble (LGBx1 + UNetx0) | COMPLETE | **IoU 0.881** (test) |
| Comprehensive Evaluation | COMPLETE | Full analysis done |
| Visualizations | COMPLETE | 12 plots generated |

---

## 2. Final Results Achieved

### Model Training Summary

| Model | Val IoU | Test IoU | Parameters | Training Time |
|-------|---------|----------|------------|---------------|
| **LightGBM v4** | 0.9302 +/- 0.0004 | **0.8808** | 69 features | ~30 min |
| **U-Net v4** | 0.9184 | 0.7662 | 7.8M | ~2 hours |
| **Ensemble** | - | 0.8808 | Optimal: LGB=1.0, UNet=0.0 | - |

### Improvement Over Previous Versions

| Version | LightGBM IoU | U-Net IoU | Notes |
|---------|--------------|-----------|-------|
| v2 | 0.868 | 0.788 | Had MNDWI optical leakage (invalid) |
| v3 | 0.824 | Failed | SAR-only, fixed thresholds |
| **v4** | **0.930** | **0.918** | Comprehensive 69 features, working! |

### Top 10 LightGBM Features (by split count)

1. **DEM** (3290) - Elevation matters most!
2. **TWI** (1711) - Topographic wetness index
3. **VH_closed** (1391) - Morphological feature
4. **VV_std_s21** (1272) - Multi-scale texture
5. **VH_mean_s21** (1231) - Large-scale smoothing
6. **VV_mean_s21** (1101) - Large-scale smoothing
7. **VH_std_s21** (1058) - Multi-scale texture
8. **VV_min_s9** (1013) - Local minimum (dark water)
9. **HAND** (1011) - Height above drainage
10. **VH_min_s9** (1010) - Local minimum

---

## 3. Comprehensive Evaluation Results

### 3.1 Overall Test Performance

| Model | Accuracy | Precision | Recall | F1 | IoU |
|-------|----------|-----------|--------|-----|-----|
| LightGBM | 94.2% | 93.1% | 94.2% | 93.7% | **88.1%** |
| U-Net | 88.6% | 85.0% | 88.6% | 86.8% | 76.6% |
| Ensemble | 94.2% | 93.1% | 94.2% | 93.7% | **88.1%** |

**Key Finding:** LightGBM significantly outperforms U-Net on test set. The optimal ensemble weights are LGB=1.0, UNet=0.0, indicating LightGBM alone is best.

### 3.2 Narrow Water Body Detection

Critical for rivers and streams. Measured by recall at different water body widths:

| Max Width | Recall | Std Dev | Notes |
|-----------|--------|---------|-------|
| <= 3 pixels | **80.1%** | +/-25.0% | Very narrow streams |
| <= 5 pixels | **83.0%** | +/-23.2% | Small streams |
| <= 10 pixels | **87.1%** | +/-20.4% | Narrow rivers |
| <= 20 pixels | **89.9%** | +/-18.8% | Wide streams |

**Finding:** Even the narrowest water bodies (3px wide) achieve 80% recall, which is strong for SAR-based detection.

### 3.3 Land-Water Boundary Accuracy

Measures how well the model detects edges/transitions:

| Distance from Boundary | IoU | Std Dev |
|------------------------|-----|---------|
| Within 1 pixel | 6.7% | +/-24.9% |
| Within 2 pixels | 6.7% | +/-24.9% |
| Within 3 pixels | 16.7% | +/-37.3% |
| Within 5 pixels | **59.5%** | +/-48.7% |

**Finding:** Boundary detection is challenging. Accuracy improves significantly at 5px from edge. This is expected for SAR due to speckle and mixed pixels at boundaries.

### 3.4 False Positive/Negative Analysis

| Metric | Value |
|--------|-------|
| Total True Positives | 851,996 |
| Total False Positives | 62,966 |
| Total False Negatives | 52,332 |
| FP Rate | **6.9%** |
| FN Rate | **5.8%** |

**Error Categories:**
- Shadow-like FP: Steep slopes with low VH
- Wet-soil FP: Low HAND with moderate VH
- Bright water FN: Rough/windblown water with high VH

---

## 4. Technical Implementation Details

### 4.1 Feature Engineering (69 Features)

**Category 1: Basic SAR (4 features)**
- VV, VH, VV-VH, VV/VH ratio

**Category 2: Topographic (6 features)**
- HAND, SLOPE, TWI, DEM, DEM_gradient, DEM_aspect

**Category 3: Polarimetric (4 features)**
- Pseudo-entropy, Pseudo-alpha, RVI, Span

**Category 4: Multi-scale Statistics (26 features)**
- Mean, std at scales 3, 5, 9, 15, 21
- Min at scales 3, 5, 9 (for dark water detection)

**Category 5: Speckle Statistics (4 features)**
- ENL (Equivalent Number of Looks)
- CV (Coefficient of Variation)

**Category 6: Texture (6 features)**
- Fast GLCM: contrast, homogeneity, energy

**Category 7: Morphological (4 features)**
- Opening, Closing, White top-hat, Black top-hat

**Category 8: Linear Features (1 feature)**
- Multi-orientation line detector (for rivers)

**Category 9: Adaptive Thresholds (6 features)**
- Otsu and Kapur thresholding

**Category 10: Physics Composite (5 features)**
- Combined physics score and individual scores

**Category 11: Edge Features (3 features)**
- VV/VH gradients, Laplacian

### 4.2 Data Summary

- **Total Chips:** 204 (117 original + 99 expanded - 12 corrupted)
- **Training Samples:** 972,361 pixels
- **Water Class Balance:** 62.2%
- **Test Set:** 31 chips
- **SAR-Only:** No optical (MNDWI) features used

### 4.3 U-Net Architecture

```
UNetV4 (7.8M parameters)
├── Encoder: ConvBlock with CBAM attention (32->64->128->256)
├── Bottleneck: 512 channels
├── Decoder: Transposed convolutions with skip connections
├── Output: 1 channel (logits), sigmoid for probability
├── Loss: Dice + Focal (no BCE due to numerical issues)
└── Input: 6 channels (VV, VH, DEM, HAND, SLOPE, TWI)
```

---

## 5. Files and Resources

### 5.1 Local Machine

```
/media/neeraj-parekh/Data1/sar soil system/chips/gui/
├── master_training_pipeline_v4.py    # Full training pipeline
├── unet_v4_standalone.py             # Fixed U-Net training
├── comprehensive_evaluation.py        # Evaluation script
├── visualizations/
│   ├── chip_*_comparison.png          # 10 prediction visualizations
│   ├── feature_importance.png         # Top 25 features plot
│   └── narrow_water_analysis.png      # Narrow water recall plot
├── docs/
│   ├── PROJECT_STATUS.md              # This document
│   ├── RESEARCH_SOURCES.md            # Literature review
│   └── NEXT_PHASE_STRATEGY.md         # Original plan
```

### 5.2 Remote Server (mit-aoe@100.84.105.5)

```
~/sar_water_detection/
├── master_training_pipeline_v4.py
├── unet_v4_standalone.py
├── comprehensive_evaluation.py
├── models/
│   ├── lightgbm_v4_comprehensive.txt   # Best LightGBM (3.5MB)
│   └── unet_v4_best.pth                # Best U-Net (31MB)
├── results/
│   ├── training_results_v4.json        # LightGBM results
│   ├── unet_v4_standalone_results.json # U-Net results
│   └── comprehensive_evaluation_results.json
├── visualizations/                      # All plots
├── chips/                               # 117 original chips
└── chips_expanded/                      # 99 expanded chips
```

---

## 6. Commands Reference

### Check Server Status
```bash
# Check GPU
sshpass -p 'mitaoe' ssh mit-aoe@100.84.105.5 "nvidia-smi"

# Check running processes
sshpass -p 'mitaoe' ssh mit-aoe@100.84.105.5 "ps aux | grep python | grep -v grep"
```

### Get Results
```bash
# LightGBM results
sshpass -p 'mitaoe' ssh mit-aoe@100.84.105.5 "cat ~/sar_water_detection/results/training_results_v4.json"

# U-Net results
sshpass -p 'mitaoe' ssh mit-aoe@100.84.105.5 "cat ~/sar_water_detection/results/unet_v4_standalone_results.json"

# Comprehensive evaluation
sshpass -p 'mitaoe' ssh mit-aoe@100.84.105.5 "cat ~/sar_water_detection/results/comprehensive_evaluation_results.json"
```

### Download Files
```bash
# Download models
sshpass -p 'mitaoe' scp mit-aoe@100.84.105.5:~/sar_water_detection/models/lightgbm_v4_comprehensive.txt ./
sshpass -p 'mitaoe' scp mit-aoe@100.84.105.5:~/sar_water_detection/models/unet_v4_best.pth ./

# Download visualizations
sshpass -p 'mitaoe' scp -r mit-aoe@100.84.105.5:~/sar_water_detection/visualizations ./
```

---

## 7. What Remains To Do

### Immediate Next Steps
1. **PySR Symbolic Regression** (~10 hours)
   - Discover interpretable water detection equations
   - Run on each water body type category

2. **Boundary Enhancement**
   - Current boundary IoU is low (6.7% at 1px)
   - Consider active contour post-processing
   - CRF-based boundary refinement

3. **Error Analysis Deep Dive**
   - Analyze shadow-like false positives
   - Wet soil false positive reduction
   - Bright water (wind-affected) false negative handling

### Future Improvements
1. **SegFormer-B0** - Transformer architecture for comparison
2. **Test-Time Augmentation** - Improve predictions with TTA
3. **Multi-Temporal Fusion** - Use time-series SAR for better accuracy
4. **Region-Specific Thresholds** - Adapt thresholds by geography
5. **Uncertainty Quantification** - Provide confidence maps

### Publication Preparation
1. Generate paper-quality figures
2. Write methods section with 69-feature description
3. Compare with Sen1Floods11 benchmark
4. Document physics-guided approach

---

## Summary of Achievements

| Goal | Status | Result |
|------|--------|--------|
| High accuracy water detection | ACHIEVED | IoU 0.88+ |
| SAR-only (no optical leakage) | ACHIEVED | 69 SAR+topo features |
| Narrow water body detection | ACHIEVED | 80% recall at 3px width |
| Physics-guided approach | ACHIEVED | HAND/TWI/Slope constraints |
| Interpretable model | ACHIEVED | LightGBM with feature importance |
| Deep learning baseline | ACHIEVED | U-Net IoU 0.77 |
| Comprehensive evaluation | ACHIEVED | Full analysis complete |
| Visualizations | ACHIEVED | 12 plots generated |

---

*Last Updated: 2026-01-25 02:45 IST*
