# Research Sources & Literature Review

**Project:** SAR Water Detection for India  
**Date:** 2026-01-24  
**Purpose:** Document all research consulted during project development

---

## 1. Primary Research Papers

### 1.1 Physics-Guided Machine Learning for Water Detection
**Citation:** Gierszewska, M., & Berezowski, T. (2024). Physics-guided machine learning for SAR-based water detection. arXiv:2410.08837

**Key Findings:**
- Used SAR time-series + water gauge measurements
- Physics loss: Pearson correlation between predicted water area and gauge level
- Achieved **IoU 0.89** for water detection
- Outperformed standard BCE-only training

**How We Used It:**
- Implemented physics-constrained loss function
- Added HAND correlation loss
- Added slope exclusion loss
- Added backscatter constraint loss

**Relevance:** HIGH - Direct validation of user's physics-guided ML idea

---

### 1.2 Gradient Boosting for SAR Flood Detection
**Citation:** Iselborn, K., et al. (2023). Benchmarking machine learning methods for SAR-based flood detection on Sen1Floods11.

**Key Findings:**
- Tested multiple ML methods on Sen1Floods11 dataset
- **LightGBM achieved IoU 0.875**
- Deep learning methods (U-Net) achieved IoU ~0.70
- Feature engineering crucial for gradient boosting success

**How We Used It:**
- Chose LightGBM as baseline model
- Adopted pixel-wise feature extraction approach
- Implemented 12 physics-meaningful features

**Relevance:** HIGH - Direct evidence that simple models can beat deep learning

---

### 1.3 Symbolic Regression for Remote Sensing
**Citation:** Yu, X., et al. (2025). SatelliteFormula: Symbolic Regression for Satellite Image Analysis.

**Key Findings:**
- First application of symbolic regression to satellite imagery
- Discovers interpretable equations for classification
- Works with multispectral data
- Computational cost: ~10-20 hours GPU time

**How We Used It:**
- Designed PySR-based equation search script
- Configured search space for SAR-specific features
- Added physics constraints to search

**Relevance:** MEDIUM - Validates equation search approach, no SAR-specific results

---

### 1.4 Sen12MS: SAR-Optical Paired Dataset
**Citation:** Schmitt, M., et al. (2019). Sen12MS - A Curated Dataset of Georeferenced Multi-Spectral Sentinel-1/2 Imagery. ISPRS.

**Key Findings:**
- 282,384 triplets of Sentinel-1/2/DEM data
- Global coverage, all seasons
- Enables SAR-optical matching research

**How We Used It:**
- Validated user's SAR-optical matching idea
- Reference for potential future data augmentation

**Relevance:** MEDIUM - Future enhancement option

---

### 1.5 Lightweight Neural Networks for Segmentation
**Citation:** Various sources on 1D-Justo-LiuNet, FastSCNN, MobileNetV3

**Key Findings:**
- 1D-Justo-LiuNet: Only 4,563 parameters
- FastSCNN: 1.1M parameters, 123.5 FPS
- MobileNetV3: 2.5M parameters, efficient h-swish activation

**How We Used It:**
- Designed lightweight U-Net (~500K parameters)
- Justified small model approach for limited data

**Relevance:** MEDIUM - Architectural inspiration

---

## 2. Technical References

### 2.1 SAR Backscatter Physics
**Source:** ESA Sentinel-1 Technical Guide

**Key Facts:**
- Water appears dark in SAR (specular reflection)
- VH polarization: water typically < -18 dB
- VV polarization: water typically < -15 dB
- Roughness affects backscatter (wind, waves)

**How We Used It:**
- Backscatter constraint in physics loss
- Feature normalization ranges
- Threshold values for water detection

---

### 2.2 HAND (Height Above Nearest Drainage)
**Citation:** Rennó, C.D., et al. (2008). HAND, a new terrain descriptor using SRTM-DEM.

**Key Facts:**
- HAND measures vertical distance to nearest stream
- Water bodies have HAND ≈ 0
- Values > 10m unlikely to contain water
- Better than raw DEM for hydrological analysis

**How We Used It:**
- HAND attention module in U-Net
- HAND correlation loss in physics loss
- Feature for LightGBM

---

### 2.3 TWI (Topographic Wetness Index)
**Citation:** Beven, K.J., & Kirkby, M.J. (1979). A physically based, variable contributing area model of basin hydrology.

**Key Facts:**
- TWI = ln(a / tan(b)) where a = upslope area, b = slope
- High TWI indicates water accumulation potential
- Values > 10 highly likely to contain water

**How We Used It:**
- Feature for LightGBM
- Input band for neural networks

---

### 2.4 JRC Global Surface Water
**Source:** EC Joint Research Centre

**Key Facts:**
- 38 years of Landsat imagery (1984-2022)
- Global water occurrence, change, seasonality
- Used as truth mask for training

**How We Used It:**
- Truth masks for all training chips
- Validation reference

---

## 3. Datasets Referenced

| Dataset | Size | Source | Use Case |
|---------|------|--------|----------|
| **India Chips** | 118 chips | Our GEE exports | Primary training |
| **Sen1Floods11** | 4,831 chips | Cloud-to-Street | Benchmark reference |
| **Sen12MS** | 282,384 triplets | TU Munich | Future SAR-optical matching |
| **JRC Water** | Global | EC JRC | Truth mask source |
| **SRTM DEM** | Global 30m | NASA | Terrain features |
| **FABDEM** | Global 30m | U Bristol | HAND computation |

---

## 4. Software & Libraries

### 4.1 Machine Learning
| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.x | Deep learning framework |
| LightGBM | 4.x | Gradient boosting |
| scikit-learn | 1.x | Metrics, preprocessing |
| PySR | 0.x | Symbolic regression |

### 4.2 Remote Sensing
| Library | Purpose |
|---------|---------|
| Google Earth Engine | Data download |
| rasterio | Geospatial I/O |
| numpy | Array operations |
| scipy | Image processing |

### 4.3 Visualization
| Library | Purpose |
|---------|---------|
| matplotlib | Plotting |
| seaborn | Statistical plots |

---

## 5. Key Equations & Formulas

### 5.1 Water Detection Indices
```
MNDWI = (Green - SWIR) / (Green + SWIR)
NDWI = (Green - NIR) / (Green + NIR)
AWEI = 4*(Green - SWIR) - (0.25*NIR + 2.75*SWIR)
```

### 5.2 SAR Water Detection (Literature)
```
# Simple threshold (Bioresita et al.)
water = VH < -18 dB

# Ratio-based (Martinis et al.)
water = (VV/VH > 0.7) AND (VH < -20 dB)

# Multi-feature (our approach)
water = f(VV, VH, VV-VH, VV/VH, HAND, SLOPE, TWI)
```

### 5.3 Physics Loss Function
```python
L_total = L_BCE + α*L_HAND + β*L_slope + γ*L_backscatter

where:
  L_HAND = mean(pred * hand)           # Penalize water at high elevation
  L_slope = mean(pred * (slope > 15))  # Penalize water on steep slopes
  L_backscatter = mean(pred * (VH > -15))  # Penalize water where bright
```

### 5.4 Evaluation Metrics
```
IoU (Jaccard) = TP / (TP + FP + FN)
F1 Score = 2*TP / (2*TP + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
```

---

## 6. Research Gaps Identified

### 6.1 Opportunities for Novel Contribution

| Gap | Our Approach | Novelty |
|-----|--------------|---------|
| No symbolic regression for SAR water | PySR equation search | First SAR-specific equations |
| Limited India-specific models | Train on India chips | Regional specialization |
| Physics loss not combined with multi-task | Multi-task + physics | Novel architecture |
| No interpretable uncertainty | Confidence head + physics | Explainable AI |

### 6.2 Limitations of Current Literature

1. **Most papers use single loss function** - We combine BCE + physics
2. **Few papers target narrow rivers** - Our multi-task approach addresses this
3. **Limited work on SAR equation discovery** - Our PySR approach is novel
4. **No India-specific SAR water models** - Our dataset fills this gap

---

## 7. Citation Format for Paper

If publishing results, cite as:

```bibtex
@article{our_sar_water_2026,
  title={Physics-Guided Multi-Task Learning for SAR Water Detection in India},
  author={[Authors]},
  journal={[Journal]},
  year={2026},
  note={Using LightGBM baseline and physics-constrained U-Net}
}

@article{gierszewska2024physics,
  title={Physics-guided machine learning for SAR-based water detection},
  author={Gierszewska, M. and Berezowski, T.},
  journal={arXiv preprint arXiv:2410.08837},
  year={2024}
}

@article{iselborn2023benchmarking,
  title={Benchmarking ML methods for SAR flood detection on Sen1Floods11},
  author={Iselborn, K. and others},
  year={2023}
}
```

---

*This document serves as a comprehensive reference for all research consulted during the SAR Water Detection project development.*
