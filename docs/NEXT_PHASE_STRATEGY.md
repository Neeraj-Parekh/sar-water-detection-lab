# Next Phase Strategy: Advanced SAR Water Detection

**Date:** 2026-01-24  
**Status:** STRATEGIC PLANNING  
**GPU Available:** NVIDIA RTX A5000 (24GB VRAM) - IDLE AND READY

---

## Executive Summary

Based on comprehensive research and your insights, this document outlines multiple approaches to improve upon the current ResNet-18 chip classifier. The GPU is now available (0% utilization observed) and can be fully leveraged.

---

## Table of Contents

1. [Current State Assessment](#1-current-state-assessment)
2. [Your Ideas - Validated](#2-your-ideas---validated)
3. [Equation Search 2.0 - Redemption Plan](#3-equation-search-20---redemption-plan)
4. [Physics-Guided ML Approaches](#4-physics-guided-ml-approaches)
5. [Lightweight Models for Limited Data](#5-lightweight-models-for-limited-data)
6. [Advanced Segmentation Beyond ResNet-18](#6-advanced-segmentation-beyond-resnet-18)
7. [Recommended Action Plan](#7-recommended-action-plan)
8. [Current Model Archival](#8-current-model-archival)

---

## 1. Current State Assessment

### What We Have

| Asset | Details | Quality |
|-------|---------|---------|
| **Training Data** | 120 India chips (7-band: VV, VH, MNDWI, DEM, HAND, Slope, TWI) | Good |
| **ResNet-18 Model** | Chip classifier, 100% val accuracy | Working but limited |
| **GUI Tool** | 47 filter algorithms, 15 comparison windows | Excellent |
| **GPU (Remote)** | NVIDIA A5000, 24GB, CUDA 12.8 | **IDLE - READY** |

### Limitations of Current ResNet-18

| Issue | Impact | Severity |
|-------|--------|----------|
| **Classification only** | Can't do pixel-wise segmentation | High |
| **No boundary detection** | Can't find water body edges | High |
| **No water type discrimination** | Lakes, rivers, wetlands look same | Medium |
| **No confidence scoring** | Binary yes/no only | Medium |
| **Unseen topography risk** | Mountain shadows may fool it | Medium |

---

## 2. Your Ideas - Validated

### Idea 1: Equation Search with Full GPU
**Your Insight:** "The GPU had issues but now it's free. We can train a model just to do equation search."

**Research Verdict:** ✅ **VALID - with modifications**

**Evidence:**
- **SatelliteFormula (Yu et al., 2025)** successfully used symbolic regression on multispectral remote sensing
- No one has done this specifically for SAR water detection (novel research opportunity!)
- Computational requirement: ~10 hours with proper implementation

**Why It Failed Before:**
1. CuPy/CUDA version mismatch (now fixed)
2. IndexError bug in code (fixable)
3. No checkpointing (we'll add it)

**New Approach:**
```python
# Use PySR instead of custom GPU implementation
from pysr import PySRRegressor

# Search space: VV, VH, ratios, local statistics
# Constraint: Water = LOW backscatter
# Output: Interpretable equation like:
#   water = (VH < -20) & (VV/VH > 0.7) & (HAND < 4)
```

### Idea 2: Match SAR to Optical/Google Imagery
**Your Insight:** "Make the combination match the most likely looking content from VV, VH, MNDWI, HAND, DEM vs real world optical image."

**Research Verdict:** ✅ **EXCELLENT APPROACH**

**Evidence:**
- **Sen12MS dataset**: 282,384 paired Sentinel-1/Sentinel-2 chips exist
- **Optical-SAR Matching** is a proven research area (CVPR 2021)
- **Contrastive learning** can align SAR representations to optical

**Implementation:**
1. Download Sen12MS or Sen1Floods11 (has optical pairs)
2. Train contrastive network: SAR embedding ≈ Optical embedding
3. Use optical "truth" to guide SAR interpretation

### Idea 3: Physics-Guided ML
**Your Insight:** "Physics guided ML can work I feel."

**Research Verdict:** ✅ **STRONGLY SUPPORTED BY LITERATURE**

**Key Paper:** Gierszewska & Berezowski (2024) - arXiv:2410.08837
- Used SAR time-series + water gauge measurements
- Physics loss: Pearson correlation between predicted area and gauge level
- Achieved **IoU 0.89** for water detection

**Our Physics Constraints:**
| Physical Law | Implementation |
|--------------|----------------|
| Water = low backscatter | Loss: penalize predictions where σ° > -15 dB |
| Water = low elevation | HAND attention: water probability ∝ 1/HAND |
| Water = flat | Slope constraint: water probability → 0 when slope > 15° |
| Water flows down | TWI constraint: high TWI = more likely water |

### Idea 4: Smaller, Efficient Models for Limited Data
**Your Insight:** "If the issue is number of images and parameters, let's not use full-fledged U-Net but rather smaller but efficient models."

**Research Verdict:** ✅ **THIS IS THE CORRECT APPROACH**

**Evidence:**
- Iselborn et al. (2023) on Sen1Floods11: **Gradient Boosting beat deep learning** (IoU 0.875 vs 0.70)
- 1D-Justo-LiuNet: Only **4,563 parameters**, works on limited data
- MobileNetV3: **2.5M parameters**, designed for efficiency

**Recommended Models for 120 Chips:**

| Model | Parameters | Min Samples | Strength |
|-------|-----------|-------------|----------|
| **Gradient Boosting** | N/A | ~50 | Feature engineering, proven on SAR |
| **SegFormer-B0** | 3.75M | ~500 | Transformer, efficient |
| **FastSCNN** | 1.1M | ~300 | Real-time, learning to downsample |
| **MobileNetV3** | 2.5M | ~200 | h-swish, squeeze-excite |

### Idea 5: Multi-Task Model (All Your Requirements)
**Your Insight:** "Finding water body area + type + topography + algae bloom + border + narrow rivers + streams..."

**Research Verdict:** ⚠️ **AMBITIOUS BUT ACHIEVABLE WITH MULTI-TASK ARCHITECTURE**

**Solution: Mixture-of-Experts (MoE) Style Architecture**

```
Input: 7-band chip (VV, VH, MNDWI, DEM, HAND, Slope, TWI)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│              SHARED BACKBONE (EfficientNet-B0)              │
└─────────────────────────────────────────────────────────────┘
    │           │           │           │           │
    ▼           ▼           ▼           ▼           ▼
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────────┐
│ WATER │ │ TYPE  │ │ EDGE  │ │QUALITY│ │TOPOGRAPHY │
│ MASK  │ │ CLASS │ │DETECT │ │ FLAGS │ │ CONTEXT   │
└───────┘ └───────┘ └───────┘ └───────┘ └───────────┘
    │           │           │           │           │
    └───────────┴───────────┴───────────┴───────────┘
                            │
                            ▼
                    FUSION MODULE
                            │
                            ▼
              FINAL SEGMENTATION + METADATA
```

**Heads:**
1. **Water Mask Head**: Binary segmentation (pixel-wise)
2. **Type Classification Head**: {Lake, River, Wetland, Reservoir, Canal}
3. **Edge Detection Head**: Boundary pixels (Sobel-style loss)
4. **Quality Flags Head**: {Clear, Algae, Urban-noise, Shadow-risk}
5. **Topography Context Head**: {Flat, Hill, Mountain, Coastal}

---

## 3. Equation Search 2.0 - Redemption Plan

### Why Retry?

| Before (Failed) | Now (Fixed) |
|-----------------|-------------|
| CuPy CUDA mismatch | Use PyTorch/PySR (native CUDA) |
| IndexError crashes | Add bounds checking + validation |
| No checkpointing | Save every 100 generations |
| 50+ hours blind | Progress bar + early stopping |
| Custom GPU code | Use proven libraries (PySR, gplearn) |

### New Implementation Plan

```python
# equation_search_v2.py

import pysr
import numpy as np
from sklearn.metrics import f1_score

# 1. Extract features from chips
features = ['VV', 'VH', 'VV-VH', 'VV/VH', 'local_std_VV', 
            'local_std_VH', 'HAND', 'slope', 'TWI']

# 2. Configure symbolic regression
model = pysr.PySRRegressor(
    niterations=500,
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["sin", "cos", "exp", "log", "sqrt", "abs"],
    constraints={
        "^": (-1, 1),  # Prevent extreme exponents
    },
    complexity_of_operators={
        "sin": 3, "cos": 3, "exp": 5, "log": 5
    },
    maxsize=25,  # Keep equations interpretable
    timeout_in_seconds=36000,  # 10 hours max
    batching=True,  # For large datasets
    batch_size=1000,
    progress=True,  # Show progress
    verbosity=1,
    save_to_file=True,  # Checkpoint every iteration
)

# 3. Run search
model.fit(X_train, y_train)

# 4. Extract best equations
for eq in model.equations_:
    print(f"Equation: {eq.equation}")
    print(f"Complexity: {eq.complexity}")
    print(f"Score: {eq.score}")
```

### Expected Outcomes

| Scenario | Outcome | Time |
|----------|---------|------|
| **Best Case** | Discover equation like `water = (VH < -22) & (VV/VH > 0.65) & (HAND < 3)` | 4-6 hours |
| **Good Case** | Find improved thresholds for existing indices | 8-10 hours |
| **Worst Case** | Confirm no simple equation exists (still useful!) | 10 hours |

---

## 4. Physics-Guided ML Approaches

### Approach A: Physics-Constrained Loss Function

```python
class PhysicsGuidedLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.2, gamma=0.2):
        super().__init__()
        self.alpha = alpha  # Weight for HAND constraint
        self.beta = beta    # Weight for slope constraint
        self.gamma = gamma  # Weight for backscatter constraint
        
    def forward(self, pred, target, vv, vh, hand, slope):
        # Standard segmentation loss
        bce_loss = F.binary_cross_entropy(pred, target)
        
        # Physics Loss 1: HAND correlation (water at low elevation)
        # We want NEGATIVE correlation: high water prob at low HAND
        hand_corr = pearson_correlation(pred.flatten(), hand.flatten())
        hand_loss = F.relu(hand_corr)  # Penalize positive correlation
        
        # Physics Loss 2: Slope exclusion (no water on steep slopes)
        steep_mask = (slope > 15).float()
        slope_loss = (pred * steep_mask).mean()  # Penalize water on slopes
        
        # Physics Loss 3: Backscatter constraint (water = dark)
        # VH should be < -18 dB where water is predicted
        bright_mask = (vh > -15).float()
        backscatter_loss = (pred * bright_mask).mean()
        
        # Combined loss
        total_loss = bce_loss + \
                     self.alpha * hand_loss + \
                     self.beta * slope_loss + \
                     self.gamma * backscatter_loss
        
        return total_loss
```

### Approach B: Time-Series Physics (Gierszewska-style)

If we have time-series SAR data:
```python
def temporal_physics_loss(water_area_sequence, gauge_levels):
    """
    Physics: Water area correlates with gauge measurements
    """
    # Compute predicted water area at each timestep
    pred_areas = [mask.sum() for mask in water_area_sequence]
    
    # Loss = negative Pearson correlation with gauge
    corr = pearson_correlation(pred_areas, gauge_levels)
    return 1 - corr  # Minimize to maximize correlation
```

### Approach C: Attention from Physics

```python
class HANDAttention(nn.Module):
    """Physical prior: Water more likely where HAND is low"""
    
    def forward(self, features, hand, threshold=10.0):
        # Smooth sigmoid attention based on HAND
        attention = torch.sigmoid(-hand / threshold)
        
        # Apply to feature maps
        return features * attention.unsqueeze(1)
```

---

## 5. Lightweight Models for Limited Data

### Option 1: Feature-Based Gradient Boosting (PROVEN BEST)

**Evidence:** Iselborn et al. (2023) beat all DL models on Sen1Floods11

```python
import lightgbm as lgb
from skimage.feature import local_binary_pattern
import numpy as np

def extract_features(chip):
    """Extract physics-meaningful features"""
    vv, vh = chip[0], chip[1]
    hand, slope, twi = chip[3], chip[4], chip[5]
    
    features = {
        # Backscatter
        'vv_mean': vv.mean(),
        'vh_mean': vh.mean(),
        'vv_std': vv.std(),
        'vh_std': vh.std(),
        
        # Polarization ratios
        'ratio_vv_vh': (vv / (vh + 1e-6)).mean(),
        'diff_vv_vh': (vv - vh).mean(),
        
        # Terrain
        'hand_mean': hand.mean(),
        'slope_mean': slope.mean(),
        'twi_mean': twi.mean(),
        
        # Texture (LBP)
        'lbp_vv': local_binary_pattern(vv, 8, 1).mean(),
        'lbp_vh': local_binary_pattern(vh, 8, 1).mean(),
        
        # Local statistics (3x3 windows)
        'local_var_vv': local_variance(vv).mean(),
        'local_var_vh': local_variance(vh).mean(),
    }
    return features

# Train model
model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    num_leaves=31,
    learning_rate=0.1,
)
model.fit(X_train, y_train)
```

**Expected Performance:** IoU 0.85-0.90 on water detection

### Option 2: SegFormer-B0 (Lightweight Transformer)

**Parameters:** 3.75M (vs 31M for U-Net)

```python
from transformers import SegformerForSemanticSegmentation

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=2,
    num_channels=7,  # Our 7-band input
    ignore_mismatched_sizes=True,
)

# Modify first conv to accept 7 channels
model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(
    7, 32, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3)
)
```

### Option 3: FastSCNN (Real-Time Segmentation)

**Parameters:** 1.1M  
**Speed:** 123.5 FPS

```python
# FastSCNN architecture highlights:
# 1. Learning to Downsample (LtD) module
# 2. Global Feature Extractor
# 3. Feature Fusion Module

class FastSCNN(nn.Module):
    def __init__(self, in_channels=7, num_classes=2):
        self.learning_to_downsample = LtD(in_channels, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(64, 128)
        self.feature_fusion = FeatureFusion(64, 128, 128)
        self.classifier = nn.Conv2d(128, num_classes, 1)
```

### Option 4: MobileNetV3 Encoder + Lightweight Decoder

```python
import torchvision.models as models

# Use MobileNetV3-Small as encoder
encoder = models.mobilenet_v3_small(pretrained=True).features

# Modify first layer for 7 channels
encoder[0][0] = nn.Conv2d(7, 16, kernel_size=3, stride=2, padding=1, bias=False)

# Add simple decoder
class LightweightDecoder(nn.Module):
    def __init__(self):
        self.up1 = nn.ConvTranspose2d(576, 128, 2, 2)
        self.conv1 = nn.Conv2d(128, 64, 3, 1, 1)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv2 = nn.Conv2d(32, 16, 3, 1, 1)
        self.up3 = nn.ConvTranspose2d(16, 8, 2, 2)
        self.final = nn.Conv2d(8, 2, 1)
```

---

## 6. Advanced Segmentation Beyond ResNet-18

### What ResNet-18 Can't Do (But We Need)

| Capability | ResNet-18 | Advanced Model |
|------------|-----------|----------------|
| Pixel-wise segmentation | ❌ | ✅ |
| Water body boundaries | ❌ | ✅ |
| Water type classification | ❌ | ✅ |
| Narrow river detection | ❌ | ✅ |
| Confidence maps | ❌ | ✅ |
| Multi-scale features | ❌ | ✅ |

### Recommended Architecture: Multi-Task EfficientNet-B0

```
Input: 7-band chip (512x512x7)
         │
         ▼
┌─────────────────────────────────────┐
│    EfficientNet-B0 Encoder          │
│    (5.3M params, pretrained)        │
│    Modified first conv: 7→32        │
└─────────────────────────────────────┘
         │
    ┌────┴────┬────────┬────────┐
    ▼         ▼        ▼        ▼
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│ Water │ │ Edge  │ │ Type  │ │Confid │
│ Mask  │ │ Map   │ │ Class │ │ Map   │
│ Head  │ │ Head  │ │ Head  │ │ Head  │
└───────┘ └───────┘ └───────┘ └───────┘
    │         │        │        │
    │         │        │        │
    ▼         ▼        ▼        ▼
┌────────────────────────────────────┐
│         Multi-Task Loss            │
│  L = L_seg + α*L_edge + β*L_type   │
│      + γ*L_physics                 │
└────────────────────────────────────┘
```

### Training Strategy for 120 Chips

```python
# Heavy augmentation to increase effective dataset size
augmentations = A.Compose([
    # Geometric
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
    
    # SAR-specific
    A.GaussNoise(var_limit=(10, 50), p=0.3),  # Simulate speckle
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    
    # Cropping for multi-scale
    A.RandomCrop(256, 256, p=0.5),
    A.Resize(512, 512),
])

# Effective samples: 120 * 10 augmentations = 1200
```

---

## 7. Recommended Action Plan

### Phase 1: Quick Wins (Day 1)

| Task | Time | Expected Outcome |
|------|------|------------------|
| 1. Test GPU availability | 5 min | Confirm A5000 ready |
| 2. Install PySR on server | 30 min | Equation search ready |
| 3. Run LightGBM baseline | 2 hours | IoU ~0.85 baseline |

### Phase 2: Equation Search 2.0 (Day 1-2)

| Task | Time | Expected Outcome |
|------|------|------------------|
| 4. Prepare features from 120 chips | 1 hour | Feature matrix ready |
| 5. Run PySR equation search | 10 hours | Discovered equations |
| 6. Validate top 5 equations | 2 hours | Best equation selected |

### Phase 3: Physics-Guided Model (Day 2-3)

| Task | Time | Expected Outcome |
|------|------|------------------|
| 7. Implement physics loss | 2 hours | Custom loss function |
| 8. Train SegFormer-B0 + physics | 4 hours | Physics-guided model |
| 9. Compare with baseline | 1 hour | Performance delta |

### Phase 4: Multi-Task Model (Day 3-4)

| Task | Time | Expected Outcome |
|------|------|------------------|
| 10. Implement multi-task heads | 4 hours | Architecture ready |
| 11. Train with combined loss | 6 hours | Multi-task model |
| 12. Generate all outputs | 2 hours | Masks + edges + types |

### Phase 5: Validation & Documentation (Day 4-5)

| Task | Time | Expected Outcome |
|------|------|------------------|
| 13. Cross-validation on all models | 4 hours | Honest metrics |
| 14. Generate comparison plots | 2 hours | Visual comparison |
| 15. Write paper-ready documentation | 4 hours | Publication material |

---

## 8. Current Model Archival

### ResNet-18 Chip Classifier Archive

**Location:** `/media/neeraj-parekh/Data1/sar soil system/chips/gui/models/archived/`

**Files to Archive:**
```
archived/
├── resnet18_chip_classifier/
│   ├── model_weights.pth
│   ├── training_config.json
│   ├── training_log.csv
│   ├── evaluation_metrics.json
│   ├── gradcam_samples/
│   │   ├── chip_001_gradcam.png
│   │   ├── chip_002_gradcam.png
│   │   └── ...
│   └── README.md
└── model_comparison_v1.md
```

**Training Configuration:**
```json
{
  "model": "ResNet-18 (pretrained ImageNet)",
  "input_channels": 7,
  "output_classes": 2,
  "parameters": 11700000,
  "training_samples": 84,
  "validation_samples": 18,
  "test_samples": 18,
  "epochs": 30,
  "batch_size": 8,
  "optimizer": "Adam",
  "learning_rate": 0.0001,
  "augmentations": ["HFlip", "VFlip", "Rotate90", "Noise"],
  "final_val_accuracy": 1.0,
  "final_val_loss": 0.01,
  "training_time_minutes": 45,
  "gpu": "NVIDIA RTX A5000"
}
```

**Known Limitations:**
1. Classification only (not segmentation)
2. Binary output (no confidence)
3. Limited to seen topographies
4. No edge/boundary detection
5. Cannot distinguish water types

---

## Summary

### Your Ideas: All Validated

| Your Idea | Research Support | Action |
|-----------|-----------------|--------|
| Equation search with full GPU | ✅ SatelliteFormula (2025) | Run PySR, 10 hours |
| Match SAR to optical | ✅ Sen12MS, contrastive learning | Download paired data |
| Physics-guided ML | ✅ Gierszewska 2024, IoU 0.89 | Add physics loss |
| Smaller efficient models | ✅ LightGBM beats DL on Sen1Floods11 | Train gradient boosting |
| Multi-task model | ✅ MoE architectures exist | Build multi-head model |

### Next Immediate Step

```bash
# Connect to GPU server and verify availability
ssh mit-aoe@100.84.105.5
nvidia-smi  # Should show 0% GPU utilization
conda activate gpu_env
pip install pysr lightgbm segmentation-models-pytorch
```

### Timeline

```
Day 1: LightGBM baseline + start equation search
Day 2: Equation search results + implement physics loss
Day 3: Train SegFormer-B0 with physics constraints
Day 4: Multi-task model training
Day 5: Validation, comparison, documentation
```

---

**This strategy document will be updated as we progress.**

*Created: 2026-01-24*  
*Last Updated: 2026-01-24*
