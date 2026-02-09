# SAR Water Detection System for India

## Overview

Physics-Guided SAR Water Detection System using Sentinel-1 imagery for India.

**Best Model:** LightGBM v9 (IoU: 0.882, Precision: 0.962, Recall: 0.914)

## Quick Start

```python
from production_package import SARWaterDetector

# Initialize detector
detector = SARWaterDetector(model_path='models/lightgbm_v9_clean_mndwi.txt')

# Load chip (H, W, 8) numpy array
import numpy as np
chip = np.load('your_chip.npy')

# Predict
result = detector.predict(chip)
water_mask = result['water_mask']
water_probability = result['water_probability']
```

## Input Format

The system expects chips as numpy arrays with shape `(H, W, 8)`:

| Channel | Description | Units | Range |
|---------|-------------|-------|-------|
| 0 | VV backscatter | dB | -30 to 10 |
| 1 | VH backscatter | dB | -35 to 0 |
| 2 | DEM | meters | -100 to 5000 |
| 3 | Slope | degrees | 0 to 90 |
| 4 | HAND | meters | 0 to 500 |
| 5 | TWI | - | 0 to 30 |
| 6 | Label | binary | 0 or 1 |
| 7 | MNDWI | - | -1 to 1 |

## Output Format

The `predict()` method returns a dictionary:

```python
{
    'water_mask': np.ndarray,        # Binary mask (H, W)
    'water_probability': np.ndarray, # Probability 0-1 (H, W)
    'confidence': np.ndarray,        # Confidence 0-1 (H, W)
    'physics_score': np.ndarray,     # Physics-based score (H, W)
    'veto_mask': np.ndarray,         # Areas vetoed by physics (H, W)
    'lgb_raw': np.ndarray            # Raw LightGBM output (H, W)
}
```

## Model Performance

### Overall Results (99 India Chips)

| Metric | Value |
|--------|-------|
| IoU | 0.882 |
| Precision | 0.962 |
| Recall | 0.914 |
| F1 Score | 0.937 |

### Performance by Water Body Type

| Type | Count | Avg IoU |
|------|-------|---------|
| Lakes/Reservoirs | 45 | 0.92 |
| Rivers | 25 | 0.85 |
| Coastal/Estuaries | 15 | 0.88 |
| Wetlands | 8 | 0.78 |
| Flood Areas | 6 | 0.65 |

## Architecture

### Feature Engineering (74 Features)

1. **Raw SAR** (2): VV, VH backscatter
2. **SAR Indices** (4): VV/VH ratio, VV-VH diff, NDWI-like, RVI
3. **Multi-scale Texture** (40): Mean, std, min, max at scales 3,5,9,15,21
4. **Gradient Features** (4): VV/VH gradient magnitude, Laplacian
5. **Morphological** (4): Opening, closing for VV/VH
6. **Statistical** (4): Otsu-like diff, local contrast
7. **GLCM-like** (5): Contrast, homogeneity, pseudo-entropy
8. **Terrain** (4): DEM, Slope, HAND, TWI
9. **Physics Scores** (3): HAND, slope, TWI sigmoid scores
10. **MNDWI** (4): Raw, binary, mean, std

### Physics Constraints

**Veto Rules** (areas that CANNOT be water):
- HAND > 100m
- Slope > 45 degrees
- HAND > 30m AND Slope > 20 degrees

**Physics Score** (Mahalanobis-inspired):
```
score = 0.4 * hand_score + 0.4 * slope_score + 0.2 * twi_score
```

## Models Trained

| Model | IoU | Notes |
|-------|-----|-------|
| LightGBM v9 | 0.882 | **Production model** |
| U-Net v11 | 0.700 | Best deep learning |
| U-Net v9 | 0.685 | AttentionGates |
| U-Net v10 | 0.604 | Failed (CBAM) |
| Simple Equation | 0.761 | Physics-only backup |

## Directory Structure

```
sar_water_detection/
├── production_package/
│   ├── __init__.py
│   └── sar_water_detector.py
├── models/
│   ├── lightgbm_v9_clean_mndwi.txt  # Production model
│   ├── attention_unet_v11_best.pth   # Best U-Net
│   └── attention_unet_v9_sota_best.pth
├── chips_expanded_npy/               # 99 India chips
├── results/
│   ├── pipeline_v2_test/
│   ├── mst_postprocessing_results.json
│   └── problem_chip_analysis.json
└── logs/
```

## Command Line Usage

```bash
# Single chip prediction
python -m production_package.sar_water_detector \
    --input chip.npy \
    --output results/ \
    --threshold 0.5

# Batch prediction with evaluation
python -m production_package.sar_water_detector \
    --input chips_expanded_npy/ \
    --output results/ \
    --evaluate
```

## API Reference

### SARWaterDetector

```python
class SARWaterDetector:
    def __init__(self, config=None, model_path=None):
        """
        Initialize detector.
        
        Args:
            config: DetectorConfig object
            model_path: Path to LightGBM model (overrides config)
        """
    
    def predict(self, chip: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict water mask from chip."""
    
    def evaluate(self, chip: np.ndarray, label=None) -> Dict:
        """Predict and compute metrics against ground truth."""
    
    def predict_batch(self, chips: List[np.ndarray]) -> List[Dict]:
        """Batch prediction."""
```

### DetectorConfig

```python
@dataclass
class DetectorConfig:
    model_path: str = "models/lightgbm_v9_clean_mndwi.txt"
    min_water_size: int = 50      # Min water body size (pixels)
    min_hole_size: int = 50       # Min hole size to fill
    physics_weight: float = 0.1   # Weight for physics score
    lgb_weight: float = 0.9       # Weight for LightGBM
    threshold: float = 0.5        # Binary threshold
    use_physics_veto: bool = True # Apply physics veto
```

## Known Issues & Problem Chips

37 chips have IoU < 0.7, categorized as:

| Category | Count | Recommended Action |
|----------|-------|-------------------|
| MINOR_ISSUES | 11 | Post-processing |
| POOR_SEPARATION | 8 | Texture features |
| FLOOD_DYNAMICS | 8 | Exclude (temporal) |
| SEASONAL_DRY | 4 | Exclude |
| MINIMAL_WATER | 3 | Exclude |
| MODERATE_FAILURE | 2 | Investigate |
| MISALIGNED | 1 | SAR-only |

## Future Work

1. **Edge Case Handlers**: Specialized models for flood/seasonal/mangrove
2. **Deep Learning**: DeepLabV3+, HRNet, SegFormer
3. **Self-Training**: Pseudo-labeling for more training data
4. **Adaptive Thresholding**: Per water body type thresholds

## Citation

```
SAR Water Detection System for India
Physics-Guided Machine Learning Approach
2026
```

## License

MIT License

## Contact

SAR Water Detection Project Team
