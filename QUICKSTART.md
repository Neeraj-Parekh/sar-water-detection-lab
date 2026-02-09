# SAR Water Detection Lab - Quick Start Guide

## Installation & Setup

### Option 1: Docker (Recommended for Production)

```bash
# 1. Navigate to the project directory
cd chips/gui

# 2. Build and run with Docker Compose
docker-compose up -d

# 3. Access the application
open http://localhost:8501

# 4. Stop the application
docker-compose down
```

### Option 2: Local Installation (For Development)

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (copy and edit)
cp .env.example .env
# Edit .env and set SAR_DATA_ROOT to your data directory

# 4. Run the application
streamlit run app.py

# 5. Access at http://localhost:8501
```

## Preparing Your Data

### Required Directory Structure

```
your_data_directory/
├── chips/
│   └── processed/
│       ├── features_7band/
│       │   ├── chip_001_features_7band_f32.tif
│       │   ├── chip_002_features_7band_f32.tif
│       │   └── ...
│       ├── labels_verified/
│       │   ├── chip_001_label_verified_u8.tif
│       │   └── ...
│       └── exports/
│           └── (generated outputs will go here)
├── models/
│   └── (optional: pre-trained models)
└── results/
    └── (analysis results)
```

### Data Format

Each chip must be a 7-band GeoTIFF with 32-bit float values:

- **Band 1**: VV polarization (dB, range: -35 to 5)
- **Band 2**: VH polarization (dB, range: -35 to 5)
- **Band 3**: MNDWI (range: -1 to 1)
- **Band 4**: DEM (meters, range: 0 to 3000)
- **Band 5**: HAND (meters, range: 0 to 100)
- **Band 6**: Slope (degrees, range: 0 to 90)
- **Band 7**: TWI (range: 0 to 30)

## Basic Usage

### 1. Load a Chip

1. Launch the application
2. Use the sidebar to select a chip from the dropdown
3. Navigate with Prev/Next buttons

### 2. Apply Filters

1. Click on a filter window (1-15)
2. Select a filter type from the dropdown
3. Adjust parameters using sliders
4. Check "Include in Composite" to activate
5. View the result in the window

### 3. Combine Filters

Choose a fusion mode:
- **Union (OR)**: Combines all detections
- **Intersection (AND)**: Only where all agree
- **Majority Vote**: Require N/2 filters to agree

### 4. Export Results

1. Configure your filters
2. Click "Export Result"
3. Find outputs in `exports/` directory:
   - `*_mask.png`: Binary water mask
   - `*_recipe.json`: Configuration file

## Example Workflows

### Workflow 1: Simple VH Thresholding

```
Window 1: Simple Threshold
  - Band: VH
  - Threshold: -19.0 dB

Fusion Mode: Union
Result: Basic water detection
```

### Workflow 2: Multi-Parameter Ensemble

```
Window 1: Simple Threshold (VH < -19)
Window 2: HAND Definite (HAND < 5m)
Window 3: TWI High (TWI > 10)

Fusion Mode: Majority Vote
Result: High-confidence water bodies
```

### Workflow 3: Custom Equation

```
Window 1: Custom Equation
  Equation: (vh < -19.0) & (hand < 5.0) & (slope < 10.0)

Result: Terrain-constrained water detection
```

## Troubleshooting

### Issue: "No chips found"

**Solution**: 
1. Check that `SAR_FEATURES_DIR` is set correctly in your `.env`
2. Verify chips exist in the directory
3. Ensure chips follow naming convention: `chip_XXX_features_7band_f32.tif`

### Issue: "Module not found"

**Solution**:
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Issue: Docker container won't start

**Solution**:
```bash
# Check logs
docker-compose logs sar-lab

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## Advanced Features

### Load Presets

1. Go to sidebar > Presets
2. Select from dropdown (e.g., "Urban Areas High Confidence")
3. Click "Load Configuration"

### QA System

1. Review detection quality
2. Click traffic light buttons:
   - 🟢 Green: Pass
   - 🟡 Yellow: Warning
   - 🔴 Red: Fail
3. Audit log saved to `audit_log.json`

### Target Matcher

1. Click "Target Matcher" in sidebar
2. Enter desired water percentage
3. System suggests filter configurations

## Next Steps

- **Customize**: Edit `presets.py` to add your own filter combinations
- **Extend**: Add new filters to `filter_engine_complete.py`
- **Train Models**: Use generated masks to train ML models
- **Scale Up**: Process entire study areas using batch scripts

## Support

- **Issues**: https://github.com/Neeraj-Parekh/sar-water-detection-lab/issues
- **Docs**: See `README.md` for full documentation
- **Examples**: Check `docs/` for tutorials

---

Happy water mapping! 🌊
