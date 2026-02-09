# SAR Water Detection Lab

<div align="center">

![SAR Water Detection](https://img.shields.io/badge/SAR-Water%20Detection-blue)
![Python](https://img.shields.io/badge/Python-3.11%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Interactive tool for SAR-based water body detection using 47+ algorithms and deep learning**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Documentation](#documentation) • [Contributing](#contributing)

</div>

---

## 🌊 Overview

SAR Water Detection Lab is a production-ready, interactive web application for detecting water bodies from Synthetic Aperture Radar (SAR) imagery. It combines classical signal processing, machine learning, and deep learning approaches to provide researchers and practitioners with a comprehensive toolkit for water mapping.

### Key Capabilities

- **47+ Water Detection Algorithms**: From simple thresholding to advanced active contours
- **Interactive Filter Tuning**: Real-time parameter adjustment with visual feedback  
- **Ensemble Methods**: Combine multiple algorithms for robust detection
- **Deep Learning Integration**: U-Net and attention-based architectures
- **QA/Audit System**: Built-in quality assurance and version control
- **Export Pipeline**: Generate production-ready masks and recipes

## ✨ Features

### Algorithm Categories

1. **Pre-Processing**
   - RFI Filter, Refined Lee, Frost, Gamma MAP, BayesShrink Wavelet

2. **Radiometric Thresholding**
   - Otsu, Kittler-Illingworth, Triangle, Hysteresis, Sauvola Adaptive

3. **Derived Indices**
   - Cross-Pol Ratio, SDWI, SWI

4. **Texture Analysis**
   - GLCM Entropy/Variance, Coefficient of Variation

5. **Geometric Methods**
   - Touzi Edge, Frangi Vesselness, SRAD, Morphological Snake

6. **Hydro-Geomorphic**
   - HAND-based filtering, Shadow/Layover masks, TWI integration

7. **Machine Learning**
   - K-Means, LightGBM, Attention U-Net

### Interactive Features

- **15 Configurable Filter Windows**: Build complex multi-algorithm pipelines
- **Custom Equation Engine**: Write Python expressions for novel filters
- **Histogram Analysis**: Visualize data distributions and thresholds
- **Target Matcher**: Reverse-solve to achieve desired water coverage
- **Preset Library**: Pre-tuned configurations for different scenarios
- **Session Persistence**: Save/load your configurations

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/sar-water-detection-lab.git
cd sar-water-detection-lab/chips/gui

# Start with Docker Compose
docker-compose up -d

# Access at http://localhost:8501
```

### Option 2: Local Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment (optional)
export SAR_DATA_ROOT=/path/to/your/data

# Run application
streamlit run app.py
```

## 📁 Project Structure

```
sar-water-detection-lab/
├── app.py                      # Main Streamlit application
├── config.py                   # Centralized configuration
├── filter_engine_complete.py   # 47+ water detection algorithms
├── analysis_module.py          # Analysis utilities
├── qa_module.py                # Quality assurance system
├── presets.py                  # Pre-configured filter combinations
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Docker orchestration
├── .gitignore                  # Git ignore patterns
└── README.md                   # This file
```

## 🔧 Configuration

### Environment Variables

```bash
# Data directories
export SAR_DATA_ROOT=/path/to/data
export SAR_CHIP_DIR=$SAR_DATA_ROOT/chips/processed
export SAR_FEATURES_DIR=$SAR_DATA_ROOT/chips/processed/features_7band
export SAR_LABELS_DIR=$SAR_DATA_ROOT/chips/processed/labels_verified
export SAR_MODEL_DIR=$SAR_DATA_ROOT/chips/models

# Application settings
export SAR_ENV=production  # development, production, docker
export SAR_MAX_WORKERS=4
```

### Input Data Format

The application expects 7-band GeoTIFF files with the following structure:

```
Band 1: VV polarization (dB)
Band 2: VH polarization (dB)
Band 3: MNDWI (Modified Normalized Difference Water Index)
Band 4: DEM (Digital Elevation Model)
Band 5: HAND (Height Above Nearest Drainage)
Band 6: Slope (degrees)
Band 7: TWI (Topographic Wetness Index)
```

Place your chips in: `processed/features_7band/chip_XXX_features_7band_f32.tif`

## 📊 Usage Examples

### Basic Workflow

1. **Select a Chip**: Use the sidebar to choose your data chip
2. **Configure Filters**: Enable filter windows and adjust parameters
3. **Set Fusion Mode**: Choose Union (OR), Intersection (AND), or Majority Vote
4. **Visualize Results**: View the composite water mask
5. **Export**: Save your configuration and results

### Advanced: Custom Equations

```python
# Example: Combine VH threshold with terrain constraints
(vh < vh_t) & (hand < hand_t) & (slope < slope_t)

# Example: Use texture for wetland detection
(vh < -19.0) & (fe.glcm_entropy(vh, window_size=5) < 2.0)
```

### Preset Configurations

```python
# Load India-specific preset
from presets import INDIA_PRESETS
preset = INDIA_PRESETS['Urban Areas High Confidence']
```

## 🧪 Testing

```bash
# Run tests (requires pytest)
pytest tests/

# Run specific test
pytest tests/test_filter_engine.py -v
```

## 📈 Performance

- **Throughput**: ~10-15 chips/minute (512x512px, 7 bands)
- **Memory**: ~2-4GB per worker  
- **GPU**: Optional for deep learning models (8GB+ recommended)

## 🛠️ Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run with auto-reload
streamlit run app.py --server.runOnSave=true
```

### Code Style

```bash
# Format code
black *.py

# Lint
flake8 *.py

# Type check
mypy app.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **ESA Sentinel-1**: SAR imagery data source
- **Streamlit**: Interactive web framework
- **GDAL/Rasterio**: Geospatial data processing

## 📧 Contact

For questions, issues, or feature requests, please open an issue on GitHub.

## 🔗 Related Projects

- [Pune SAR Water Monitor](https://github.com/yourusername/pune-sar-water-monitor) - Full production pipeline
- [SAR Processing Toolkit](https://github.com/yourusername/sar-processing-toolkit) - SNAP-based preprocessing

---

**Built with ❤️ for the remote sensing community**
