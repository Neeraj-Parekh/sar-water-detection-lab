# Architecture Diagrams

This directory contains the system architecture diagrams for the SAR Water Detection Lab.

## Files

- **architecture.dot** - GraphViz DOT source file (located in project root)
- **architecture.png** - High-resolution PNG diagram (239 KB)
- **architecture.svg** - Scalable SVG diagram (28 KB)

## Architecture Overview

The SAR Water Detection Lab follows a 3-layer architecture:

### 1. UI Layer (Streamlit)
- **Sidebar**: Chip selection, presets loader, QA controls, settings
- **Main Canvas**: 15 filter windows, live preview, visual feedback, overlay modes
- **Controls Panel**: Fusion modes (OR/AND/Vote), export pipeline, QA validation, config save

### 2. Core Processing Engine
- **filter_engine_complete.py**: 47+ water detection algorithms
  - Radiometric (Otsu, CFAR, Kittler-Illingworth, Triangle)
  - Texture (GLCM Entropy/Variance, Touzi Edge, Haralick)
  - Morphological (Active Contours, Top-Hat, Area Filters, Watershed)
  - Geomorphic (HAND, TWI, Slope, Curvature)
  - ML/DL (Attention U-Net, LightGBM, Random Forest, SVM)
  - Custom (Equation Engine, Python Expressions, User-Defined)

- **Support Modules**:
  - analysis_module.py: Statistics & metrics
  - qa_module.py: Quality assurance system
  - presets.py: Filter presets management

### 3. Data Layer
- **config.py**: Configuration management
- **Features**: 7-band SAR data (VV/VH polarization, GLCM textures, derived indices)
- **Ground Truth**: JRC Water Layer, manual labels, validation data
- **Models**: Attention U-Net, LightGBM, pretrained weights

## Regenerating Diagrams

If you modify the architecture, regenerate the diagrams using:

```bash
# From project root directory
dot -Tpng architecture.dot -o docs/diagrams/architecture.png
dot -Tsvg architecture.dot -o docs/diagrams/architecture.svg
```

### Requirements
- GraphViz (install: `sudo apt install graphviz` or `brew install graphviz`)

## Color Scheme

- **Blue (#E3F2FD)**: UI Layer components
- **Orange (#FFF3E0)**: Core processing engine
- **Pink (#FCE4EC)**: Algorithm categories
- **Green (#F1F8E9)**: Data layer components
- **Gray (#E0E0E0)**: External systems

## Usage in Documentation

The architecture diagram is referenced in:
- README.md (main project documentation)
- DEPLOYMENT_GUIDE.md
- CONTRIBUTING.md
- LinkedIn posts and presentations

## License

These diagrams are part of the SAR Water Detection Lab project and are licensed under the MIT License.
