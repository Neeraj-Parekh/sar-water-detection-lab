"""
SAR Water Detection Lab - Configuration
========================================

Centralized configuration management for all paths and settings.
Automatically detects environment and uses appropriate paths.
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Central configuration for SAR Water Detection Lab"""
    
    # Environment detection
    ENV = os.getenv('SAR_ENV', 'development')  # development, production, docker
    
    # Base directories
    BASE_DIR = Path(__file__).parent.resolve()
    ROOT_DIR = BASE_DIR.parent.parent
    
    # Data directories (configurable via environment variables)
    DATA_ROOT = os.getenv('SAR_DATA_ROOT', str(ROOT_DIR))
    
    CHIP_DIR = Path(os.getenv('SAR_CHIP_DIR', f'{DATA_ROOT}/chips/processed'))
    FEATURES_DIR = Path(os.getenv('SAR_FEATURES_DIR', f'{DATA_ROOT}/chips/processed/features_7band'))
    LABELS_DIR = Path(os.getenv('SAR_LABELS_DIR', f'{DATA_ROOT}/chips/processed/labels_verified'))
    EXPORT_DIR = Path(os.getenv('SAR_EXPORT_DIR', f'{DATA_ROOT}/chips/processed/exports'))
    
    # Model directories
    MODEL_DIR = Path(os.getenv('SAR_MODEL_DIR', f'{DATA_ROOT}/chips/models'))
    RESULTS_DIR = Path(os.getenv('SAR_RESULTS_DIR', f'{DATA_ROOT}/chips/results'))
    VIZ_DIR = Path(os.getenv('SAR_VIZ_DIR', f'{DATA_ROOT}/chips/visualizations'))
    
    # Session and state
    SESSION_FILE = BASE_DIR / 'session_state.json'
    QA_FILE = BASE_DIR / 'audit_log.json'
    GOLDEN_SET_FILE = BASE_DIR / 'golden_set.json'
    
    # Application settings
    APP_TITLE = "SAR Water Detection Lab"
    APP_ICON = "🌊"
    MAX_WORKERS = int(os.getenv('SAR_MAX_WORKERS', '4'))
    
    # Band configuration
    BAND_NAMES = ['VV', 'VH', 'MNDWI', 'DEM', 'HAND', 'Slope', 'TWI']
    VV_IDX, VH_IDX, MNDWI_IDX = 0, 1, 2
    DEM_IDX, HAND_IDX, SLOPE_IDX, TWI_IDX = 3, 4, 5, 6
    
    # Filter configuration
    NUM_WINDOWS = 15
    
    @classmethod
    def ensure_directories(cls):
        """Create all necessary directories if they don't exist"""
        for directory in [
            cls.CHIP_DIR,
            cls.FEATURES_DIR,
            cls.LABELS_DIR,
            cls.EXPORT_DIR,
            cls.MODEL_DIR,
            cls.RESULTS_DIR,
            cls.VIZ_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_path(cls, model_name: str) -> Path:
        """Get full path for a model file"""
        return cls.MODEL_DIR / model_name
    
    @classmethod
    def get_chip_path(cls, chip_name: str, suffix: str = '_features_7band_f32.tif') -> Path:
        """Get full path for a chip file"""
        return cls.FEATURES_DIR / f'{chip_name}{suffix}'
    
    @classmethod
    def get_label_path(cls, chip_name: str) -> Path:
        """Get full path for a label file"""
        return cls.LABELS_DIR / f'{chip_name}_label_verified_u8.tif'
    
    @classmethod
    def summary(cls) -> str:
        """Return configuration summary"""
        return f"""
SAR Water Detection Lab Configuration
======================================
Environment: {cls.ENV}
Base Directory: {cls.BASE_DIR}
Data Root: {cls.DATA_ROOT}

Directories:
- Chips: {cls.CHIP_DIR}
- Features: {cls.FEATURES_DIR}
- Labels: {cls.LABELS_DIR}
- Models: {cls.MODEL_DIR}
- Results: {cls.RESULTS_DIR}
- Exports: {cls.EXPORT_DIR}

Settings:
- Max Workers: {cls.MAX_WORKERS}
- Bands: {', '.join(cls.BAND_NAMES)}
- Filter Windows: {cls.NUM_WINDOWS}
"""


# Initialize directories on import
Config.ensure_directories()
