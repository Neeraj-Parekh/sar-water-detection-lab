"""
SAR Water Detection Lab - Streamlit Application v3
===================================================

Full-featured application with:
- Reference Strip + 15 Filter Windows + Composite
- Per-window parameter sliders
- Target Matcher (reverse solver)
- Interactive histograms
- Manual editing (drawable canvas)
- Geo-sync map with location
- Session state persistence
- Export pipeline

Usage:
    cd chips/gui
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import rasterio
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
from datetime import datetime
from PIL import Image
import io

# Import filter engine
import filter_engine_complete as fe
import presets as pr
import qa_module as qa
import analysis_module as am
import numpy as np
import re
from config import Config


def extract_equation_params(equation_text):
    """
    Parse an equation string to extract numeric literals for dynamic slider creation.
    Returns a dict of {param_name: {'value': float, 'min': float, 'max': float, 'step': float}}
    
    Example: "(vv < -19.0) & (fe.glcm_entropy(vv, window_size=9) > 1.2)"
    Returns: {'p1': {'value': -19.0, ...}, 'window_size': {'value': 9, ...}, 'p2': {'value': 1.2, ...}}
    """
    params = {}
    counter = 1
    
    # Pattern 1: Named parameters like window_size=9, levels=32
    named_pattern = r'(\w+)\s*=\s*(-?[\d.]+)'
    for match in re.finditer(named_pattern, equation_text):
        name = match.group(1)
        value = float(match.group(2))
        
        # Determine range based on typical parameter usage
        if 'window' in name.lower() or 'size' in name.lower():
            params[name] = {'value': int(value), 'min': 3, 'max': 21, 'step': 2, 'is_int': True}
        elif 'level' in name.lower():
            params[name] = {'value': int(value), 'min': 8, 'max': 64, 'step': 8, 'is_int': True}
        elif 'iter' in name.lower():
            params[name] = {'value': int(value), 'min': 5, 'max': 50, 'step': 5, 'is_int': True}
        else:
            # Generic numeric parameter
            params[name] = {'value': value, 'min': value - 10, 'max': value + 10, 'step': 0.5, 'is_int': False}
    
    # Pattern 2: Comparison thresholds like < -19.0 or > 1.2
    # Exclude numbers that are part of named params (already captured)
    comparison_pattern = r'([<>]=?)\s*(-?[\d.]+)(?!\s*[,\)])'  # Avoid capturing func args
    for match in re.finditer(comparison_pattern, equation_text):
        operator = match.group(1)
        value = float(match.group(2))
        
        # Check if this value is already captured as named param
        already_captured = any(abs(p['value'] - value) < 0.001 for p in params.values())
        if already_captured:
            continue
        
        param_name = f'thresh_{counter}'
        counter += 1
        
        # Determine range based on value magnitude
        if value < -10:  # Likely dB threshold
            params[param_name] = {'value': value, 'min': -40.0, 'max': 0.0, 'step': 0.5, 'is_int': False}
        elif value < 0:
            params[param_name] = {'value': value, 'min': value - 5, 'max': 0.0, 'step': 0.1, 'is_int': False}
        elif value < 1:  # Likely ratio or normalized value
            params[param_name] = {'value': value, 'min': 0.0, 'max': 2.0, 'step': 0.05, 'is_int': False}
        elif value < 10:  # Small positive threshold
            params[param_name] = {'value': value, 'min': 0.0, 'max': 20.0, 'step': 0.5, 'is_int': False}
        else:  # Larger value
            params[param_name] = {'value': value, 'min': 0.0, 'max': value * 2, 'step': 1.0, 'is_int': False}
    
    return params


def substitute_equation_params(equation_text, params_dict):
    """
    Substitute numeric literals in equation with param values from sliders.
    """
    result = equation_text
    
    # Substitute named parameters first
    for name, spec in params_dict.items():
        if not name.startswith('thresh_'):
            # Named param like window_size=9 -> window_size=NEW_VALUE
            pattern = rf'({name}\s*=\s*)(-?[\d.]+)'
            replacement = rf'\g<1>{spec["value"]}'
            result = re.sub(pattern, replacement, result)
    
    # Substitute threshold values (trickier - need to match by position)
    # For simplicity, we replace in order found
    threshold_params = [(n, s) for n, s in params_dict.items() if n.startswith('thresh_')]
    threshold_params.sort(key=lambda x: int(x[0].split('_')[1]))
    
    for name, spec in threshold_params:
        # Find first numeric literal in comparisons and replace
        pattern = r'([<>]=?)\s*(-?[\d.]+)'
        def replacer(match):
            return f"{match.group(1)} {spec['value']}"
        result = re.sub(pattern, replacer, result, count=1)
    
    return result


# =============================================================================
# CONFIGURATION (from config.py)
# =============================================================================

# Use centralized configuration
FEATURES_DIR = Config.FEATURES_DIR
LABELS_DIR = Config.LABELS_DIR
EXPORT_DIR = Config.EXPORT_DIR
SESSION_FILE = Config.SESSION_FILE

# Band indices
BAND_NAMES = Config.BAND_NAMES
VV_IDX, VH_IDX, MNDWI_IDX = Config.VV_IDX, Config.VH_IDX, Config.MNDWI_IDX
DEM_IDX, HAND_IDX, SLOPE_IDX, TWI_IDX = Config.DEM_IDX, Config.HAND_IDX, Config.SLOPE_IDX, Config.TWI_IDX

NUM_WINDOWS = Config.NUM_WINDOWS
WATER_CMAP = LinearSegmentedColormap.from_list('water', ['white', '#1E90FF'])

# Filter specifications with parameter types
FILTER_SPECS = {
    'None': {},
    
    # --- 1. PRE-PROCESSING ---
    # Note: Pre-processing filters return filtered data, which is then thresholded
    'RFI Filter': {
        'z_threshold': {'type': 'slider', 'min': 2.0, 'max': 5.0, 'default': 3.0, 'step': 0.1},
        'threshold': {'type': 'slider', 'min': -30.0, 'max': -5.0, 'default': -17.0, 'step': 0.5}
    },
    'Refined Lee': {
        'window_size': {'type': 'slider', 'min': 3, 'max': 11, 'default': 7, 'step': 2},
        'threshold': {'type': 'slider', 'min': -30.0, 'max': -5.0, 'default': -17.0, 'step': 0.5}
    },
    'Frost Filter': {
        'window_size': {'type': 'slider', 'min': 3, 'max': 11, 'default': 5, 'step': 2},
        'damping': {'type': 'slider', 'min': 0.1, 'max': 5.0, 'default': 2.0, 'step': 0.1},
        'threshold': {'type': 'slider', 'min': -30.0, 'max': -5.0, 'default': -17.0, 'step': 0.5}
    },
    'Gamma MAP': {
        'window_size': {'type': 'slider', 'min': 3, 'max': 11, 'default': 5, 'step': 2},
        'num_looks': {'type': 'slider', 'min': 1, 'max': 10, 'default': 4, 'step': 1},
        'threshold': {'type': 'slider', 'min': -30.0, 'max': -5.0, 'default': -17.0, 'step': 0.5}
    },
    'BayesShrink Wavelet': {
        'levels': {'type': 'slider', 'min': 1, 'max': 5, 'default': 3, 'step': 1},
        'threshold': {'type': 'slider', 'min': -30.0, 'max': -5.0, 'default': -17.0, 'step': 0.5}
    },
    
    # --- 2. RADIOMETRIC THRESHOLDING ---
    'Simple Threshold': {
        'band': {'type': 'select', 'options': ['vh', 'vv', 'mndwi'], 'default': 'vh'},
        'threshold': {'type': 'slider', 'min': -30.0, 'max': 0.0, 'default': -17.0, 'step': 0.5}
    },
    'Otsu Threshold': {
        'band': {'type': 'select', 'options': ['vh', 'vv'], 'default': 'vh'}
    },
    'Kittler-Illingworth': {
        'band': {'type': 'select', 'options': ['vh', 'vv'], 'default': 'vh'}
    },
    'K-Dist CFAR': {
        'band': {'type': 'select', 'options': ['vh', 'vv'], 'default': 'vh'},
        'pfa': {'type': 'select', 'options': [1e-3, 1e-4, 1e-5], 'default': 1e-4},
        'num_looks': {'type': 'slider', 'min': 1, 'max': 10, 'default': 4, 'step': 1}
    },
    'G0-Distribution': {
        'band': {'type': 'select', 'options': ['vh', 'vv'], 'default': 'vh'}
    },
    'S1 (VV)': { # Alias for Simple Threshold VV
        'threshold': {'type': 'slider', 'min': -25.0, 'max': -5.0, 'default': -12.0, 'step': 0.5}
    },
    'S2 (VH)': { # Alias for Simple Threshold VH
        'threshold': {'type': 'slider', 'min': -30.0, 'max': -10.0, 'default': -19.0, 'step': 0.5}
    },
    'Adaptive (Sauvola)': { # Alias
        'window_size': {'type': 'slider', 'min': 11, 'max': 51, 'default': 31, 'step': 2},
        'k': {'type': 'slider', 'min': 0.1, 'max': 0.5, 'default': 0.2, 'step': 0.05}
    },
    'Triangle Method': {
        'band': {'type': 'select', 'options': ['vh', 'vv'], 'default': 'vh'}
    },
    'Hysteresis': {
        'band': {'type': 'select', 'options': ['vh', 'vv'], 'default': 'vh'},
        'low': {'type': 'slider', 'min': -30.0, 'max': -10.0, 'default': -21.0, 'step': 0.5},
        'high': {'type': 'slider', 'min': -25.0, 'max': -5.0, 'default': -16.0, 'step': 0.5}
    },
    'Sauvola Adaptive': {
        'band': {'type': 'select', 'options': ['vh', 'vv'], 'default': 'vh'},
        'window_size': {'type': 'slider', 'min': 11, 'max': 51, 'default': 31, 'step': 2},
        'k': {'type': 'slider', 'min': 0.1, 'max': 0.5, 'default': 0.2, 'step': 0.05}
    },
    'Maximum Entropy': {
        'band': {'type': 'select', 'options': ['vh', 'vv'], 'default': 'vh'}
    },
    
    # --- 3. INDICES ---
    'Cross-Pol Ratio': {
        'threshold': {'type': 'slider', 'min': 0.1, 'max': 2.0, 'default': 0.5, 'step': 0.1}
    },
    'SDWI': {
        'threshold': {'type': 'slider', 'min': -5.0, 'max': 5.0, 'default': 0.0, 'step': 0.5}
    },
    'SWI': {
        'threshold': {'type': 'slider', 'min': -20.0, 'max': 0.0, 'default': -10.0, 'step': 1.0}
    },
    
    # --- 4. TEXTURE ---
    'GLCM Entropy': {
        'band': {'type': 'select', 'options': ['vh', 'vv'], 'default': 'vh'},
        'window_size': {'type': 'slider', 'min': 3, 'max': 9, 'default': 5, 'step': 2},
        'threshold': {'type': 'slider', 'min': 0.0, 'max': 10.0, 'default': 2.0, 'step': 0.1}
    },
    'GLCM Variance': {
        'band': {'type': 'select', 'options': ['vh', 'vv'], 'default': 'vh'},
        'threshold': {'type': 'slider', 'min': 0.0, 'max': 50.0, 'default': 10.0, 'step': 1.0}
    },
    'Coeff of Variation': {
        'band': {'type': 'select', 'options': ['vh', 'vv'], 'default': 'vh'},
        'threshold': {'type': 'slider', 'min': 0.1, 'max': 2.0, 'default': 0.5, 'step': 0.1}
    },
    
    # --- 5. GEOMETRIC ---
    'Touzi Edge': {
        'window_size': {'type': 'slider', 'min': 3, 'max': 11, 'default': 7, 'step': 2},
        'threshold': {'type': 'slider', 'min': 1.0, 'max': 10.0, 'default': 3.0, 'step': 0.1}
    },
    'Frangi Vesselness': {
        'band': {'type': 'select', 'options': ['vh', 'vv'], 'default': 'vh'},
        'sigma': {'type': 'slider', 'min': 0.5, 'max': 5.0, 'default': 2.0, 'step': 0.5},
        'threshold': {'type': 'slider', 'min': 0.01, 'max': 0.5, 'default': 0.1, 'step': 0.01}
    },
    'SRAD': {
        'num_iter': {'type': 'slider', 'min': 5, 'max': 30, 'default': 15, 'step': 5},
        'dt': {'type': 'slider', 'min': 0.01, 'max': 0.2, 'default': 0.05, 'step': 0.01},
        'threshold': {'type': 'slider', 'min': -30.0, 'max': -5.0, 'default': -17.0, 'step': 0.5}
    },
    'Morph Snake': {
        'num_iter': {'type': 'slider', 'min': 10, 'max': 200, 'default': 50, 'step': 10},
        'balloon': {'type': 'slider', 'min': -1.0, 'max': 1.0, 'default': 1.0, 'step': 0.1}
    },
    
    # --- 6. MORPHOLOGICAL ---
    'Area Opening': {
        'min_size': {'type': 'slider', 'min': 10, 'max': 200, 'default': 50, 'step': 10}
    },
    'Area Closing': {
        'min_size': {'type': 'slider', 'min': 10, 'max': 200, 'default': 50, 'step': 10}
    },
    'White Top Hat': {
        'band': {'type': 'select', 'options': ['mndwi', 'vh'], 'default': 'mndwi'},
        'size': {'type': 'slider', 'min': 5, 'max': 25, 'default': 15, 'step': 2},
        'threshold': {'type': 'slider', 'min': 0.01, 'max': 0.2, 'default': 0.08, 'step': 0.01}
    },
    'Black Top Hat': {
        'band': {'type': 'select', 'options': ['vh', 'vv'], 'default': 'vh'},
        'size': {'type': 'slider', 'min': 5, 'max': 25, 'default': 15, 'step': 2},
        'threshold': {'type': 'slider', 'min': 0.1, 'max': 5.0, 'default': 2.0, 'step': 0.1}
    },
    
    # --- 7. HYDRO-GEOMORPHIC ---
    'HAND Definite': {
        'hand_thresh': {'type': 'slider', 'min': 1.0, 'max': 10.0, 'default': 3.0, 'step': 0.5}
    },
    'HAND Probable': {
        'hand_thresh': {'type': 'slider', 'min': 2.0, 'max': 15.0, 'default': 5.0, 'step': 0.5}
    },
    'HAND Possible': {
        'hand_thresh': {'type': 'slider', 'min': 5.0, 'max': 20.0, 'default': 10.0, 'step': 1.0}
    },
    'Shadow Mask': {
        'azimuth': {'type': 'slider', 'min': 0, 'max': 360, 'default': 100, 'step': 10},
        'elevation': {'type': 'slider', 'min': 10, 'max': 80, 'default': 45, 'step': 5}
    },
    'Layover Mask': {
        'incidence': {'type': 'slider', 'min': 20, 'max': 50, 'default': 35, 'step': 1}
    },
    'TWI High': {
        'min_twi': {'type': 'slider', 'min': 5.0, 'max': 20.0, 'default': 10.0, 'step': 1.0}
    },
    
    # --- 8. OTHERS ---
    'K-Means Clustering': {
        'k': {'type': 'slider', 'min': 2, 'max': 5, 'default': 2, 'step': 1}
    },
    'Hole Analysis': {
        'threshold': {'type': 'slider', 'min': -30.0, 'max': -10.0, 'default': -17.0, 'step': 0.5},
        'band': {'type': 'select', 'options': ['vh', 'vv'], 'default': 'vh'}
    },
    'Custom Equation': {
        'equation': {'type': 'text', 'default': '(vh < vh_t) & (hand < hand_t)'},
        # --- Dynamic: General thresholds ---
        'T': {'type': 'slider', 'min': -35.0, 'max': 0.0, 'default': -17.0, 'step': 0.5, 'dynamic': True},
        'S': {'type': 'slider', 'min': 0.0, 'max': 20.0, 'default': 5.0, 'step': 0.5, 'dynamic': True},
        'win': {'type': 'slider', 'min': 3, 'max': 15, 'default': 5, 'step': 2, 'dynamic': True},
        # --- Dynamic: Band-specific thresholds ---
        'vv_t': {'type': 'slider', 'min': -35.0, 'max': 0.0, 'default': -15.0, 'step': 0.5, 'dynamic': True},
        'vh_t': {'type': 'slider', 'min': -35.0, 'max': 0.0, 'default': -17.0, 'step': 0.5, 'dynamic': True},
        'mndwi_t': {'type': 'slider', 'min': -1.0, 'max': 1.0, 'default': 0.0, 'step': 0.05, 'dynamic': True},
        'hand_t': {'type': 'slider', 'min': 0.0, 'max': 30.0, 'default': 5.0, 'step': 1.0, 'dynamic': True},
        'slope_t': {'type': 'slider', 'min': 0.0, 'max': 45.0, 'default': 15.0, 'step': 1.0, 'dynamic': True},
        'twi_t': {'type': 'slider', 'min': 0.0, 'max': 20.0, 'default': 10.0, 'step': 0.5, 'dynamic': True},
        'dem_t': {'type': 'slider', 'min': 0.0, 'max': 500.0, 'default': 100.0, 'step': 10.0, 'dynamic': True},
        # --- Dynamic: Derived index thresholds ---
        'cpr_t': {'type': 'slider', 'min': 0.0, 'max': 2.0, 'default': 0.5, 'step': 0.05, 'dynamic': True},
        'sdwi_t': {'type': 'slider', 'min': -25.0, 'max': 0.0, 'default': -15.0, 'step': 0.5, 'dynamic': True},
        'swi_t': {'type': 'slider', 'min': -5.0, 'max': 5.0, 'default': 0.0, 'step': 0.2, 'dynamic': True},
        'entropy_t': {'type': 'slider', 'min': 0.0, 'max': 5.0, 'default': 2.0, 'step': 0.1, 'dynamic': True},
        # --- Dynamic: Window sizes ---
        'glcm_win': {'type': 'slider', 'min': 3, 'max': 15, 'default': 5, 'step': 2, 'dynamic': True},
    },
    
    
    # --- 9. HYBRID ---
    'Split-Logic Fusion': {
        'vh_thresh_open': {'type': 'slider', 'min': -30.0, 'max': -10.0, 'default': -21.0, 'step': 0.5},
        'hand_thresh_open': {'type': 'slider', 'min': 5.0, 'max': 30.0, 'default': 15.0, 'step': 1.0},
        'ratio_thresh_urban': {'type': 'slider', 'min': 1.0, 'max': 10.0, 'default': 5.0, 'step': 0.5},
        'hand_thresh_urban': {'type': 'slider', 'min': 1.0, 'max': 15.0, 'default': 5.0, 'step': 0.5}
    },
    'Lake Filter': {
        'min_size': {'type': 'slider', 'min': 100, 'max': 5000, 'default': 500, 'step': 100}
    },
    'Wetland Filter': {
        'texture_thresh': {'type': 'slider', 'min': 0.1, 'max': 1.0, 'default': 0.3, 'step': 0.05}
    },
    'Dominant Filter': {
        'window_size': {'type': 'slider', 'min': 3, 'max': 11, 'default': 5, 'step': 2}
    },
    'Fusion Preset': {
        'mode': {'type': 'select', 'options': ['high_conf', 'med_conf'], 'default': 'high_conf'}
    }
}

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def get_chip_list():
    """Get list of available chips."""
    chips = sorted(FEATURES_DIR.glob('chip_*_features_7band_f32.tif'))
    return [f.stem.replace('_features_7band_f32', '') for f in chips]


@st.cache_data
def load_chip(chip_name):
    """Load a chip's 7-band features."""
    filepath = FEATURES_DIR / f'{chip_name}_features_7band_f32.tif'
    if not filepath.exists():
        return None
    
    with rasterio.open(filepath) as src:
        data = src.read()
        # Get geotransform for coordinates
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
    
    return {
        'vv': data[VV_IDX],
        'vh': data[VH_IDX],
        'mndwi': data[MNDWI_IDX],
        'dem': data[DEM_IDX],
        'hand': data[HAND_IDX],
        'slope': data[SLOPE_IDX],
        'twi': data[TWI_IDX],
        'raw': data,
        'bounds': bounds,
        'crs': str(crs) if crs else None
    }


@st.cache_data
def load_verified_label(chip_name):
    """Load verified label if exists."""
    filepath = LABELS_DIR / f'{chip_name}_label_verified_u8.tif'
    if not filepath.exists():
        return None
    
    with rasterio.open(filepath) as src:
        label = src.read(1)
    
    return label > 0



def load_preset(preset_name):
    """Load a preset configuration."""
    preset = pr.INDIA_PRESETS.get(preset_name)
    if not preset:
        return
    
    # Apply windows
    valid_windows = preset['windows']
    # Clear existing
    for i in range(NUM_WINDOWS):
        st.session_state[f'window_{i}_include'] = False
    
    # Set new
    for i, win_cfg in enumerate(valid_windows):
        st.session_state[f'window_{i}_filter'] = win_cfg['filter']
        st.session_state[f'window_{i}_params'] = win_cfg['params']
        st.session_state[f'window_{i}_include'] = True
        
    st.session_state.composite_mode = preset['fusion_mode']
    st.success(f"Loaded preset: {preset_name}")
    st.rerun()


# =============================================================================
# FILTER HISTORY (UNDO/REDO)
# =============================================================================

def save_filter_snapshot():
    """Save current filter configuration to history."""
    snapshot = {
        'windows': [],
        'composite_mode': st.session_state.composite_mode,
        'timestamp': str(np.datetime64('now'))
    }
    for i in range(NUM_WINDOWS):
        snapshot['windows'].append({
            'filter': st.session_state[f'window_{i}_filter'],
            'params': st.session_state[f'window_{i}_params'].copy(),
            'include': st.session_state[f'window_{i}_include']
        })
    
    # Add to history (limit to last 20 states)
    if 'filter_history' not in st.session_state:
        st.session_state.filter_history = []
    st.session_state.filter_history.append(snapshot)
    if len(st.session_state.filter_history) > 20:
        st.session_state.filter_history = st.session_state.filter_history[-20:]
    st.session_state.filter_history_index = len(st.session_state.filter_history) - 1


def restore_filter_snapshot(index):
    """Restore filter configuration from history."""
    if 'filter_history' not in st.session_state:
        return False
    if index < 0 or index >= len(st.session_state.filter_history):
        return False
    
    snapshot = st.session_state.filter_history[index]
    for i, win_cfg in enumerate(snapshot['windows']):
        st.session_state[f'window_{i}_filter'] = win_cfg['filter']
        st.session_state[f'window_{i}_params'] = win_cfg['params'].copy()
        st.session_state[f'window_{i}_include'] = win_cfg['include']
    st.session_state.composite_mode = snapshot['composite_mode']
    st.session_state.filter_history_index = index
    return True


def undo_filter():
    """Undo to previous filter state."""
    if 'filter_history' not in st.session_state or len(st.session_state.filter_history) < 2:
        return False
    current_idx = st.session_state.get('filter_history_index', len(st.session_state.filter_history) - 1)
    if current_idx > 0:
        return restore_filter_snapshot(current_idx - 1)
    return False


# =============================================================================
# SESSION STATE
# =============================================================================

def save_session():
    """Save session state to JSON."""
    state = {
        'current_chip': st.session_state.current_chip,
        'composite_mode': st.session_state.composite_mode,
        'timestamp': datetime.now().isoformat(),
        'windows': {}
    }
    
    for i in range(NUM_WINDOWS):
        state['windows'][f'window_{i}'] = {
            'filter': st.session_state.get(f'window_{i}_filter', 'None'),
            'params': st.session_state.get(f'window_{i}_params', {}),
            'include': st.session_state.get(f'window_{i}_include', False)
        }
    
    with open(SESSION_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def load_session():
    """Load session state from JSON."""
    if not SESSION_FILE.exists():
        return False
    
    try:
        with open(SESSION_FILE, 'r') as f:
            state = json.load(f)
        
        st.session_state.current_chip = state.get('current_chip')
        st.session_state.composite_mode = state.get('composite_mode', 'Union (OR)')
        
        for i in range(NUM_WINDOWS):
            win_state = state.get('windows', {}).get(f'window_{i}', {})
            st.session_state[f'window_{i}_filter'] = win_state.get('filter', 'None')
            st.session_state[f'window_{i}_params'] = win_state.get('params', {})
            st.session_state[f'window_{i}_include'] = win_state.get('include', False)
        
        return True
    except:
        return False


# =============================================================================
# FILTER APPLICATION
# =============================================================================

def apply_filter(filter_name, chip_data, params):
    """Apply a filter and return binary mask."""
    if filter_name == 'None':
        return None

    # --- 1. PRE-PROCESSING ---
    # NOTE: Pre-processing filters return FILTERED data, not masks.
    # We apply a threshold to the filtered data to create a mask.
    # The threshold should be user-adjustable if possible.
    
    elif filter_name == 'RFI Filter':
        z = params.get('z_threshold', 3.0)
        cleaned, rfi_mask = fe.rfi_filter_simple(chip_data['vh'], z_threshold=z)
        # Apply threshold to CLEANED data (not the RFI mask itself)
        thresh = params.get('threshold', -17.0)
        return cleaned < thresh
        
    elif filter_name == 'Refined Lee':
        win = params.get('window_size', 7)
        filtered = fe.refined_lee_filter(chip_data['vh'], window_size=win)
        thresh = params.get('threshold', -17.0)
        return filtered < thresh
        
    elif filter_name == 'Frost Filter':
        win = params.get('window_size', 5)
        damp = params.get('damping', 2.0)
        filtered = fe.frost_filter(chip_data['vh'], window_size=win, damping=damp)
        thresh = params.get('threshold', -17.0)
        return filtered < thresh
        
    elif filter_name == 'Gamma MAP':
        win = params.get('window_size', 5)
        looks = params.get('num_looks', 4)
        filtered = fe.gamma_map_filter(chip_data['vh'], window_size=win, num_looks=looks)
        thresh = params.get('threshold', -17.0)
        return filtered < thresh
        
    elif filter_name == 'BayesShrink Wavelet':
        levels = params.get('levels', 3)
        filtered = fe.bayesshrink_wavelet(chip_data['vh'], levels=levels)
        thresh = params.get('threshold', -17.0)
        return filtered < thresh
        # Wait, pre-processing filters return FILTERED DATA, not MASK.
        # This architecture assumes filters return MASKS.
        # We should default to thresholding the filtered result.
        # Adding a default threshold of -17dB for speckle filters.
        # filtered = fe.refined_lee_filter(chip_data['vh'], window_size=win)
        # return filtered < -17.0
    
    # --- 2. RADIOMETRIC THRESHOLDING ---
    elif filter_name == 'Simple Threshold':
        band = params.get('band', 'vh')
        return chip_data[band] < params.get('threshold', -17.0)
    
    elif filter_name == 'Otsu Threshold':
        band = params.get('band', 'vh')
        thresh = fe.numpy_otsu(chip_data[band])
        return chip_data[band] < thresh
        
    elif filter_name == 'Kittler-Illingworth':
        band = params.get('band', 'vh')
        thresh = fe.kittler_illingworth(chip_data[band])
        return chip_data[band] < thresh
        
    elif filter_name == 'K-Dist CFAR':
        band = params.get('band', 'vh')
        return fe.k_distribution_cfar(chip_data[band], pfa=params.get('pfa', 1e-4), num_looks=params.get('num_looks', 4))
        
    elif filter_name == 'G0-Distribution':
        band = params.get('band', 'vh')
        return fe.g0_distribution_threshold(chip_data[band])
        
    elif filter_name == 'Triangle Method':
        band = params.get('band', 'vh')
        thresh = fe.triangle_threshold(chip_data[band])
        return chip_data[band] < thresh
        
    elif filter_name == 'Hysteresis':
        band = params.get('band', 'vh')
        return fe.hysteresis_threshold(chip_data[band], low=params.get('low', -21.0), high=params.get('high', -16.0))
        
    elif filter_name == 'Sauvola Adaptive':
        band = params.get('band', 'vh')
        return fe.sauvola_threshold(chip_data[band], window_size=params.get('window_size', 31), k=params.get('k', 0.2))
        
    elif filter_name == 'Maximum Entropy':
        band = params.get('band', 'vh')
        thresh = fe.maximum_entropy_threshold(chip_data[band])
        return chip_data[band] < thresh

    # --- 3. INDICES ---
    elif filter_name == 'Cross-Pol Ratio':
        # Water has LOW VH/VV ratio (specular reflection, minimal depolarization)
        # High CPR indicates vegetation/volume scattering
        cpr = fe.cross_pol_ratio(chip_data['vv'], chip_data['vh'])
        return cpr < params.get('threshold', 0.5)  # Changed > to < for water detection
        
    elif filter_name == 'SDWI':
        val = fe.sdwi(chip_data['vv'], chip_data['vh'])
        return val > params.get('threshold', 0.0)
        
    elif filter_name == 'SWI':
        val = fe.swi(chip_data['vv'], chip_data['vh'])
        return val > params.get('threshold', -10.0)

    # --- 4. TEXTURE ---
    elif filter_name == 'GLCM Entropy':
        band = params.get('band', 'vh')
        ent = fe.glcm_entropy(chip_data[band], window_size=params.get('window_size', 5))
        return ent < params.get('threshold', 2.0) # Low entropy = smooth = water
        
    elif filter_name == 'GLCM Variance':
        band = params.get('band', 'vh')
        var = fe.glcm_variance(chip_data[band])
        return var < params.get('threshold', 10.0)
        
    elif filter_name == 'Coeff of Variation':
        cov = fe.coefficient_of_variation(chip_data['vh'])
        return cov < params.get('threshold', 0.5)

    # --- 5. GEOMETRIC ---
    elif filter_name == 'Touzi Edge':
        # Edge detector returns edges, not water. 
        # Typically used to bound water. 
        edges = fe.touzi_ratio_edge(chip_data['vh'], window_size=params.get('window_size', 7))
        return edges > params.get('threshold', 3.0)
        
    elif filter_name == 'Frangi Vesselness':
        band = params.get('band', 'vh')
        vessel = fe.frangi_vesselness(chip_data[band], sigma=params.get('sigma', 2.0))
        return vessel > params.get('threshold', 0.1)
        
    elif filter_name == 'SRAD':
        # Speckle filter, return thresholded
        # Now returns dB
        filtered = fe.srad_filter(chip_data['vh'], num_iter=params.get('num_iter', 15), dt=params.get('dt', 0.05))
        return filtered < params.get('threshold', -17.0)
        
    elif filter_name == 'Morph Snake':
        # Active contour requires init. Use simple threshold as init.
        init_mask = chip_data['vh'] < -17.0
        return fe.morphological_snake(chip_data['vh'], init_mask, num_iter=params.get('num_iter', 50), balloon=params.get('balloon', 1.0))

    # --- 6. MORPHOLOGICAL ---
    elif filter_name == 'Area Opening':
        base = chip_data['vh'] < -17.0
        return fe.area_opening(base, min_size=params.get('min_size', 50))
        
    elif filter_name == 'Area Closing':
        base = chip_data['vh'] < -17.0
        return fe.area_closing(base, min_size=params.get('min_size', 50))
        
    elif filter_name == 'White Top Hat':
        band = params.get('band', 'mndwi')
        thresh = params.get('threshold', 0.08)
        mask, _ = fe.white_top_hat(np.nan_to_num(chip_data[band], nan=-1), size=params.get('size', 15), threshold=thresh)
        return mask
        
    elif filter_name == 'Black Top Hat':
        band = params.get('band', 'vh')
        mask, _ = fe.black_top_hat(chip_data[band], size=params.get('size', 15), threshold=params.get('threshold', 2.0))
        return mask

    # --- 7. HYDRO-GEOMORPHIC ---
    elif filter_name == 'HAND Definite':
        # Re-implement using simple threshold on HAND
        # fe.hand_definite is logic-based, let's use direct params
        return (chip_data['hand'] < params.get('hand_thresh', 3.0)) & (chip_data['vh'] < -15.0)
        
    elif filter_name == 'HAND Probable':
        return (chip_data['hand'] < params.get('hand_thresh', 5.0)) & (chip_data['vh'] < -15.0)
        
    elif filter_name == 'HAND Possible':
        return chip_data['hand'] < params.get('hand_thresh', 10.0)
        
    elif filter_name == 'Shadow Mask':
        return fe.shadow_mask_raycast(chip_data['dem'], azimuth=params.get('azimuth', 100), elevation=params.get('elevation', 45))
        
    elif filter_name == 'Layover Mask':
        return fe.layover_mask_simple(chip_data['dem'], incidence_angle=params.get('incidence', 35))
        
    elif filter_name == 'TWI High':
        return chip_data['twi'] > params.get('min_twi', 10.0)
    
    # --- 8. OTHERS ---
    elif filter_name == 'K-Means Clustering':
        labels, centers = fe.simple_kmeans(chip_data['vh'], k=params.get('k', 2))
        # Assume lowest center is water
        water_label = np.argmin(centers)
        return labels == water_label

    elif filter_name == 'Hole Analysis':
        # Step 1: Generate binary mask using threshold
        threshold = params.get('threshold', -17.0)
        band = params.get('band', 'vh')
        initial_mask = chip_data[band] < threshold
        
        # Step 2: Analyze holes
        analysis = fe.hole_compactness_analysis(initial_mask)
        
        # Step 3: Return irregular holes (potential false negatives)
        # These are holes inside water that might actually be water
        return analysis['irregular_holes_mask']

    elif filter_name == 'Custom Equation':
        equation = params.get('equation', '(vh < vh_t) & (hand < hand_t)')
        try:
            # Get user-controllable parameters - General
            T = params.get('T', -17.0)
            S = params.get('S', 5.0)
            win = params.get('win', 5)
            glcm_win = params.get('glcm_win', 5)
            
            # Get band-specific thresholds
            vv_t = params.get('vv_t', -15.0)
            vh_t = params.get('vh_t', -17.0)
            mndwi_t = params.get('mndwi_t', 0.0)
            hand_t = params.get('hand_t', 5.0)
            slope_t = params.get('slope_t', 15.0)
            twi_t = params.get('twi_t', 10.0)
            dem_t = params.get('dem_t', 100.0)
            
            # Get derived index thresholds
            cpr_t = params.get('cpr_t', 0.5)
            sdwi_t = params.get('sdwi_t', -15.0)
            swi_t = params.get('swi_t', 0.0)
            entropy_t = params.get('entropy_t', 2.0)
            
            # Pre-compute derived arrays for use in equations
            glcm_entropy = fe.glcm_entropy(chip_data['vh'], window_size=glcm_win)
            cpr = fe.cross_pol_ratio(chip_data['vv'], chip_data['vh'])
            sdwi = fe.sdwi(chip_data['vv'], chip_data['vh'])
            swi = fe.swi(chip_data['vv'], chip_data['vh'])
            
            local_vars = {
                # Raw bands
                'vv': chip_data['vv'],
                'vh': chip_data['vh'],
                'mndwi': chip_data['mndwi'],
                'hand': np.nan_to_num(chip_data['hand'], nan=999),
                'slope': chip_data['slope'],
                'twi': chip_data['twi'],
                'dem': chip_data['dem'],
                # Pre-computed derived arrays
                'glcm_entropy': glcm_entropy,
                'cpr': cpr,
                'sdwi': sdwi,
                'swi': swi,
                # General parameters
                'T': T,
                'S': S,
                'win': win,
                'glcm_win': glcm_win,
                # Band-specific thresholds
                'vv_t': vv_t,
                'vh_t': vh_t,
                'mndwi_t': mndwi_t,
                'hand_t': hand_t,
                'slope_t': slope_t,
                'twi_t': twi_t,
                'dem_t': dem_t,
                # Derived index thresholds
                'cpr_t': cpr_t,
                'sdwi_t': sdwi_t,
                'swi_t': swi_t,
                'entropy_t': entropy_t,
                # Utilities
                'np': np,
                'fe': fe  # Access to all filters
            }
            
            # Add auto-detected params (prefixed with _auto_)
            for key, value in params.items():
                if key.startswith('_auto_'):
                    # Extract the actual param name (after _auto_)
                    auto_name = key[6:]  # Remove '_auto_' prefix
                    local_vars[auto_name] = value
            
            # Substitute auto-detected numeric literals in equation with slider values
            auto_params = extract_equation_params(equation)
            if auto_params:
                substituted_params = {}
                for ap_name, ap_spec in auto_params.items():
                    # Use slider value if available, else original
                    slider_val = params.get(f'_auto_{ap_name}', ap_spec['value'])
                    substituted_params[ap_name] = {'value': slider_val, **ap_spec}
                equation = substitute_equation_params(equation, substituted_params)
            
            result = eval(equation, {"__builtins__": {}}, local_vars)
            return result.astype(bool)
        except Exception as e:
            st.error(f"Equation error: {e}")
            return None
            
    elif filter_name == 'Split-Logic Fusion':
        return fe.split_logic_fusion(
            chip_data['vv'], 
            chip_data['vh'], 
            chip_data['hand'], 
            mndwi=chip_data['mndwi'],
            vh_thresh_open=params.get('vh_thresh_open', -21.0),
            hand_thresh_open=params.get('hand_thresh_open', 15.0),
            ratio_thresh_urban=params.get('ratio_thresh_urban', 5.0),
            hand_thresh_urban=params.get('hand_thresh_urban', 5.0)
        )
            
    return None


# =============================================================================
# FUSION FUNCTIONS
# =============================================================================

def fusion_union(*masks):
    """Union (OR) of all masks."""
    if not masks:
        return None
    result = masks[0].copy()
    for m in masks[1:]:
        result = result | m
    return result


def fusion_intersection(*masks):
    """Intersection (AND) of all masks."""
    if not masks:
        return None
    result = masks[0].copy()
    for m in masks[1:]:
        result = result & m
    return result


def fusion_vote(masks, min_votes=2):
    """Majority voting fusion."""
    if not masks:
        return None
    stack = np.stack([m.astype(int) for m in masks], axis=0)
    votes = stack.sum(axis=0)
    return votes >= min_votes


def morphological_cleanup(mask, min_size=50):
    """Clean up small objects and holes."""
    from scipy.ndimage import binary_opening, binary_closing
    cleaned = binary_opening(mask, structure=np.ones((3, 3)))
    cleaned = binary_closing(cleaned, structure=np.ones((3, 3)))
    return cleaned


def calc_water_pct(mask):
    """Calculate water percentage."""
    if mask is None:
        return 0.0
    return mask.sum() / mask.size * 100


def threshold_simple(data, threshold=-17.0):
    """Simple threshold."""
    return data < threshold


def hand_definite(hand, mndwi, vh, hand_thresh=3.0):
    """HAND definite water detection."""
    return (hand < hand_thresh) & (vh < -15.0)


def threshold_hysteresis(data, low=-21.0, high=-16.0):
    """Hysteresis thresholding."""
    return fe.hysteresis_threshold(data, low=low, high=high)


# =============================================================================
# TARGET MATCHER
# =============================================================================

def target_matcher(chip_data, target_pct, tolerance=2.0):
    """Find filter configurations that achieve target water percentage."""
    results = []
    
    for thresh in np.arange(-25.0, -10.0, 0.5):
        mask = threshold_simple(chip_data['vh'], threshold=thresh)
        pct = mask.sum() / mask.size * 100
        if abs(pct - target_pct) <= tolerance:
            results.append({
                'filter': 'Simple Threshold',
                'params': {'band': 'vh', 'threshold': thresh},
                'water_pct': pct
            })
    
    for hand_t in np.arange(1.0, 15.0, 1.0):
        mask = hand_definite(chip_data['hand'], chip_data['mndwi'], chip_data['vh'], hand_thresh=hand_t)
        pct = mask.sum() / mask.size * 100
        if abs(pct - target_pct) <= tolerance:
            results.append({
                'filter': 'HAND Definite',
                'params': {'hand_thresh': hand_t},
                'water_pct': pct
            })
    
    for low in np.arange(-25.0, -15.0, 1.0):
        for high in np.arange(low + 3, -10.0, 1.0):
            mask = threshold_hysteresis(chip_data['vh'], low=low, high=high)
            pct = mask.sum() / mask.size * 100
            if abs(pct - target_pct) <= tolerance:
                results.append({
                    'filter': 'Hysteresis',
                    'params': {'band': 'vh', 'low': low, 'high': high},
                    'water_pct': pct
                })
                break
    
    results.sort(key=lambda x: abs(x['water_pct'] - target_pct))
    return results[:15]


# =============================================================================
# EXPORT (Golden Zipper)
# =============================================================================

def export_result(chip_name, composite_mask, recipe, mode):
    """Export result package."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f"{chip_name}_{timestamp}"
    
    # Export mask as PNG
    mask_path = EXPORT_DIR / f"{base_name}_mask.png"
    mask_img = (composite_mask.astype(np.uint8) * 255)
    Image.fromarray(mask_img).save(mask_path)
    
    # Export recipe as JSON
    recipe_path = EXPORT_DIR / f"{base_name}_recipe.json"
    with open(recipe_path, 'w') as f:
        json.dump({
            'chip_name': chip_name,
            'mode': mode,
            'recipe': recipe,
            'water_pct': composite_mask.sum() / composite_mask.size * 100,
            'timestamp': timestamp
        }, f, indent=2)
    
    return mask_path, recipe_path


# =============================================================================
# VISUALIZATION
# =============================================================================

def render_band(data, cmap='gray', vmin=None, vmax=None):
    """Render a band as matplotlib figure."""
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig


def render_rgb_composite(chip_data):
    """
    Render Physics-Based False Color Composite.
    Red: VV (Surface Scattering) - normalized
    Green: VH (Volume Scattering) - normalized
    Blue: VV/VH Ratio (Double Bounce/Water) - normalized
    """
    # Normalize helper
    def norm(data, min_v, max_v):
        return np.clip((data - min_v) / (max_v - min_v), 0, 1)

    # 1. Red = VV (Range: -25 to 0 dB typical)
    R = norm(chip_data['vv'], -25, 0)
    
    # 2. Green = VH (Range: -30 to -10 dB typical)
    G = norm(chip_data['vh'], -30, -10)
    
    # 3. Blue = Ratio (Range: 0 to 1 for calculation, but typically 0.2 to 2.0 linear)
    # We compute Log Ratio for visualization: 10*log10(VV/VH) = VV_dB - VH_dB
    ratio_db = chip_data['vv'] - chip_data['vh']
    # Range: 0 dB to 15 dB
    B = norm(ratio_db, 0, 12)
    
    # Stack
    rgb = np.dstack((R, G, B))
    
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(rgb)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig


def render_mask(mask, water_pct=None):
    """Render binary mask."""
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(mask, cmap=WATER_CMAP, interpolation='nearest')
    ax.axis('off')
    if water_pct is not None:
        ax.set_title(f'{water_pct:.1f}%', fontsize=10)
    plt.tight_layout(pad=0)
    return fig


def render_histogram(data, threshold=None, title='Histogram'):
    """Render histogram with threshold line."""
    fig, ax = plt.subplots(figsize=(4, 2))
    flat = data[~np.isnan(data)].flatten()
    ax.hist(flat, bins=100, color='steelblue', alpha=0.7, edgecolor='none')
    if threshold is not None:
        ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    return fig


def render_swipe_comparison(base_data, mask, position=0.5):
    """Render swipe comparison view."""
    fig, ax = plt.subplots(figsize=(6, 6))
    h, w = base_data.shape
    split = int(w * position)
    
    # Left: raw data
    ax.imshow(base_data, cmap='gray', vmin=-30, vmax=-10)
    
    # Right: mask overlay
    mask_rgba = np.zeros((h, w, 4))
    mask_rgba[mask, 0] = 0.12  # R
    mask_rgba[mask, 1] = 0.56  # G
    mask_rgba[mask, 2] = 1.0   # B
    mask_rgba[mask, 3] = 0.7   # Alpha
    mask_rgba[:, :split, 3] = 0  # Hide left side
    
    ax.imshow(mask_rgba)
    ax.axvline(x=split, color='yellow', linewidth=2)
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig


# =============================================================================
# MAP VIEW
# =============================================================================

def render_location_map(bounds, mask=None):
    """Render a simple location map using matplotlib with boundaries."""
    if bounds is None:
        return None
    
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Draw bounding box
    minx, miny, maxx, maxy = bounds
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    
    ax.set_xlim(minx - 0.01, maxx + 0.01)
    ax.set_ylim(miny - 0.01, maxy + 0.01)
    
    # Draw chip boundary
    from matplotlib.patches import Rectangle
    rect = Rectangle((minx, miny), maxx - minx, maxy - miny,
                     linewidth=2, edgecolor='red', facecolor='lightblue', alpha=0.3)
    ax.add_patch(rect)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Location: {center_y:.4f}°N, {center_x:.4f}°E', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# SESSION STATE INIT
# =============================================================================

def init_session_state():
    """Initialize session state."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.current_chip = None
        st.session_state.chip_data = None
        st.session_state.show_histogram = False
        st.session_state.show_map = False
        st.session_state.show_swipe = False
        st.session_state.target_matcher_results = None
        st.session_state.manual_edits = None
        st.session_state.edit_mode = 'None'
        
        # Filter history for undo functionality
        st.session_state.filter_history = []  # List of snapshots
        st.session_state.filter_history_index = -1  # Current position
        
        for i in range(NUM_WINDOWS):
            st.session_state[f'window_{i}_filter'] = 'None'
            st.session_state[f'window_{i}_params'] = {}
            st.session_state[f'window_{i}_include'] = False
            st.session_state[f'window_{i}_mask'] = None
        
        st.session_state.composite_mode = 'Union (OR)'
        
        load_session()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="SAR Water Detection Lab",
        page_icon="🌊",
        layout="wide"
    )
    
    init_session_state()
    
    # Custom CSS
    st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    div[data-testid="stExpander"] { background-color: #1e1e2e; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)
    
    # =========================================================================
    # SIDEBAR
    # =========================================================================
    with st.sidebar:
        st.title("🌊 SAR Lab")
        
        # Chip Selection
        st.header("📂 Chip Selection")
        chips = get_chip_list()
        if not chips:
            st.error("No chips found!")
            return
        
        chip_idx = 0
        if st.session_state.current_chip and st.session_state.current_chip in chips:
            chip_idx = chips.index(st.session_state.current_chip)
        
        selected_chip = st.selectbox("Chip", chips, index=chip_idx, label_visibility='collapsed')
        
        if selected_chip != st.session_state.current_chip:
            st.session_state.current_chip = selected_chip
            st.session_state.chip_data = load_chip(selected_chip)
            st.session_state.manual_edits = None
        
        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Prev", use_container_width=True):
                idx = chips.index(st.session_state.current_chip)
                st.session_state.current_chip = chips[(idx - 1) % len(chips)]
                st.session_state.chip_data = None
                st.session_state.manual_edits = None
                st.rerun()
        with col2:
            if st.button("➡️ Next", use_container_width=True):
                idx = chips.index(st.session_state.current_chip)
                st.session_state.current_chip = chips[(idx + 1) % len(chips)]
                st.session_state.chip_data = None
                st.session_state.manual_edits = None
                st.rerun()
        
        st.divider()
        
        # Composite Mode
        st.header("🔧 Fusion Mode")
        st.session_state.composite_mode = st.radio(
            "Mode",
            ['Union (OR)', 'Intersection (AND)', 'Majority Vote'],
            index=['Union (OR)', 'Intersection (AND)', 'Majority Vote'].index(st.session_state.composite_mode),
            label_visibility='collapsed'
        )
        
        st.divider()
        

        st.sidebar.divider()
        st.sidebar.subheader("📚 Presets")
        preset_name = st.sidebar.selectbox("Load Preset", ["Select..."] + list(pr.INDIA_PRESETS.keys()))
        if preset_name != "Select...":
            if st.sidebar.button("Load Configuration"):
                load_preset(preset_name)
                
        st.sidebar.divider()
        st.sidebar.subheader("🚦 QA & Audit")
        
        # QA Status
        current_status = qa.get_chip_status(st.session_state.current_chip)['status']
        
        # Traffic Light Visual
        colors = {'green': '🟢', 'yellow': '🟡', 'red': '🔴', 'none': '⚪'}
        st.markdown(f"#### Status: {colors.get(current_status, '⚪')} {current_status.upper()}")
        
        c1, c2, c3 = st.columns(3)
        if c1.button("🟢 Pass", use_container_width=True):
            qa.set_chip_status(st.session_state.current_chip, 'green')
            st.toast("Status: Green (Pass)")
            st.rerun()
        if c2.button("🟡 Warn", use_container_width=True):
            qa.set_chip_status(st.session_state.current_chip, 'yellow')
            st.toast("Status: Yellow (Warning)")
            st.rerun()
        if c3.button("🔴 Fail", use_container_width=True):
            qa.set_chip_status(st.session_state.current_chip, 'red')
            st.toast("Status: Red (Fail)")
            st.rerun()
            
        # Analysis Probe
        st.sidebar.divider()
        with st.sidebar.expander("🔍 Pixel Probe"):
            if st.session_state.chip_data:
                chip_data = st.session_state.chip_data
                proow = st.number_input("Row", 0, chip_data['raw'].shape[1]-1, 0)
                pcol = st.number_input("Col", 0, chip_data['raw'].shape[2]-1, 0)
                if st.button("Probe Pixel"):
                    stats_text = am.pixel_probe_text(chip_data, proow, pcol)
                    st.text(stats_text)
            
        # Advanced Analysis
        with st.sidebar.expander("🔬 Advanced Analysis"):
            if st.session_state.chip_data:
                chip_data = st.session_state.chip_data
                # Flow Director
                st.markdown("**Flow Director**")
                if st.button("Show Flow Arrows"):
                    if 'dem' in chip_data:
                        # Overlay flow on DEM
                        pass # Visualization logic would go here, for now calculate
                        try:
                            # Calculate
                            x, y, u, v = am.flow_direction_overlay(chip_data['dem'])
                            
                            # Create plot
                            fig, ax = plt.subplots(figsize=(6, 6))
                            ax.imshow(chip_data['dem'], cmap='terrain', alpha=0.6)
                            ax.quiver(x, y, u, v, color='blue', scale=50, alpha=0.6)
                            ax.axis('off')
                            st.session_state.stability_overlay = None # Clear stable if any
                            st.image(fig, caption="Dominant Flow Direction (Blue Arrows)", use_column_width=True)
                            plt.close(fig)
                            st.success("Flow direction calculated")
                        except Exception as e:
                            st.error(f"Flow error: {e}")
                    else:
                        st.error("No DEM data")

                # Eco Classifier
                st.markdown("**Eco-Classifier**")
                if st.button("Classify Water Type"):
                    if 'composite_mask' in st.session_state and st.session_state.composite_mask is not None:
                        eco_map, eco_legend = am.eco_classifier(chip_data, st.session_state.composite_mask)
                        
                        # Render eco map
                        fig, ax = plt.subplots(figsize=(6, 6))
                        # Custom colormap: 0=bg, 1=Lake(Blue), 2=Wetland(Green)
                        import matplotlib.colors as mcolors
                        cmap = mcolors.ListedColormap(['none', '#1E90FF', '#32CD32'])
                        ax.imshow(chip_data['vh'], cmap='gray', vmin=-30, vmax=-10)
                        ax.imshow(eco_map, cmap=cmap, alpha=0.6, interpolation='nearest')
                        ax.axis('off')
                        
                        # Legend
                        patches = [plt.Rectangle((0,0),1,1, color=color) for color in ['#1E90FF', '#32CD32']]
                        ax.legend(patches, ['Lake (Open)', 'Wetland (Vegetated)'], loc='upper right')
                        
                        st.image(fig, caption="Eco-Classification", use_column_width=True)
                        plt.close(fig)
                    else:
                        st.warning("Generate a Water Mask first (filters > composite).")

                # Best Band
                st.markdown("**Best Band Selector**")
                if st.button("Run Fisher Analysis"):
                    best, scores = am.best_band_selector(chip_data)
                    st.info(f"Recommended: {best}")
                    st.json(scores)
                
                # Auto-Noise
                st.divider()
                st.markdown("**Auto-Noise Tuner**")
                if st.button("Calculate Noise Floor"):
                    thresh = am.auto_noise_tuner(chip_data['vh'], chip_data['hand'])
                    st.info(f"Optimal Threshold: {thresh:.2f} dB")
                    
                # Stability
                st.divider()
                st.markdown("**Stability Check**")
                if st.button("Generate Heatmap"):
                    overlay, score = am.generate_stability_overlay(chip_data['vh'])
                    # Store in session to render in main view
                    st.session_state.stability_overlay = overlay
                    st.success("Stability layer generated! (See main view)")
            else:
                st.warning("Load chip first")

        # Target Matcher
        st.header("🎯 Target Matcher")
        target_pct = st.number_input("Target %", 0.0, 100.0, 15.0, 1.0)
        tolerance = st.slider("Tolerance", 1.0, 10.0, 2.0, 0.5)
        
        if st.button("🔍 Find Matching", use_container_width=True):
            if st.session_state.chip_data:
                results = target_matcher(st.session_state.chip_data, target_pct, tolerance)
                st.session_state.target_matcher_results = results
                st.success(f"Found {len(results)} matches") if results else st.warning("No matches")
        
        if st.session_state.target_matcher_results:
            if st.button("📥 Apply to Windows", use_container_width=True):
                for i, result in enumerate(st.session_state.target_matcher_results[:NUM_WINDOWS]):
                    st.session_state[f'window_{i}_filter'] = result['filter']
                    st.session_state[f'window_{i}_params'] = result['params']
                st.rerun()
        
        st.divider()
        
        # Quick Actions
        st.header("⚡ Quick Actions")
        
        # Smart Paste
        with st.expander("✨ Smart Paste / Magic Import", expanded=False):
            smart_text = st.text_area("Paste recommendations here...", height=100, 
                placeholder="e.g. refined_lee(window=7), T_low=-22.5")
            
            if smart_text:
                parsed_configs = am.parse_smart_text(smart_text)
                
                if parsed_configs:
                    st.caption("🔍 Glimpse (Preview):")
                    # Show preview table
                    preview_data = [{'Filter': c['filter'], 'Params': str(c['params'])} for c in parsed_configs]
                    st.dataframe(preview_data, use_container_width=True, hide_index=True)
                    
                    if st.button("🚀 Apply Detected Filters", use_container_width=True):
                        save_filter_snapshot()
                        # Apply starting from window 0
                        for i in range(NUM_WINDOWS):
                            st.session_state[f'window_{i}_include'] = False # Reset all first
                            
                        for i, cfg in enumerate(parsed_configs):
                            if i < NUM_WINDOWS:
                                st.session_state[f'window_{i}_filter'] = cfg['filter']
                                st.session_state[f'window_{i}_params'] = cfg['params']
                                st.session_state[f'window_{i}_include'] = True
                        
                        st.success(f"Applied {len(parsed_configs)} filters!")
                        st.rerun()
                else:
                    st.info("No recognizable filters found in text.")
        
        if st.button("🎲 Auto-Populate (Smart Variety)", use_container_width=True):
            if st.session_state.chip_data:
                # Save current state before changing
                save_filter_snapshot()
                
                configs = am.generate_variety_configs(st.session_state.chip_data, NUM_WINDOWS)
                for i, cfg in enumerate(configs):
                    if i < NUM_WINDOWS:
                        st.session_state[f'window_{i}_filter'] = cfg['filter']
                        st.session_state[f'window_{i}_params'] = cfg['params']
                        st.session_state[f'window_{i}_include'] = True
                
                # Show reasoning for each filter
                reasons = [f"• {cfg['filter']}: {cfg.get('reason', 'N/A')}" for cfg in configs[:5]]
                st.success(f"Populated {len(configs)} filters based on chip analysis:\n" + "\n".join(reasons))
                st.rerun()
            else:
                st.error("No chip loaded")
        
        # Undo/History Row
        col_undo, col_history = st.columns(2)
        with col_undo:
            history_len = len(st.session_state.get('filter_history', []))
            if st.button(f"↩️ Undo ({history_len})", use_container_width=True, disabled=(history_len < 2)):
                if undo_filter():
                    st.success("Restored previous filter state")
                    st.rerun()
        with col_history:
            if st.button("📋 View History", use_container_width=True, disabled=(history_len < 1)):
                st.session_state.show_filter_history = True
        
        # Filter History Modal (if shown)
        if st.session_state.get('show_filter_history', False):
            with st.expander("📜 Filter Change History", expanded=True):
                history = st.session_state.get('filter_history', [])
                for idx, snapshot in enumerate(reversed(history)):
                    actual_idx = len(history) - 1 - idx
                    is_current = actual_idx == st.session_state.get('filter_history_index', -1)
                    filters_used = [w['filter'] for w in snapshot['windows'] if w['filter'] != 'None'][:3]
                    label = f"{'→ ' if is_current else ''}{snapshot['timestamp'][:19]} - {', '.join(filters_used)}..."
                    if st.button(label, key=f"restore_{actual_idx}", use_container_width=True):
                        restore_filter_snapshot(actual_idx)
                        st.session_state.show_filter_history = False
                        st.rerun()
                if st.button("Close", use_container_width=True):
                    st.session_state.show_filter_history = False
                    st.rerun()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Select All", use_container_width=True):
                save_filter_snapshot()
                for i in range(NUM_WINDOWS):
                    if st.session_state[f'window_{i}_filter'] != 'None':
                        st.session_state[f'window_{i}_include'] = True
                st.rerun()
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                save_filter_snapshot()
                for i in range(NUM_WINDOWS):
                    st.session_state[f'window_{i}_filter'] = 'None'
                    st.session_state[f'window_{i}_include'] = False
                st.rerun()
        
        st.divider()
        
        # View Options
        st.header("👁️ View Options")
        st.session_state.show_histogram = st.checkbox("📊 Histograms", st.session_state.show_histogram)
        st.session_state.show_map = st.checkbox("🗺️ Location Map", st.session_state.show_map)
        st.session_state.show_swipe = st.checkbox("🔀 Swipe Compare", st.session_state.show_swipe)
        
        st.divider()
        
        # Session Management
        st.header("💾 Session")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save", use_container_width=True):
                save_session()
                st.success("Saved!")
        with col2:
            if st.button("📂 Load", use_container_width=True):
                if load_session():
                    st.success("Loaded!")
                    st.rerun()
    
    # =========================================================================
    # MAIN CONTENT
    # =========================================================================
    
    if st.session_state.chip_data is None:
        st.session_state.chip_data = load_chip(st.session_state.current_chip)
    
    chip_data = st.session_state.chip_data
    if chip_data is None:
        st.error("Failed to load chip")
        return
    
    # Header
    st.title(f"🌊 {st.session_state.current_chip}")
    
    # =========================================================================
    # REFERENCE STRIP
    # =========================================================================
    st.subheader("📊 Reference Strip")
    
    ref_cols = st.columns(7)
    ref_bands = [
        ('VV', chip_data['vv'], 'gray', -25, -5),
        ('VH', chip_data['vh'], 'gray', -30, -10),
        ('MNDWI', chip_data['mndwi'], 'RdBu', -0.5, 0.5),
        ('DEM', chip_data['dem'], 'terrain', None, None),
        ('HAND', chip_data['hand'], 'Blues_r', 0, 20),
        ('Slope', chip_data['slope'], 'magma', 0, 20),
        ('TWI', chip_data['twi'], 'YlGnBu', 0, 15)
    ]
    
    for col, (name, data, cmap, vmin, vmax) in zip(ref_cols, ref_bands):
        with col:
            st.caption(name)
            fig = render_band(data, cmap=cmap, vmin=vmin, vmax=vmax)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    
    # Optional Views
    if st.session_state.show_histogram:
        st.subheader("📊 Band Histograms & Statistics")
        
        # Generate comprehensive statistics for all bands
        band_names = ['vv', 'vh', 'mndwi', 'dem', 'hand', 'slope', 'twi']
        band_labels = ['VV (dB)', 'VH (dB)', 'MNDWI', 'DEM (m)', 'HAND (m)', 'Slope (°)', 'TWI']
        default_thresholds = [-17.0, -17.0, 0.0, None, 5.0, 15.0, 10.0]
        
        # Build comprehensive stats report
        # Safe access to chip_files
        chip_files = st.session_state.get('chip_files', [])
        chip_idx = st.session_state.get('chip_idx', 0)
        chip_id = chip_files[chip_idx] if chip_files and chip_idx < len(chip_files) else 'Unknown'
        
        stats_report = {
            'chip_id': chip_id,
            'timestamp': str(np.datetime64('now')),
            'bands': {}
        }
        
        for band_name, band_label in zip(band_names, band_labels):
            if band_name in chip_data and chip_data[band_name] is not None:
                data = chip_data[band_name]
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    stats_report['bands'][band_name] = {
                        'label': band_label,
                        'shape': list(data.shape),
                        'dtype': str(data.dtype),
                        'min': float(np.nanmin(data)),
                        'max': float(np.nanmax(data)),
                        'mean': float(np.nanmean(data)),
                        'std': float(np.nanstd(data)),
                        'median': float(np.nanmedian(data)),
                        'percentile_5': float(np.nanpercentile(data, 5)),
                        'percentile_25': float(np.nanpercentile(data, 25)),
                        'percentile_75': float(np.nanpercentile(data, 75)),
                        'percentile_95': float(np.nanpercentile(data, 95)),
                        'valid_pixels': int(len(valid_data)),
                        'nan_pixels': int(np.sum(np.isnan(data))),
                        'total_pixels': int(data.size)
                    }
                    # Add histogram bins and counts
                    hist_counts, hist_edges = np.histogram(valid_data, bins=50)
                    stats_report['bands'][band_name]['histogram'] = {
                        'counts': hist_counts.tolist(),
                        'bin_edges': hist_edges.tolist()
                    }
        
        # Add bounds if available
        if chip_data.get('bounds'):
            stats_report['bounds'] = chip_data['bounds']
        
        # Display histograms in rows of 4
        hist_row1 = st.columns(4)
        hist_row2 = st.columns(4)
        all_cols = hist_row1 + hist_row2[:3]
        
        for i, (band_name, band_label, default_thresh) in enumerate(zip(band_names, band_labels, default_thresholds)):
            if band_name in chip_data and chip_data[band_name] is not None and i < len(all_cols):
                with all_cols[i]:
                    # Use last active threshold if available, else default
                    active_thresh = st.session_state.get('last_thresholds', {}).get(band_name.lower(), default_thresh)
                    fig = render_histogram(chip_data[band_name], active_thresh, f'{band_label}')
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
        
        # Copy Stats Button
        import json
        stats_json = json.dumps(stats_report, indent=2)
        
        st.divider()
        col_stats1, col_stats2 = st.columns([3, 1])
        
        with col_stats1:
            with st.expander("📋 View Full Statistics Report (JSON)", expanded=False):
                st.code(stats_json, language='json')
        
        with col_stats2:
            st.download_button(
                label="⬇️ Download Stats JSON",
                data=stats_json,
                file_name=f"chip_stats_{stats_report['chip_id'].replace('/', '_').replace('.', '_')}.json",
                mime="application/json"
            )
        
        # Quick Stats Table
        st.markdown("### 📈 Quick Stats Summary")
        quick_stats_data = []
        for band_name in band_names:
            if band_name in stats_report['bands']:
                b = stats_report['bands'][band_name]
                quick_stats_data.append({
                    'Band': b['label'],
                    'Min': f"{b['min']:.2f}",
                    'Max': f"{b['max']:.2f}",
                    'Mean': f"{b['mean']:.2f}",
                    'Std': f"{b['std']:.2f}",
                    'P5': f"{b['percentile_5']:.2f}",
                    'P95': f"{b['percentile_95']:.2f}",
                    'Valid %': f"{100*b['valid_pixels']/b['total_pixels']:.1f}%"
                })
        if quick_stats_data:
            st.dataframe(quick_stats_data, use_container_width=True)
    
    if st.session_state.show_map and chip_data.get('bounds'):
        with st.expander("🗺️ Location Map", expanded=True):
            fig = render_location_map(chip_data['bounds'])
            if fig:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                
                # Google Maps link
                bounds = chip_data['bounds']
                center_lat = (bounds[1] + bounds[3]) / 2
                center_lon = (bounds[0] + bounds[2]) / 2
                maps_url = f"https://www.google.com/maps/@{center_lat},{center_lon},15z"
                st.markdown(f"[🔗 Open in Google Maps]({maps_url})")
    
    st.divider()
    
    # =========================================================================
    # FILTER WINDOWS
    # =========================================================================
    st.subheader("🔬 Filter Windows")
    
    # Stability Overlay Display
    if 'stability_overlay' in st.session_state and st.session_state.stability_overlay is not None:
        st.info("Showing Stability Analysis Overlay (Blue=Stable, Yellow=Unstable)")
        st.image(st.session_state.stability_overlay, caption="Stability Heatmap", clamp=True, use_column_width=True)
        if st.button("❌ Close Overlay"):
            del st.session_state.stability_overlay
            st.rerun()
            
    filter_options = list(FILTER_SPECS.keys())
    included_masks = []
    window_recipe = []
    
    for row in range(3):
        cols = st.columns(5)
        for col_idx, col in enumerate(cols):
            win_idx = row * 5 + col_idx
            
            with col:
                current_filter = st.session_state.get(f'window_{win_idx}_filter', 'None')
                new_filter = st.selectbox(
                    f"W{win_idx + 1}",
                    filter_options,
                    index=filter_options.index(current_filter) if current_filter in filter_options else 0,
                    key=f'win_{win_idx}_select',
                    label_visibility='collapsed'
                )
                
                if new_filter != current_filter:
                    st.session_state[f'window_{win_idx}_filter'] = new_filter
                    st.session_state[f'window_{win_idx}_params'] = {}
                    # Clear synced widget keys to prevent stale parameter carry-over
                    for key in list(st.session_state.keys()):
                        if key.startswith(f'win_{win_idx}_'):
                            del st.session_state[key]
                    st.rerun()
                
                params = st.session_state.get(f'window_{win_idx}_params', {}).copy()
                filter_spec = FILTER_SPECS.get(new_filter, {})
                
                if filter_spec:
                    with st.expander("⚙️ Parameters", expanded=True):
                        # --- Accurate Distribution Plotting ---
                        active_band = params.get('band', 'vh').lower()
                        
                        # Handle specific filter data transformations for accurate histogram visualization
                        display_data = chip_data.get(active_band)
                        plot_title = f"{active_band.upper()} Distribution"
                        
                        if new_filter == 'Refined Lee' and 'vh' in chip_data:
                            display_data = fe.refined_lee_filter(chip_data['vh'], window_size=params.get('window_size', 7))
                            plot_title = "Refined Lee Filtered (dB)"
                        elif new_filter == 'SDWI' and 'vv' in chip_data and 'vh' in chip_data:
                            display_data = fe.sdwi(chip_data['vv'], chip_data['vh'])
                            plot_title = "SDWI Values"
                        elif new_filter == 'Cross-Pol Ratio' and 'vv' in chip_data and 'vh' in chip_data:
                            display_data = fe.cross_pol_ratio(chip_data['vv'], chip_data['vh'])
                            plot_title = "CPR Values (Linear)"
                        elif new_filter == 'GLCM Entropy' and active_band in chip_data:
                            display_data = fe.glcm_entropy(chip_data[active_band], window_size=params.get('window_size', 5))
                            plot_title = "GLCM Entropy"
                        elif new_filter == 'SRAD' and 'vh' in chip_data:
                            display_data = fe.srad_filter(chip_data['vh'], num_iter=params.get('num_iter', 15))
                            plot_title = "SRAD Filtered (dB)"
                            
                        if display_data is not None:
                            # Try to find a threshold parameter to show on hist
                            current_thresh = None
                            for p_n, p_s in filter_spec.items():
                                if 'threshold' in p_n.lower() or p_n.lower() in ['low', 'high', 't_low', 'hand_thresh']:
                                    current_thresh = params.get(p_n, p_s['default'])
                                    break
                            
                            hist_fig = render_histogram(display_data, current_thresh, plot_title)
                            st.pyplot(hist_fig, use_container_width=True)
                            plt.close(hist_fig)
                            st.divider()

                        # -- Dynamic visibility helper for Custom Equation --
                        equation_text = params.get('equation', '') if new_filter == 'Custom Equation' else ''
                        
                        for param_name, param_spec in filter_spec.items():
                            # Skip dynamic params if not mentioned in equation
                            if param_spec.get('dynamic', False):
                                if param_name not in equation_text:
                                    continue  # Hide this parameter
                            
                            current_val = params.get(param_name, param_spec['default'])
                            
                            if param_spec['type'] == 'slider':
                                # Unified Slider + Text Input with robust sync using partial
                                min_v = param_spec['min']
                                max_v = param_spec['max']
                                step_v = param_spec.get('step', 1)
                                
                                # Keys for sync
                                slider_key = f'win_{win_idx}_{param_name}_slide'
                                num_key = f'win_{win_idx}_{param_name}_num'
                                
                                # Initialize keys if missing
                                if slider_key not in st.session_state:
                                    st.session_state[slider_key] = current_val
                                if num_key not in st.session_state:
                                    st.session_state[num_key] = current_val
                                
                                # Sync callbacks using partial to avoid closure capture
                                from functools import partial
                                
                                def _sync_slider_to_num(sk, nk):
                                    st.session_state[nk] = st.session_state[sk]
                                
                                def _sync_num_to_slider(sk, nk):
                                    st.session_state[sk] = st.session_state[nk]
                                    
                                c1, c2 = st.columns([3, 1])
                                with c1:
                                    val_slider = st.slider(
                                        param_name,
                                        min_v, max_v,
                                        key=slider_key,
                                        step=step_v,
                                        on_change=partial(_sync_slider_to_num, slider_key, num_key)
                                    )
                                with c2:
                                    val_text = st.number_input(
                                        "",
                                        min_v, max_v,
                                        key=num_key,
                                        step=step_v,
                                        on_change=partial(_sync_num_to_slider, slider_key, num_key),
                                        label_visibility='collapsed'
                                    )
                                
                                params[param_name] = st.session_state[slider_key]
                                
                                # Track "last active" threshold for Stats View accuracy
                                if 'threshold' in param_name.lower() or param_name == 'T':
                                    band_ref = params.get('band', 'vh') if isinstance(params.get('band'), str) else 'vh'
                                    if 'last_thresholds' not in st.session_state:
                                        st.session_state.last_thresholds = {}
                                    st.session_state.last_thresholds[band_ref.lower()] = params[param_name]
                                
                            elif param_spec['type'] == 'select':
                                idx = param_spec['options'].index(current_val)
                                params[param_name] = st.selectbox(
                                    param_name, param_spec['options'], idx,
                                    key=f'win_{win_idx}_{param_name}'
                                )
                            elif param_spec['type'] == 'text':
                                params[param_name] = st.text_input(
                                    param_name,
                                    current_val,
                                    key=f'win_{win_idx}_{param_name}'
                                )
                        
                        # --- Auto-Detected Sliders for Custom Equation ---
                        if new_filter == 'Custom Equation':
                            auto_params = extract_equation_params(params.get('equation', ''))
                            
                            if auto_params:
                                st.caption("📊 Auto-Detected Parameters:")
                                
                                for ap_name, ap_spec in auto_params.items():
                                    # Skip if already handled by predefined params
                                    if ap_name in params:
                                        continue
                                    
                                    ap_key_slide = f'win_{win_idx}_auto_{ap_name}_slide'
                                    ap_key_num = f'win_{win_idx}_auto_{ap_name}_num'
                                    
                                    # Initialize with detected value
                                    if ap_key_slide not in st.session_state:
                                        st.session_state[ap_key_slide] = ap_spec['value']
                                    if ap_key_num not in st.session_state:
                                        st.session_state[ap_key_num] = ap_spec['value']
                                    
                                    from functools import partial
                                    
                                    def _sync_auto_slide(sk, nk):
                                        st.session_state[nk] = st.session_state[sk]
                                    
                                    def _sync_auto_num(sk, nk):
                                        st.session_state[sk] = st.session_state[nk]
                                    
                                    ac1, ac2 = st.columns([3, 1])
                                    with ac1:
                                        ap_val = st.slider(
                                            ap_name,
                                            float(ap_spec['min']), float(ap_spec['max']),
                                            key=ap_key_slide,
                                            step=float(ap_spec['step']),
                                            on_change=partial(_sync_auto_slide, ap_key_slide, ap_key_num)
                                        )
                                    with ac2:
                                        st.number_input(
                                            "",
                                            float(ap_spec['min']), float(ap_spec['max']),
                                            key=ap_key_num,
                                            step=float(ap_spec['step']),
                                            on_change=partial(_sync_auto_num, ap_key_slide, ap_key_num),
                                            label_visibility='collapsed'
                                        )
                                    
                                    # Store in params for use in equation
                                    params[f'_auto_{ap_name}'] = st.session_state[ap_key_slide]
                        
                        st.session_state[f'window_{win_idx}_params'] = params
                
                include = st.checkbox("✓", st.session_state.get(f'window_{win_idx}_include', False),
                                     key=f'win_{win_idx}_include')
                st.session_state[f'window_{win_idx}_include'] = include
                
                mask = apply_filter(new_filter, chip_data, params)
                st.session_state[f'window_{win_idx}_mask'] = mask
                
                if include and mask is not None:
                    included_masks.append(mask)
                    window_recipe.append({'filter': new_filter, 'params': params})
                
                if mask is not None:
                    pct = calc_water_pct(mask)
                    fig = render_mask(mask, pct)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.info("Select filter")
    
    st.divider()
    
    # =========================================================================
    # COMPOSITE RESULT
    # =========================================================================
    st.subheader("🎯 Composite Result")
    
    res_cols = st.columns([3, 1])
    
    with res_cols[0]:
        if included_masks:
            mode = st.session_state.composite_mode
            
            if 'Union' in mode:
                composite = fusion_union(*included_masks)
            elif 'Intersection' in mode:
                composite = fusion_intersection(*included_masks)
            else:
                min_votes = len(included_masks) // 2 + 1
                composite = fusion_vote(included_masks, min_votes=min_votes)
            
            composite = morphological_cleanup(composite)
            water_pct = calc_water_pct(composite)
            
            # Swipe comparison
            if st.session_state.show_swipe:
                swipe_pos = st.slider("Swipe Position", 0.0, 1.0, 0.5, 0.05)
                fig = render_swipe_comparison(chip_data['vh'], composite, swipe_pos)
            else:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(composite, cmap=WATER_CMAP, interpolation='nearest')
                ax.set_title(f"Composite ({mode}): {water_pct:.2f}% Water", fontsize=14)
                ax.axis('off')
            
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Check ✓ on windows to include in composite")
            composite = None
            water_pct = 0
    
    with res_cols[1]:
        st.metric("Windows", len(included_masks))
        st.metric("Mode", st.session_state.composite_mode.split()[0])
        if included_masks:
            st.metric("Water %", f"{water_pct:.2f}")
        
        # JSON Export for Composite Result
        if composite is not None:
            import json
            composite_data = {
                'chip_id': st.session_state.get('current_chip', 'Unknown'),
                'timestamp': str(np.datetime64('now')),
                'composite_mode': st.session_state.composite_mode,
                'num_windows': len(included_masks),
                'water_percentage': float(water_pct),
                'shape': list(composite.shape),
                'total_pixels': int(composite.size),
                'water_pixels': int(composite.sum()),
                'land_pixels': int((~composite).sum()),
                'windows_used': [
                    {
                        'index': i,
                        'filter': st.session_state[f'window_{i}_filter'],
                        'params': st.session_state[f'window_{i}_params']
                    }
                    for i in range(NUM_WINDOWS)
                    if st.session_state[f'window_{i}_include']
                ],
                # Optionally include mask as list (can be large!)
                # 'mask': composite.astype(int).tolist()
            }
            
            json_str = json.dumps(composite_data, indent=2)
            st.download_button(
                label="📥 Export Composite JSON",
                data=json_str,
                file_name=f"composite_{st.session_state.get('current_chip', 'result')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Validation Metrics
        verified_label = load_verified_label(st.session_state.current_chip)
        if verified_label is not None and composite is not None:
            # Resize label if needed (simple check)
            if verified_label.shape == composite.shape:
                # Jaccard (IoU)
                intersection = np.logical_and(composite, verified_label).sum()
                union = np.logical_or(composite, verified_label).sum()
                iou = intersection / (union + 1e-8)
                
                # Dice (F1)
                dice = 2 * intersection / (composite.sum() + verified_label.sum() + 1e-8)
                
                st.divider()
                st.caption("Validation vs Verified Label")
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("IoU", f"{iou:.3f}")
                col_m2.metric("F1", f"{dice:.3f}")
            else:
                st.warning("Label shape mismatch")
        
        st.divider()
        
        # RGB Visualizer
        with st.expander("🎨 Physics RGB"):
            rgb_fig = render_rgb_composite(chip_data)
            st.pyplot(rgb_fig, use_container_width=True)
            st.caption("R:VV( Surf), G:VH(Vol), B:Ratio(Dbl)")
            plt.close(rgb_fig)
        
        st.divider()
        
        # Export
        if st.button("📦 Export Package", use_container_width=True, disabled=composite is None):
            if composite is not None:
                mask_path, recipe_path = export_result(
                    st.session_state.current_chip,
                    composite,
                    window_recipe,
                    st.session_state.composite_mode
                )
                st.success(f"Exported to {EXPORT_DIR}")
        
        if st.button("💾 Save & Next", use_container_width=True):
            save_session()
            idx = chips.index(st.session_state.current_chip)
            st.session_state.current_chip = chips[(idx + 1) % len(chips)]
            st.session_state.chip_data = None
            st.rerun()


if __name__ == '__main__':
    main()
