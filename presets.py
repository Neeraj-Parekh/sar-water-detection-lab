"""
SAR Water Detection - Presets Module
=====================================

Quick-access preset configurations for different scenarios.
"""

# =============================================================================
# FUSION PRESETS
# =============================================================================

FUSION_PRESETS = {
    'Urban Flood': {
        'description': 'Detects urban flooding using double-bounce + HAND',
        'windows': [
            {'filter': 'Simple Threshold', 'params': {'band': 'vv', 'threshold': -5.0}},
            {'filter': 'HAND Definite', 'params': {'hand_thresh': 5.0}},
            {'filter': 'Slope Flat', 'params': {'max_slope': 3.0}},
        ],
        'fusion_mode': 'Intersection (AND)'
    },
    
    'Open Water': {
        'description': 'Large water bodies like lakes and reservoirs',
        'windows': [
            {'filter': 'Simple Threshold', 'params': {'band': 'vh', 'threshold': -18.0}},
            {'filter': 'HAND Definite', 'params': {'hand_thresh': 3.0}},
            {'filter': 'Continuity Filter', 'params': {'min_length': 50}},
        ],
        'fusion_mode': 'Intersection (AND)'
    },
    
    'River Detection': {
        'description': 'Narrow rivers using width and curvature',
        'windows': [
            {'filter': '5-Stage Fusion', 'params': {}},
            {'filter': 'White Top Hat', 'params': {'band': 'mndwi', 'size': 7, 'threshold': 0.05}},
            {'filter': 'Width Filter', 'params': {'min_width': 3, 'max_width': 30}},
        ],
        'fusion_mode': 'Union (OR)'
    },
    
    'Disaster Response': {
        'description': 'Quick assessment with relaxed thresholds',
        'windows': [
            {'filter': 'Simple Threshold', 'params': {'band': 'vh', 'threshold': -15.0}},
            {'filter': 'HAND Possible', 'params': {'hand_thresh': 15.0}},
        ],
        'fusion_mode': 'Union (OR)'
    },
    
    'Wetland Detection': {
        'description': 'Wetlands and marshy areas',
        'windows': [
            {'filter': 'TWI High', 'params': {'min_twi': 10.0}},
            {'filter': 'HAND Probable', 'params': {'hand_thresh': 8.0}},
            {'filter': 'Slope Flat', 'params': {'max_slope': 2.0}},
        ],
        'fusion_mode': 'Intersection (AND)'
    },
    
    'Arid/Sparse Water': {
        'description': 'Sparse water in arid regions',
        'windows': [
            {'filter': 'Simple Threshold', 'params': {'band': 'vh', 'threshold': -20.0}},
            {'filter': 'HAND Definite', 'params': {'hand_thresh': 2.0}},
            {'filter': 'Continuity Filter', 'params': {'min_length': 20}},
        ],
        'fusion_mode': 'Intersection (AND)'
    },
    
    'Conservative': {
        'description': 'High precision, may miss faint water',
        'windows': [
            {'filter': 'Simple Threshold', 'params': {'band': 'vh', 'threshold': -21.0}},
            {'filter': 'HAND Definite', 'params': {'hand_thresh': 3.0}},
            {'filter': 'Hysteresis', 'params': {'band': 'vh', 'low': -22.0, 'high': -18.0}},
        ],
        'fusion_mode': 'Intersection (AND)'
    },
    
    'Aggressive': {
        'description': 'High recall, may include noise',
        'windows': [
            {'filter': 'Simple Threshold', 'params': {'band': 'vh', 'threshold': -14.0}},
            {'filter': 'HAND Possible', 'params': {'hand_thresh': 15.0}},
        ],
        'fusion_mode': 'Union (OR)'
    }
}


# =============================================================================
# INDIA-SPECIFIC PRESETS
# =============================================================================

INDIA_PRESETS = {
    'Gangetic Plains': {
        'description': 'Wide braided rivers, seasonal floods',
        'windows': [
            {'filter': 'HAND Probable', 'params': {'hand_thresh': 8.0}},
            {'filter': 'Width Filter', 'params': {'min_width': 10, 'max_width': 100}},
            {'filter': 'Slope Flat', 'params': {'max_slope': 2.0}},
        ],
        'fusion_mode': 'Intersection (AND)',
        'notes': 'Low gradient, extensive floodplains'
    },
    
    'Peninsular Rivers': {
        'description': 'Narrow valleys, rocky terrain',
        'windows': [
            {'filter': 'HAND Definite', 'params': {'hand_thresh': 5.0}},
            {'filter': 'Width Filter', 'params': {'min_width': 4, 'max_width': 40}},
            {'filter': 'Curvature Filter', 'params': {}},
        ],
        'fusion_mode': 'Intersection (AND)',
        'notes': 'Variable flow, confined channels'
    },
    
    'Western Ghats': {
        'description': 'Mountain streams, steep slopes',
        'windows': [
            {'filter': 'HAND Definite', 'params': {'hand_thresh': 3.0}},
            {'filter': 'Simple Threshold', 'params': {'band': 'vh', 'threshold': -19.0}},
            {'filter': 'Continuity Filter', 'params': {'min_length': 20}},
        ],
        'fusion_mode': 'Intersection (AND)',
        'notes': 'High slope, narrow streams'
    },
    
    'Rajasthan Arid': {
        'description': 'Ephemeral channels, sparse water',
        'windows': [
            {'filter': 'Simple Threshold', 'params': {'band': 'vh', 'threshold': -20.0}},
            {'filter': 'HAND Definite', 'params': {'hand_thresh': 2.0}},
            {'filter': 'Continuity Filter', 'params': {'min_length': 30}},
        ],
        'fusion_mode': 'Intersection (AND)',
        'notes': 'Low water, high sand noise'
    },
    
    'Coastal Delta': {
        'description': 'Mangroves, estuaries, tidal zones',
        'windows': [
            {'filter': 'HAND Possible', 'params': {'hand_thresh': 5.0}},
            {'filter': 'TWI High', 'params': {'min_twi': 12.0}},
            {'filter': 'Simple Threshold', 'params': {'band': 'vh', 'threshold': -16.0}},
        ],
        'fusion_mode': 'Union (OR)',
        'notes': 'Mixed water, low elevation'
    },
    
    'Northeast Monsoon': {
        'description': 'Flash floods during monsoon',
        'windows': [
            {'filter': 'HAND Possible', 'params': {'hand_thresh': 10.0}},
            {'filter': 'Simple Threshold', 'params': {'band': 'vh', 'threshold': -15.0}},
            {'filter': 'Slope Flat', 'params': {'max_slope': 8.0}},
        ],
        'fusion_mode': 'Union (OR)',
        'notes': 'Rapid changes, extended flood extent'
    }
}


# =============================================================================
# WATER TYPE PRESETS (Based on chip categories)
# =============================================================================

WATER_TYPE_PRESETS = {
    'large_lakes': {
        'recommended_filters': ['Simple Threshold', 'HAND Definite', 'Continuity Filter'],
        'vh_threshold': -18.0,
        'hand_threshold': 3.0,
        'min_size': 100
    },
    'rivers_wide': {
        'recommended_filters': ['5-Stage Fusion', 'Width Filter', 'HAND Probable'],
        'vh_threshold': -17.0,
        'hand_threshold': 5.0,
        'width_range': (10, 80)
    },
    'rivers_narrow': {
        'recommended_filters': ['White Top Hat', 'Curvature Filter', 'HAND Definite'],
        'vh_threshold': -17.0,
        'hand_threshold': 4.0,
        'width_range': (3, 30)
    },
    'wetlands': {
        'recommended_filters': ['TWI High', 'HAND Possible', 'Slope Flat'],
        'vh_threshold': -16.0,
        'hand_threshold': 10.0,
        'twi_threshold': 10.0
    },
    'reservoirs': {
        'recommended_filters': ['Simple Threshold', 'HAND Definite', 'Continuity Filter'],
        'vh_threshold': -19.0,
        'hand_threshold': 3.0,
        'min_size': 50
    },
    'sparse_arid': {
        'recommended_filters': ['Simple Threshold', 'HAND Definite', 'Continuity Filter'],
        'vh_threshold': -20.0,
        'hand_threshold': 2.0,
        'min_size': 30
    }
}


def get_water_type_from_chip_name(chip_name):
    """Extract water type from chip name."""
    for wtype in WATER_TYPE_PRESETS.keys():
        if wtype in chip_name.lower():
            return wtype
    return None


def get_recommended_preset(chip_name):
    """Get recommended preset based on chip name."""
    water_type = get_water_type_from_chip_name(chip_name)
    if water_type:
        return WATER_TYPE_PRESETS.get(water_type)
    return None
