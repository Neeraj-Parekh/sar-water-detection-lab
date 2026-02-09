
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from filter_engine_complete import (
    refined_lee_filter, k_distribution_cfar, 
    rfi_filter_simple, frost_filter,
    numpy_otsu, cross_pol_ratio
)

# Mock Data Fixtures
@pytest.fixture
def clean_chip():
    """Rank-2 array of random floats (simulating dB or Intensity)."""
    return np.random.normal(loc=-20, scale=5, size=(100, 100))

@pytest.fixture
def nan_chip():
    """Chip with NaNs."""
    data = np.random.normal(-20, 5, (50, 50))
    data[10:20, 10:20] = np.nan
    return data

@pytest.fixture
def constant_chip():
    """Constant value chip (zero variance)."""
    return np.full((50, 50), -15.0)

@pytest.fixture
def tiny_chip():
    """Tiny chip (smaller than typical window)."""
    return np.random.randn(5, 5)

# =============================================================================
# 1. Refined Lee Tests (Vectorized)
# =============================================================================
def test_refined_lee_shape(clean_chip):
    res = refined_lee_filter(clean_chip, window_size=7)
    assert res.shape == clean_chip.shape
    assert not np.isnan(res).all()

def test_refined_lee_constant(constant_chip):
    # Should maintain constant value
    res = refined_lee_filter(constant_chip)
    assert np.allclose(res, constant_chip, atol=1e-5)

def test_refined_lee_tiny(tiny_chip):
    # Should not crash on tiny input
    res = refined_lee_filter(tiny_chip, window_size=7) # Window > Image
    assert res.shape == tiny_chip.shape

# =============================================================================
# 2. CFAR Tests (Local Stats)
# =============================================================================
def test_cfar_output_type(clean_chip):
    # Should return boolean mask
    mask = k_distribution_cfar(clean_chip)
    assert mask.dtype == bool
    assert mask.shape == clean_chip.shape

def test_cfar_constant(constant_chip):
    # Constant input -> Variance 0 -> Should be False (no detection)
    mask = k_distribution_cfar(constant_chip)
    assert not mask.any()

# =============================================================================
# 3. Frost Tests
# =============================================================================
def test_frost_output(clean_chip):
    res = frost_filter(clean_chip)
    assert res.shape == clean_chip.shape

# =============================================================================
# 4. Indices Tests
# =============================================================================
def test_cpr_bounds(clean_chip):
    vv = clean_chip
    vh = clean_chip - 5 # simulated logical relation
    cpr = cross_pol_ratio(vv, vh)
    assert cpr.min() >= 0
    assert cpr.max() <= 10 # clipped

# =============================================================================
# 5. Nan Handling
# =============================================================================
def test_rfi_nan_handling(nan_chip):
    res, mask = rfi_filter_simple(nan_chip)
    assert not np.isnan(res).all() # Should handle NaNs
    assert mask.shape == nan_chip.shape

if __name__ == "__main__":
    # Integration smoke test
    data = np.random.randn(100, 100)
    print("Running smoke tests...")
    refined_lee_filter(data)
    k_distribution_cfar(data)
    print("Passed.")
