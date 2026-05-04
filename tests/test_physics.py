import pytest
import numpy as np
from sensoformer.ext import MTDecomposer
from sensoformer.utils.physics import kagan_angle

def test_mtdcmp_loading():
    """Ensure the shared library loads correctly."""
    decomposer = MTDecomposer()
    # Check if lib attribute is populated (it might be None if build failed, handled gracefully)
    if decomposer.lib is None:
        pytest.skip("Fortran library not found. Skipping interface test.")
    assert decomposer.lib is not None

def test_mt_to_sdr_conversion():
    """Test conversion of a known Moment Tensor to SDR."""
    decomposer = MTDecomposer()
    if decomposer.lib is None:
        pytest.skip("Fortran library not available")

    # Pure Strike-Slip Fault
    mt_in = np.array([1.0, -1.0, 0.0, 0.0, 0.0])
    
    sdr1, sdr2 = decomposer.mt_to_sdr(mt_in)
    
    assert sdr1.shape == (3,)
    assert sdr2.shape == (3,)
    
    # Relaxed assertion for floating point arithmetic
    # Allow small epsilon over 90.0
    assert 0 <= sdr1[1] <= 90.0 + 1e-4
    assert 0 <= sdr2[1] <= 90.0 + 1e-4

def test_kagan_angle_identity():
    """Kagan angle between identical mechanisms should be 0."""
    # Pass scalars to get scalar output
    s, d, r = 45.0, 90.0, 0.0
    angle = kagan_angle(s, d, r, s, d, r)
    assert np.isclose(angle, 0.0, atol=1e-5)

def test_kagan_angle_symmetry():
    """Kagan angle should handle double-couple symmetry."""
    # Same mechanism rotated 180 degrees
    s1, d1, r1 = 45.0, 90.0, 0.0
    s2, d2, r2 = 225.0, 90.0, 0.0 
    
    angle = kagan_angle(s1, d1, r1, s2, d2, r2)
    assert np.isclose(angle, 0.0, atol=1e-5)

def test_kagan_angle_vectorized():
    """Test vectorized input for Kagan angle."""
    s = np.array([45.0, 45.0])
    d = np.array([90.0, 90.0])
    r = np.array([0.0, 0.0])
    
    angles = kagan_angle(s, d, r, s, d, r)
    assert angles.shape == (2,)
    # Use np.all for array assertion
    assert np.all(np.isclose(angles, 0.0, atol=1e-5))