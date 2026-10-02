#!/usr/bin/env python3
"""
Unit tests for OrientationCosineLoss (scale-invariant deviatoric-MT
orientation loss, a smooth Kagan-angle surrogate).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import pytest

PROJ_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJ_ROOT / "src"))

from sensoformer.utils.metrics import OrientationCosineLoss


def full_tensor(v):
    """(5,) [Mxx, Myy, Mxy, Mxz, Myz] -> symmetric 3x3 with zero trace."""
    mxx, myy, mxy, mxz, myz = v
    mzz = -(mxx + myy)
    return np.array([[mxx, mxy, mxz],
                     [mxy, myy, myz],
                     [mxz, myz, mzz]])


def five_vec(m):
    return torch.tensor([m[0, 0], m[1, 1], m[0, 1], m[0, 2], m[1, 2]],
                        dtype=torch.float64)


def rotate_mt(m, axis, angle_deg):
    """Rotate a 3x3 tensor about a coordinate axis."""
    t = np.deg2rad(angle_deg)
    c, s = np.cos(t), np.sin(t)
    if axis == "z":
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    elif axis == "x":
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    else:
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return R @ m @ R.T


DC = full_tensor([1.0, -1.0, 0.0, 0.0, 0.0])  # pure double couple


def test_zero_for_identical():
    loss = OrientationCosineLoss()
    v = torch.tensor([[0.4, -0.2, 0.3, 0.1, -0.5]], dtype=torch.float64)
    assert loss(v, v).item() == pytest.approx(0.0, abs=1e-10)


def test_scale_invariance():
    loss = OrientationCosineLoss()
    v = torch.tensor([[0.4, -0.2, 0.3, 0.1, -0.5]], dtype=torch.float64)
    assert loss(3.7 * v, v).item() == pytest.approx(0.0, abs=1e-10)


def test_two_for_antialigned():
    loss = OrientationCosineLoss()
    v = torch.tensor([[0.4, -0.2, 0.3, 0.1, -0.5]], dtype=torch.float64)
    assert loss(-v, v).item() == pytest.approx(2.0, abs=1e-10)


def test_frobenius_uses_full_tensor():
    """The 5-vector inner product must equal the true 3x3 Frobenius product
    (off-diagonals doubled, Mzz reconstructed)."""
    rng = np.random.RandomState(0)
    a5, b5 = rng.randn(5), rng.randn(5)
    A, B = full_tensor(a5), full_tensor(b5)
    expected = float(np.sum(A * B))
    got = OrientationCosineLoss._frobenius_inner(
        torch.tensor(a5), torch.tensor(b5)).item()
    assert got == pytest.approx(expected, rel=1e-10)


def test_monotone_in_rotation_angle():
    """For a DC mechanism rotated about the z-axis, the loss must increase
    monotonically with rotation angle in (0°, 90°)."""
    loss = OrientationCosineLoss()
    target = five_vec(DC).unsqueeze(0)
    values = []
    for ang in (5, 15, 30, 45, 60, 85):
        pred = five_vec(rotate_mt(DC, "z", ang)).unsqueeze(0)
        values.append(loss(pred, target).item())
    assert all(v2 > v1 for v1, v2 in zip(values, values[1:]))
    assert values[0] > 0


def test_gradients_flow():
    loss = OrientationCosineLoss()
    pred = torch.tensor([[0.4, -0.2, 0.3, 0.1, -0.5]], requires_grad=True)
    target = torch.tensor([[1.0, -1.0, 0.0, 0.0, 0.0]])
    out = loss(pred, target)
    out.backward()
    assert pred.grad is not None and pred.grad.abs().max() > 0


def test_batch_mean_reduction():
    loss = OrientationCosineLoss()
    v = torch.tensor([[0.4, -0.2, 0.3, 0.1, -0.5]], dtype=torch.float64)
    batch_pred = torch.cat([v, -v])
    batch_target = torch.cat([v, v])
    # (0 + 2) / 2 = 1
    assert loss(batch_pred, batch_target).item() == pytest.approx(1.0, abs=1e-10)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
