#!/usr/bin/env python3
"""
Unit tests for the continuous relative-geometry attention bias
(GeometryAttentionBias) and its integration into Sensoformer.
"""
import sys
from pathlib import Path

import torch
import pytest

PROJ_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJ_ROOT / "src"))

from sensoformer.models.network import GeometryAttentionBias, Sensoformer


def make_cfg(geometry_bias: bool):
    return {
        "model": {
            "name": "sensoformer",
            "in_channels": 6,
            "num_scalar_features": 20,
            "embed_dim": 32,
            "heads": 4,
            "layers": 2,
            "dropout": 0.0,
            "feedforward_dim": 64,
            "geometry_bias": geometry_bias,
        }
    }


def make_batch(B=2, S=6, F=20, L=64, n_valid=4, seed=0):
    torch.manual_seed(seed)
    waveforms = torch.randn(B, S, 12, L)
    features = torch.randn(B, S, F)
    # Physically plausible geometry columns
    features[..., 0] = torch.rand(B, S) * 300.0          # distance (km)
    features[..., 1] = torch.rand(B, S) * 360.0          # azimuth (deg)
    features[..., 2] = -121.0 + torch.rand(B, S) * 6.0   # lon
    features[..., 3] = 33.0 + torch.rand(B, S) * 4.0     # lat
    mask = torch.zeros(B, S)
    mask[:, :n_valid] = 1.0
    return waveforms, features, mask


def test_bias_shape_and_zero_init():
    gb = GeometryAttentionBias(num_heads=4)
    _, features, _ = make_batch()
    bias = gb(features)
    assert bias.shape == (2 * 4, 6, 6)
    # Final layer zero-initialised -> bias must be exactly zero at init
    assert torch.all(bias == 0)


def test_bias_nonzero_after_perturbing_weights():
    gb = GeometryAttentionBias(num_heads=4)
    torch.nn.init.normal_(gb.mlp[-1].weight, std=0.5)
    _, features, _ = make_batch()
    bias = gb(features)
    assert bias.abs().max() > 0


def test_geo_model_matches_baseline_at_init():
    """With the bias MLP zero-initialised, the geo model's forward output
    must be identical to the baseline given identical remaining weights."""
    torch.manual_seed(0)
    base = Sensoformer(make_cfg(False)).eval()
    geo = Sensoformer(make_cfg(True)).eval()
    # Copy all shared weights from base into geo (geo has extra bias params)
    geo.load_state_dict(base.state_dict(), strict=False)

    wf, ft, mask = make_batch()
    with torch.no_grad():
        pred_base, attn_base = base(wf, ft, mask)
        pred_geo, attn_geo = geo(wf, ft, mask)
    assert torch.allclose(pred_base, pred_geo, atol=1e-6)
    assert torch.allclose(attn_base, attn_geo, atol=1e-6)


def test_geo_model_permutation_invariance():
    """Event-level predictions must be invariant to station order, including
    the geometry bias (which must permute consistently with the stations)."""
    torch.manual_seed(0)
    model = Sensoformer(make_cfg(True)).eval()
    torch.nn.init.normal_(model.geometry_bias.mlp[-1].weight, std=0.5)

    wf, ft, mask = make_batch(n_valid=6)  # all stations valid
    perm = torch.randperm(6)
    with torch.no_grad():
        pred, _ = model(wf, ft, mask)
        pred_perm, _ = model(wf[:, perm], ft[:, perm], mask[:, perm])
    assert torch.allclose(pred, pred_perm, atol=1e-5)


def test_geo_model_padding_isolation():
    """Garbage content in padded stations must not change predictions."""
    torch.manual_seed(0)
    model = Sensoformer(make_cfg(True)).eval()
    torch.nn.init.normal_(model.geometry_bias.mlp[-1].weight, std=0.5)

    wf, ft, mask = make_batch(n_valid=4)
    wf2, ft2 = wf.clone(), ft.clone()
    wf2[:, 4:] = 123.0
    ft2[:, 4:] = -55.0
    with torch.no_grad():
        pred, _ = model(wf, ft, mask)
        pred2, _ = model(wf2, ft2, mask)
    assert torch.allclose(pred, pred2, atol=1e-5)


def test_geo_model_backward():
    """Gradients must flow into the bias MLP through the attention mask."""
    model = Sensoformer(make_cfg(True)).train()
    torch.nn.init.normal_(model.geometry_bias.mlp[-1].weight, std=0.5)
    wf, ft, mask = make_batch()
    pred, _ = model(wf, ft, mask)
    pred.sum().backward()
    grads = [p.grad for p in model.geometry_bias.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().max() > 0 for g in grads)


def test_state_dict_unchanged_when_flag_off():
    """geometry_bias=false must not introduce any new state_dict keys, so
    existing baseline checkpoints stay loadable with strict=True."""
    base = Sensoformer(make_cfg(False))
    assert not any(k.startswith("geometry_bias") for k in base.state_dict())


if __name__ == "__main__":
    pytest.main(["-v", __file__])
