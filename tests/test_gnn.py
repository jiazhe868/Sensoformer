#!/usr/bin/env python3
"""
Unit tests for the GNN message-passing layers, in particular DenseGCNLayer
(the non-attention "MPNN" baseline conv layer).
"""
import sys
from pathlib import Path

import torch
import pytest

PROJ_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJ_ROOT / "src"))

from sensoformer.models.gnn import DenseGCNLayer, DenseGATLayer


@pytest.fixture
def toy_graph():
    torch.manual_seed(0)
    B, S, D = 2, 5, 8
    x = torch.randn(B, S, D)
    # Stations 0-3 form a chain with self-loops; station 4 is padding (isolated).
    adj = torch.zeros(B, S, S)
    for i in range(4):
        adj[:, i, i] = 1
    adj[:, 0, 1] = adj[:, 1, 0] = 1
    adj[:, 1, 2] = adj[:, 2, 1] = 1
    adj[:, 2, 3] = adj[:, 3, 2] = 1
    return x, adj


def test_gcn_layer_has_no_attention_params():
    layer = DenseGCNLayer(in_dim=8, out_dim=8, dropout=0.0)
    names = [n for n, _ in layer.named_parameters()]
    assert not any("a_src" in n or "a_dst" in n for n in names)
    assert set(names) == {"W.weight", "W.bias", "layer_norm.weight", "layer_norm.bias"}


def test_gcn_layer_output_shape(toy_graph):
    x, adj = toy_graph
    layer = DenseGCNLayer(in_dim=8, out_dim=8, dropout=0.0).eval()
    out = layer(x, adj)
    assert out.shape == x.shape


def test_gcn_layer_padding_isolation(toy_graph):
    """A padding node (no real edges to any valid station) must not leak
    into real stations' outputs, regardless of its own input content."""
    x, adj = toy_graph
    layer = DenseGCNLayer(in_dim=8, out_dim=8, dropout=0.0).eval()

    out = layer(x, adj)

    x_garbage = x.clone()
    x_garbage[:, 4, :] = 999.0
    out_garbage = layer(x_garbage, adj)

    assert torch.allclose(out[:, :4], out_garbage[:, :4], atol=1e-5)


def test_gcn_layer_permutation_equivariant(toy_graph):
    x, adj = toy_graph
    layer = DenseGCNLayer(in_dim=8, out_dim=8, dropout=0.0).eval()

    out = layer(x, adj)

    perm = torch.tensor([2, 0, 3, 1, 4])
    x_perm = x[:, perm, :]
    adj_perm = adj[:, perm][:, :, perm]
    out_perm = layer(x_perm, adj_perm)
    out_perm_unshuffled = out_perm[:, perm.argsort(), :]

    assert torch.allclose(out, out_perm_unshuffled, atol=1e-5)


def test_gat_layer_has_attention_params():
    """Sanity check that DenseGATLayer (used elsewhere) is distinguishable
    from DenseGCNLayer by actually having learned attention parameters."""
    layer = DenseGATLayer(in_dim=8, out_dim=8, heads=2, dropout=0.0)
    names = [n for n, _ in layer.named_parameters()]
    assert any("a_src" in n for n in names)
    assert any("a_dst" in n for n in names)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
