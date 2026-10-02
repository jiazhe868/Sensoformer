#!/usr/bin/env python3
"""
Lock in the architecture numbers documented in docs/ARCHITECTURE.md and
published with the release checkpoints. If a change to the model alters any of
these, the documentation (and the model card) must be updated with it.
"""
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from sensoformer.hub import _SENSOFORMER_ARCH, build_model
from sensoformer.models.network import Sensoformer

# docs/ARCHITECTURE.md, section 5
EXPECTED_TOTAL = 2_063_015
EXPECTED_BLOCKS = {
    "station_encoder": 262_880,
    "event_aggregator": 1_779_072,
    "attention_pooling": 8_321,
}


def n_params(module):
    return sum(p.numel() for p in module.parameters())


@pytest.fixture
def model():
    return build_model(_SENSOFORMER_ARCH)


def test_total_parameter_count(model):
    assert n_params(model) == EXPECTED_TOTAL


def test_per_block_parameter_counts(model):
    for name, expected in EXPECTED_BLOCKS.items():
        assert n_params(getattr(model, name)) == expected, name
    heads = n_params(model.magnitude_head) + n_params(model.moment_tensor_head)
    assert heads == 12_742


def test_attention_dimensions(model):
    layer = model.event_aggregator.layers[0]
    attn = layer.self_attn
    assert attn.embed_dim == 128
    assert attn.num_heads == 4
    assert attn.head_dim == 32
    # Packed QKV projection: 3 * d_model rows
    assert tuple(attn.in_proj_weight.shape) == (384, 128)
    assert tuple(attn.out_proj.weight.shape) == (128, 128)
    assert len(model.event_aggregator.layers) == 3


def test_feedforward_dimension_is_2048(model):
    """The released checkpoint was trained with FFN=2048 (the code default),
    not the 256 quoted in an early draft of the paper."""
    layer = model.event_aggregator.layers[0]
    assert tuple(layer.linear1.weight.shape) == (2048, 128)
    assert tuple(layer.linear2.weight.shape) == (128, 2048)


def test_no_positional_encoding_parameters(model):
    """Set semantics come from the *absence* of positional encodings."""
    names = [n for n, _ in model.named_parameters()]
    assert not any("pos" in n.lower() or "embed_positions" in n.lower()
                   for n in names)


def test_attention_pooling_is_additive_not_qkv(model):
    """Pooling is a scoring MLP (128->64->1), not a multi-head QKV block."""
    shapes = [tuple(p.shape) for p in model.attention_pooling.parameters()]
    assert (64, 128) in shapes and (1, 64) in shapes
    assert not hasattr(model.attention_pooling, "in_proj_weight")


def test_forward_shapes_and_permutation_invariance(model):
    model.eval()
    torch.manual_seed(0)
    B, S = 2, 9
    wf, ft, mask = torch.randn(B, S, 12, 101), torch.randn(B, S, 20), torch.ones(B, S)
    with torch.no_grad():
        pred, attn = model(wf, ft, mask)
        assert pred.shape == (B, 6) and attn.shape == (B, S)
        perm = torch.randperm(S)
        pred_perm, _ = model(wf[:, perm], ft[:, perm], mask[:, perm])
    assert torch.allclose(pred, pred_perm, atol=1e-5), \
        "event-level predictions must not depend on station order"


def test_feedforward_dim_is_configurable():
    cfg = OmegaConf.create({"model": dict(_SENSOFORMER_ARCH, feedforward_dim=256)})
    small = Sensoformer(cfg)
    assert tuple(small.event_aggregator.layers[0].linear1.weight.shape) == (256, 128)
    assert n_params(small) < EXPECTED_TOTAL
