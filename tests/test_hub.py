#!/usr/bin/env python3
"""The weight loader must actually transfer weights (regression test: an
earlier version silently skipped load_state_dict on the strict path, leaving
a randomly initialised model that still ran a forward pass)."""
from pathlib import Path

import pytest
import torch

from sensoformer.hub import CHECKPOINTS, build_model, load_pretrained, _SENSOFORMER_ARCH


def test_registry_entries_are_well_formed():
    for name, entry in CHECKPOINTS.items():
        assert {"filename", "arch", "stage", "use_for"} <= set(entry)
        assert entry["filename"].endswith(".pth")


def test_load_pretrained_transfers_weights(tmp_path):
    ref = build_model(_SENSOFORMER_ARCH)
    # Make the reference weights clearly distinct from any fresh init.
    with torch.no_grad():
        for prm in ref.parameters():
            prm.fill_(0.015625)
    ckpt = tmp_path / "toy.pth"
    torch.save({"state_dict": ref.state_dict(), "arch": _SENSOFORMER_ARCH}, ckpt)

    loaded = load_pretrained(str(ckpt), device="cpu")
    for (k, a), (_, b) in zip(ref.state_dict().items(), loaded.state_dict().items()):
        assert torch.equal(a, b), f"parameter {k} was not loaded"
    assert not loaded.training


def test_load_pretrained_accepts_bare_state_dict(tmp_path):
    ref = build_model(_SENSOFORMER_ARCH)
    ckpt = tmp_path / "bare.pth"
    torch.save(ref.state_dict(), ckpt)            # legacy format, no metadata
    loaded = load_pretrained(str(ckpt), device="cpu")
    assert torch.equal(ref.state_dict()["magnitude_head.0.weight"],
                       loaded.state_dict()["magnitude_head.0.weight"])


def test_load_pretrained_strips_dataparallel_prefix(tmp_path):
    ref = build_model(_SENSOFORMER_ARCH)
    ckpt = tmp_path / "dp.pth"
    torch.save({"module." + k: v for k, v in ref.state_dict().items()}, ckpt)
    loaded = load_pretrained(str(ckpt), device="cpu")
    assert torch.equal(ref.state_dict()["magnitude_head.0.weight"],
                       loaded.state_dict()["magnitude_head.0.weight"])


def test_unknown_name_raises_with_hint():
    with pytest.raises(FileNotFoundError, match="Known checkpoints"):
        load_pretrained("no-such-model")
