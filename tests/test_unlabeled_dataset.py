#!/usr/bin/env python3
"""The dataset must accept HDF5 files without ground-truth attributes
(pure-inference use) and signal their absence with NaN targets."""
import numpy as np
import pytest
import torch
import h5py

from sensoformer.data.dataset import SeismicDataset, collate_fn


def _write(path, labeled):
    with h5py.File(path, "w") as f:
        for i in range(3):
            g = f.create_group(f"ev{i}")
            g.create_dataset("waveforms", data=np.random.randn(7, 12, 101).astype("f4"))
            g.create_dataset("features", data=np.random.randn(7, 20).astype("f4"))
            if labeled:
                g.attrs["magnitude"] = 3.5
                for k in ("Mxx", "Myy", "Mxy", "Mxz", "Myz"):
                    g.attrs[k] = 0.1


@pytest.mark.parametrize("labeled", [True, False])
def test_dataset_handles_missing_labels(tmp_path, labeled):
    path = tmp_path / "toy.hdf5"
    _write(path, labeled)
    ds = SeismicDataset(str(path), [f"ev{i}" for i in range(3)], mode="test",
                        augmentation=False, config={})
    wf, ft, mask, target, eid = ds[0]
    assert wf.shape == (7, 12, 101) and ft.shape == (7, 20)
    assert np.isnan(target.numpy()).all() is not labeled or labeled
    if labeled:
        assert not np.isnan(target.numpy()).any()
    else:
        assert np.isnan(target.numpy()).all()
    batch = collate_fn([ds[i] for i in range(3)])
    assert batch[0].shape[0] == 3 and batch[2].shape == (3, 7)
