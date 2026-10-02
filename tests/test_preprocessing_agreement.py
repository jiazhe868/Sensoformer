#!/usr/bin/env python3
"""
Agreement tests: the standardized preprocessing scripts must reproduce the
contents of the original (released) training HDF5 files from the raw data.

- Real pipeline is fully deterministic -> waveforms/features must match the
  released file exactly (verified bitwise-equal at creation time; tested here
  with a small float tolerance to stay robust to library updates).
- Synthetic pipeline contains PSDR randomization (coda injection, P/S
  amplitude factors), so only its deterministic outputs are required to match
  exactly (station selection, geometry feature columns, attributes); the
  stochastic waveforms are required to correlate strongly with the released
  ones (median per-station correlation of the P-window Z component > 0.5 --
  the same level obtained when re-running the ORIGINAL script twice, since
  the random draws differ run to run).

These tests are skipped automatically when the raw archives or the released
HDF5 files are not present on the machine.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJ_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ_ROOT / "scripts" / "preprocessing"))

# These tests compare the shipped preprocessing against the original released
# HDF5 files, so they need both the raw SAC archives and those files. Point the
# environment variables below at them to enable the comparison; the tests skip
# cleanly otherwise (e.g. in CI or on a fresh clone).
REAL_ORIG = os.environ.get("SENSOFORMER_TEST_REAL_HDF5", "")
REAL_ROOT = os.environ.get("SENSOFORMER_TEST_REAL_SAC", "")
SYN_ORIG = os.environ.get("SENSOFORMER_TEST_SYN_HDF5", "")
SYN_ROOT = os.environ.get("SENSOFORMER_TEST_SYN_SAC", "")

N_EVENTS = 2  # events per test (kept small: each takes ~10-30 s)


def _available(*paths):
    return all(p and os.path.exists(p) for p in paths)


def _decode(names):
    return [s.decode() if isinstance(s, bytes) else s for s in names]


@pytest.mark.skipif(not _available(REAL_ORIG, REAL_ROOT),
                    reason="set SENSOFORMER_TEST_REAL_{SAC,HDF5} to run this comparison")
def test_real_preprocessing_reproduces_released_hdf5():
    import h5py
    from preprocess_real_hdf5 import process_event

    with h5py.File(REAL_ORIG) as f:
        ids = [k for k in list(f.keys())
               if os.path.isdir(os.path.join(REAL_ROOT, k))][:N_EVENTS]
        assert ids, "no overlapping events found"
        for eid in ids:
            _, status, data = process_event(({"event_id": eid}, REAL_ROOT, 5))
            assert status == "success", data
            _, wf_new, ft_new, names_new = data
            g = f[eid]
            assert names_new == _decode(g["station_names"][:])
            assert wf_new.shape == g["waveforms"].shape
            assert np.allclose(wf_new, g["waveforms"][:], atol=1e-6)
            assert np.allclose(ft_new, g["features"][:], atol=1e-5)


@pytest.mark.skipif(not _available(SYN_ORIG, SYN_ROOT),
                    reason="set SENSOFORMER_TEST_SYN_{SAC,HDF5} to run this comparison")
def test_synthetic_preprocessing_agrees_with_released_hdf5():
    import h5py
    from preprocess_synthetic_hdf5 import process_event
    from sdr_utils import parse_synthetic_event_name

    np.random.seed(0)
    with h5py.File(SYN_ORIG) as f:
        ids = [k for k in list(f.keys())
               if os.path.isdir(os.path.join(SYN_ROOT, k))][:N_EVENTS]
        assert ids, "no overlapping events found"
        for eid in ids:
            ev = parse_synthetic_event_name(eid)
            assert ev is not None
            _, status, data = process_event((ev, SYN_ROOT, 5))
            assert status == "success", data
            _, wf_new, ft_new, names_new = data
            g = f[eid]
            # Deterministic parts: station selection, geometry features, attrs
            assert names_new == _decode(g["station_names"][:])
            assert wf_new.shape == g["waveforms"].shape
            assert np.allclose(ft_new[:, :5], g["features"][:][:, :5], atol=1e-5)
            assert ev["magnitude"] == pytest.approx(g.attrs["magnitude"])
            assert ev["depth"] == pytest.approx(g.attrs["depth"])
            # Stochastic parts: strong correlation with the released waveforms
            wf_old = g["waveforms"][:]
            cors = [np.corrcoef(wf_new[i, 0], wf_old[i, 0])[0, 1]
                    for i in range(len(wf_new))]
            assert np.median(cors) > 0.5, f"median P-Z corr {np.median(cors):.3f}"


def test_sdr2mxyz_matches_known_mechanisms():
    """Pure double-couple sanity checks of the SDR->MT conversion."""
    from sdr_utils import sdr2mxyz_norm
    # Vertical strike-slip (strike 0, dip 90, rake 0): Mxy = +1... convention:
    mxx, myy, mxy, mxz, myz = sdr2mxyz_norm(0.0, 90.0, 0.0)
    assert abs(mxx) < 1e-12 and abs(myy) < 1e-12
    assert abs(abs(mxy) - 1.0) < 1e-12
    assert abs(mxz) < 1e-12 and abs(myz) < 1e-12
    # 45-degree dip-slip (strike 0, dip 45, rake 90): pure Myy/Mzz couple
    mxx, myy, mxy, mxz, myz = sdr2mxyz_norm(0.0, 45.0, 90.0)
    mzz = -(mxx + myy)
    assert abs(myy + 1.0) < 1e-12 and abs(mzz - 1.0) < 1e-12
    assert abs(mxx) < 1e-12 and abs(mxy) < 1e-12


if __name__ == "__main__":
    pytest.main(["-v", __file__])
