#!/usr/bin/env python3
"""
Agreement tests for the data-acquisition chain (scripts/data_acquisition/):

1. The documented pick conventions must hold on the existing archives: the
   P pick stored in the SAC headers (analyst t1 after 03_add_picks_to_sac.sh,
   or the automatic 'a' marker written by STP TRIG) must agree with the
   analyst pick in the per-event phase file ({evid}.dat) to within 0.5 s.
   Exact equality is NOT required: the archives mix pick sources (analyst
   phase files, automatic pickers), which legitimately differ by a few
   tenths of a second for the same arrival. What this test guards is the
   documented convention -- seconds relative to the event origin in the same
   reference as the phase files; a wrong reference would be off by tens of
   seconds.
2. The auxiliary-plane conversion in merge_catalog_mechanisms.py must
   preserve the moment tensor exactly (a focal mechanism is invariant under
   nodal-plane exchange) while bringing rake into [-90, 90].

Archive-dependent tests are skipped when the data is not on this machine.
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJ_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJ_ROOT / "scripts" / "preprocessing"))
sys.path.insert(0, str(PROJ_ROOT / "scripts" / "data_acquisition"))

# Point SENSOFORMER_TEST_SAC_ARCHIVE at a raw SAC archive (a directory holding
# per-event folders plus their {evid}.dat phase files) and
# SENSOFORMER_TEST_SAC_EVENT at one event id in it to enable the header check.
# Set SENSOFORMER_TEST_SAC_ANALYST=1 if that archive carries analyst t1/t2
# picks rather than only automatic 'a' markers.
ARCHIVES = [(
    os.environ.get("SENSOFORMER_TEST_SAC_ARCHIVE", ""),
    os.environ.get("SENSOFORMER_TEST_SAC_EVENT", ""),
    os.environ.get("SENSOFORMER_TEST_SAC_ANALYST", "") == "1",
)]


def parse_phase_file(path):
    """{evid}.dat -> {(net, sta): {'P': t, 'S': t}} (first pick per phase)."""
    picks = {}
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            t = line.split()
            if len(t) < 13:
                continue
            key, phase = (t[0], t[1]), t[7].upper()
            picks.setdefault(key, {})
            if phase in ("P", "S") and phase not in picks[key]:
                picks[key][phase] = float(t[12])
    return picks


@pytest.mark.parametrize("root,evid,analyst_picks", ARCHIVES)
def test_phase_file_picks_present_in_sac_headers(root, evid, analyst_picks):
    phase_file = os.path.join(root, f"{evid}.dat")
    event_dir = os.path.join(root, evid)
    if not (root and evid and os.path.isfile(phase_file)
            and os.path.isdir(event_dir)):
        pytest.skip("set SENSOFORMER_TEST_SAC_ARCHIVE / _EVENT to run this check")
    from obspy import read

    picks = parse_phase_file(phase_file)
    checked, agree = 0, 0
    for (net, sta), ph in picks.items():
        if "P" not in ph:
            continue
        z_files = glob.glob(os.path.join(event_dir, f"{evid}.{net}.{sta}.*Z.sac"))
        if not z_files:
            continue
        hdr = read(z_files[0], headonly=True)[0].stats.sac
        stored = hdr.get("t1") if analyst_picks else hdr.get("a")
        if stored is None:
            stored = hdr.get("a")
        if stored is None:
            continue
        checked += 1
        if abs(stored - ph["P"]) <= 0.5:
            agree += 1
        if checked >= 10:
            break
    assert checked >= 3, "too few stations verified"
    assert agree / checked >= 0.8, (
        f"only {agree}/{checked} stations' stored P picks agree with the "
        "phase file within 0.5 s -- time reference or units mismatch?")


def test_auxiliary_plane_preserves_moment_tensor():
    from merge_catalog_mechanisms import calculate_auxiliary_plane
    from sdr_utils import sdr2mxyz_norm

    rng = np.random.RandomState(0)
    for _ in range(200):
        strike = rng.uniform(0, 360)
        dip = rng.uniform(1, 89)
        rake = rng.uniform(90, 270)          # deliberately outside [-90, 90]
        if rake > 180:
            rake -= 360                       # -> (-180, -90) or (90, 180)
        s2, d2, r2 = calculate_auxiliary_plane(strike, dip, rake)
        assert -90.001 <= r2 <= 90.001, f"aux rake {r2} outside [-90, 90]"
        assert 0 <= d2 <= 90.001
        mt_orig = np.array(sdr2mxyz_norm(strike, dip, rake))
        mt_aux = np.array(sdr2mxyz_norm(s2, d2, r2))
        assert np.allclose(mt_orig, mt_aux, atol=1e-6), (
            f"MT changed: ({strike:.1f},{dip:.1f},{rake:.1f}) -> "
            f"({s2:.1f},{d2:.1f},{r2:.1f})")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
