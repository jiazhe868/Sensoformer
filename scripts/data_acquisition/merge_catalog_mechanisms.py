#!/usr/bin/env python3
"""
Step 4: Merge the STP event list with YSH focal-mechanism solutions to
produce the metadata used downstream (standardized port of the original
gen_events_wmec.py, translated and parameterized).

For mechanisms whose rake falls outside [-90, 90], the AUXILIARY nodal plane
is substituted (a focal mechanism is invariant under this exchange), so all
stored rakes lie in [-90, 90] — the convention assumed by the synthetic
source sampler and by sdr2mxyz_norm targets.

Usage:
  python merge_catalog_mechanisms.py \
      --events events_cleaned.dat --mechanisms ysh_all.log \
      --output events_wmeca.dat
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from sensoformer.hub import CATALOGS, resolve_catalog  # noqa: E402


def calculate_auxiliary_plane(strike, dip, rake):
    """Return (strike, dip, rake) of the auxiliary nodal plane, degrees.
    Direct port of the original awk-derived algorithm; the returned dip is
    guaranteed to lie in [0, 90]."""
    d2r, r2d = math.pi / 180.0, 180.0 / math.pi
    phi, delta, lam = strike * d2r, dip * d2r, rake * d2r

    l1 = -math.sin(lam) * math.sin(delta)
    l2 = math.cos(lam) * math.cos(phi) + math.cos(delta) * math.sin(lam) * math.sin(phi)
    l3 = -(math.cos(lam) * math.sin(phi) - math.cos(delta) * math.sin(lam) * math.cos(phi))
    n1 = -math.cos(delta)
    n2 = -math.sin(delta) * math.sin(phi)
    n3 = -math.sin(delta) * math.cos(phi)

    if l1 > 0:
        l1, l2, l3 = -l1, -l2, -l3
        n1, n2, n3 = -n1, -n2, -n3

    delta2 = math.acos(max(-1.0, min(1.0, -l1)))
    fai2 = math.atan2(-l2, -l3)
    cos_lam2 = n2 * math.cos(fai2) - n3 * math.sin(fai2)
    sin_lam2 = 0.0
    if abs(math.sin(delta2)) > 1e-6:
        sin_lam2 = max(-1.0, min(1.0, -n1 / math.sin(delta2)))
    lam2 = math.atan2(sin_lam2, cos_lam2)

    return (fai2 * r2d) % 360.0, delta2 * r2d, lam2 * r2d


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--events", default="events_cleaned.dat",
                        help="Cleaned STP event list (event ID in column 11)")
    parser.add_argument("--mechanisms", default="ysh_all.log",
                        help="YSH-format mechanism catalog: a path, or a registry "
                             f"name ({', '.join(CATALOGS)}) for the mirrored copy")
    parser.add_argument("--output", default="events_wmeca.dat")
    args = parser.parse_args()

    mechanisms = {}
    with open(resolve_catalog(args.mechanisms)) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 21:
                continue
            event_id, quality = parts[6], parts[20]
            try:
                strike, dip, rake = (float(parts[11]), float(parts[12]),
                                     float(parts[13]))
                if not (-90 <= rake <= 90) and strike <= 360:
                    strike, dip, rake = calculate_auxiliary_plane(strike, dip, rake)
                mechanisms[event_id] = (f"{strike:.0f}", f"{dip:.0f}",
                                        f"{rake:.0f}", quality)
            except ValueError:
                mechanisms[event_id] = (parts[11], parts[12], parts[13], quality)
    print(f"Loaded {len(mechanisms)} mechanism solutions.")

    n_out = 0
    with open(args.events) as fin, open(args.output, "w") as fout:
        for line in fin:
            original = line.rstrip("\n")
            parts = original.split()
            if len(parts) < 11:
                fout.write(original + "\n")
                continue
            meca = mechanisms.get(parts[10], ("-999", "-999", "-999", "Z"))
            fout.write(f"{original} {meca[0]:>5s} {meca[1]:>5s} "
                       f"{meca[2]:>5s} {meca[3]:>2s}\n")
            n_out += 1
    print(f"Wrote {n_out} merged events to {args.output}")


if __name__ == "__main__":
    main()
