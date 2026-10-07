#!/usr/bin/env python3
"""
Validate a focal-mechanism catalog against the YHS column layout expected by
scripts/preprocessing/preprocess_real_hdf5.py.

Run this first if preprocessing reports a catalog format error, or when
adapting a catalog from another source/region.

    python scripts/data_acquisition/check_catalog_format.py ysh_all.log
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing"))
from preprocess_real_hdf5 import VALID_GRADES, YSH_COL, YSH_MIN_COLUMNS  # noqa: E402

LABELS = [
    (0, "year"), (1, "month"), (2, "day"), (3, "hour"), (4, "minute"),
    (5, "second"), (6, "event id"), (7, "latitude"), (8, "longitude"),
    (9, "depth (km)"), (10, "magnitude"), (11, "strike"), (12, "dip"),
    (13, "rake"), (14, "nodal-plane uncertainty 1"),
    (15, "nodal-plane uncertainty 2"), (20, "quality grade"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("catalog", type=Path)
    ap.add_argument("--show", type=int, default=2, help="sample lines to dissect")
    args = ap.parse_args()

    if not args.catalog.exists():
        sys.exit(f"File not found: {args.catalog}\n"
                 "See docs/DATA_PIPELINE.md for where to obtain the YHS catalog.")

    rows = [ln for ln in args.catalog.read_text(errors="replace").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    if not rows:
        sys.exit("File contains no data lines.")

    widths = Counter(len(r.split()) for r in rows)
    print(f"file            : {args.catalog}")
    print(f"data lines      : {len(rows)}")
    print(f"column counts   : " + ", ".join(
        f"{n} cols x{c}" for n, c in widths.most_common(4)))

    ok = True
    if widths.most_common(1)[0][0] < YSH_MIN_COLUMNS:
        print(f"  PROBLEM: expected at least {YSH_MIN_COLUMNS} columns")
        ok = False

    print(f"\nfirst {args.show} line(s), column by column:")
    for r in rows[: args.show]:
        t = r.split()
        print("  " + "-" * 58)
        for idx, name in LABELS:
            val = t[idx] if idx < len(t) else "<missing>"
            print(f"  col {idx:>2}  {name:<28} {val}")

    grades = Counter(t.split()[YSH_COL["quality"]] for t in rows
                     if len(t.split()) > YSH_COL["quality"])
    print(f"\nquality grades  : {dict(grades.most_common())}")
    bad = set(grades) - VALID_GRADES
    if bad:
        print(f"  PROBLEM: column {YSH_COL['quality']} holds {sorted(bad)}, "
              f"expected grades from {sorted(VALID_GRADES)}.")
        print("           The column order likely differs from the YHS layout.")
        ok = False

    try:
        mags = [float(r.split()[YSH_COL["magnitude"]]) for r in rows[:5000]
                if len(r.split()) > YSH_COL["quality"]]
        dips = [float(r.split()[YSH_COL["dip"]]) for r in rows[:5000]
                if len(r.split()) > YSH_COL["quality"]]
        print(f"magnitude range : {min(mags):.2f} to {max(mags):.2f}")
        print(f"dip range       : {min(dips):.1f} to {max(dips):.1f} "
              f"(should lie within 0-90)")
        if not (0 <= min(dips) and max(dips) <= 90.5):
            print("  PROBLEM: dip outside 0-90 deg -- columns may be shifted.")
            ok = False
    except (ValueError, IndexError) as exc:
        print(f"  PROBLEM: numeric columns did not parse ({exc})")
        ok = False

    print("\n" + ("OK: this file matches the expected YHS layout."
                  if ok else
                  "NOT USABLE as-is. See docs/DATA_PIPELINE.md "
                  "(\"Obtaining the YHS focal-mechanism catalog\")."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
