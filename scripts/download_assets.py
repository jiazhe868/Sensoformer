#!/usr/bin/env python3
"""
Download Sensoformer release assets (pretrained weights and/or datasets)
from the Hugging Face Hub.

Weights are small (~8 MB each); datasets are large (0.3-12 GB), so they are
opt-in by name.

Examples
--------
python scripts/download_assets.py --weights                    # both models
python scripts/download_assets.py --datasets socal-real        # 0.26 GB
python scripts/download_assets.py --weights --datasets socal-real synthetic-psdr
python scripts/download_assets.py --catalogs yhs-socal         # 31 MB label catalog
python scripts/download_assets.py --list                       # just show what exists
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sensoformer.hub import (CATALOGS, CHECKPOINTS, DATASETS, HF_DATA_REPO,
                             HF_REPO, LOCAL_DATA_DIR, resolve_catalog,
                             resolve_checkpoint, resolve_dataset)


def show():
    print(f"Model repo  : {HF_REPO}")
    for n, e in CHECKPOINTS.items():
        print(f"  {n:<28} {e['filename']:<36} {e['stage']}")
    print(f"\nDataset repo: {HF_DATA_REPO}")
    for n, e in DATASETS.items():
        print(f"  {n:<18} {e['filename']}")
        print(f"  {'':<18} {e['about']}")
    print(f"\nLabel catalogs (same dataset repo):")
    for n, e in CATALOGS.items():
        print(f"  {n:<18} {e['filename']}")
        print(f"  {'':<18} {e['about']}")
        print(f"  {'':<18} source: {e['source']}")
    print(f"\nLocal dataset dir ($SENSOFORMER_DATA): {LOCAL_DATA_DIR}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", action="store_true", help="download all checkpoints")
    p.add_argument("--checkpoint", nargs="*", default=[], choices=list(CHECKPOINTS),
                   help="download specific checkpoints")
    p.add_argument("--datasets", nargs="*", default=[], choices=list(DATASETS),
                   help="download specific datasets (large)")
    p.add_argument("--catalogs", nargs="*", default=[], choices=list(CATALOGS),
                   help="download label catalogs (small text files)")
    p.add_argument("--data-dir", type=Path, default=LOCAL_DATA_DIR,
                   help=f"where datasets are placed (default {LOCAL_DATA_DIR})")
    p.add_argument("--symlink", action="store_true",
                   help="symlink datasets from the HF cache instead of copying")
    p.add_argument("--list", action="store_true", help="list available assets and exit")
    args = p.parse_args()

    if args.list or not (args.weights or args.checkpoint or args.datasets
                         or args.catalogs):
        show()
        if not args.list:
            print("\nNothing requested. Use --weights and/or --datasets NAME.")
        return

    for name in (list(CHECKPOINTS) if args.weights else args.checkpoint):
        path, _ = resolve_checkpoint(name)
        print(f"weights  {name:<28} -> {path}")

    if args.datasets or args.catalogs:
        args.data_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        cached = resolve_dataset(name)
        dst = args.data_dir / DATASETS[name]["filename"]
        if cached.resolve() == dst.resolve() or dst.exists():
            print(f"dataset  {name:<18} -> {dst} (already present)")
            continue
        if args.symlink:
            dst.symlink_to(cached.resolve())
            print(f"dataset  {name:<18} -> {dst} (symlink)")
        else:
            print(f"dataset  {name:<18} copying {cached.stat().st_size/2**30:.2f} GB ...")
            shutil.copy2(cached, dst)
            print(f"dataset  {name:<18} -> {dst}")

    for name in args.catalogs:
        cached = resolve_catalog(name)
        dst = args.data_dir / CATALOGS[name]["filename"]
        if cached.resolve() != dst.resolve() and not dst.exists():
            shutil.copy2(cached, dst)
        print(f"catalog  {name:<18} -> {dst}")
        print(f"{'':<29} cite: {CATALOGS[name]['cite']}")

    print("\nReady. The Hydra data configs resolve $SENSOFORMER_DATA "
          f"(default ./data), e.g.\n  export SENSOFORMER_DATA={args.data_dir}")


if __name__ == "__main__":
    main()
