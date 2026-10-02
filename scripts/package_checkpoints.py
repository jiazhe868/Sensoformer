#!/usr/bin/env python3
"""
Package raw training checkpoints into self-describing release checkpoints.

Training writes bare `state_dict` files. For distribution we wrap them with
the architecture definition and provenance so that `load_pretrained()` needs
no external configuration:

    {"state_dict": ..., "arch": {...}, "sensoformer_version": "1.0.0",
     "stage": "...", "notes": "...", "metrics": {...}}

Example
-------
python scripts/package_checkpoints.py \\
    --finetuned /path/to/best_finetuned_model_v3.pth \\
    --pretrained /path/to/best_sensoformer_model_v3.pth \\
    --out-dir release_assets/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sensoformer import __version__
from sensoformer.hub import CHECKPOINTS, _SENSOFORMER_ARCH

# Published evaluation numbers (244-event held-out SoCal test split, seed=42).
METRICS = {
    "sensoformer-v3-finetuned": {
        "test_set": "244 held-out real SoCal events (seed=42, 80/10/10 split)",
        "kagan_mean_deg": 23.89, "kagan_median_deg": 19.72,
        "magnitude_mae": 0.0996, "frac_kagan_lt30": 0.766,
    },
    "sensoformer-v3-pretrained": {
        "test_set": "held-out PSDR synthetic events",
        "kagan_mean_deg": 6.6, "kagan_median_deg": 5.0,
        "magnitude_mae": 0.066,
    },
}


def package(src: Path, name: str, out_dir: Path) -> Path:
    blob = torch.load(src, map_location="cpu", weights_only=False)
    state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    entry = CHECKPOINTS[name]
    payload = {
        "state_dict": state,
        "arch": dict(_SENSOFORMER_ARCH),
        "sensoformer_version": __version__,
        "checkpoint_name": name,
        "stage": entry["stage"],
        "use_for": entry["use_for"],
        "metrics": METRICS.get(name, {}),
        "source_file": src.name,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / entry["filename"]
    torch.save(payload, dst)
    size_mb = dst.stat().st_size / 1048576
    n_params = sum(v.numel() for v in state.values() if hasattr(v, "numel"))
    print(f"  {name:<28} -> {dst.name:<36} {size_mb:6.1f} MB  "
          f"{n_params:,} params")
    return dst


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--finetuned", type=Path,
                   help="raw checkpoint for sensoformer-v3-finetuned")
    p.add_argument("--pretrained", type=Path,
                   help="raw checkpoint for sensoformer-v3-pretrained")
    p.add_argument("--out-dir", type=Path, default=Path("release_assets"))
    args = p.parse_args()

    if not (args.finetuned or args.pretrained):
        p.error("give at least one of --finetuned / --pretrained")
    print("Packaging checkpoints:")
    written = []
    if args.finetuned:
        written.append(package(args.finetuned, "sensoformer-v3-finetuned", args.out_dir))
    if args.pretrained:
        written.append(package(args.pretrained, "sensoformer-v3-pretrained", args.out_dir))

    # Round-trip check: the packaged file must load through the public API.
    from sensoformer import load_pretrained
    for path in written:
        m = load_pretrained(str(path), device="cpu")
        print(f"  verified loadable: {path.name} "
              f"({sum(q.numel() for q in m.parameters()):,} params)")
    manifest = args.out_dir / "manifest.json"
    manifest.write_text(json.dumps(
        {"sensoformer_version": __version__,
         "files": [p_.name for p_ in written]}, indent=2))
    print(f"\nDone. Upload with: python scripts/upload_assets_to_hf.py "
          f"--weights-dir {args.out_dir}")


if __name__ == "__main__":
    main()
