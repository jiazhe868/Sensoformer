#!/usr/bin/env python3
"""
Minimal, self-contained example: load a pretrained Sensoformer, run it over an
HDF5 event file, and convert the output to strike/dip/rake.

    # with the released weights (downloaded from the Hugging Face Hub)
    python examples/quickstart_inference.py --input socal-real --limit 5

    # with your own file and checkpoint, no network access needed
    python examples/quickstart_inference.py --input my_events.hdf5 \
        --checkpoint outputs/my_run/best_finetuned_model.pth --limit 5
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sensoformer import load_pretrained
from sensoformer.data.dataset import SeismicDataset, collate_fn
from sensoformer.ext import MTDecomposer
from sensoformer.hub import resolve_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="socal-real",
                    help="HDF5 path or registry dataset name")
    ap.add_argument("--checkpoint", default="sensoformer-v3-finetuned")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # 1. Resolve inputs (both accept local paths or registry names)
    path = resolve_dataset(args.input)
    with h5py.File(path, "r") as f:
        event_ids = list(f.keys())[: args.limit]
    print(f"file   : {path}")
    print(f"events : {len(event_ids)}  ({', '.join(event_ids[:5])})")

    # 2. Load the model (one line; downloads and caches on first use)
    model = load_pretrained(args.checkpoint, device=args.device)
    print(f"model  : {args.checkpoint} on {args.device}, "
          f"{sum(p.numel() for p in model.parameters()):,} parameters\n")

    # 3. Batch the variable-station events (collate_fn pads and builds the mask)
    ds = SeismicDataset(str(path), event_ids, mode="test",
                        augmentation=False, config={"max_stations": 50})
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)

    dec = MTDecomposer()
    print(f"{'event':>12} {'n_sta':>5} {'Mw':>6} {'strike':>7} {'dip':>6} {'rake':>7}"
          f"  {'top-attention stations':>24}")
    with torch.no_grad():
        for wf, ft, mask, target, ids in loader:
            pred, attn = model(wf.to(args.device), ft.to(args.device),
                               mask.to(args.device))
            mw = (pred[:, 0].cpu().numpy() + 1) / 2 * 6 + 2      # denormalize
            mt = pred[:, 1:].cpu().numpy()                        # Mxx..Myz
            a = attn.cpu().numpy()
            n_sta = mask.sum(1).int().numpy()
            for i, eid in enumerate(ids):
                if dec.is_available:
                    s1, s2 = dec.mt_to_sdr(mt[i])
                    s, d, r = s1 if -90 <= s1[2] <= 90 else s2
                    sdr = f"{s:7.1f} {d:6.1f} {r:7.1f}"
                else:
                    sdr = f"{'--':>7} {'--':>6} {'--':>7}"       # run `make build`
                top = np.argsort(a[i][: n_sta[i]])[::-1][:3]
                print(f"{eid:>12} {n_sta[i]:5d} {mw[i]:6.2f} {sdr}"
                      f"  {str(list(top)):>24}")

    if not dec.is_available:
        print("\nNote: strike/dip/rake need the Fortran kernel -> run `make build`.")
    print("\nThe attention column lists the station indices the model weighted most "
          "heavily for each event.")


if __name__ == "__main__":
    main()
