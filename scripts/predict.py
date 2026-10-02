#!/usr/bin/env python3
"""
Sensoformer inference: run a trained model over an HDF5 event file and write
per-event source parameters (and, optionally, a catalog file and figures).

The input HDF5 must follow the schema in docs/DATA_FORMAT.md. Ground-truth
attributes are optional: when they are present, error metrics (Kagan angle,
magnitude error) are reported as well; when absent, predictions are still
written (pure inference on new events).

Examples
--------
# Headline model on a prepared file, auto-downloading weights from the Hub:
python scripts/predict.py --input data/socal_m25plus_all.hdf5 --out-dir results/

# A local checkpoint, GPU, with a catalog file and comparison figures:
python scripts/predict.py --input my_events.hdf5 \\
    --checkpoint outputs/my_run/best_finetuned_model.pth \\
    --out-dir results/my_run --device cuda --catalog --figures
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Keep multiprocessing scratch off a possibly-full /tmp (opt-in via env).
if os.environ.get("SENSOFORMER_TMPDIR"):
    os.environ["TMPDIR"] = os.environ["SENSOFORMER_TMPDIR"]

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sensoformer.data.dataset import SeismicDataset, collate_fn
from sensoformer.ext import MTDecomposer
from sensoformer.hub import load_pretrained, resolve_dataset
from sensoformer.utils.physics import kagan_angle

MAG_MIN, MAG_MAX = 2.0, 8.0
MT_NAMES = ["Mxx", "Myy", "Mxy", "Mxz", "Myz"]


def denorm_mag(x):
    """[-1, 1] -> [MAG_MIN, MAG_MAX]."""
    return (np.asarray(x) + 1.0) / 2.0 * (MAG_MAX - MAG_MIN) + MAG_MIN


def pick_plane(pair):
    """Of the two conjugate nodal planes, return the one with rake in [-90, 90]."""
    sdr1, sdr2 = pair
    return sdr1 if -90 <= sdr1[2] <= 90 else sdr2


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
                   help="HDF5 path, or a registry dataset name "
                        "(socal-real | synthetic-psdr | synthetic-clean)")
    p.add_argument("--checkpoint", default="sensoformer-v3-finetuned",
                   help="Registry name or local .pth path "
                        "(default: sensoformer-v3-finetuned)")
    p.add_argument("--out-dir", default="results",
                   help="Directory for predictions.csv and other outputs")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-stations", type=int, default=50,
                   help="Station cap per event, as used in training (default 50)")
    p.add_argument("--min-stations", type=int, default=0,
                   help="Skip events with fewer usable stations than this")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N events (smoke tests)")
    p.add_argument("--event-list", default=None,
                   help="Text file with one event id per line; restricts inference")
    p.add_argument("--catalog", action="store_true",
                   help="Also write catalog.txt (origin/hypocenter from the HDF5 "
                        "attributes + the predicted mechanism)")
    p.add_argument("--figures", action="store_true",
                   help="Also write scatter / Kagan-histogram / beachball figures "
                        "(requires ground-truth attributes)")
    return p.parse_args(argv)


def select_events(h5_path, args):
    with h5py.File(h5_path, "r") as f:
        ids = list(f.keys())
        nsta = {i: f[i]["features"].shape[0] for i in ids}
    if args.event_list:
        wanted = [ln.strip() for ln in Path(args.event_list).read_text().split()
                  if ln.strip()]
        missing = [w for w in wanted if w not in nsta]
        if missing:
            raise SystemExit(f"{len(missing)} requested events are not in the file, "
                             f"e.g. {missing[:5]}")
        ids = wanted
    if args.min_stations:
        ids = [i for i in ids if nsta[i] >= args.min_stations]
    if args.limit:
        ids = ids[: args.limit]
    if not ids:
        raise SystemExit("No events left after filtering.")
    return ids, nsta


def run(args):
    h5_path = resolve_dataset(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids, nsta = select_events(h5_path, args)
    print(f"Input      : {h5_path}")
    print(f"Events     : {len(ids)}")

    model = load_pretrained(args.checkpoint, device=args.device)
    print(f"Checkpoint : {args.checkpoint}  "
          f"({sum(p.numel() for p in model.parameters()):,} params)")

    ds = SeismicDataset(str(h5_path), ids, mode="test", augmentation=False,
                        config={"max_stations": args.max_stations})
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=args.num_workers)

    preds, trues, order = [], [], []
    with torch.no_grad():
        for wf, ft, mask, target, batch_ids in tqdm(loader, desc="Inference"):
            out, _ = model(wf.to(args.device), ft.to(args.device), mask.to(args.device))
            preds.append(out.cpu().numpy())
            trues.append(target.numpy())
            order.extend(batch_ids)
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    labeled = not np.isnan(trues).all()

    df = pd.DataFrame({"event_id": order,
                       "n_stations": [nsta[i] for i in order],
                       "pred_magnitude": denorm_mag(preds[:, 0])})
    for j, nm in enumerate(MT_NAMES):
        df[f"pred_{nm}"] = preds[:, j + 1]

    dec = MTDecomposer()
    if dec.is_available:
        sdr = np.array([pick_plane(dec.mt_to_sdr(preds[i, 1:]))
                        for i in range(len(preds))])
        df["pred_strike"], df["pred_dip"], df["pred_rake"] = sdr.T
    else:
        print("NOTE: Fortran kernel unavailable (run `make build`); writing moment "
              "tensors only, without strike/dip/rake or Kagan angles.")

    metrics = {"n_events": int(len(df)), "labeled": bool(labeled),
               "checkpoint": args.checkpoint, "input": str(h5_path)}
    if labeled:
        df["true_magnitude"] = denorm_mag(trues[:, 0])
        df["magnitude_error"] = (df["pred_magnitude"] - df["true_magnitude"]).abs()
        for j, nm in enumerate(MT_NAMES):
            df[f"true_{nm}"] = trues[:, j + 1]
        if dec.is_available:
            df["kagan_angle"] = [
                kagan_angle(*pick_plane(dec.mt_to_sdr(trues[i, 1:])),
                            *pick_plane(dec.mt_to_sdr(preds[i, 1:])))
                for i in range(len(preds))]
            k = df["kagan_angle"].to_numpy()
            metrics.update(kagan_mean=float(k.mean()),
                           kagan_median=float(np.median(k)),
                           frac_kagan_lt30=float((k < 30).mean()))
        metrics["magnitude_mae"] = float(df["magnitude_error"].mean())
        metrics["magnitude_bias"] = float(
            (df["pred_magnitude"] - df["true_magnitude"]).mean())

    csv_path = out_dir / "predictions.csv"
    df.to_csv(csv_path, index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nWrote {csv_path}  ({len(df)} rows, {len(df.columns)} columns)")
    print(json.dumps(metrics, indent=2))

    if args.catalog:
        write_catalog(h5_path, df, out_dir / "catalog.txt", dec)
    if args.figures:
        write_figures(df, preds, trues, labeled, out_dir, dec)
    return df, metrics


def write_catalog(h5_path, df, path, dec):
    """Catalog file: network origin/hypocenter (from HDF5 attrs) + predicted
    mechanism. Columns are documented in the file header."""
    if not dec.is_available:
        print("Skipping catalog: strike/dip/rake unavailable without the "
              "Fortran kernel.")
        return
    with h5py.File(h5_path, "r") as f:
        meta = {i: dict(f[i].attrs) for i in df["event_id"]}
    have_loc = all("evlo" in meta[i] and "evla" in meta[i] for i in df["event_id"])
    with open(path, "w") as fh:
        fh.write("# Sensoformer mechanism catalog\n")
        fh.write("# Hypocenter/magnitude columns are carried over from the input "
                 "file's attributes (the model does not locate events);\n"
                 "# strike/dip/rake are predicted (nodal plane with rake in "
                 "[-90, 90]); mag_pred is the predicted magnitude.\n")
        fh.write("# evid lat lon depth mag_cat mag_pred strike dip rake nsta\n")
        for r in df.itertuples():
            m = meta[r.event_id]
            lat = float(m.get("evla", np.nan)) if have_loc else np.nan
            lon = float(m.get("evlo", np.nan)) if have_loc else np.nan
            dep = float(m.get("depth", np.nan))
            magc = float(m.get("magnitude", np.nan))
            fh.write(f"{r.event_id:>12s} {lat:9.4f} {lon:11.4f} {dep:7.2f} "
                     f"{magc:5.2f} {r.pred_magnitude:5.2f} {r.pred_strike:6.1f} "
                     f"{r.pred_dip:5.1f} {r.pred_rake:7.1f} {int(r.n_stations):4d}\n")
    print(f"Wrote {path}")


def write_figures(df, preds, trues, labeled, out_dir, dec):
    if not labeled:
        print("Skipping figures: the input file has no ground-truth attributes.")
        return
    from sensoformer.utils.visualization import (
        plot_beachball_comparison, plot_kagan_histogram, plot_scatter_matrix)
    p = np.column_stack([df["pred_magnitude"].to_numpy(), preds[:, 1:]])
    t = np.column_stack([df["true_magnitude"].to_numpy(), trues[:, 1:]])
    names = ["Mw"] + MT_NAMES
    plot_scatter_matrix(p, t, names, event_ids=list(df["event_id"]),
                        save_path=str(out_dir / "scatter_results.pdf"))
    if "kagan_angle" in df:
        plot_kagan_histogram(df["kagan_angle"].to_numpy(),
                             save_path=str(out_dir / "kagan_histogram.pdf"))
    if dec.is_available:
        np.random.seed(0)
        plot_beachball_comparison(
            trues[:, 1:], preds[:, 1:], list(df["event_id"]),
            df["true_magnitude"].to_numpy(), dec, num_samples=min(100, len(df)),
            save_path=str(out_dir / "beachball_grid.pdf"))
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    run(parse_args())
