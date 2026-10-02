#!/usr/bin/env python3
"""
Regenerate the explanatory figures used in README.md and the docs.

Everything is produced from the released checkpoint and dataset, so the
figures in the repository can be reproduced (and kept honest) with:

    python scripts/make_readme_figures.py --out-dir docs/figures

Requires the Fortran kernel (`make build`) for beachballs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from obspy.imaging.beachball import beach

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sensoformer import load_pretrained
from sensoformer.data.dataset import SeismicDataset, collate_fn
from sensoformer.ext import MTDecomposer
from sensoformer.hub import resolve_dataset
from sensoformer.utils.physics import kagan_angle

DEC = MTDecomposer()
TITLE_FS, LABEL_FS, TICK_FS = 15, 13, 11


def plane(mt):
    s1, s2 = DEC.mt_to_sdr(np.asarray(mt, dtype=float))
    return s1 if -90 <= s1[2] <= 90 else s2


def sdr_to_mt5(strike, dip, rake):
    """Strike/dip/rake -> normalized deviatoric [Mxx, Myy, Mxy, Mxz, Myz]."""
    d2r = np.pi / 180
    s, d, r = strike * d2r, dip * d2r, rake * d2r
    mxx = -(np.sin(d) * np.cos(r) * np.sin(2 * s)
            + np.sin(2 * d) * np.sin(r) * np.sin(s) ** 2)
    myy = (np.sin(d) * np.cos(r) * np.sin(2 * s)
           - np.sin(2 * d) * np.sin(r) * np.cos(s) ** 2)
    mxy = (np.sin(d) * np.cos(r) * np.cos(2 * s)
           + 0.5 * np.sin(2 * d) * np.sin(r) * np.sin(2 * s))
    mxz = -(np.cos(d) * np.cos(r) * np.cos(s) + np.cos(2 * d) * np.sin(r) * np.sin(s))
    myz = (np.cos(2 * d) * np.sin(r) * np.cos(s) - np.cos(d) * np.cos(r) * np.sin(s))
    return np.array([mxx, myy, mxy, mxz, myz])


# ----------------------------------------------------------------------
# 1. Beachball primer: what a focal mechanism is, and what a Kagan angle means
# ----------------------------------------------------------------------
def fig_beachball_primer(out):
    kinds = [("Strike-slip\n(blocks slide past each other)", (0, 90, 0)),
             ("Normal\n(crust pulled apart)", (0, 45, -90)),
             ("Reverse / thrust\n(crust pushed together)", (0, 45, 90))]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.8),
                             gridspec_kw={"width_ratios": [1, 1, 1, 0.15, 1.9]})
    for ax, (name, sdr) in zip(axes[:3], kinds):
        ax.add_collection(beach(list(sdr), xy=(0, 0), width=1.6,
                                facecolor="0.15", linewidth=1.0))
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.5, 1.15)
        ax.set_aspect(1); ax.axis("off")
        ax.set_title(name, fontsize=LABEL_FS)
    axes[3].axis("off")

    # Kagan angle: two mechanisms and the single number separating them
    ax = axes[4]
    a, b = (30.0, 70.0, 10.0), (55.0, 60.0, 30.0)
    ang = kagan_angle(*a, *b)
    ax.add_collection(beach(list(a), xy=(-1.0, -0.15), width=1.4,
                            facecolor="0.15", linewidth=1.0))
    ax.add_collection(beach(list(b), xy=(1.0, -0.15), width=1.4,
                            facecolor="tab:blue", linewidth=1.0))
    ax.annotate("", xy=(0.25, -0.15), xytext=(-0.25, -0.15),
                arrowprops=dict(arrowstyle="<->", lw=1.8, color="crimson"))
    ax.text(0, 0.95, f"Kagan angle = {ang:.0f}°", ha="center", fontsize=LABEL_FS,
            color="crimson", fontweight="bold")
    ax.text(0, 0.58, "0° = identical, ~120° = maximally different",
            ha="center", fontsize=TICK_FS, color="0.3")
    ax.text(-1.0, -1.15, "catalog", ha="center", fontsize=TICK_FS)
    ax.text(1.0, -1.15, "prediction", ha="center", fontsize=TICK_FS, color="tab:blue")
    ax.set_xlim(-2.0, 2.0); ax.set_ylim(-1.5, 1.15)
    ax.set_aspect(1); ax.axis("off")
    ax.set_title('One number for "how different are two mechanisms?"',
                 fontsize=LABEL_FS)

    fig.suptitle('Reading a "beachball": each sphere shows the orientation of the '
                 "fault and slip direction.\nDark quadrants = ground pushed outward "
                 "(compression), white = pulled inward (dilatation).",
                 fontsize=TITLE_FS, y=1.06)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------------
# 2. What the model sees for one event, and what it outputs
# ----------------------------------------------------------------------
def fig_what_the_model_sees(out, hdf5, model, device):
    import h5py
    with h5py.File(hdf5, "r") as f:
        # an event with a good number of stations, for a legible map
        eid = next(k for k in list(f.keys())[:60] if f[k]["features"].shape[0] >= 25)
        g = f[eid]
        feats = g["features"][:]
        waves = g["waveforms"][:]
        evlo, evla = float(g.attrs["evlo"]), float(g.attrs["evla"])
        mt_true = np.array([g.attrs[k] for k in ("Mxx", "Myy", "Mxy", "Mxz", "Myz")])
        mag_true = float(g.attrs["magnitude"])

    ds = SeismicDataset(str(hdf5), [eid], mode="test", augmentation=False,
                        config={"max_stations": 50})
    wf, ft, mask, _, _ = collate_fn([ds[0]])
    with torch.no_grad():
        pred, attn = model(wf.to(device), ft.to(device), mask.to(device))
    pred = pred[0].cpu().numpy(); attn = attn[0].cpu().numpy()
    n = int(mask[0].sum())
    mag_pred = (pred[0] + 1) / 2 * 6 + 2
    mt_pred = pred[1:]
    ang = kagan_angle(*plane(mt_true), *plane(mt_pred))

    stlo, stla = feats[:n, 2], feats[:n, 3]
    fig = plt.figure(figsize=(16, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.25, 0.9], wspace=0.28)

    # (a) network geometry, stations shaded by the model's attention weight
    ax = fig.add_subplot(gs[0])
    sc = ax.scatter(stlo, stla, c=attn[:n], s=120, marker="v",
                    cmap="viridis", edgecolor="k", linewidth=0.5, zorder=3)
    ax.plot(evlo, evla, marker="*", ms=26, color="crimson",
            markeredgecolor="k", zorder=4)
    for x, y in zip(stlo, stla):
        ax.plot([evlo, x], [evla, y], color="0.75", lw=0.5, zorder=1)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046)
    cb.set_label("attention weight", fontsize=TICK_FS)
    ax.set_xlabel("Longitude", fontsize=LABEL_FS); ax.set_ylabel("Latitude", fontsize=LABEL_FS)
    ax.set_aspect(1 / np.cos(np.deg2rad(evla)))
    ax.tick_params(labelsize=TICK_FS); ax.grid(alpha=0.25, ls=":")
    ax.set_title(f"(a) Input: {n} stations, irregular geometry\n"
                 f"star = earthquake, triangles = seismometers", fontsize=LABEL_FS)
    ax.legend(handles=[Line2D([], [], marker="*", ls="", color="crimson",
                              markeredgecolor="k", ms=16, label="event"),
                       Line2D([], [], marker="v", ls="", color="0.5",
                              markeredgecolor="k", ms=10, label="station")],
              fontsize=TICK_FS, loc="best")

    # (b) the waveforms themselves, for the three most-attended stations
    ax = fig.add_subplot(gs[1])
    order = np.argsort(attn[:n])[::-1][:3]
    t_p = np.arange(101) * 0.1 - 5
    for i, idx in enumerate(order):
        p = waves[idx, 0]; s = waves[idx, 3]
        off = -i * 2.4
        ax.plot(t_p, p / (np.abs(p).max() + 1e-9) + off, lw=1.0, color="tab:blue")
        ax.plot(t_p + 13, s / (np.abs(s).max() + 1e-9) + off, lw=1.0, color="tab:red")
        ax.text(-6.4, off, f"stn {idx}", fontsize=TICK_FS, va="center", ha="right")
    ax.axvline(0, color="tab:blue", ls=":", lw=1)
    ax.axvline(13, color="tab:red", ls=":", lw=1)
    ax.text(0, 1.6, "P arrival", color="tab:blue", fontsize=TICK_FS, ha="center")
    ax.text(13, 1.6, "S arrival", color="tab:red", fontsize=TICK_FS, ha="center")
    ax.set_yticks([]); ax.set_xlabel("seconds around each arrival", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.set_xlim(-7.5, 19); ax.set_ylim(-6.2, 2.2)
    ax.set_title("(b) Each station contributes two short\nground-motion windows "
                 "(P and S)", fontsize=LABEL_FS)

    # (c) output
    ax = fig.add_subplot(gs[2])
    ax.add_collection(beach(list(plane(mt_true)), xy=(-0.9, 0), width=1.5,
                            facecolor="0.15", linewidth=1.0))
    ax.add_collection(beach(list(plane(mt_pred)), xy=(0.9, 0), width=1.5,
                            facecolor="tab:blue", linewidth=1.0))
    ax.text(-0.9, -1.15, f"catalog\nM{mag_true:.2f}", ha="center", fontsize=TICK_FS)
    ax.text(0.9, -1.15, f"Sensoformer\nM{mag_pred:.2f}", ha="center",
            fontsize=TICK_FS, color="tab:blue")
    ax.text(0, 1.15, f"Kagan {ang:.0f}°", ha="center", fontsize=LABEL_FS,
            color="crimson", fontweight="bold")
    ax.set_xlim(-1.95, 1.95); ax.set_ylim(-1.7, 1.5); ax.set_aspect(1); ax.axis("off")
    ax.set_title("(c) Output: magnitude +\nfocal mechanism", fontsize=LABEL_FS)

    fig.suptitle(f"What Sensoformer consumes and produces for one earthquake "
                 f"(event {eid})", fontsize=TITLE_FS, y=1.02)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------------
# 3. Example predictions: catalog vs model for a sample of events
# ----------------------------------------------------------------------
def fig_examples(out, hdf5, model, device, n_show=16):
    import h5py
    with h5py.File(hdf5, "r") as f:
        ids = list(f.keys())[:200]
        mags = {i: float(f[i].attrs["magnitude"]) for i in ids}
        truth = {i: np.array([f[i].attrs[k] for k in
                              ("Mxx", "Myy", "Mxy", "Mxz", "Myz")]) for i in ids}
    ds = SeismicDataset(str(hdf5), ids, mode="test", augmentation=False,
                        config={"max_stations": 50})
    loader = torch.utils.data.DataLoader(ds, batch_size=64, collate_fn=collate_fn)
    preds, order = [], []
    with torch.no_grad():
        for wf, ft, mask, _, bids in loader:
            p, _ = model(wf.to(device), ft.to(device), mask.to(device))
            preds.append(p.cpu().numpy()); order.extend(bids)
    preds = np.concatenate(preds)

    rng = np.random.RandomState(0)
    pick = rng.choice(len(order), n_show, replace=False)
    cols = 8; rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.5 * rows))
    for ax, k in zip(axes.flat, pick):
        eid = order[k]
        pt, pp = plane(truth[eid]), plane(preds[k, 1:])
        ang = kagan_angle(*pt, *pp)
        ax.add_collection(beach(list(pt), xy=(-0.55, 0), width=1.0,
                                facecolor="0.15", linewidth=0.7))
        ax.add_collection(beach(list(pp), xy=(0.55, 0), width=1.0,
                                facecolor="tab:blue", linewidth=0.7))
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.1, 1.1); ax.set_aspect(1); ax.axis("off")
        ax.set_title(f"M{mags[eid]:.1f}   {ang:.0f}°", fontsize=10)
    for ax in axes.flat[n_show:]:
        ax.axis("off")
    fig.suptitle("Catalog (black) vs Sensoformer (blue) focal mechanisms, with the "
                 "Kagan angle between them", fontsize=TITLE_FS, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="docs/figures")
    ap.add_argument("--input", default="socal-real")
    ap.add_argument("--checkpoint", default="sensoformer-v3-finetuned")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if not DEC.is_available:
        sys.exit("The Fortran kernel is required for beachballs: run `make build`.")

    fig_beachball_primer(out / "beachball_primer.png")
    hdf5 = resolve_dataset(args.input)
    model = load_pretrained(args.checkpoint, device=args.device)
    fig_what_the_model_sees(out / "what_the_model_sees.png", hdf5, model, args.device)
    fig_examples(out / "example_predictions.png", hdf5, model, args.device)


if __name__ == "__main__":
    main()
