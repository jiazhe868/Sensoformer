#!/usr/bin/env python3
"""
Stage 1 of the synthetic (PSDR) data pipeline: generate raw synthetic SAC
waveforms for random moment-tensor sources on REAL station geometries.

Standardized, parameterized port of the original gen_single_event.py
(gen_single_event.py). The generation logic — source
parameter sampling, geometry templating, velocity-model selection, FK
synthesis via the external `syn` binary, real-noise injection, and
travel-time perturbation — is kept equivalent so new datasets match the
distribution the released models were pre-trained on.

The PSDR randomization enters here as:
  - Generative environment (phi): each event uses the Green's-function
    library gridpoint (1-D velocity model) closest to its template's real
    epicenter -- sources spread over the region therefore sample the full
    velocity-model library.
  - Signal distortion (T): P-arrival time scaled by U(0.92, 1.08) per
    station; epicenter jittered by U(-0.2, 0.2) degrees.
  - Realistic noise (n): a noise segment from the SAME real station/channel
    is superimposed, amplitude-clamped to [0.03, 0.3] x synthetic max.
  (Variable-geometry masking (M) and the remaining distortions -- coda,
   amplitude scaling -- are applied later, in preprocess_synthetic_hdf5.py
   and in the training-time Dataset augmentation.)

Requirements:
  - `syn` binary (FK package of Zhu & Rivera) on PATH;
  - Green's function library laid out as
    {gf_root}/{lat}_{lon}/vmodel_{depth_km}/{dist_km}.grn.0;
  - a real-geometry template HDF5 (per-event 'features' with dist/az columns,
    'station_names', attrs depth/evla/evlo);
  - a noise HDF5 with matching event IDs holding raw pre-event noise
    waveforms ('waveforms' (S, >=3, T), 'station_names').

Each generated event is written to {out_root}/ev_{strike}_{dip}_{rake}_{depth}_{mag}/
as per-station .z/.r/.t SAC files; all source parameters are recoverable from
the directory name (see sdr_utils.parse_synthetic_event_name).

Example:
  python scripts/preprocessing/generate_synthetic_events.py \
      --template-hdf5 .../socal_mlmag_data_20s_wmeca_test_rtz_lp1.hdf5 \
      --noise-hdf5   .../socal_mlmag_data_20s_wmeca_test_rtz_lp1_noise.hdf5 \
      --gf-root      .../gf_socal \
      --out-root     /scratch/syn_events \
      --nevents 100000 --workers 60
"""
import argparse
import math
import os
import random
import subprocess
from multiprocessing import Pool

import h5py
import numpy as np
from obspy import read, Trace
from tqdm import tqdm

WINDOW_PRE_T1 = 5.0
WINDOW_POST_T1 = 30.0
TARGET_SAMPLING_RATE = 10.0  # Hz of the noise HDF5 waveforms

# Globals initialized per worker
g_templates = None
g_gf_models = None
g_cfg = None


def find_closest_gf_model(event_lat, event_lon, available_models):
    min_dist_sq, closest = float("inf"), None
    for model_lat, model_lon in available_models:
        d = ((event_lat - float(model_lat)) ** 2
             + ((event_lon - float(model_lon)) * np.cos(np.deg2rad(event_lon))) ** 2)
        if d < min_dist_sq:
            min_dist_sq, closest = d, (model_lat, model_lon)
    return closest


def init_worker(templates, gf_models, cfg):
    global g_templates, g_gf_models, g_cfg
    g_templates, g_gf_models, g_cfg = templates, gf_models, cfg


def sample_source(cfg):
    """Random source parameters: magnitude from a clipped gamma distribution,
    uniform orientation (rake restricted to [-90, 90]: the auxiliary plane
    covers the other half-space)."""
    mag = -1
    while not (cfg["min_mag"] <= mag <= cfg["max_mag"]):
        mag = round(np.random.gamma(cfg["mag_gamma_shape"], cfg["mag_gamma_scale"]), 1)
    strike = round(random.uniform(0.0, 360.0), 1)
    dip = round(random.uniform(0.0, 90.0), 1)
    rake = round(random.uniform(-90.0, 90.0), 1)
    return mag, strike, dip, rake


def process_event(event_num):
    cfg = g_cfg
    event_id = "unknown"
    try:
        template = random.choice(g_templates)
        np.random.seed()  # re-seed per task (fork inherits parent state)

        mag, strike, dip, rake = sample_source(cfg)
        snapped_depth = max(1, min(20, int(round(template["depth"]))))
        event_id = f"ev_{strike}_{dip}_{rake}_{snapped_depth}_{mag}"
        event_dir = os.path.join(cfg["out_root"], event_id)
        os.makedirs(event_dir, exist_ok=True)

        gf_lat, gf_lon = find_closest_gf_model(
            template["evla"], template["evlo"], g_gf_models)

        with h5py.File(cfg["noise_hdf5"], "r") as h5f:
            if template["real_event_id"] not in h5f:
                return f"skip {event_num}: {template['real_event_id']} not in noise file"
            g = h5f[template["real_event_id"]]
            noise_waveforms = g["waveforms"][:]
            noise_names = [s.decode("utf-8") for s in g["station_names"][:]]
        name_to_idx = {n: i for i, n in enumerate(noise_names)}

        real_evla, real_evlo = template["evla"], template["evlo"]
        for real_dist, real_az, station_key in template["stations"]:
            real_stlo = real_evlo + (real_dist * np.sin(math.radians(real_az))) / (
                111.2 * np.cos(math.radians(real_evla)))
            real_stla = real_evla + (real_dist * np.cos(math.radians(real_az))) / 111.2
            snapped_distance = max(2, round(real_dist / 2) * 2)
            if snapped_distance > 200:
                continue
            perturbation = random.uniform(0.92, 1.08)
            azimuth = int(round(real_az))
            duration = max(0.3, 10 ** (mag * 0.5 - 2.5))

            gf_path = os.path.join(cfg["gf_root"], f"{gf_lat}_{gf_lon}",
                                   f"vmodel_{snapped_depth}",
                                   f"{snapped_distance}.grn.0")
            if not os.path.exists(gf_path):
                continue
            base = os.path.join(event_dir, f"{event_id}.{station_key}")
            subprocess.run(
                ["syn", f"-M{mag}/{strike}/{dip}/{rake}", f"-D{duration}",
                 f"-A{azimuth}", f"-O{base}.z", f"-G{gf_path}"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            paths = {c: f"{base}.{c.lower()}" for c in ["Z", "R", "T"]}
            if not all(os.path.exists(p) for p in paths.values()):
                continue
            if station_key not in name_to_idx:
                continue
            noise_3c = noise_waveforms[name_to_idx[station_key], 0:3, :]

            for i, comp in enumerate(["Z", "R", "T"]):
                tr = read(paths[comp])[0]
                tr.stats.sac.nzyear, tr.stats.sac.nzjday = 1970, 1
                tr.stats.sac.nzhour = tr.stats.sac.nzmin = 0
                tr.stats.sac.nzsec = tr.stats.sac.nzmsec = 0
                tr.stats.sac.evlo = real_evlo + random.uniform(-0.2, 0.2)
                tr.stats.sac.evla = real_evla + random.uniform(-0.2, 0.2)
                tr.stats.sac.stlo, tr.stats.sac.stla = real_stlo, real_stla
                t1 = tr.stats.sac.get("t1", -12345)
                b = tr.stats.sac.get("b", 0)
                delta = tr.stats.delta
                if t1 == -12345 or not delta:
                    continue
                t1 *= perturbation
                tr.stats.sac.t1 = t1
                p_sample = int(round((t1 - b) / delta))
                w_start = max(0, p_sample - int(round(WINDOW_PRE_T1 / delta)))
                w_end = min(len(tr.data),
                            p_sample + int(round(WINDOW_POST_T1 / delta)))
                npts = w_end - w_start
                if npts <= 0:
                    continue
                npts_noise = int((WINDOW_PRE_T1 + WINDOW_POST_T1) * TARGET_SAMPLING_RATE)
                if noise_3c.shape[1] < npts_noise:
                    continue
                tr_noise = Trace(data=noise_3c[i, -npts_noise:],
                                 header={"sampling_rate": TARGET_SAMPLING_RATE})
                tr_noise.resample(sampling_rate=tr.stats.sampling_rate)
                noise = np.resize(tr_noise.data, npts)
                seg = tr.data[w_start:w_end]
                max_syn = np.max(np.abs(seg))
                max_noise = np.max(np.abs(noise))
                if max_noise > max_syn * 0.3:
                    noise = noise * max_syn / max_noise * 0.3
                elif max_noise < max_syn * 0.03:
                    noise = noise * max_syn / max_noise * 0.03
                tr.data[w_start:w_end] += noise
                tr.write(paths[comp], format="SAC")
        return f"ok {event_num}"
    except Exception as e:
        return f"error {event_num} ({event_id}): {e}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--template-hdf5", required=True,
                        help="Real-geometry template HDF5")
    parser.add_argument("--noise-hdf5", required=True,
                        help="Real noise HDF5 (matching event IDs)")
    parser.add_argument("--gf-root", required=True,
                        help="Green's function library root")
    parser.add_argument("--out-root", required=True,
                        help="Output directory for per-event SAC folders")
    parser.add_argument("--nevents", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--min-mag", type=float, default=2.8)
    parser.add_argument("--max-mag", type=float, default=7.0)
    parser.add_argument("--mag-gamma-shape", type=float, default=2.5)
    parser.add_argument("--mag-gamma-scale", type=float, default=1.0)
    args = parser.parse_args()

    cfg = {"out_root": args.out_root, "gf_root": args.gf_root,
           "noise_hdf5": args.noise_hdf5,
           "min_mag": args.min_mag, "max_mag": args.max_mag,
           "mag_gamma_shape": args.mag_gamma_shape,
           "mag_gamma_scale": args.mag_gamma_scale}
    os.makedirs(args.out_root, exist_ok=True)

    print(f"Loading geometry templates from {args.template_hdf5}")
    templates = []
    with h5py.File(args.template_hdf5, "r") as h5f:
        for event_id in h5f.keys():
            g = h5f[event_id]
            if not ("features" in g and "station_names" in g
                    and all(k in g.attrs for k in ["depth", "evla", "evlo"])):
                continue
            features = g["features"][:]
            names = [s.decode("utf-8") for s in g["station_names"][:]]
            if features.shape[0] and features.shape[0] == len(names):
                templates.append({
                    "real_event_id": event_id,
                    "stations": list(zip(features[:, 0], features[:, 1], names)),
                    "depth": g.attrs["depth"],
                    "evla": g.attrs["evla"], "evlo": g.attrs["evlo"]})
    if not templates:
        raise SystemExit("No usable geometry templates found.")
    print(f"Loaded {len(templates)} templates.")

    gf_models = []
    for d in os.listdir(args.gf_root):
        if os.path.isdir(os.path.join(args.gf_root, d)) and "_" in d:
            try:
                lat, lon = d.split("_")
                gf_models.append((lat, lon))
            except ValueError:
                continue
    if not gf_models:
        raise SystemExit(f"No GF models found in {args.gf_root}")
    print(f"{len(gf_models)} GF models. Generating {args.nevents} events "
          f"with {args.workers} workers...")

    with Pool(processes=args.workers, initializer=init_worker,
              initargs=(templates, gf_models, cfg)) as pool:
        results = list(tqdm(pool.imap_unordered(
            process_event, range(1, args.nevents + 1)), total=args.nevents))
    n_ok = sum(1 for r in results if r.startswith("ok"))
    print(f"Done: {n_ok}/{args.nevents} events generated under {args.out_root}")


if __name__ == "__main__":
    main()
