#!/usr/bin/env python3
"""
Stage 2 of the synthetic (PSDR) data pipeline: per-event SAC folders ->
model-ready HDF5.

Standardized, parameterized port of the original
1_preprocess_hdf5_parallel_30s_test_30w.py
(1_preprocess_hdf5_parallel_30s_test_30w.py), which
produced the pre-training file syn_mt_data_realgeom_realvn_10w_ps_wcoda_wlola.hdf5.
Per-station processing is kept equivalent so new datasets match the
distribution the released models were pre-trained on:

  bandpass 0.1-2.0 Hz -> synthetic coda injection at BOTH the P and S
  arrivals (band-limited noise under an exponential decay envelope; part of
  the PSDR signal-distortion module) -> P/S windows (5 s pre / 5 s post,
  dt=0.1 s, 101 samples) -> random P/S amplitude scaling (alpha_P ~ U(0.5,2),
  alpha_S ~ U(1,1.5)*alpha_P) -> amplitude spectra -> 20-D scalar features
  [dist, az, stlo, stla, depth, 3 P amps, 3 S amps, 3 P-spec max, 3 S-spec
  max, 3 log P/S ratios] -> event-level max-amplitude normalization.

Event source parameters are parsed from the directory names produced by
generate_synthetic_events.py (ev_{strike}_{dip}_{rake}_{depth}_{mag}); an
explicit metafile (columns: name mag strike dip rake depth) can override.

Example:
  python scripts/preprocessing/preprocess_synthetic_hdf5.py \
      --data-root /scratch/syn_events \
      --output /scratch/hdf5/syn_mt_data_new.hdf5 --workers 32
"""
import argparse
import concurrent.futures
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from obspy import read, UTCDateTime
from scipy.signal import butter, filtfilt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from sdr_utils import sdr2mxyz_norm, parse_synthetic_event_name

# ----------------------------------------------------------------------
# Processing constants -- MUST match the original pre-training preprocessing.
# ----------------------------------------------------------------------
BANDPASS = dict(freqmin=0.1, freqmax=2.0, corners=4, zerophase=True)
WINDOW_PRE_P = 5
WINDOW_POST_P = 5
WINDOW_PRE_S = 5
WINDOW_POST_S = 5
TARGET_DT = 0.1
TARGET_SAMPLE_P = int((WINDOW_PRE_P + WINDOW_POST_P) / TARGET_DT + 1)
TARGET_SAMPLE_S = int((WINDOW_PRE_S + WINDOW_POST_S) / TARGET_DT + 1)
SAC_UNDEFINED = -12345
P_AMP_RANGE = (0.5, 2.0)
S_AMP_EXTRA_RANGE = (1.0, 1.5)
CODA_AMP_RANGE = (0.5, 1.0)
CODA_DECAY_RANGE = (0.1, 0.3)


def generate_truncated_normal(num_samples, lower=-1, upper=1):
    samples = np.array([])
    while samples.size < num_samples:
        new = np.random.randn(num_samples)
        samples = np.concatenate((samples, new[(new >= lower) & (new <= upper)]))
    return samples[:num_samples]


def add_synthetic_coda(st, arrival_time, sampling_rate):
    """Inject an exponentially decaying, band-limited noise coda after the
    given arrival on all three components (PSDR signal distortion)."""
    try:
        if len(st) < 3:
            return st
        window = st.copy().trim(arrival_time - 2, arrival_time + 2)
        amp = max(max(np.max(np.abs(tr.data)) for tr in window), 1e-9)
        idx = int((arrival_time - st[0].stats.starttime) * sampling_rate)
        if idx < 0 or idx >= len(st[0].data):
            return st
        n = len(st[0].data) - idx
        nyq = 0.5 * sampling_rate
        b_f, a_f = butter(4, [0.1 / nyq, 2.0 / nyq], btype="bandpass")
        t = np.arange(n) / sampling_rate
        decay = np.random.uniform(*CODA_DECAY_RANGE)
        out = st.copy()
        for tr in out:
            noise = filtfilt(b_f, a_f, generate_truncated_normal(n))
            env = np.random.uniform(*CODA_AMP_RANGE) * amp * np.exp(-decay * t)
            tr.data[idx:] += env * noise
        return out
    except Exception:
        return st


def process_event(args):
    ev, data_root, min_sta = args
    event_name = ev["event_id"]
    event_dir = os.path.join(data_root, event_name)
    if not os.path.isdir(event_dir):
        return (event_name, "error", f"Directory not found: {event_dir}")

    sac_files = glob.glob(os.path.join(event_dir, "*.[rtz]"))
    stations = defaultdict(list)
    for f in sac_files:
        try:
            parts = os.path.basename(f).split(".")
            stations[f"{parts[5]}.{parts[6]}"].append(f)
        except IndexError:
            continue

    waveforms_out, features_out, names_out = [], [], []
    for station_key, files in stations.items():
        if len(files) < 3:
            continue
        channel_map = {}
        for f in files:
            c = os.path.basename(f).split(".")[-1].lower()
            if c in ["r", "t", "z"]:
                channel_map[c] = f
        if len(channel_map) < 3:
            continue
        try:
            st = read(channel_map["z"]) + read(channel_map["r"]) + read(channel_map["t"])
            if st[0].stats.sac.nzyear == SAC_UNDEFINED:
                for tr in st:
                    tr.stats.starttime = UTCDateTime("1970-01-01T00:00:00")
            if len(st) < 3:
                continue
            st.filter("bandpass", **BANDPASS)
            header = st[0].stats.sac
            t1, t2, b, delta, dist, az, stlo, stla = (
                header.get(k, SAC_UNDEFINED) for k in
                ["t1", "t2", "b", "delta", "dist", "az", "stlo", "stla"])
            if any(v == SAC_UNDEFINED for v in [t1, b, delta, dist, az]):
                continue
            if t2 == SAC_UNDEFINED:
                t2 = t1 * 1.75

            t0 = st[0].stats.starttime
            p_arrival = t0 + (t1 - b)
            s_arrival = t0 + (t2 - b)
            sr = st[0].stats.sampling_rate
            st = add_synthetic_coda(st, p_arrival, sr)
            st = add_synthetic_coda(st, s_arrival, sr)

            st_p = st.copy().trim(p_arrival - WINDOW_PRE_P, p_arrival + WINDOW_POST_P,
                                  pad=True, fill_value=0)
            st_p.detrend("linear").detrend("demean").taper(max_percentage=0.05, type="cosine")
            st_p.interpolate(sampling_rate=1 / TARGET_DT, npts=TARGET_SAMPLE_P)
            st_s = st.copy().trim(s_arrival - WINDOW_PRE_S, s_arrival + WINDOW_POST_S,
                                  pad=True, fill_value=0)
            st_s.detrend("linear").detrend("demean").taper(max_percentage=0.05, type="cosine")
            st_s.interpolate(sampling_rate=1 / TARGET_DT, npts=TARGET_SAMPLE_S)

            p_factor = np.random.uniform(*P_AMP_RANGE)
            s_factor = np.random.uniform(*S_AMP_EXTRA_RANGE) * p_factor
            for tr in st_p:
                tr.data *= p_factor
            for tr in st_s:
                tr.data *= s_factor

            if len(st_p) < 3 or len(st_s) < 3:
                raise ValueError("P/S windows are empty")
            waveform_p = np.vstack([tr.data for tr in st_p]).astype(np.float32)
            waveform_s = np.vstack([tr.data for tr in st_s]).astype(np.float32)
            spectra_p = np.abs(np.fft.rfft(waveform_p, axis=1))
            spectra_s = np.abs(np.fft.rfft(waveform_s, axis=1))
            spectra_p = np.pad(spectra_p, ((0, 0), (0, TARGET_SAMPLE_P - spectra_p.shape[1])), "constant")
            spectra_s = np.pad(spectra_s, ((0, 0), (0, TARGET_SAMPLE_S - spectra_s.shape[1])), "constant")
            arrays = [waveform_p, waveform_s, spectra_p, spectra_s]
            if any(np.isnan(a).any() or np.isinf(a).any() for a in arrays):
                continue
            combined = np.vstack(arrays).astype(np.float32)

            st_p_amp = st_p.copy().trim(p_arrival - 2, p_arrival + 3, pad=True, fill_value=0)
            st_s_amp = st_s.copy().trim(s_arrival - 2, s_arrival + 3, pad=True, fill_value=0)
            p_amps = [np.max(np.abs(tr.data)) if len(tr.data) > 0 else 0 for tr in st_p_amp]
            s_amps = [np.max(np.abs(tr.data)) if len(tr.data) > 0 else 0 for tr in st_s_amp]
            eps = 1e-6
            ratios = [float(np.log(max(0.01, min(10, p / (s + eps)))))
                      for p, s in zip(p_amps, s_amps)]
            scalar_features = np.array(
                [dist, az, stlo, stla, ev["depth"]] + list(p_amps) + list(s_amps)
                + list(np.max(spectra_p, axis=1)) + list(np.max(spectra_s, axis=1))
                + ratios, dtype=np.float32)

            waveforms_out.append(combined)
            features_out.append(scalar_features)
            names_out.append(station_key)
        except Exception:
            continue

    if not waveforms_out:
        return (event_name, "skipped", "No valid station waveforms found")
    if len(waveforms_out) < min_sta:
        return (event_name, "skipped",
                f"Not enough valid stations ({len(waveforms_out)} < {min_sta})")

    waveforms_array = np.array(waveforms_out)
    waveforms_array[:, 0:6, :] /= np.max(np.abs(waveforms_array[:, 0:6, :])) + 1e-8
    waveforms_array[:, 6:12, :] /= np.max(np.abs(waveforms_array[:, 6:12, :])) + 1e-8
    return (event_name, "success",
            (ev, waveforms_array, np.array(features_out), names_out))


def collect_events(data_root, metafile=None):
    if metafile:
        events = []
        with open(metafile) as f:
            for line in f:
                try:
                    name, mag, strike, dip, rake, depth = line.split()
                    events.append({"event_id": name, "magnitude": float(mag),
                                   "strike": float(strike), "dip": float(dip),
                                   "rake": float(rake), "depth": float(depth)})
                except ValueError:
                    print(f"Skipping malformed metafile line: {line.strip()}")
        return events
    events = []
    for name in os.listdir(data_root):
        if os.path.isdir(os.path.join(data_root, name)):
            ev = parse_synthetic_event_name(name)
            if ev:
                events.append(ev)
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data-root", required=True,
                        help="Root of per-event synthetic SAC directories")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument("--metafile", default=None,
                        help="Optional metafile (name mag strike dip rake depth); "
                             "default: parse parameters from directory names")
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--min-sta", type=int, default=5)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the stochastic augmentations (coda, "
                             "amplitude scaling); default: unseeded")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    events = collect_events(args.data_root, args.metafile)
    if args.max_events:
        events = events[: args.max_events]
    print(f"Events to process: {len(events)}")

    tasks = [(e, args.data_root, args.min_sta) for e in events]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in tqdm(ex.map(process_event, tasks), total=len(tasks),
                        desc="Processing events"):
            if res[1] == "success":
                results.append(res)
            else:
                print(f"  {res[0]}: {res[1]} - {res[2]}")

    print(f"\nWriting {len(results)} events to {args.output}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    string_dt = h5py.special_dtype(vlen=str)
    with h5py.File(args.output, "w") as h5f:
        for event_name, _, (ev, waveforms, features, names) in tqdm(
                results, desc="Writing HDF5"):
            mxx, myy, mxy, mxz, myz = sdr2mxyz_norm(
                ev["strike"], ev["dip"], ev["rake"])
            g = h5f.create_group(event_name)
            g.create_dataset("waveforms", data=waveforms, compression="gzip")
            g.create_dataset("features", data=features, compression="gzip")
            g.create_dataset("station_names", data=names, dtype=string_dt,
                             compression="gzip")
            g.attrs["magnitude"] = ev["magnitude"]
            g.attrs["Mxx"], g.attrs["Myy"] = mxx, myy
            g.attrs["Mxy"], g.attrs["Mxz"], g.attrs["Myz"] = mxy, mxz, myz
            g.attrs["depth"] = ev["depth"]
            g.attrs["strike"], g.attrs["dip"], g.attrs["rake"] = (
                ev["strike"], ev["dip"], ev["rake"])
    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
