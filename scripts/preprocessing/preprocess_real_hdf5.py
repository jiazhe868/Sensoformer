#!/usr/bin/env python3
"""
Standardized real-data preprocessing: SAC event directories + YSH catalog ->
model-ready HDF5.

This is a parameterized, reusable port of the original one-off script
(1_preprocess_hdf5_parallel_30s_test_ps.py) that
produced the training file socal_mxyz_data_rtz_lp2_ampr_ps_wlola.hdf5. The
per-station processing (filtering, windowing, rotation, spectra, scalar
features, event-level normalization) is kept byte-for-byte equivalent so that
new HDF5 files are drawn from the SAME input distribution the model was
trained on. Only the event selection and I/O plumbing are new:

  - reads the YSH catalog format directly (no intermediate metafile);
  - filters by magnitude, mechanism quality grade, and data availability;
  - can exclude events already present in another HDF5 (e.g. the training
    catalog) to guarantee a disjoint evaluation set;
  - selects the N most recent qualifying events;
  - writes gzip-compressed HDF5 with full provenance attributes
    (magnitude, MT components, depth, lon/lat, strike/dip/rake, grade, date).

Example (200 most recent M>=2.5 grade-A/B events, disjoint from training):
  python scripts/preprocess_real_hdf5.py \
      --catalog /path/to/archive/ysh_all.log \
      --data-root /path/to/archive \
      --output data/socal_m25_recent200.hdf5 \
      --min-mag 2.5 --grades AB --max-events 200 \
      --exclude-hdf5 data/socal_mxyz_data_rtz_lp2_ampr_ps_wlola.hdf5
"""
import argparse
import concurrent.futures
import glob
import os
from collections import defaultdict

import h5py
import numpy as np
from obspy import read, Stream
from tqdm import tqdm

# ----------------------------------------------------------------------
# Processing constants -- MUST match the original training preprocessing.
# ----------------------------------------------------------------------
BANDPASS = dict(freqmin=0.2, freqmax=2.0, corners=4, zerophase=True)
WINDOW_PRE_P = 5
WINDOW_POST_P = 5
WINDOW_PRE_S = 5
WINDOW_POST_S = 5
TARGET_DT = 0.1
TARGET_SAMPLE_P = int((WINDOW_PRE_P + WINDOW_POST_P) / TARGET_DT + 1)
TARGET_SAMPLE_S = int((WINDOW_PRE_S + WINDOW_POST_S) / TARGET_DT + 1)
SAC_UNDEFINED = -12345
MAX_RAW_AMP = 5


import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent))
from sdr_utils import sdr2mxyz_norm


# Column layout of the Yang-Hauksson-Shearer (YHS) focal-mechanism catalog.
# See docs/DATA_PIPELINE.md ("Obtaining the YHS focal-mechanism catalog") for
# where to get the file and the full column description.
YSH_COL = {"year": 0, "month": 1, "day": 2, "hour": 3, "minute": 4, "second": 5,
           "event_id": 6, "lat": 7, "lon": 8, "depth": 9, "magnitude": 10,
           "strike": 11, "dip": 12, "rake": 13, "fpu1": 14, "fpu2": 15,
           "quality": 20}
YSH_MIN_COLUMNS = 21
VALID_GRADES = set("ABCDZ")


class CatalogFormatError(ValueError):
    """The catalog file does not match the expected YHS column layout."""


def parse_ysh_catalog(path, strict=True):
    """Parse a YHS-format focal-mechanism catalog.

    Expects whitespace-separated columns in the order documented in
    YSH_COL (21 columns; see docs/DATA_PIPELINE.md). Returns a list of dicts
    in file order (chronological).

    Rather than guessing, this fails loudly when the file does not look like
    a YHS catalog: a silently mis-parsed catalog would attach the wrong
    mechanism or quality grade to every event.
    """
    events, skipped = [], 0
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            t = line.split()
            if len(t) < YSH_MIN_COLUMNS:
                skipped += 1
                if strict and skipped <= 3:
                    raise CatalogFormatError(
                        f"{path}:{lineno}: expected at least "
                        f"{YSH_MIN_COLUMNS} whitespace-separated columns, got "
                        f"{len(t)}.\n  line: {line.strip()[:120]}\n"
                        f"  This does not look like a YHS catalog. See "
                        f"docs/DATA_PIPELINE.md, or run\n"
                        f"  python scripts/data_acquisition/check_catalog_format.py "
                        f"{path}")
                continue
            try:
                grade = t[YSH_COL["quality"]]
                if strict and grade not in VALID_GRADES:
                    raise CatalogFormatError(
                        f"{path}:{lineno}: column {YSH_COL['quality']} should be "
                        f"the quality grade (one of {sorted(VALID_GRADES)}), "
                        f"found {grade!r}. Column order probably differs from "
                        f"the expected YHS layout; see docs/DATA_PIPELINE.md.")
                events.append({
                    "date": (int(t[YSH_COL["year"]]) * 10000
                             + int(t[YSH_COL["month"]]) * 100
                             + int(t[YSH_COL["day"]])),
                    "event_id": t[YSH_COL["event_id"]],
                    "lat": float(t[YSH_COL["lat"]]),
                    "lon": float(t[YSH_COL["lon"]]),
                    "depth": float(t[YSH_COL["depth"]]),
                    "magnitude": float(t[YSH_COL["magnitude"]]),
                    "strike": float(t[YSH_COL["strike"]]),
                    "dip": float(t[YSH_COL["dip"]]),
                    "rake": float(t[YSH_COL["rake"]]),
                    "grade": grade,
                })
            except (ValueError, IndexError) as exc:
                if isinstance(exc, CatalogFormatError):
                    raise
                skipped += 1
                continue
    if not events:
        raise CatalogFormatError(
            f"No usable rows parsed from {path}. Check the file with\n"
            f"  python scripts/data_acquisition/check_catalog_format.py {path}")
    if skipped:
        print(f"  note: skipped {skipped} unparseable catalog lines")
    return events


# ----------------------------------------------------------------------
# Per-event processing (faithful port of the original process_event)
# ----------------------------------------------------------------------
def process_event(args):
    ev, data_root, min_sta = args
    event_name = ev["event_id"]
    event_dir = os.path.join(data_root, event_name)
    if not os.path.isdir(event_dir):
        return (event_name, "error", f"Directory not found: {event_dir}")

    sac_files = glob.glob(os.path.join(event_dir, "*.sac"))
    stations = defaultdict(list)
    for f in sac_files:
        try:
            parts = os.path.basename(f).split(".")
            stations[f"{parts[1]}.{parts[2]}"].append(f)
        except IndexError:
            continue

    event_station_waveforms = []
    event_station_features = []
    event_station_names = []
    event_depth = event_lo = event_la = None

    for station_key, files in stations.items():
        if len(files) < 3:
            continue
        channel_map = {}
        for f in files:
            ch = os.path.basename(f).split(".")[3][-1].upper()
            if ch in ["E", "N", "Z"]:
                channel_map[ch] = f
        if len(channel_map) < 3:
            continue
        try:
            st = read(channel_map["Z"]) + read(channel_map["N"]) + read(channel_map["E"])
            st.merge(method=1, fill_value=0)
            if len(st) < 3:
                continue
            st.filter("bandpass", **BANDPASS)
            try:
                trace_z = st.select(component="Z")[0]
                trace_n = st.select(component="N")[0]
                trace_e = st.select(component="E")[0]
            except IndexError:
                continue
            hz, hn, he = trace_z.stats.sac, trace_n.stats.sac, trace_e.stats.sac
            t1z, bz, ez, am_z, delta, dist, az, evdp, evlo, evla, stlo, stla = (
                hz.get(k, SAC_UNDEFINED) for k in
                ["t1", "b", "e", "a", "delta", "dist", "az", "evdp", "evlo",
                 "evla", "stlo", "stla"])
            t2n, bn, en, am_n = (hn.get(k, SAC_UNDEFINED) for k in ["t2", "b", "e", "a"])
            t2e, be, ee, am_e = (he.get(k, SAC_UNDEFINED) for k in ["t2", "b", "e", "a"])
            if any(v == SAC_UNDEFINED for v in [delta, dist, az, evdp, evlo, evla]):
                continue
            if event_depth is None:
                event_depth, event_lo, event_la = evdp, evlo, evla
            if t1z < 0:
                if am_z != SAC_UNDEFINED and am_z > 0:
                    t1z = am_z
                else:
                    continue
            elif am_z > 0 and am_z < t1z:
                t1z = am_z
            if t2e < 0:
                if am_e != SAC_UNDEFINED and am_e > 0:
                    t2e = am_e
                elif t2n != SAC_UNDEFINED and t2n > 0:
                    t2e = t2n
                elif am_n != SAC_UNDEFINED and am_n > 0:
                    t2e = am_n
                else:
                    continue
            if t2e <= t1z * 1.3:
                t2e = 1.75 * t1z
            if t2e + 5 > en:
                continue

            p_arrival_time = trace_z.stats.starttime + (t1z - bz)
            st.trim(starttime=p_arrival_time - 10, endtime=p_arrival_time + 50,
                    pad=True, fill_value=0)
            try:
                trace_z = st.select(component="Z")[0]
                trace_n = st.select(component="N")[0]
                trace_e = st.select(component="E")[0]
            except IndexError:
                continue

            azimuth_rad = np.deg2rad((az + 0) % 360)
            cos_baz, sin_baz = np.cos(azimuth_rad), np.sin(azimuth_rad)
            r_data = trace_n.data * cos_baz + trace_e.data * sin_baz
            t_data = -trace_n.data * sin_baz + trace_e.data * cos_baz
            trace_r = trace_n.copy(); trace_r.data = r_data; trace_r.stats.channel = "HHR"
            trace_t = trace_e.copy(); trace_t.data = t_data; trace_t.stats.channel = "HHT"
            st = Stream([trace_z, trace_r, trace_t])
            if np.max(np.abs(np.vstack([tr.data for tr in st]))) >= MAX_RAW_AMP:
                continue

            p_window_start = p_arrival_time - WINDOW_PRE_P
            p_window_end = p_arrival_time + WINDOW_POST_P
            s_arrival_time = p_arrival_time - t1z + t2e
            s_window_start = s_arrival_time - WINDOW_PRE_S
            s_window_end = s_arrival_time + WINDOW_POST_S
            st_p = st.copy().trim(p_window_start, p_window_end, pad=True, fill_value=0)
            st_p.detrend("linear").detrend("demean").taper(max_percentage=0.05, type="cosine")
            st_p.interpolate(sampling_rate=1 / TARGET_DT, npts=TARGET_SAMPLE_P)
            st_s = st.copy().trim(s_window_start, s_window_end, pad=True, fill_value=0)
            st_s.detrend("linear").detrend("demean").taper(max_percentage=0.05, type="cosine")
            st_s.interpolate(sampling_rate=1 / TARGET_DT, npts=TARGET_SAMPLE_S)
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

            st_p_amp = st_p.copy().trim(p_arrival_time - 2, p_arrival_time + 3, pad=True, fill_value=0)
            st_s_amp = st_s.copy().trim(s_arrival_time - 2, s_arrival_time + 3, pad=True, fill_value=0)
            p_amps = [np.max(np.abs(tr.data)) if len(tr.data) > 0 else 0 for tr in st_p_amp]
            s_amps = [np.max(np.abs(tr.data)) if len(tr.data) > 0 else 0 for tr in st_s_amp]
            eps = 1e-6
            ratios = [float(np.log(max(0.01, min(10, p / (s + eps)))))
                      for p, s in zip(p_amps, s_amps)]
            scalar_features = np.array(
                [dist, az, stlo, stla, evdp] + list(p_amps) + list(s_amps)
                + list(np.max(spectra_p, axis=1)) + list(np.max(spectra_s, axis=1))
                + ratios, dtype=np.float32)

            event_station_waveforms.append(combined)
            event_station_features.append(scalar_features)
            event_station_names.append(station_key)
        except Exception:
            continue

    if not event_station_waveforms:
        return (event_name, "skipped", "No valid station waveforms found")
    if len(event_station_waveforms) < min_sta:
        return (event_name, "skipped",
                f"Not enough valid stations ({len(event_station_waveforms)} < {min_sta})")

    waveforms_array = np.array(event_station_waveforms)
    waveforms_array[:, 0:6, :] /= np.max(np.abs(waveforms_array[:, 0:6, :])) + 1e-8
    waveforms_array[:, 6:12, :] /= np.max(np.abs(waveforms_array[:, 6:12, :])) + 1e-8
    features_array = np.array(event_station_features)
    return (event_name, "success",
            (ev, waveforms_array, features_array, event_station_names))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--catalog", required=True,
                        help="YHS focal-mechanism catalog (e.g. ysh_all.log). "
                             "Not shipped with this repo -- download it from SCEDC; "
                             "see docs/DATA_PIPELINE.md")
    parser.add_argument("--data-root", required=True, help="Root of per-event SAC directories")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument("--min-mag", type=float, default=2.5)
    parser.add_argument("--max-mag", type=float, default=None)
    parser.add_argument("--grades", default="AB",
                        help="Accepted mechanism quality grades, e.g. 'AB'")
    parser.add_argument("--max-events", type=int, default=200,
                        help="Keep the N most recent qualifying events")
    parser.add_argument("--exclude-hdf5", default=None,
                        help="Exclude event IDs present in this HDF5 (e.g. training file)")
    parser.add_argument("--min-sta", type=int, default=5)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()

    events = parse_ysh_catalog(args.catalog)
    print(f"Catalog events: {len(events)}")

    exclude = set()
    if args.exclude_hdf5:
        with h5py.File(args.exclude_hdf5, "r") as f:
            exclude = set(f.keys())
        print(f"Excluding {len(exclude)} events present in {args.exclude_hdf5}")

    sel = [e for e in events
           if e["magnitude"] >= args.min_mag
           and (args.max_mag is None or e["magnitude"] < args.max_mag)
           and e["grade"] in set(args.grades)
           and e["event_id"] not in exclude
           and os.path.isdir(os.path.join(args.data_root, e["event_id"]))]
    sel.sort(key=lambda e: e["date"], reverse=True)
    sel = sel[: args.max_events]
    print(f"Selected {len(sel)} events "
          f"(dates {sel[-1]['date']}..{sel[0]['date']}, "
          f"mags {min(e['magnitude'] for e in sel):.2f}.."
          f"{max(e['magnitude'] for e in sel):.2f})")

    tasks = [(e, args.data_root, args.min_sta) for e in sel]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for res in tqdm(ex.map(process_event, tasks), total=len(tasks),
                        desc="Processing events"):
            if res[1] == "success":
                results.append(res)
            else:
                print(f"  {res[0]}: {res[1]} - {res[2]}")

    print(f"\nWriting {len(results)} events to {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
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
            g.attrs["evlo"], g.attrs["evla"] = ev["lon"], ev["lat"]
            g.attrs["strike"], g.attrs["dip"], g.attrs["rake"] = (
                ev["strike"], ev["dip"], ev["rake"])
            g.attrs["grade"] = ev["grade"]
            g.attrs["date"] = ev["date"]
    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
