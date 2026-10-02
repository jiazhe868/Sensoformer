# HDF5 data format

Everything in this repo — training, fine-tuning, inference — reads a single HDF5 layout.
Bring your own data by matching it. To build such a file from raw SAC archives, use
`scripts/preprocessing/` ([DATA_PIPELINE.md](DATA_PIPELINE.md)).

## Layout

```
/<event_id>/
    waveforms       (S, 12, 101) float32     # per station, see channel table
    features        (S, 20)      float32     # per station scalars, see column table
    station_names   (S,)         vlen str    # e.g. "CI.PASC" (optional but recommended)
    attrs:
        magnitude   float    # Mw            ─┐
        Mxx, Myy, Mxy, Mxz, Myz  float       ─┤ ground truth: required for training,
                                              │ optional for inference
        depth       float    # km            ─┘
        evlo, evla  float    # event lon/lat  (optional; used by --catalog and maps)
        strike, dip, rake    (optional; provenance)
        grade       str      (optional; catalog quality flag)
        date        int      (optional; YYYYMMDD)
```

`S` (number of stations) varies per event — that is the point of the architecture. Events
are stored as top-level groups whose name is the event id.

**Labels are optional.** If `magnitude` and the five `M··` attributes are absent, the
dataset yields NaN targets and `scripts/predict.py` runs in pure-inference mode (no error
metrics). This is the path for applying the model to new, unlabeled events.

## Waveform channels (axis 1, length 12)

| Index | Content |
| :---: | :--- |
| 0–2 | P-wave window, Z / R / T components |
| 3–5 | S-wave window, Z / R / T components |
| 6–8 | amplitude spectrum of the P window, Z / R / T |
| 9–11 | amplitude spectrum of the S window, Z / R / T |

Each window is 101 samples: 5 s before to 5 s after the arrival at dt = 0.1 s.
Horizontals are rotated to radial/transverse using the source–station azimuth.

**Normalization (applied per event, not per station):** time-domain channels 0–5 are
divided by the maximum absolute amplitude over those channels across all stations of the
event; spectral channels 6–11 likewise. This preserves *relative* amplitudes between
stations — which is what carries the radiation pattern — while removing the absolute
scale.

## Scalar features (axis 1, length 20)

| Index | Content | Group |
| :---: | :--- | :--- |
| 0 | source–station distance (km) | geometry |
| 1 | source–station azimuth (deg) | geometry |
| 2 | station longitude | geometry |
| 3 | station latitude | geometry |
| 4 | event depth (km) | geometry |
| 5–7 | P-window max amplitude, Z / R / T | amplitude |
| 8–10 | S-window max amplitude, Z / R / T | amplitude |
| 11–13 | P-spectrum max, Z / R / T | amplitude |
| 14–16 | S-spectrum max, Z / R / T | amplitude |
| 17–19 | log P/S amplitude ratio, Z / R / T | amplitude |

The split matters for ablations: `data.aug_params.zero_amplitude_features=true` zeroes
indices 5–19 and keeps 0–4 (the constants are `GEOMETRY_FEATURE_INDICES` and
`AMPLITUDE_FEATURE_INDICES` in `data/dataset.py`). Doing so roughly doubles the magnitude
error while leaving the mechanism nearly unchanged — amplitudes carry magnitude, waveform
shape and geometry carry the mechanism.

## Target convention

- **Magnitude** is scaled to [−1, 1] assuming the range [2, 8]:
  `y = 2·(Mw − 2)/6 − 1`, inverse `Mw = (y + 1)/2·6 + 2`.
  Note the released model was trained on M ≥ 3.0 only; predictions floor near Mw ≈ 2.7
  for smaller events (see the zero-shot section of [RESULTS.md](RESULTS.md)).
- **Moment tensor**: five normalized deviatoric components `[Mxx, Myy, Mxy, Mxz, Myz]`
  with unit scalar moment and `Mzz = −(Mxx + Myy)`. Converted from strike/dip/rake by
  `scripts/preprocessing/sdr_utils.py:sdr2mxyz_norm` (Aki & Richards convention).
- **Nodal-plane convention**: mechanisms are stored with rake in [−90°, 90°]; where a
  catalog gives the conjugate plane, the auxiliary plane is substituted (the moment
  tensor is invariant under that exchange — verified in
  `tests/test_acquisition_agreement.py`).

## Minimal example

```python
import h5py, numpy as np

with h5py.File("my_events.hdf5", "w") as f:
    g = f.create_group("event_0001")
    g.create_dataset("waveforms", data=np.zeros((12, 12, 101), "f4"), compression="gzip")
    g.create_dataset("features",  data=np.zeros((12, 20),      "f4"), compression="gzip")
    g.create_dataset("station_names",
                     data=[f"NET.ST{i:02d}" for i in range(12)],
                     dtype=h5py.special_dtype(vlen=str))
    g.attrs["depth"] = 8.0
    g.attrs["evlo"], g.attrs["evla"] = -117.5, 34.1
    # ground-truth attrs omitted -> inference-only file
```

Then:

```bash
python scripts/predict.py --input my_events.hdf5 --out-dir results/mine --catalog
```
