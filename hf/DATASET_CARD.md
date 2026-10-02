---
license: mit
tags:
  - seismology
  - earthquake
  - moment-tensor
  - sim-to-real
pretty_name: Sensoformer datasets (synthetic PSDR + real Southern California)
---

# Sensoformer datasets

Preprocessed, model-ready HDF5 datasets for
[Sensoformer](https://github.com/jiazhe868/Sensoformer): amortized inference of
earthquake moment tensors and magnitudes from variable-geometry station sets.

| File | Events | Size | Role |
| :--- | :---: | :---: | :--- |
| `socal_mxyz_data_rtz_lp2_ampr_ps_wlola.hdf5` | 2,435 | 0.26 GB | real Southern California catalog (M ≥ 3.0) with analyst mechanisms — fine-tuning and evaluation |
| `syn_mt_data_realgeom_realvn_10w_ps_wcoda_wlola.hdf5` | ~100k | 11.4 GB | PSDR synthetics on real station geometries — pre-training |
| `syn_mt_data_noaug.hdf5` | ~50k | 5.4 GB | clean synthetics, no randomization — the ablation baseline |

## Format

Per event, one HDF5 group:

```
/<event_id>/
    waveforms      (S, 12, 101) float32   # P/S windows Z,R,T + their amplitude spectra
    features       (S, 20)      float32   # distance, azimuth, station lon/lat, depth,
                                          # P/S max amplitudes, spectral maxima, log P/S ratios
    station_names  (S,)         vlen str
    attrs: magnitude, Mxx, Myy, Mxy, Mxz, Myz, depth [, evlo, evla, strike, dip, rake, grade, date]
```

`S` (station count) varies per event. Windows are 5 s before to 5 s after each arrival at
dt = 0.1 s; horizontals are rotated to radial/transverse; waveform normalization is
**per event** so inter-station relative amplitudes (which carry the radiation pattern) are
preserved. Moment tensors are normalized deviatoric components with
`Mzz = −(Mxx + Myy)`; magnitude is scaled to [−1, 1] over [2, 8] at load time. Full
specification:
[DATA_FORMAT.md](https://github.com/jiazhe868/Sensoformer/blob/main/docs/DATA_FORMAT.md).

## Usage

```bash
pip install git+https://github.com/jiazhe868/Sensoformer.git
python scripts/download_assets.py --datasets socal-real
python scripts/predict.py --input socal-real --out-dir results/
```

```python
import h5py
with h5py.File("socal_mxyz_data_rtz_lp2_ampr_ps_wlola.hdf5") as f:
    g = f[list(f)[0]]
    print(g["waveforms"].shape, g["features"].shape, dict(g.attrs))
```

## Provenance

- **Real data**: waveforms from the Southern California Earthquake Data Center (SCEDC),
  with analyst focal mechanisms from the Yang–Hauksson–Shearer catalog. Downloaded and
  preprocessed with the scripts in `scripts/data_acquisition/` and
  `scripts/preprocessing/`; filtering 0.2–2.0 Hz.
- **Synthetics**: frequency-wavenumber synthesis on **real** station geometries, with
  Physics-Structured Domain Randomization — velocity model sampled per event from a
  CRUST1.0-derived library, stochastic travel-time and amplitude perturbation, injected
  real ambient noise, synthetic scattering coda, and station dropout. Filtering
  0.1–2.0 Hz. The clean dataset omits all randomization.

The preprocessing in the repo reproduces these files **byte-for-byte** from the raw
archives (verified by `tests/test_preprocessing_agreement.py`).

Please cite the SCEDC and the source-mechanism catalog in addition to the Sensoformer
paper when using the real data.

## Citation

```bibtex
@article{jia2026sensoformer,
  title   = {Sensoformer: Robust Sim-to-Real Inference on Variable-Geometry
             Sensor Sets via Physics-Structured Randomization},
  author  = {Jia, Zhe and Zhang, Xiaotian and Li, Junpeng},
  year    = {2026}
}
```
