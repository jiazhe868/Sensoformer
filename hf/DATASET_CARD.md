---
license: other
license_name: mixed-mit-code-and-scedc-attribution-data
license_link: https://scedc.caltech.edu/data/alt-2011-yang-hauksson-shearer.html
tags:
  - seismology
  - earthquake
  - moment-tensor
  - focal-mechanism
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
| `ysh_all.log` | 280,889 | 31 MB | Yang–Hauksson–Shearer focal-mechanism catalog, 1981–2024 — the **labels** used to build the real dataset (plain text, not HDF5) |

## The label catalog (`ysh_all.log`)

The real HDF5 above is waveforms **plus labels**; the labels come from this catalog,
so it is included to make the pipeline reproducible end to end. It is a plain
whitespace-separated text file, one line per event, 21 columns:

| col | field | | col | field |
| :-- | :-- | :-- | :-- | :-- |
| 0–5 | year, month, day, hour, minute, second | | 11–13 | **strike, dip, rake** (deg) |
| 6 | event id (SCSN) | | 14–15 | fault-plane / auxiliary fault-plane uncertainty (deg) |
| 7–9 | latitude, longitude, depth (km) | | 16–19 | # P first motions, their misfit, # S/P ratios, their misfit |
| 10 | magnitude | | 20 | **quality grade** `A`/`B`/`C`/`D` |

```
1981  1  1  4 13 55.710  3301565  33.25517 -115.96750   5.680  2.260  318  57 -168  37  39   18  0.17    0  0.00 C
2024 12 31 23 33 51.090 40831071  36.18400 -118.08167   3.950  1.560  169  55 -151  30  30    9  0.26   16  0.00 B
```

Grades follow the HASH convention and are a hard cap on the mean nodal-plane
uncertainty (columns 14–15), which you can read straight off this file:

| grade | events | mean fault-plane uncertainty | cap |
| :-- | --: | --: | --: |
| A | 24,282 | 19.7° | ≤ 25° |
| B | 57,158 | 28.1° | ≤ 35° |
| C | 86,744 | 36.2° | ≤ 45° |
| D | 112,705 | 43.5° | — |

The repo's experiments train and evaluate against A/B only; the C/D events — 71% of
the catalog — are where Sensoformer adds mechanisms the catalog does not reliably
provide.

Validate a copy before using it:

```bash
python scripts/data_acquisition/check_catalog_format.py ysh_all.log
```


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

# the label catalog, if you are rebuilding the HDF5 from raw SAC archives
python scripts/download_assets.py --catalogs yhs-socal
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

## Licensing and redistribution

The Sensoformer **code** is MIT. The **data here is not ours to relicense**:

- `ysh_all.log` is the Yang–Hauksson–Shearer catalog, distributed by the SCEDC and
  redistributed here unmodified so that the pipeline is reproducible. SCEDC's
  citation policy asks you to cite SCEDC and the references listed on the
  catalog's page — see below.
- The real HDF5 is a derived product of SCEDC waveforms and the same catalog; the
  same attribution applies.
- The synthetic HDF5 files are generated by this project and are MIT.

If you are redistributing further, go back to the SCEDC for the authoritative copy
and its current terms rather than treating this mirror as the source.

Please cite the SCEDC and the source-mechanism catalog in addition to the Sensoformer
paper when using the real data:

```bibtex
@misc{scedc2013,
  author    = {{SCEDC}},
  title     = {Southern California Earthquake Data Center},
  publisher = {Caltech},
  year      = {2013},
  doi       = {10.7909/C3WD3xH1}
}
@article{hauksson2012relocated,
  title   = {Waveform Relocated Earthquake Catalog for Southern California
             (1981 to June 2011)},
  author  = {Hauksson, Egill and Yang, Wenzheng and Shearer, Peter M.},
  journal = {Bulletin of the Seismological Society of America},
  volume  = {102}, number = {5}, pages = {2239--2244}, year = {2012},
  doi     = {10.1785/0120120010}
}
@article{yang2012ysh,
  title   = {Computing a Large Refined Catalog of Focal Mechanisms for Southern
             California (1981--2010): Temporal Stability of the Style of Faulting},
  author  = {Yang, Wenzheng and Hauksson, Egill and Shearer, Peter M.},
  journal = {Bulletin of the Seismological Society of America},
  volume  = {102}, number = {3}, pages = {1179--1194}, year = {2012},
  doi     = {10.1785/0120110311}
}
```

The published catalog covers 1981–2010; the SCEDC extends it with one file per year
since 2011, and the file here runs through 2024-12-31. Rebuilt from SCEDC's files today,
1981–2023 matches this mirror byte for byte; 2024 differs because SCEDC re-issued it
(marked *updated 02/27/2026*; median mechanism change 2.8°). Use this mirror to reproduce
the Sensoformer results and a fresh SCEDC download for the newest solutions. The
step-by-step rebuild is in the
[repo README](https://github.com/jiazhe868/Sensoformer#option-b--build-it-yourself-from-scedc).

## Citation

```bibtex
@article{jia2026sensoformer,
  title   = {Sensoformer: Robust Sim-to-Real Inference on Variable-Geometry
             Sensor Sets via Physics-Structured Randomization},
  author  = {Jia, Zhe and Zhang, Xiaotian and Li, Junpeng},
  year    = {2026}
}
```
