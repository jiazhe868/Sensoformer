---
license: mit
tags:
  - seismology
  - earthquake
  - moment-tensor
  - set-transformer
  - domain-randomization
  - sim-to-real
  - pytorch
library_name: pytorch
pipeline_tag: other
---

# Sensoformer — pretrained weights

Set-attention inference of earthquake **moment tensor** and **moment magnitude** from a
variable-size set of seismic station recordings.

- **Code, docs and CLI:** https://github.com/jiazhe868/Sensoformer
- **Datasets:** https://huggingface.co/datasets/jiazhe868/sensoformer-data
- **Architecture:** dual-tower 1D-ResNet station encoder → 3-layer transformer encoder
  over stations (128-d, 4 heads, no positional encoding) → additive attention pooling →
  magnitude and moment-tensor heads. 2,063,015 parameters.

## Files

| File | Stage | Use for |
| :--- | :--- | :--- |
| `sensoformer_v3_finetuned.pth` | PSDR synthetic pre-training **+ real fine-tuning** | inference on real data |
| `sensoformer_v3_psdr_pretrained.pth` | PSDR synthetic pre-training only | fine-tuning on a new catalog or region |

Each file is a self-describing checkpoint:
`{state_dict, arch, sensoformer_version, stage, metrics}`.

## Usage

```bash
pip install git+https://github.com/jiazhe868/Sensoformer.git
```

```python
from sensoformer import load_pretrained
model = load_pretrained("sensoformer-v3-finetuned", device="cuda")  # downloads from here
predictions, attention = model(waveforms, features, mask)
# predictions: (B, 6) = [scaled Mw, Mxx, Myy, Mxy, Mxz, Myz];  Mw = (y+1)/2*6 + 2
```

Inputs must follow the documented HDF5 / tensor layout: per station a (12, 101) waveform
block (P and S windows, Z/R/T, plus their amplitude spectra) and 20 scalar features
(distance, azimuth, station lon/lat, depth, amplitude summaries). See
[DATA_FORMAT.md](https://github.com/jiazhe868/Sensoformer/blob/main/docs/DATA_FORMAT.md).

End-to-end CLI:

```bash
python scripts/predict.py --input my_events.hdf5 --out-dir results/ --catalog
```

## Evaluation

244 held-out real Southern California events (M ≥ 3.0; seed=42, 80/10/10 split):

| Metric | Value |
| :--- | :---: |
| Median Kagan angle | **19.7°** |
| Mean Kagan angle | 23.9° |
| Magnitude MAE | **0.100** |
| Fraction with Kagan < 30° | 0.77 |

The median equals the analyst catalog's own median 1σ nodal-plane uncertainty (19.5°),
i.e. the model is at the label noise floor. Baselines under the identical protocol: MPNN
24.3°, DeepSets 29.1°, DeepONet 28.5°, no pre-training 25.2° (median Kagan). Full tables:
[RESULTS.md](https://github.com/jiazhe868/Sensoformer/blob/main/docs/RESULTS.md).

## Intended use and limitations

Intended for research on amortized seismic source inversion and as a starting point for
fine-tuning on other catalogs/networks.

- **Trained on M ≥ 3.0 Southern California events.** Below that range magnitudes floor
  near Mw ≈ 2.7, giving a roughly constant positive bias (+0.28 at M2.5–3.0, +0.71 at
  M2.0–2.5); mechanisms degrade only gradually (22.5° and 28.3° median). Remove the
  constant offset, or fine-tune with small events, before using magnitudes out of range.
- **Error is geometry-limited**: median Kagan ranges from ~17° for well-surrounded events
  (azimuthal gap < 45°) to ~33° beyond 180°. Stratify by azimuthal gap or station count
  rather than quoting a single number.
- **Region/network transfer is untested** beyond Southern California. The architecture is
  geometry-agnostic, but re-fine-tuning and re-calibration are recommended.
- The model does **not** locate events; hypocenter and origin time must come from a
  catalog or locator.
- Reported conformal intervals (90%: ±0.22 magnitude, 44.8° Kagan ball) are calibrated
  for this catalog population — recalibrate elsewhere.

## Citation

```bibtex
@article{jia2026sensoformer,
  title   = {Sensoformer: Robust Sim-to-Real Inference on Variable-Geometry
             Sensor Sets via Physics-Structured Randomization},
  author  = {Jia, Zhe and Zhang, Xiaotian and Li, Junpeng},
  year    = {2026}
}
```
