# Sensoformer

**Set-attention inference of earthquake source parameters from variable-geometry seismic networks.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Models on HF](https://img.shields.io/badge/%F0%9F%A4%97%20models-jiazhe868%2Fsensoformer-yellow)](https://huggingface.co/jiazhe868/sensoformer)
[![Data on HF](https://img.shields.io/badge/%F0%9F%A4%97%20data-sensoformer--data-yellow)](https://huggingface.co/datasets/jiazhe868/sensoformer-data)

Sensoformer infers the **moment tensor** and **moment magnitude** of an earthquake
directly from the waveforms of however many stations happen to have recorded it.
The station set is treated as an unordered, variable-cardinality **set** — no grid,
no fixed station list, no interpolation — and the sim-to-real gap is bridged by
**Physics-Structured Domain Randomization (PSDR)**: pre-training on synthetics whose
*physics* (velocity model, scattering, noise, network availability) is randomized,
then fine-tuning on a small real catalog.

On a held-out set of 244 real Southern California events it reaches a **median Kagan
angle of 19.7°**, which is the uncertainty level of the human-analyst catalog it is
compared against.

---

## Results at a glance

244 held-out real SoCal events (seed=42 split), all models trained through the
identical two-stage pipeline. Full tables, CIs and ablations: [docs/RESULTS.md](docs/RESULTS.md).

| Model | Mag MAE | Mean Kagan | Median Kagan |
| :--- | :---: | :---: | :---: |
| **Sensoformer** | **0.100** | **23.9°** | **19.7°** |
| MPNN (GCN, 5-NN graph) | 0.139 | 31.2° | 24.3° |
| MPNN (GAT) | 0.139 | 30.3° | 23.7° |
| DeepSets (no interaction) | 0.140 | 35.7° | 29.1° |
| DeepONet (neural operator) | 0.141 | 35.4° | 28.5° |
| No synthetic pre-training | 0.185 | 32.9° | 25.2° |
| MLP on pooled inputs | 0.138 | 41.9° | 38.2° |
| Linear regression | 0.494 | 41.1° | 37.0° |

Other headline findings, all reproducible with this repo:

- **PSDR shrinks the domain gap by 78%** in embedding space (MMD² 0.721 → 0.157, p = 0.001).
- **Label efficiency**: PSDR pre-training + 10% of the real labels (25.8° median Kagan)
  beats training from scratch on 100% of them (31.8°).
- **Clean synthetics are not enough**: a model pre-trained without any randomization
  fits idealized simulations almost perfectly (4.4° median Kagan in-domain) yet only
  reaches 33.6° on real data after the same fine-tuning.
- **Below the training magnitude range**: zero-shot on M2.5–3.0 events gives 22.5°
  median Kagan; the magnitude error is a single constant offset (remove it and the
  residual MAE is 0.10, the in-distribution value).
- **Calibrated uncertainty**: conformal prediction gives 90% magnitude intervals of
  ±0.22 and 90% Kagan-angle "orientation balls" of 44.8°, with empirical coverage
  matching nominal to ±0.005.

---

## Install

```bash
git clone https://github.com/jiazhe868/Sensoformer.git
cd sensoformer
pip install -e .            # or: pip install -r requirements.txt
make build                  # compile the Fortran moment-tensor kernel (needs gfortran)
make test                   # optional: 40+ unit tests
```

`make build` compiles `src/sensoformer/ext/mtdcmp.f`, which converts moment tensors to
strike/dip/rake and computes Kagan angles. Without it the model still runs and predicts
moment tensors; only the strike/dip/rake and Kagan-angle outputs are skipped.

---

## Quickstart

### 1. Inference on prepared data (weights download automatically)

```bash
# Fetch the pretrained weights (~8 MB) and the real SoCal catalog (0.26 GB)
python scripts/download_assets.py --weights --datasets socal-real

# Predict: per-event source parameters + metrics + optional catalog/figures
python scripts/predict.py --input socal-real --out-dir results/demo --figures
```

Writes `results/demo/predictions.csv` (magnitude, 5 moment-tensor components,
strike/dip/rake, and — when the input has ground truth — Kagan angle and magnitude
error), plus `metrics.json` and figures. `make demo` runs exactly this.

### 2. Python API

```python
from sensoformer import load_pretrained

model = load_pretrained("sensoformer-v3-finetuned", device="cuda")   # cached after first call
predictions, attention = model(waveforms, features, mask)
# predictions: (B, 6) = [scaled Mw, Mxx, Myy, Mxy, Mxz, Myz]
# attention:   (B, S) = per-station pooling weights (interpretable; see docs/ARCHITECTURE.md)
```

### 3. Your own events

Any HDF5 following [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) works, with or without
ground-truth labels:

```bash
python scripts/predict.py --input my_events.hdf5 --out-dir results/mine --catalog
```

To go from raw SAC archives to such a file (SCEDC/STP download, phase picks, windowing,
feature construction), see [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md).

### 4. Fine-tune on a new catalog or region

```bash
python scripts/train.py \
    model=sensoformer data=real_socal training=finetune \
    data.path=/path/to/your_catalog.hdf5 \
    training.pretrained_ckpt=$(python -c "from sensoformer.hub import resolve_checkpoint; print(resolve_checkpoint('sensoformer-v3-pretrained')[0])") \
    hydra.run.dir=./outputs/my_finetune
```

Full walkthrough, hyperparameters and pitfalls: [docs/FINETUNING.md](docs/FINETUNING.md).

---

## Pretrained models and datasets

Weights and data are hosted on the Hugging Face Hub (too large for git) and are fetched
on demand into a local cache. Both `scripts/*.py` and the Python API accept either a
registry name or a plain local path, so the repo also works fully offline.

| Registry name | What it is | Size |
| :--- | :--- | :---: |
| `sensoformer-v3-finetuned` | PSDR pre-trained **+ real fine-tuned** — use this for inference | 8 MB |
| `sensoformer-v3-pretrained` | PSDR synthetic pre-trained — use this to fine-tune on new data | 8 MB |
| `socal-real` | 2,435 real SoCal events (M≥3.0) with analyst mechanisms | 0.26 GB |
| `synthetic-psdr` | ~100k PSDR synthetic events on real station geometries | 11.4 GB |
| `synthetic-clean` | ~50k clean synthetics (no randomization), for the baseline | 5.4 GB |

```bash
python scripts/download_assets.py --list      # show everything available
```

See [docs/HUGGINGFACE.md](docs/HUGGINGFACE.md) for the hosting layout and, for
maintainers, how to publish new assets.

---

## Repository map

```text
sensoformer/
├── src/sensoformer/
│   ├── hub.py              # weight/dataset resolution + load_pretrained()
│   ├── models/             # Sensoformer + every baseline in the paper
│   │   ├── network.py          Sensoformer (set transformer + attention pooling)
│   │   ├── gnn.py              MPNN baselines (GCN and GAT message passing)
│   │   ├── deeponet.py         Neural-operator baseline
│   │   ├── simple_baselines.py Linear / MLP on pooled inputs
│   │   └── posterior_flow.py   Conditional flow for amortized posteriors
│   ├── data/dataset.py     # variable-station dataset + dynamic-padding collate
│   ├── utils/              # physics (Kagan, beachballs), losses, figures
│   └── ext/                # Fortran moment-tensor kernel (mtdcmp.f)
├── scripts/
│   ├── predict.py          # ← inference CLI
│   ├── train.py            # ← pre-training / fine-tuning (Hydra)
│   ├── download_assets.py  # fetch weights + data from the Hub
│   ├── preprocessing/      # raw SAC → model-ready HDF5 (synthetic + real)
│   └── data_acquisition/   # SCEDC/STP download, phase picks, catalog merge
├── configs/                # Hydra configs (model / data / training)
├── docs/                   # see below
└── tests/                  # unit + data-agreement tests
```

## Documentation

| Document | Contents |
| :--- | :--- |
| [QUICKSTART.md](docs/QUICKSTART.md) | The 5-minute paths for inference and fine-tuning |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Exact tensor shapes, QKV dimensions, attention pooling, parameter counts |
| [DATA_FORMAT.md](docs/DATA_FORMAT.md) | HDF5 schema — what you need to bring your own data |
| [DATA_PIPELINE.md](docs/DATA_PIPELINE.md) | Raw SAC acquisition → preprocessing → HDF5, for both domains |
| [FINETUNING.md](docs/FINETUNING.md) | Fine-tuning and pre-training recipes, hyperparameters, diagnostics |
| [INFERENCE.md](docs/INFERENCE.md) | `predict.py` reference, outputs, catalog format, how to read the numbers |
| [RESULTS.md](docs/RESULTS.md) | Full benchmark tables, ablations, uncertainty and generalization studies |
| [HUGGINGFACE.md](docs/HUGGINGFACE.md) | Asset hosting; publishing new weights/datasets |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Known pitfalls and their fixes |

---

## Citation

```bibtex
@article{jia2026sensoformer,
  title   = {Sensoformer: Robust Sim-to-Real Inference on Variable-Geometry
             Sensor Sets via Physics-Structured Randomization},
  author  = {Jia, Zhe and Zhang, Xiaotian and Li, Junpeng},
  year    = {2026}
}
```

Machine-readable metadata is in [CITATION.cff](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
