# Quickstart

Two paths: **run the released model** (minutes) and **fine-tune on your own data**
(hours). Both assume the install from the [README](../README.md).

## 0. Install check

```bash
pip install -e .
make build                     # Fortran kernel: strike/dip/rake + Kagan angles
python -c "import sensoformer; print(sensoformer.__version__)"
python -c "from sensoformer.ext import MTDecomposer; print('kernel:', MTDecomposer().is_available)"
```

If the kernel prints `False`, install `gfortran` and re-run `make build`. The model still
works without it; only strike/dip/rake and Kagan angles are unavailable.

## 1. Inference in three commands

```bash
python scripts/download_assets.py --weights --datasets socal-real   # ~0.26 GB
python scripts/predict.py --input socal-real --out-dir results/demo --figures
head -2 results/demo/predictions.csv
```

Expected (the full 2,435-event file includes events used for training, so this is not a
clean test score — see [RESULTS.md](RESULTS.md) for the held-out numbers):

```json
{"kagan_median": ~17-20, "magnitude_mae": ~0.10, "frac_kagan_lt30": ~0.8}
```

Useful flags: `--limit N` (quick smoke test), `--device cpu`, `--catalog`
(write a catalog file), `--event-list file.txt` (specific events),
`--min-stations 10` (skip poorly recorded events). Full reference:
[INFERENCE.md](INFERENCE.md).

## 2. Inference on your own events

Build an HDF5 matching [DATA_FORMAT.md](DATA_FORMAT.md) — from raw SAC with

```bash
python scripts/preprocessing/preprocess_real_hdf5.py \
    --catalog /path/to/catalog.log --data-root /path/to/sac_archive \
    --output my_events.hdf5 --min-mag 2.5 --grades AB --workers 16
```

— then predict. Labels are optional; without them you get predictions and no metrics.

```bash
python scripts/predict.py --input my_events.hdf5 --out-dir results/mine --catalog
```

## 3. Fine-tune on your own catalog

```bash
# Where is the PSDR-pretrained checkpoint? (downloads on first call)
CKPT=$(python -c "from sensoformer.hub import resolve_checkpoint; print(resolve_checkpoint('sensoformer-v3-pretrained')[0])")

python scripts/train.py \
    model=sensoformer data=real_socal training=finetune \
    data.path=/path/to/my_catalog.hdf5 \
    training.pretrained_ckpt=$CKPT \
    training.eval_test=true training.val_size=0.1 training.test_size=0.1 \
    seed=42 device=cuda \
    hydra.run.dir=./outputs/my_finetune
```

Always pass an explicit `hydra.run.dir` — see
[TROUBLESHOOTING.md](TROUBLESHOOTING.md#concurrent-runs-overwrite-each-other).
Add `training.smoke_test=true` first: it truncates to 20 events and 2 epochs and
exercises the whole path in about a minute.

Outputs in the run directory: `best_finetuned_model.pth`, `metrics_summary.json`,
`per_event_predictions.csv`, learning curves and figures.
Details and hyperparameters: [FINETUNING.md](FINETUNING.md).

## 4. Pre-train from scratch (optional, GPU-hours)

```bash
python scripts/download_assets.py --datasets synthetic-psdr      # 11.4 GB
python scripts/train.py model=sensoformer data=synthetic training=pretrain \
    seed=42 device=cuda hydra.run.dir=./outputs/my_pretrain
```

Then fine-tune as in step 3 with
`training.pretrained_ckpt=./outputs/my_pretrain/best_sensoformer_pretrain.pth`.
Check the pre-training actually converged before using it — the failure mode and its
diagnostic are in [TROUBLESHOOTING.md](TROUBLESHOOTING.md#pre-training-stalls-at-the-initial-loss-plateau).
