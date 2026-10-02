# Troubleshooting

Problems we actually hit, and what they look like.

## Concurrent runs overwrite each other

**Symptom.** Two trainings launched at the same time; one crashes with
`RuntimeError: Error(s) in loading state_dict ... Missing key(s)`, or a run's results are
silently wrong.

**Cause.** Hydra's default output directory is a timestamp with second resolution
(`outputs/YYYY-MM-DD/HH-MM-SS`). Two jobs started in the same second share a directory
and race on the same checkpoint filename.

**Fix.** Always pass an explicit, unique output directory:

```bash
python scripts/train.py ... hydra.run.dir=./outputs/my_run_a
```

Never rely on the timestamp default when launching more than one job.

## Pre-training stalls at the initial loss plateau

**Symptom.** Synthetic pre-training ends with val loss ≈ 0.22 and ~75° mean Kagan on
synthetics — i.e. the mechanisms are no better than random — and fine-tuning from it
gives ~30°+ on real data.

**Cause.** The pre-training recipe has no LR warmup and no gradient clipping, and plain
set-transformer pre-training occasionally fails to escape the initial plateau. In our
runs it escaped on 1 of 3 seeds for the unmodified model, while every variant with extra
inductive structure escaped.

**Fix.** Check before using a checkpoint:

```bash
python -c "import json; h=json.load(open('outputs/my_pretrain/loss_history.json')); \
v=h.get('val_loss',h.get('val')); print('best val', min(v), 'at epoch', v.index(min(v))+1)"
```

A converged run reaches ≲ 0.02 (and ≲ 10° mean Kagan in `metrics_summary.json`). If it
stalled, re-run with a different `seed`. Adding warmup + gradient clipping is a sensible
local change if you are modifying the training loop anyway.

## Late-epoch NaN losses

Transformer pre-training can produce NaN losses in late epochs (again: no gradient
clipping). The *best* checkpoint is saved before that happens, so completed runs are
usable — but verify `metrics_summary.json` rather than assuming the last epoch is fine.

## `Could not override 'training.foo'`

Hydra's structured configs reject keys that are not declared. Either add the key to the
relevant YAML in `configs/`, or use the append syntax:

```bash
python scripts/train.py +training.my_new_flag=true
```

## `MTDecomposer` warns that the library is missing

**Symptom.** `Fortran library not found at .../mtdcmp.so` and no strike/dip/rake or
Kagan angles in the output.

**Fix.** `make build` (requires `gfortran`). Check with:

```bash
python -c "from sensoformer.ext import MTDecomposer; print(MTDecomposer().is_available)"
```

A prebuilt `mtdcmp.so` is committed for convenience but will not load on a different
architecture or libc — rebuild it rather than debugging the loader.

## CUDA is available in `nvidia-smi` but `torch.cuda.is_available()` is False

A PyTorch build newer than the driver. Check the pairing:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```

Install a torch wheel whose CUDA version the driver supports (e.g. `cu124` for driver
555.x), or run with `device=cpu`.

## `No space left on device` from a DataLoader

PyTorch multiprocessing uses `/tmp` for shared memory. On a full or small `/tmp`:

```bash
export SENSOFORMER_TMPDIR=/dev/shm      # honoured by scripts/predict.py
export TMPDIR=/dev/shm                  # for everything else
```

Also lower `data.num_workers`.

## `weights_only` error from `torch.load`

Torch ≥ 2.6 defaults to `weights_only=True`. The release checkpoints are plain
dicts of tensors/strings and load fine through `load_pretrained()`. If you load a
checkpoint manually and hit this, pass `weights_only=False` for files you trust.

## Predictions look random (≈75–80° Kagan)

Two usual causes:

1. **Weights never loaded.** Verify the parameter values actually changed, not just that
   the call returned — a forward pass works perfectly well with random weights.
   `load_pretrained()` is covered by `tests/test_hub.py` for exactly this reason.
2. **Feature-column mismatch.** If you built the HDF5 yourself, confirm the 20 scalar
   columns and 12 waveform channels are in the documented order
   ([DATA_FORMAT.md](DATA_FORMAT.md)) and that normalization is per *event*, not per
   station.

Sanity check against a known-good file:

```bash
python scripts/predict.py --input socal-real --limit 300 --out-dir /tmp/check
# expect kagan_median ~17-20 and magnitude_mae ~0.10
```

## Magnitudes are systematically too large for small events

Expected: the released model saw only M ≥ 3.0 and floors near Mw ≈ 2.7. Below the
training range the bias is roughly constant (+0.28 at M2.5–3.0, +0.71 at M2.0–2.5) while
the *relative* scaling stays accurate. Subtract the mean bias measured on a labeled
sample, or fine-tune briefly including small events.

## Modifying attention: float masks silently break in eval mode

If you add an additive attention bias through `attn_mask`, note that
`nn.TransformerEncoder`'s fused fast path (taken under `eval()`/`no_grad`) mishandles
float 3-D masks and can produce NaNs. `models/network.py` works around this with
`BiasedTransformerEncoderLayer`, which forces the standard path — follow that pattern.

## Data-agreement tests skip

`tests/test_preprocessing_agreement.py` and `tests/test_acquisition_agreement.py` compare
the preprocessing against the original archives and skip when those raw SAC archives are
not mounted. That is expected on any machine without them; the rest of the suite still
runs.
