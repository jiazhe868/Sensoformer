# Training and fine-tuning

Sensoformer is trained in two stages: **pre-train on PSDR synthetics**, then
**fine-tune on a real catalog**. Most users only need the second stage, starting from
the released `sensoformer-v3-pretrained` weights.

Training is driven by [Hydra](https://hydra.cc): `scripts/train.py` composes
`configs/{model,data,training}/*.yaml` and every value is overridable on the command
line.

## Stage 2 only: fine-tune on your catalog (the common case)

```bash
CKPT=$(python -c "from sensoformer.hub import resolve_checkpoint; print(resolve_checkpoint('sensoformer-v3-pretrained')[0])")

python scripts/train.py \
    model=sensoformer data=real_socal training=finetune \
    data.path=/path/to/my_catalog.hdf5 \
    training.pretrained_ckpt=$CKPT \
    training.eval_test=true training.val_size=0.1 training.test_size=0.1 \
    seed=42 device=cuda \
    hydra.run.dir=./outputs/my_finetune
```

Run it once with `training.smoke_test=true` first (20 events, 2 epochs, ~1 minute) to
validate paths and shapes before committing GPU hours.

### What the fine-tune recipe does

| Setting | Value | Why |
| :--- | :--- | :--- |
| optimizer | AdamW, weight decay 1e-4 | — |
| `lr_backbone`, `lr_head` | 2e-6 both | deliberately tiny: the pre-trained features are the asset; large LRs destroy them |
| loss | Focal-L1 on the moment tensor (γ=1.5) + MSE on magnitude | down-weights easy events, focuses on hard mechanisms |
| sampler | `WeightedRandomSampler` over mechanism type | real catalogs are dominated by strike-slip; this prevents mode collapse on rare mechanisms |
| batch size | 64 | — |
| epochs / patience | 150 / 50 (early stopping on val loss) | — |
| station augmentation | dropout to 30–50 stations, 3% noise | the training-time half of PSDR |

### Splits

`seed` controls **both** weight initialization and the train/val/test split:

```
train, temp = train_test_split(all_ids, test_size=val_size+test_size, random_state=seed)
val,   test = train_test_split(temp,    test_size=test_size/(val_size+test_size), random_state=seed)
```

The canonical protocol in the paper is `seed=42, val_size=0.1, test_size=0.1`, which
yields 244 test events on the released SoCal catalog. **Models you intend to compare must
share `seed` and the split sizes** — otherwise they are scored on different events.

### Outputs

In `hydra.run.dir`:

| File | Contents |
| :--- | :--- |
| `best_finetuned_model.pth` | best-val-loss weights (a bare `state_dict`) |
| `metrics_summary.json` | test magnitude MAE, mean/median Kagan angle, early-stop epoch |
| `per_event_predictions.csv` | per-event Kagan angle and magnitude error, keyed by `event_id` |
| `loss_history.json` | train/val loss per epoch |
| `learning_curve.png`, `scatter_results.pdf`, `kagan_histogram.pdf`, `beachball_grid.pdf` | diagnostics |
| `.hydra/config.yaml` | the fully resolved configuration (always check this when a run surprises you) |

To package the result for distribution or `load_pretrained()`, run
`scripts/package_checkpoints.py` (see [HUGGINGFACE.md](HUGGINGFACE.md)).

## Both stages from scratch

```bash
# Stage 1: PSDR synthetic pre-training (GPU-hours; 11.4 GB dataset)
python scripts/download_assets.py --datasets synthetic-psdr
python scripts/train.py model=sensoformer data=synthetic training=pretrain \
    seed=42 device=cuda hydra.run.dir=./outputs/my_pretrain

# Stage 2: fine-tune from your own stage-1 checkpoint
python scripts/train.py model=sensoformer data=real_socal training=finetune \
    training.pretrained_ckpt=./outputs/my_pretrain/best_sensoformer_pretrain.pth \
    training.eval_test=true training.val_size=0.1 training.test_size=0.1 \
    seed=42 device=cuda hydra.run.dir=./outputs/my_finetune
```

Pre-training uses AdamW at lr 2e-4, MSE loss, batch 512, 150 epochs with patience 30.

> **Verify stage 1 before you trust stage 2.** With this recipe (no LR warmup, no
> gradient clipping) plain set-transformer pre-training occasionally never escapes the
> initial loss plateau. A converged run reaches val loss ≲ 0.02 and ≲ 10° mean Kagan on
> synthetics; a stalled run sits near 0.22 with ~75° Kagan (i.e. random mechanisms).
> Check `loss_history.json` / `metrics_summary.json`, and if stalled, simply retry with a
> different `seed`. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#pre-training-stalls-at-the-initial-loss-plateau).

## Training without pre-training (baseline)

```bash
python scripts/train.py model=sensoformer data=real_socal training=real_from_scratch \
    training.lr=3e-4 seed=42 device=cuda \
    training.eval_test=true training.val_size=0.1 training.test_size=0.1 \
    hydra.run.dir=./outputs/scratch
```

lr 3e-4 was selected by a validation-only sweep. Expect ~25–32° median Kagan with large
run-to-run variance — from-scratch training on ~1,700 events is unstable, which is part
of the case for PSDR pre-training.

## Other models

Any variant in [ARCHITECTURE.md](ARCHITECTURE.md#6-variants-shipped-in-this-repo) works
in the same commands, e.g. `model=mpnn`, `model=deepsets`, `model=linear_baseline`.
They share the pre-train → fine-tune protocol, so comparisons are apples-to-apples.

## Useful knobs

| Override | Effect |
| :--- | :--- |
| `training.train_fraction=0.1` | train on a random 10% of the training split (label-efficiency curves); val/test untouched |
| `data.aug_params.zero_amplitude_features=true` | zero the 15 amplitude scalars, keep the 5 geometry ones |
| `data.aug_params.augmentation=false` | disable training-time station dropout + noise |
| `data.aug_params.min_stations_keep=50` | effectively disable station dropout |
| `training.orientation_loss_weight=0.5` | add a scale-invariant deviatoric-tensor cosine loss (a smooth Kagan surrogate; research option, no significant accuracy change in our tests) |
| `data.batch_size=128`, `data.num_workers=16` | throughput tuning |

## Comparing two models honestly

Point metrics from two runs are not enough — fine-tuning has real seed variance. Score
both models on the **same** test events (same `seed`, same split sizes) and compare with
an event-level **paired** bootstrap over the per-event `kagan_angle` / `magnitude_error`
columns of their `per_event_predictions.csv` files, reporting 95% CIs of the paired
difference. In our experience several plausible architecture changes produced point-
estimate "improvements" that vanished under this test.
