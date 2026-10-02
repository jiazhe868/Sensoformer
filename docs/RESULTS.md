# Results

All real-data numbers below come from the **same protocol**: the released SoCal catalog
(2,435 events, M ≥ 3.0), split with `seed=42, val_size=0.1, test_size=0.1` → **244
held-out test events**, and every model trained through the identical two-stage pipeline
(PSDR synthetic pre-training → real fine-tuning). Confidence intervals are 10,000-sample
bootstraps; model-vs-model differences are **paired** bootstraps over the same events.

## 1. Benchmark against baselines

| Model | Mag MAE | Mean Kagan | Median Kagan | Kagan<10° | Kagan<20° | Kagan<30° |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sensoformer** | **0.0996** [0.089, 0.111] | **23.90°** [21.83, 26.06] | **19.76°** [17.26, 21.72] | 0.156 | 0.508 | 0.766 |
| MPNN (GAT, 5-NN) | 0.1390 | 30.35° | 23.68° | — | — | — |
| MPNN (GCN, 5-NN) | 0.1391 | 31.20° [28.51, 34.00] | 24.31° [21.80, 27.43] | 0.094 | 0.398 | 0.619 |
| DeepSets (no interaction) | 0.1403 | 35.67° | 29.11° | — | — | — |
| DeepONet | 0.1409 | 35.41° [32.67, 38.24] | 28.53° [25.85, 32.00] | 0.061 | 0.311 | 0.533 |
| No synthetic pre-training | 0.1849 | 32.86° [30.04, 35.75] | 25.20° [22.75, 28.81] | 0.074 | 0.385 | 0.578 |
| MLP on pooled inputs | 0.1380 | 41.94° [39.04, 44.84] | 38.20° | — | — | — |
| Linear regression | 0.4942 | 41.05° [38.13, 43.93] | 37.01° | — | — | — |
| Single tower (no P/S split) | — | 36.0° | 31.3° | — | — | — |

Every baseline is significantly worse than Sensoformer (95% CIs of the paired differences
exclude zero). Two comparisons are worth singling out:

- **Graph attention does not close the gap.** Swapping GCN for GAT message passing over
  the same 5-nearest-neighbour station graph changes nothing significantly (mean Kagan
  −0.86° [−3.04, +1.32]). What matters is *global all-to-all* connectivity, not
  attention-weighted local edges.
- **Magnitude and mechanism dissociate.** Going from linear regression to an MLP on the
  same mean-pooled inputs fixes magnitude almost entirely (0.494 → 0.138) but leaves the
  mechanism unchanged (41.0° → 41.9°). Magnitude survives pooling of amplitude features;
  the mechanism lives in the per-station azimuth-to-polarity relationship, which pooling
  destroys before any interaction can use it.

## 2. PSDR ablations (remove one randomization module from pre-training)

| Removed component | Mag MAE | Mean Kagan | Median Kagan |
| :--- | :---: | :---: | :---: |
| *(none — full PSDR)* | 0.0996 | 23.90° | 19.76° |
| Earth-structure randomization | 0.1172 | 27.40° | 20.09° |
| Variable-geometry masking | 0.1102 | 27.7° | 20.4° |
| Waveform distortion / coda | 0.1452 | 28.46° | 21.33° |
| Real ambient-noise injection | 0.1391 | 29.74° | 22.09° |

Reproduce by disabling the corresponding module in the preprocessing/augmentation
configuration — see [DATA_PIPELINE.md](DATA_PIPELINE.md).

## 3. Does the randomization actually close the domain gap?

**Embedding space (MMD).** Maximum Mean Discrepancy between synthetic and real event
embeddings (RBF kernel, median-heuristic bandwidth, N = 500 per domain, 1,000-permutation
test):

| Encoder | MMD² (synthetic ↔ real) | p |
| :--- | :---: | :---: |
| Trained on clean synthetics | 0.7213 | 0.001 |
| Trained with PSDR | 0.1566 | 0.001 |

**−78.3% domain gap.** Both gaps remain individually significant — PSDR shrinks the gap,
it does not eliminate it.

**End-to-end (the clean-synthetic reference case).** Identical architecture and two-stage
recipe, but pre-trained on clean synthetics with all randomization off:

| Stage | Evaluated on | Mag MAE | Mean Kagan | Median Kagan |
| :--- | :--- | :---: | :---: | :---: |
| Clean pre-train | clean synthetics (4,992) | 0.116 | 5.55° | **4.37°** |
| Clean pre-train + real fine-tune | 244 real events | 0.225 | 40.40° | **33.57°** |
| PSDR pre-train + real fine-tune (same seed) | 244 real events | 0.115 | 24.89° | **19.52°** |

The clean model is excellent *in its own domain* (4.4° median), so the real-data failure
is not a capacity or optimization problem — and fine-tuning on the full real training set
does not repair it. The invariances have to be learned during pre-training.

## 4. Label efficiency

Fine-tuning on a fraction of the real training split, evaluated on the fixed 244 events:

| Real labels used | PSDR pre-trained | From scratch |
| :--- | :---: | :---: |
| 0% (zero-shot) | 36.67° / 0.342 | — |
| 10% (~195 events) | **25.79° / 0.145** | 31.44° / 0.333 |
| 25% | 23.63° / 0.123 | 33.96° / 0.230 |
| 50% | 22.95° / 0.123 | 32.81° / 0.235 |
| 100% (~1,948 events) | 23.64° / 0.112 | 31.79° / 0.191 |

*(median Kagan / magnitude MAE)*. PSDR pre-training with **10% of the labels beats
from-scratch training on 100%**, and the curve saturates by ~25%. Reproduce with
`training.train_fraction=0.1`.

## 5. Where the error comes from

Measured on 2,709 non-training events with reliable (A/B-grade) analyst mechanisms:

| Usable stations | 10–15 | 15–20 | 20–25 | 25–30 | 30–40 | 40–50 | 50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Median Kagan | 26.7° | 23.7° | 21.8° | 20.8° | 19.2° | 21.2° | 16.8° |

| Azimuthal gap | <45° | 45–60° | 60–90° | 90–120° | 120–180° | >180° |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Median Kagan | 16.8° | 20.0° | 21.6° | 24.9° | 26.5° | 32.7° |

The error tail is **geometry-limited**: it concentrates on sparsely and one-sidedly
recorded events, where a focal mechanism is weakly identifiable for any method.

**The median sits at the label noise floor.** Against the catalog's own per-solution 1σ
nodal-plane uncertainty:

| Population | Sensoformer vs label | Label's own 1σ uncertainty |
| :--- | :---: | :---: |
| 244 held-out test events | median **19.7°** (mean 23.9°) | median **19.5°** (mean 20.5°) |
| 2,709 zero-shot A/B events | median 21.5° (mean 27.7°) | median 20.5° (mean 20.8°) |

Note what is being compared: the left column is model-vs-label *disagreement*, which
contains both the model's error and the label's, while the right column is the label's
own stated error alone. Even a perfect model would therefore show disagreement at about
the label's error level. The medians coincide; the means differ because of the
geometry-driven tail above.

## 6. Calibrated uncertainty

**Conformal prediction** on the point model (distribution-free, calibrated on held-out
events):

| α | Magnitude interval | Kagan "orientation ball" | Empirical coverage |
| :---: | :---: | :---: | :---: |
| 0.05 | ±0.254 | 58.7° | 0.950 |
| 0.10 | ±0.217 | 44.8° | 0.899–0.902 |
| 0.20 | ±0.154 | 32.6° | 0.800–0.803 |
| 0.30 | ±0.126 | 27.0° | 0.705–0.706 |

Coverage matches nominal to within ±0.005 at every level. Calibration data must be
untouched by training *and* model selection: calibrating on the validation split (used
for early stopping) under-covers by 3–4 points.

**Amortized posterior** (`models/posterior_flow.py`, a conditional flow on the frozen
embedding): calibrated in-simulation (simulation-based-calibration rank KS ≤ 0.06 in all
six dimensions), but only **37%** coverage of nominal-90% credible sets when applied
zero-shot to real events — the sim-to-real gap expressed as a *calibration* gap. After
real fine-tuning plus conformal recalibration of the credible radii, coverage is exactly
nominal (0.901 ± 0.038) with a median 90% ball of 42.6°, and the per-event radius tracks
both the actual error (Spearman ρ = 0.31) and the network's azimuthal gap (ρ = 0.26).

## 7. Generalization below the training magnitude range

The released model was trained on M ≥ 3.0 only. Zero-shot on more recent, strictly
smaller events (grade A/B mechanisms, disjoint event ids):

| Population | n | Mag MAE | Signed bias | Bias-corrected MAE | Mean Kagan | Median Kagan | Kagan<30° |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| M3+ test set (reference) | 244 | 0.100 | — | — | 23.89° | 19.72° | 0.77 |
| M2.5–3.0 zero-shot | 350 | 0.302 | +0.28 | **0.10** | 28.30° | 22.50° | 0.65 |
| M2.0–2.5 zero-shot | 350 | 0.706 | +0.71 | **0.12** | 36.15° | 28.29° | 0.55 |

Mechanisms degrade gradually (19.7° → 22.5° → 28.3°). The magnitude failure is a **single
constant offset**: predictions floor near Mw ≈ 2.7, so the bias grows linearly with
distance below the threshold, while the bias-corrected MAE stays at the in-distribution
value in both bins. Relative magnitude scaling transfers; only the absolute offset does
not.

## 8. Catalog completeness

Applying the model to every M ≥ 2.5 catalog event with waveform data and ≥ 10 usable
stations (1992–2024) yields 7,134 mechanisms. Agreement with the analyst catalog,
stratified by the catalog's own quality grade (training events excluded):

| Compared against | n | Median Kagan | Kagan<30° |
| :--- | :---: | :---: | :---: |
| Grade A/B (reliable) labels | 2,709 | 21.51° [20.85, 22.26] | 0.68 |
| Grade C/D (unreliable) labels | 2,699 | 26.42° | 0.56 |

At matched magnitude the model disagrees ~5.7° *more* with the unreliable labels than
with the reliable ones — evidence that the extra disagreement sits in the labels.

**Completeness gain**: 2,699 of those events have only an unreliable (C/D) analyst
solution, i.e. **+31.8%** more quality mechanisms than the 8,497 reliable solutions in the
same period (+19.8% when restricted to the M2.5–3.0 band), with expected quality at the
matched-magnitude A/B agreement level.

---

### Reproducing these numbers

Sections 1, 2, 4, 7 and the clean-synthetic reference in 3 are produced by
`scripts/train.py` + `scripts/predict.py` with the overrides named in
[FINETUNING.md](FINETUNING.md). The MMD, conformal, posterior-flow and bootstrap studies
(sections 3, 5, 6, 8) were run with additional analysis scripts that are not part of this
release; the underlying per-event prediction CSVs that they consume are exactly what
`predict.py` writes, so they are straightforward to recompute. Contact the authors if you
need the analysis code.
