# Architecture

Sensoformer maps an unordered, variable-size set of station recordings to one
6-dimensional source vector. All shapes below are the released configuration
(`configs/model/sensoformer.yaml`) and were read off the released checkpoint, not
copied from the paper.

```
 per station                                       set of S stations
 ┌───────────────────────────┐
 │ P window   (6, 101)       │──► 1D-ResNet tower ──► 128 ┐
 │ S window   (6, 101)       │──► 1D-ResNet tower ──► 128 ├─ concat 288 ─► Linear ─► h_i (128)
 │ 20 scalar features        │──► MLP 20→64→32 ──────► 32 ┘
 └───────────────────────────┘
                                                      │  tokens (B, S, 128)
                                                      ▼
                        3 × Transformer encoder layer (4 heads, d_model 128, FFN 2048)
                            all-to-all self-attention over stations, padding-masked
                            NO positional encoding  ← set semantics
                                                      │  (B, S, 128)
                                                      ▼
                        attention pooling:  a_i = softmax(w₂ᵀ tanh(W₁ h_i))
                                            z   = Σ a_i h_i                → (B, 128)
                                                      │
                             ┌────────────────────────┴────────────────────────┐
                             ▼                                                ▼
                   magnitude head 128→32→1                     moment-tensor head 128→64→5
                             └──────────── tanh ──────────────┬─────── tanh ───┘
                                                              ▼
                                               predictions (B, 6) + attention (B, S)
```

## 1. Station tokens

Each station is encoded **independently** into a 128-d token by `StationEncoder`
(`models/network.py`):

| Stream | Input | Path | Output |
| :--- | :--- | :--- | :---: |
| P wave | 6 ch × 101 samples | Conv1d(k=7, s=2) → BN → ReLU → 3 residual blocks → global avg pool | 128 |
| S wave | 6 ch × 101 samples | separate tower, identical shape | 128 |
| Metadata | 20 scalars | Linear(20→64) → ReLU → Linear(64→32) | 32 |

Concatenated (128+128+32 = 288) and fused by `Linear(288→128)`.

Two design choices matter here:

- **Separate P and S towers.** Compressional and shear energy carry different
  constraints; sharing one tower ("single tower" ablation) degrades the median Kagan
  angle from 19.7° to 31.3°.
- **Geometry enters through the token, not through a positional encoding.** The 20
  scalars include source–station distance, azimuth, station lon/lat and event depth,
  so the attention stack has continuous spatial context without any discrete position
  embedding. Column semantics are in [DATA_FORMAT.md](DATA_FORMAT.md).

## 2. Set-transformer aggregator — a standard encoder, with two deliberate omissions

`event_aggregator` is a plain `nn.TransformerEncoder` of 3 layers. Per layer:

| Component | Shape / value |
| :--- | :--- |
| `d_model` | 128 |
| heads | 4, each 32-dimensional |
| QKV projection | one packed `in_proj_weight` of (384, 128) → W_Q, W_K, W_V each 128→128 |
| attention | `softmax(QKᵀ/√32 + mask) V` per head, concat 4×32 → out-proj 128→128 |
| feed-forward | 128 → **2048** → 128, ReLU |
| norm | post-norm (`norm_first=False`), residual around both sublayers, dropout 0.1 |

Q, K and V are **all** linear projections of the same station tokens — this is ordinary
self-attention, in which every station attends to every other station. The only two
differences from a textbook encoder block are:

1. **No positional encoding.** Station order is meaningless, so omitting it makes every
   layer permutation-*equivariant*.
2. **A key-padding mask**, because events have different station counts and batches are
   padded to the batch maximum (`collate_fn` does dynamic padding).

Note this is *not* the ISAB/PMA machinery of Lee et al.'s Set Transformer: with S ≤ 50
stations, full O(S²) attention is cheap, so the plain encoder is used.

## 3. Attention pooling — not a QKV block

The aggregator outputs a *set*; regression needs one vector. Instead of a CLS token or
mean pooling, Sensoformer scores each station with a small MLP and takes a weighted mean:

```
score_i = w₂ᵀ tanh(W₁ h_i)         W₁: 128→64,  w₂: 64→1
a_i     = softmax(score_i)          over valid stations (padding → −1e9)
z       = Σ_i a_i · h_i             ∈ ℝ¹²⁸
```

Compared with the self-attention above there is **no K or V projection** (the values are
the tokens themselves), **no multi-head split**, and **no √d scaling**; the "query" is a
single *learned constant* `w₂`. This is additive (Bahdanau-style) attention, i.e. the
attention-based multiple-instance-learning pooling of Ilse et al. (2018). Its role is the
equivariance → **invariance** step: the output no longer depends on station order.

The returned `a_i` are interpretable and are what the paper's XAI analysis uses — they
reveal that the model up-weights azimuthally under-sampled stations, recovering an
optimal-experimental-design behaviour from data alone.

## 4. Heads and target parameterization

| Head | Shape | Output |
| :--- | :--- | :--- |
| magnitude | 128 → 32 → 1, ReLU + dropout, `tanh` | Mw scaled to [−1, 1] over [2, 8] |
| moment tensor | 128 → 64 → 5, ReLU + dropout, `tanh` | normalized deviatoric Mxx, Myy, Mxy, Mxz, Myz |

`Mzz = −(Mxx + Myy)` is implied by the zero-trace constraint, so 5 components fully
determine the deviatoric tensor. Denormalize magnitude with `Mw = (y + 1)/2 · 6 + 2`.

## 5. Parameter budget

| Block | Parameters |
| :--- | ---: |
| Station encoder (2 CNN towers + scalar MLP + fusion) | 262,880 |
| Transformer aggregator (3 layers) | 1,779,072 |
| Attention pooling | 8,321 |
| Regression heads | 12,742 |
| **Total** | **2,063,015** |

## 6. Variants shipped in this repo

All are selected with `model=<name>` on the `scripts/train.py` command line and
implement the same `forward(waveforms, features, mask) → (predictions, attention)`
interface, so they are interchangeable in training and inference.

| Config | Class | What changes |
| :--- | :--- | :--- |
| `sensoformer` | `Sensoformer` | the full model above |
| `ablation_deepsets` | `Sensoformer` | aggregator → `nn.Identity`: no cross-station interaction |
| `ablation_no_scalar` | `Sensoformer` | drops the 20 scalar features |
| `mpnn` | `GNNSensoformer` | GCN message passing over a 5-nearest-neighbour station graph |
| `mpnn_gat` | `GNNSensoformer` | same graph, GAT (learned attention) message passing |
| `deeponet` | `DeepONet` | branch/trunk neural operator |
| `linear_baseline` | `LinearBaseline` | one linear map on masked-mean-pooled raw inputs |
| `mlp_baseline` | `MLPBaseline` | 256→128 MLP on the same pooled inputs |
| `sensoformer_geo` | `Sensoformer` | adds a continuous relative-geometry attention bias (research variant; see note) |

Two optional research components also live in `models/`:

- **`GeometryAttentionBias`** (`network.py`, enabled by `geometry_bias: true`) adds a
  learned per-head additive bias `b_h(g_i, g_j)` computed from pairwise relative
  geometry. It is **off by default** and, when off, introduces no state-dict keys, so
  released checkpoints load with `strict=True`. In controlled experiments it did not
  change accuracy significantly; it is shipped for reproducibility, not as a recommendation.
- **`ConditionalPosteriorFlow`** (`posterior_flow.py`) is a conditional normalizing flow
  over the 6-d target, conditioned on the pooled embedding from
  `Sensoformer.get_embedding()`. It turns the point estimate into a calibrated posterior;
  see the uncertainty section of [RESULTS.md](RESULTS.md).
