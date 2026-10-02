# Sensoformer Data Pipeline

End-to-end documentation of how model-ready HDF5 datasets are produced, for
both the synthetic (PSDR pre-training) domain and the real-catalog
(fine-tuning / evaluation) domain. All scripts live in
`scripts/preprocessing/` and are parameterized — new regions, catalogs, or
magnitude ranges require only different arguments, not new code.

```
SYNTHETIC (PSDR pre-training)                REAL (fine-tuning / evaluation)
─────────────────────────────                ───────────────────────────────
Green's function library                     Per-event SAC directories
(1-D velocity models on a grid)              ({evid}.{NET}.{STA}.{CH}.sac,
        │                                     picked t1/t2 arrivals in headers)
        ▼                                            │
[1] generate_synthetic_events.py                     │
    random MT sources on real                        │
    station geometries → per-event                   │
    .z/.r/.t SAC folders                             │
        │                                            │
        ▼                                            ▼
[2] preprocess_synthetic_hdf5.py             [3] preprocess_real_hdf5.py
    bandpass 0.1–2 Hz, coda injection,           YSH catalog parsing, filters
    P/S amp randomization, windows,              (mag / grade / exclusion),
    spectra, 20-D features, HDF5                 bandpass 0.2–2 Hz, rotation,
        │                                        windows, spectra, 20-D
        │                                        features, HDF5
        ▼                                            ▼
   syn_*.hdf5  ──────── training ────────► socal_*.hdf5
   (configs/data/synthetic.yaml)           (configs/data/real_socal.yaml)
```

The remaining PSDR components (variable-geometry station masking, additional
noise scaling) are applied **at training time** by `SeismicDataset`
(`src/sensoformer/data/dataset.py`, `aug_params`), not in the HDF5 files.

---

## [0] Raw data acquisition — `scripts/data_acquisition/`

The real-data SAC archives are fetched from SCEDC with the STP client
(https://scedc.caltech.edu/data/stp/). Four standardized steps:

| Step | Script | Output |
| :--- | :--- | :--- |
| 1. Query the event catalog | `01_query_events.sh [MINMAG MAXMAG T0 T1 ...]` | `events_cleaned.dat` |
| 2. Fetch phase picks + waveforms | `02_fetch_event_data.sh [list] [radius] [channels]` | `{evid}.dat` phase files + `{evid}/` SAC dirs |
| 3. Insert analyst picks into SAC headers | `03_add_picks_to_sac.sh {evid}.dat ...` | `t1`/`t2` headers set |
| 4. Merge catalog with YSH mechanisms | `merge_catalog_mechanisms.py` | `events_wmeca.dat` |

Requirements: `stp`, `sac`, `gawk` on PATH; network access to SCEDC for
steps 1–2. Step 4 substitutes the **auxiliary nodal plane** whenever the
catalog rake falls outside [−90°, 90°] (the moment tensor is invariant under
this exchange — verified by `tests/test_acquisition_agreement.py`), so all
stored rakes follow the convention assumed downstream.

### What the raw SAC archive must contain

```
{data_root}/
    {evid}.dat                           # STP PHASE output (picks per station)
    {evid}/
        {evid}.{NET}.{STA}.{CHA}.sac     # e.g. 10000605.CI.BBS.HHZ.sac
```

Per station, three components are required (E/N/Z; the preprocessing rotates
horizontals to R/T using the source azimuth). Required SAC headers, all
written automatically by STP TRIG:

| Header | Meaning | Notes |
| :--- | :--- | :--- |
| `b`, `e`, `delta` | trace begin/end time (s, rel. origin), sample interval | |
| `dist`, `az` | epicentral distance (km), source→station azimuth (°) | |
| `evla`, `evlo`, `evdp` | event latitude, longitude, depth | |
| `stla`, `stlo` | station latitude, longitude | |
| `t1` | analyst P pick (s rel. origin), vertical files | via step 3 |
| `t2` | analyst S pick, horizontal files | via step 3 |
| `a` | automatic first-arrival marker | fallback when `t1`/`t2` absent |

Pick fallbacks in the preprocessing: P uses `t1`, else `a`; S uses `t2`,
else the horizontals' `a`, else 1.75 × t1. Archives with only automatic `a`
markers (e.g. the M2+ archive) therefore work without step 3, at the cost of
analyst-quality picks.

Agreement tests: `tests/test_acquisition_agreement.py` verifies on the
existing archives that stored picks agree with the phase files (same time
reference) and that the auxiliary-plane conversion preserves the moment
tensor; `tests/test_preprocessing_agreement.py` verifies that the
preprocessing scripts reproduce the released training HDF5 contents —
byte-exact for the deterministic real pipeline, exact geometry/metadata plus
strongly correlated waveforms for the stochastic synthetic pipeline.

---

## [1] Synthetic event generation — `generate_synthetic_events.py`

Generates raw synthetic waveforms for random moment-tensor sources placed on
**real** station geometries (drawn from a template HDF5 of real events), with
the PSDR physics randomization applied at the source:

| PSDR module | Where it enters |
| :--- | :--- |
| Generative environment (φ) | Green's functions from the velocity-model gridpoint nearest each template's epicenter |
| Signal distortion (𝒯), part 1 | P-arrival time × U(0.92, 1.08); epicenter jitter ±0.2° |
| Realistic noise (**n**) | Pre-event noise of the *same real station* superimposed, clamped to [0.03, 0.3] × signal |

Source sampling: magnitude ~ Γ(2.5, 1.0) clipped to [2.8, 7.0]; strike ~
U(0°, 360°); dip ~ U(0°, 90°); rake ~ U(−90°, 90°).

Requirements: `syn` binary (Zhu & Rivera FK package) on PATH; a Green's
function library `{gf_root}/{lat}_{lon}/vmodel_{depth}/{dist}.grn.0`; a
real-geometry template HDF5 and a matching noise HDF5.

```bash
python scripts/preprocessing/generate_synthetic_events.py \
    --template-hdf5 <real_geometry_templates.hdf5> \
    --noise-hdf5 <real_noise.hdf5> \
    --gf-root <gf_library>/ \
    --out-root <scratch>/syn_events \
    --nevents 100000 --workers 60
```

Output: one directory per event, `ev_{strike}_{dip}_{rake}_{depth}_{mag}/`,
containing per-station `.z/.r/.t` SAC files. All source parameters are
recoverable from the directory name.

## [2] Synthetic HDF5 preprocessing — `preprocess_synthetic_hdf5.py`

Faithful port of the script that produced the released pre-training file
(`syn_mt_data_realgeom_realvn_10w_ps_wcoda_wlola.hdf5`). Adds the remaining
signal-distortion randomization (synthetic coda at P and S; random P/S
amplitude factors α_P ~ U(0.5, 2), α_S ~ U(1, 1.5)·α_P), then windows
(5 s pre / 5 s post each arrival, dt = 0.1 s → 101 samples), amplitude
spectra, the 20-D scalar feature vector, and event-level normalization.
Event parameters are parsed from directory names (no metafile needed;
`--metafile` overrides).

```bash
python scripts/preprocessing/preprocess_synthetic_hdf5.py \
    --data-root <scratch>/syn_events \
    --output <hdf5_dir>/syn_mt_data_new.hdf5 --workers 32
```

## [3] Real-data HDF5 preprocessing — `preprocess_real_hdf5.py`

Faithful port of the script that produced the released fine-tuning file
(`socal_mxyz_data_rtz_lp2_ampr_ps_wlola.hdf5`). Parses a YSH-format catalog
directly and filters by magnitude, mechanism quality grade, data
availability, and (optionally) exclusion of event IDs already present in
another HDF5 — use this to guarantee evaluation sets disjoint from training.

```bash
python scripts/preprocessing/preprocess_real_hdf5.py \
    --catalog <catalog>/ysh_all.log \
    --data-root <catalog_data_root> \
    --output <hdf5_dir>/socal_new.hdf5 \
    --min-mag 3.0 --grades AB --max-events 5000 \
    --exclude-hdf5 <existing_training_file.hdf5>
```

---

## HDF5 schema (both domains)

```
/{event_id}/
    waveforms       (S, 12, 101) float32, gzip   # per station:
                                                 #   ch 0-2  P window  Z/R/T
                                                 #   ch 3-5  S window  Z/R/T
                                                 #   ch 6-8  P spectra Z/R/T
                                                 #   ch 9-11 S spectra Z/R/T
    features        (S, 20) float32, gzip       # [dist, az, stlo, stla, depth,
                                                 #  3 P amps, 3 S amps,
                                                 #  3 P-spec max, 3 S-spec max,
                                                 #  3 log P/S ratios]
    station_names   (S,) vlen str, gzip
    attrs: magnitude, Mxx, Myy, Mxy, Mxz, Myz, depth
           (+ real data: evlo, evla, strike, dip, rake, grade, date)
```

Waveform normalization: time-domain channels (0–5) divided by the event max
over those channels; spectral channels (6–11) likewise. Magnitude targets are
scaled to [−1, 1] assuming a [2, 8] range at load time (`SeismicDataset`).

## Processing-parameter differences between the two domains (intentional)

| Parameter | Synthetic | Real |
| :--- | :--- | :--- |
| Bandpass | 0.1–2.0 Hz | 0.2–2.0 Hz |
| Component source | FK synthesis outputs Z/R/T directly | N/E rotated to R/T using the source azimuth |
| Coda | Injected synthetically (PSDR) | Present naturally |
| P/S amplitude scaling | Random (PSDR) | None |
| S-arrival fallback | t2 = 1.75·t1 if unset | t2 = 1.75·t1 if t2 ≤ 1.3·t1 |
| Amplitude cap | — | Stations with raw max ≥ 5 discarded |

## Consistency guarantees

These ports preserve the original per-station processing operation-for-
operation. When building new datasets for fine-tuning or evaluation of the
released checkpoints, do not change the constants at the top of each script
(windows, bandpass, dt, feature order) — the model's input distribution
depends on them. New *selection* behavior (region, magnitude range, quality,
recency) is safely controlled by the CLI arguments.
