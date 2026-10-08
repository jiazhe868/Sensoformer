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

Steps 1–3 fetch waveforms; the focal mechanisms themselves come from a separate
catalog file, which is fetched automatically from the Hugging Face dataset repo
— see **The YHS focal-mechanism catalog** just below.

Requirements: `stp`, `sac`, `gawk` on PATH; network access to SCEDC for
steps 1–2. Step 4 substitutes the **auxiliary nodal plane** whenever the
catalog rake falls outside [−90°, 90°] (the moment tensor is invariant under
this exchange — verified by `tests/test_acquisition_agreement.py`), so all
stored rakes follow the convention assumed downstream.

### The YHS focal-mechanism catalog (`ysh_all.log`)

Several steps below take a `--catalog` file, referred to throughout as
`ysh_all.log`. This is the **Yang–Hauksson–Shearer (YHS) focal-mechanism catalog
for Southern California**, produced with the HASH first-motion method and
distributed by the Southern California Earthquake Data Center (SCEDC). It
supplies every *label* in the real-data pipeline.

It is a 31 MB text file — too large to keep in git, but small enough to ship
alongside the datasets, so **the exact copy used in this work is mirrored in
the Hugging Face dataset repo** and is fetched automatically the first time you
need it:

```bash
# explicit download (31 MB, cached afterwards)
python scripts/download_assets.py --catalogs yhs-socal

# or just run the pipeline — `--catalog` defaults to the registry name
python scripts/preprocessing/preprocess_real_hdf5.py --data-root ... --output ...
```

In Python:

```python
from sensoformer import resolve_catalog
path = resolve_catalog()            # "yhs-socal" by default
path = resolve_catalog("/my/own/ysh_all.log")   # local paths pass through
```

**This is third-party data.** The mirror exists for reproducibility, not to
replace the source; the authoritative copy and its current terms live at SCEDC:

- Catalog page: <https://scedc.caltech.edu/data/alt-2011-yang-hauksson-shearer.html>
- SCEDC data portal: <https://scedc.caltech.edu/data/>

The published catalog covers 1981–2010 (Yang, Hauksson & Shearer, 2012,
*BSSA* 102(3), 1179–1194, [doi:10.1785/0120110311](https://doi.org/10.1785/0120110311));
SCEDC distributes updated versions extending to the present. The mirrored file
spans 1981-01-01 to 2024-12-31 and holds 280,889 events. Any file with the
column layout below works — the code never assumes a particular time span or
region, so an equivalent catalog for another network can be substituted.

Column names follow SCEDC's format sheet (linked from the catalog page as
*1981-2010 Focal Mechanism Catalog Format Description*). To rebuild the file
yourself from SCEDC's per-year downloads — and for how a fresh download differs
from the mirror in 2024 — see
[README §7.1, Option B](../README.md#option-b--build-it-yourself-from-scedc).

**If you use this catalog, cite it as SCEDC's citation policy asks:** SCEDC
(2013), doi:[10.7909/C3WD3xH1](https://doi.org/10.7909/C3WD3xH1), plus the
references on the catalog page — Yang et al. (2012) above and Hauksson, Yang &
Shearer (2012), *BSSA* 102(5), 2239–2244,
[doi:10.1785/0120120010](https://doi.org/10.1785/0120120010). This applies to
the mirror as well as your own download.

#### Expected column layout

Whitespace-separated, **at least 21 columns**, one earthquake per line:

| Col | Content | Used by this code |
| :---: | :--- | :--- |
| 0–2 | year, month, day | event date / time filtering |
| 3–5 | hour, minute, second | origin time (carried into output catalogs) |
| 6 | **event ID** | matched against the SAC directory name |
| 7–8 | latitude, longitude | maps, output catalogs |
| 9 | depth (km) | scalar feature + HDF5 attribute |
| 10 | **magnitude** | `--min-mag` / `--max-mag` filtering |
| 11–13 | **strike, dip, rake** | converted to the moment-tensor training target |
| 14 | fault-plane uncertainty (deg) | label-uncertainty analysis in [RESULTS.md](RESULTS.md) |
| 15 | auxiliary fault-plane uncertainty (deg) | label-uncertainty analysis in [RESULTS.md](RESULTS.md) |
| 16 | number of P-wave first motions | not used |
| 17 | misfit of first motions | not used |
| 18 | number of S/P amplitude ratios | not used |
| 19 | average log10(S/P amplitude ratio) misfit | not used |
| 20 | **quality grade** (`A`/`B`/`C`/`D`) | `--grades` filtering |

Example line (the one used in the format tests):

```
 2004  4 15  2 28  8.620 10000605  33.94280 -116.99420  15.760  3.350  145  85  -12  20  21   45  0.08   12  1.00 A
```

Grades follow the HASH convention (Hardebeck & Shearer, 2002, *BSSA* 92(6),
2264–2276) and act as a hard cap on the mean fault-plane uncertainty in columns
14–15. Measured directly on the mirrored file:

| Grade | Events | Mean fault-plane uncertainty | Cap |
| :---: | ---: | ---: | ---: |
| A | 24,282 | 19.7° | ≤ 25° |
| B | 57,158 | 28.1° | ≤ 35° |
| C | 86,744 | 36.2° | ≤ 45° |
| D | 112,705 | 43.5° | — |

**A** and **B** are well-constrained solutions, **C** and **D** are not. The
released model was fine-tuned on A/B-grade mechanisms only; the C/D events (71%
of the catalog) are the ones the model supplies mechanisms for in
[RESULTS.md](RESULTS.md).

#### Checking your file before you run anything

```bash
python scripts/data_acquisition/check_catalog_format.py /path/to/ysh_all.log
```

(The mirrored copy already passes this check — it is the file the grade table
above was measured from. Run it on a catalog you supply yourself, or on a newer
download from SCEDC.)

This dissects the first lines column by column, reports the grade distribution,
sanity-checks magnitude and dip ranges, and exits non-zero with a specific
diagnosis if the layout does not match. The preprocessing scripts also fail
loudly rather than silently mis-parsing: a catalog whose column 20 is not a
quality grade raises `CatalogFormatError` instead of attaching wrong mechanisms
to every event.

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
(`socal_mxyz_data_rtz_lp2_ampr_ps_wlola.hdf5`). Requires the YHS catalog file
described in [The YHS focal-mechanism catalog](#the-yhs-focal-mechanism-catalog-ysh_alllog)
(fetched automatically by name). Parses a YSH-format catalog
directly and filters by magnitude, mechanism quality grade, data
availability, and (optionally) exclusion of event IDs already present in
another HDF5 — use this to guarantee evaluation sets disjoint from training.

```bash
python scripts/preprocessing/preprocess_real_hdf5.py \
    --catalog yhs-socal \
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
