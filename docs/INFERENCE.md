# Inference

`scripts/predict.py` runs a trained model over an HDF5 event file and writes per-event
source parameters, metrics (when ground truth is present), an optional catalog file and
optional figures.

## Command

```bash
python scripts/predict.py --input <hdf5|dataset-name> [options]
```

| Option | Default | Meaning |
| :--- | :--- | :--- |
| `--input` | *(required)* | HDF5 path, or a registry name (`socal-real`, `synthetic-psdr`, `synthetic-clean`) |
| `--checkpoint` | `sensoformer-v3-finetuned` | registry name or local `.pth`; downloads on first use |
| `--out-dir` | `results` | output directory |
| `--device` | `cuda` if available | `cuda` / `cpu` |
| `--batch-size` | 128 | — |
| `--num-workers` | 4 | DataLoader workers |
| `--max-stations` | 50 | station cap per event, as in training |
| `--min-stations` | 0 | skip events with fewer usable stations |
| `--limit` | — | only the first N events (smoke tests) |
| `--event-list` | — | text file of event ids, one per line |
| `--catalog` | off | also write `catalog.txt` |
| `--figures` | off | also write scatter / Kagan histogram / beachball figures (needs labels) |

## Outputs

### `predictions.csv`

| Column | Notes |
| :--- | :--- |
| `event_id`, `n_stations` | — |
| `pred_magnitude` | Mw, already denormalized from [−1, 1] |
| `pred_Mxx … pred_Myz` | normalized deviatoric moment tensor (`Mzz = −(Mxx+Myy)`) |
| `pred_strike`, `pred_dip`, `pred_rake` | nodal plane with rake in [−90°, 90°]; requires the Fortran kernel |
| `true_magnitude`, `true_Mxx …` | only if the input has ground truth |
| `magnitude_error`, `kagan_angle` | only if the input has ground truth |

### `metrics.json`

Event count, whether labels were present, and — if so — `magnitude_mae`,
`magnitude_bias`, `kagan_mean`, `kagan_median`, `frac_kagan_lt30`.

### `catalog.txt` (with `--catalog`)

```
# evid lat lon depth mag_cat mag_pred strike dip rake nsta
```

Origin, hypocenter and catalog magnitude are **carried over from the input file's
attributes** — Sensoformer does not locate events. Only the mechanism and magnitude
columns are predictions.

## Python API

```python
import torch
from sensoformer import load_pretrained
from sensoformer.data.dataset import SeismicDataset, collate_fn
from torch.utils.data import DataLoader

model = load_pretrained("sensoformer-v3-finetuned", device="cuda")
ds = SeismicDataset("my_events.hdf5", event_ids, mode="test",
                    augmentation=False, config={"max_stations": 50})
loader = DataLoader(ds, batch_size=128, collate_fn=collate_fn)

with torch.no_grad():
    for wf, ft, mask, target, ids in loader:
        pred, attn = model(wf.cuda(), ft.cuda(), mask.cuda())
        mw = (pred[:, 0] + 1) / 2 * 6 + 2        # denormalize magnitude
        mt = pred[:, 1:]                          # Mxx, Myy, Mxy, Mxz, Myz
```

`attn` holds the per-station attention-pooling weights — useful for interpreting which
stations drove a solution.

Moment tensor → strike/dip/rake and Kagan angles:

```python
from sensoformer.ext import MTDecomposer
from sensoformer.utils.physics import kagan_angle

dec = MTDecomposer()                       # dec.is_available == False -> run `make build`
sdr1, sdr2 = dec.mt_to_sdr(mt[0].cpu().numpy())
plane = sdr1 if -90 <= sdr1[2] <= 90 else sdr2
angle = kagan_angle(*plane_true, *plane_pred)
```

## Reading the numbers

- **Median Kagan ≈ 20°** on real SoCal data is at the analyst-catalog uncertainty level;
  the catalog's own per-solution 1σ nodal-plane uncertainty has a median of ~19.5°. Do
  not expect to do much better against these labels.
- **The error tail is geometry-driven, not random.** Median Kagan falls from ~27° with
  10–15 stations to ~17° with 50, and rises from ~17° at azimuthal gaps < 45° to ~33°
  beyond 180°. Stratify by station count or azimuthal gap before concluding that a model
  underperforms on a new dataset.
- **Below the training magnitude range** (the released model saw only M ≥ 3.0),
  magnitudes floor near Mw ≈ 2.7, producing a roughly constant positive bias (+0.28 at
  M2.5–3.0, +0.71 at M2.0–2.5). Remove the mean bias and the residual MAE returns to the
  in-distribution ~0.10 — relative scaling transfers, only the offset does not. A
  one-parameter recalibration, or a short fine-tune including small events, fixes it.
- **Reported conformal intervals** (90%: ±0.22 magnitude, 44.8° Kagan ball) are
  calibrated for the SoCal catalog population. Recalibrate on a labeled sample before
  quoting them for a new region or network.

Details and the underlying studies: [RESULTS.md](RESULTS.md).
