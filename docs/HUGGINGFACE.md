# Asset hosting on the Hugging Face Hub

Model weights are small (~8 MB) but the datasets are 0.26–11.4 GB, which does not belong
in git. Both live on the Hugging Face Hub and are fetched on demand into the standard HF
cache.

| Repo | Type | Contents |
| :--- | :--- | :--- |
| [`jiazhe868/sensoformer`](https://huggingface.co/jiazhe868/sensoformer) | model | pretrained checkpoints + model card |
| [`jiazhe868/sensoformer-data`](https://huggingface.co/datasets/jiazhe868/sensoformer-data) | dataset | preprocessed HDF5 datasets, the YHS label catalog (`ysh_all.log`) + dataset card |

Override either with `SENSOFORMER_HF_REPO` / `SENSOFORMER_HF_DATA_REPO` if you host your
own copies.

## Using assets (everyone)

```bash
python scripts/download_assets.py --list                        # what exists
python scripts/download_assets.py --weights                     # both checkpoints, ~16 MB
python scripts/download_assets.py --datasets socal-real         # 0.26 GB
python scripts/download_assets.py --datasets synthetic-psdr --symlink   # 11.4 GB, no copy
python scripts/download_assets.py --catalogs yhs-socal          # 31 MB label catalog
```

Weights land in the HF cache and are resolved automatically by name. Datasets are placed
in `$SENSOFORMER_DATA` (default `./data`), which the Hydra data configs resolve, so
`data=real_socal` works with no further configuration. `--symlink` avoids a second copy
of multi-GB files.

Every entry point also accepts plain paths, so nothing here is mandatory:

```bash
python scripts/predict.py --input /my/file.hdf5 --checkpoint /my/weights.pth
```

For private repos, authenticate first: `hf auth login` (on huggingface_hub < 1.0 the command is `huggingface-cli login`).

## Publishing assets (maintainers)

### 1. Package the checkpoints

Training writes bare `state_dict` files. Wrap them with the architecture and provenance
so that `load_pretrained()` needs no external config:

```bash
python scripts/package_checkpoints.py \
    --finetuned  /path/to/best_finetuned_model_v3.pth \
    --pretrained /path/to/best_sensoformer_model_v3.pth \
    --out-dir release_assets/
```

This writes `sensoformer_v3_finetuned.pth` and `sensoformer_v3_psdr_pretrained.pth`, each
containing `{state_dict, arch, sensoformer_version, stage, metrics, ...}`, and verifies
that both load back through the public API.

### 2. Dry-run the upload

```bash
python scripts/upload_assets_to_hf.py --weights-dir release_assets --dry-run
```

Prints exactly which repos would be created and which files uploaded, with sizes.

### 3. Upload

```bash
hf auth login                             # or export HF_TOKEN=hf_...
pip install hf_transfer                   # much faster for multi-GB files
export HF_HUB_ENABLE_HF_TRANSFER=1

# weights + model card
python scripts/upload_assets_to_hf.py --weights-dir release_assets

# datasets (resumable; safe to re-run)
python scripts/upload_assets_to_hf.py \
    --dataset socal-real=/path/socal_mxyz_data_rtz_lp2_ampr_ps_wlola.hdf5 \
    --dataset synthetic-psdr=/path/syn_mt_data_realgeom_realvn_10w_ps_wcoda_wlola.hdf5 \
    --dataset synthetic-clean=/path/syn_mt_data_noaug.hdf5

# the third-party label catalog (mirrored for reproducibility; cite the source)
python scripts/upload_assets_to_hf.py --catalog yhs-socal=/path/ysh_all.log
```

Uploads go over HTTP through `huggingface_hub` — **git-lfs is not required**. The cards in
`hf/MODEL_CARD.md` and `hf/DATASET_CARD.md` are uploaded as each repo's `README.md`.

### 4. Verify as a fresh user would

```bash
python - <<'PY'
from sensoformer import load_pretrained
m = load_pretrained("sensoformer-v3-finetuned")
print(sum(p.numel() for p in m.parameters()), "parameters loaded from the Hub")
PY
python scripts/predict.py --input socal-real --limit 300 --out-dir /tmp/hub_check
# expect kagan_median ~17-20, magnitude_mae ~0.10
```

### Adding a new checkpoint or dataset

Add an entry to `CHECKPOINTS` / `DATASETS` in `src/sensoformer/hub.py` (filename, arch,
description), then package and upload as above. `tests/test_hub.py` checks that registry
entries are well formed.
