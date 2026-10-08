#!/usr/bin/env python3
"""
Publish Sensoformer assets to the Hugging Face Hub (run by the maintainer).

Requires an HF token with write access:
    pip install huggingface_hub
    hf auth login                    # or export HF_TOKEN=hf_...
                                     # (older hub versions: huggingface-cli login)

Weights go to a *model* repo, datasets to a *dataset* repo. Large files are
uploaded over HTTP (no git-lfs needed); `pip install hf_transfer` and
`export HF_HUB_ENABLE_HF_TRANSFER=1` makes multi-GB uploads much faster.

Examples
--------
# dry run first: shows exactly what would be created and uploaded
python scripts/upload_assets_to_hf.py --weights-dir release_assets --dry-run

# publish the packaged weights + the model card
python scripts/upload_assets_to_hf.py --weights-dir release_assets

# publish datasets (large; resumable - safe to re-run)
python scripts/upload_assets_to_hf.py \\
    --dataset socal-real=/path/socal_mxyz_data_rtz_lp2_ampr_ps_wlola.hdf5 \\
    --dataset synthetic-psdr=/path/syn_mt_data_realgeom_realvn_10w_ps_wcoda_wlola.hdf5

# publish the third-party label catalog (small text file)
python scripts/upload_assets_to_hf.py --catalog yhs-socal=/path/ysh_all.log
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sensoformer.hub import (CATALOGS, CHECKPOINTS, DATASETS, HF_DATA_REPO,
                             HF_REPO)

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights-dir", type=Path, default=None,
                   help="directory of packaged checkpoints "
                        "(scripts/package_checkpoints.py output)")
    p.add_argument("--dataset", action="append", default=[], metavar="NAME=PATH",
                   help="dataset to upload, e.g. socal-real=/path/file.hdf5 "
                        "(repeatable)")
    p.add_argument("--catalog", action="append", default=[], metavar="NAME=PATH",
                   help="label catalog to upload, e.g. yhs-socal=/path/ysh_all.log "
                        "(repeatable)")
    p.add_argument("--model-repo", default=HF_REPO)
    p.add_argument("--dataset-repo", default=HF_DATA_REPO)
    p.add_argument("--private", action="store_true",
                   help="create the repos as private")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.weights_dir and not args.dataset and not args.catalog:
        p.error("give --weights-dir, --dataset NAME=PATH and/or --catalog NAME=PATH")

    plan = []   # (repo_id, repo_type, local_path, path_in_repo)

    if args.weights_dir:
        for name, entry in CHECKPOINTS.items():
            f = args.weights_dir / entry["filename"]
            if f.exists():
                plan.append((args.model_repo, "model", f, entry["filename"]))
            else:
                print(f"! missing (skipped): {f}")
        card = REPO_ROOT / "hf" / "MODEL_CARD.md"
        if card.exists():
            plan.append((args.model_repo, "model", card, "README.md"))

    for flag, specs, registry in (("--dataset", args.dataset, DATASETS),
                                  ("--catalog", args.catalog, CATALOGS)):
        for spec in specs:
            if "=" not in spec:
                p.error(f"{flag} expects NAME=PATH, got {spec!r}")
            name, path = spec.split("=", 1)
            if name not in registry:
                p.error(f"unknown {flag[2:]} {name!r}; "
                        f"known: {', '.join(registry)}")
            src = Path(path).expanduser()
            if not src.exists():
                p.error(f"file not found: {src}")
            plan.append((args.dataset_repo, "dataset", src,
                         registry[name]["filename"]))
    if args.dataset or args.catalog:
        card = REPO_ROOT / "hf" / "DATASET_CARD.md"
        if card.exists():
            plan.append((args.dataset_repo, "dataset", card, "README.md"))

    total = sum(f.stat().st_size for _, _, f, _ in plan)
    print(f"\nUpload plan ({len(plan)} files, {total/2**30:.2f} GB total):")
    for repo, rtype, f, dest in plan:
        print(f"  {rtype:<8} {repo:<34} {dest:<44} "
              f"{f.stat().st_size/2**20:9.1f} MB")
    if args.dry_run:
        print("\n--dry-run: nothing uploaded.")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    who = api.whoami()
    print(f"\nAuthenticated as: {who.get('name', '?')}")

    for repo_id, repo_type in {(r, t) for r, t, _, _ in plan}:
        api.create_repo(repo_id=repo_id, repo_type=repo_type,
                        private=args.private, exist_ok=True)
        print(f"repo ready: {repo_type}/{repo_id}")

    for repo_id, repo_type, f, dest in plan:
        print(f"uploading {dest} ({f.stat().st_size/2**20:.1f} MB) ...", flush=True)
        api.upload_file(path_or_fileobj=str(f), path_in_repo=dest,
                        repo_id=repo_id, repo_type=repo_type,
                        commit_message=f"Add {dest}")
    print("\nDone. Verify with:  python scripts/download_assets.py --list")


if __name__ == "__main__":
    main()
