"""
Asset resolution for Sensoformer: pretrained weights, datasets, catalogs.

Assets are too large for git, so they live on the Hugging Face Hub and are
fetched on demand (and cached) by this module. Everything here also accepts
plain local paths, so the package works fully offline once assets are present.

Typical use
-----------
    from sensoformer import load_pretrained
    model = load_pretrained("sensoformer-v3-finetuned", device="cuda")

Environment variables
---------------------
SENSOFORMER_HF_REPO       model repo id      (default: jiazhe868/sensoformer)
SENSOFORMER_HF_DATA_REPO  dataset repo id    (default: jiazhe868/sensoformer-data)
SENSOFORMER_DATA          local dataset dir  (default: ./data)
SENSOFORMER_CACHE         weight cache dir   (default: HF cache)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

HF_REPO = os.environ.get("SENSOFORMER_HF_REPO", "jiazhe868/sensoformer")
HF_DATA_REPO = os.environ.get("SENSOFORMER_HF_DATA_REPO", "jiazhe868/sensoformer-data")
LOCAL_DATA_DIR = Path(os.environ.get("SENSOFORMER_DATA", "./data"))

# Architecture of the released models. `name` selects the class in build_model().
_SENSOFORMER_ARCH: Dict[str, Any] = {
    "name": "sensoformer",
    "in_channels": 6,
    "num_scalar_features": 20,
    "embed_dim": 128,
    "heads": 4,
    "layers": 3,
    "dropout": 0.1,
    "feedforward_dim": 2048,
}

# ---------------------------------------------------------------------------
# Registry: friendly name -> released checkpoint
# ---------------------------------------------------------------------------
CHECKPOINTS: Dict[str, Dict[str, Any]] = {
    "sensoformer-v3-finetuned": {
        "filename": "sensoformer_v3_finetuned.pth",
        "arch": _SENSOFORMER_ARCH,
        "stage": "real-world fine-tuned (PSDR pre-train -> SoCal fine-tune)",
        "use_for": "inference on real data (the paper's headline model)",
    },
    "sensoformer-v3-pretrained": {
        "filename": "sensoformer_v3_psdr_pretrained.pth",
        "arch": _SENSOFORMER_ARCH,
        "stage": "PSDR synthetic pre-trained",
        "use_for": "starting point for fine-tuning on a new catalog/region",
    },
}

DATASETS: Dict[str, Dict[str, Any]] = {
    "socal-real": {
        "filename": "socal_mxyz_data_rtz_lp2_ampr_ps_wlola.hdf5",
        "about": "2,435 real SoCal events (M>=3.0) with analyst mechanisms; "
                 "the fine-tuning / evaluation set",
    },
    "synthetic-psdr": {
        "filename": "syn_mt_data_realgeom_realvn_10w_ps_wcoda_wlola.hdf5",
        "about": "~100k PSDR synthetic events on real station geometries; "
                 "the pre-training set",
    },
    "synthetic-clean": {
        "filename": "syn_mt_data_noaug.hdf5",
        "about": "~50k clean synthetics (no PSDR); the no-randomization "
                 "baseline pre-training set",
    },
}


# Third-party catalogs redistributed for reproducibility. These are *labels*,
# not waveforms: small plain-text files, each with its own citation duty.
CATALOGS: Dict[str, Dict[str, Any]] = {
    "yhs-socal": {
        "filename": "ysh_all.log",
        "about": "Yang-Hauksson-Shearer focal mechanisms for southern "
                 "California, 1981-2024 (280,889 events); the label source "
                 "for the real-data pipeline",
        "source": "SCEDC, https://scedc.caltech.edu/data/alt-2011-yang-hauksson-shearer.html",
        "cite": "SCEDC (2013), Southern California Earthquake Data Center, "
                "Caltech, doi:10.7909/C3WD3xH1; Yang, Hauksson & Shearer "
                "(2012), BSSA 102(3), 1179-1194, doi:10.1785/0120110311; "
                "Hauksson, Yang & Shearer (2012), BSSA 102(5), 2239-2244, "
                "doi:10.1785/0120120010",
    },
}


def _hf_download(repo_id: str, filename: str, repo_type: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ImportError(
            "huggingface_hub is required to fetch assets. Install it with\n"
            "    pip install huggingface_hub\n"
            "or pass an explicit local path instead of a registry name."
        ) from exc
    cache_dir = os.environ.get("SENSOFORMER_CACHE") or None
    try:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename,
                                    repo_type=repo_type, cache_dir=cache_dir))
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not fetch '{filename}' from the {repo_type} repo "
            f"'{repo_id}'.\n"
            f"  - If the assets have not been published yet, see "
            f"docs/HUGGINGFACE.md (scripts/upload_assets_to_hf.py).\n"
            f"  - If you already have the file locally, pass its path "
            f"directly instead of the registry name.\n"
            f"  - For a private repo, run `hf auth login` first.\n"
            f"Underlying error: {exc}"
        ) from exc


def resolve_checkpoint(name_or_path: str) -> Tuple[Path, Optional[Dict[str, Any]]]:
    """Return (local path, registry entry or None).

    `name_or_path` may be a registry key (downloaded from the Hub if needed)
    or any local file path.
    """
    p = Path(name_or_path).expanduser()
    if p.exists():
        return p, CHECKPOINTS.get(name_or_path)
    if name_or_path in CHECKPOINTS:
        entry = CHECKPOINTS[name_or_path]
        return _hf_download(HF_REPO, entry["filename"], "model"), entry
    raise FileNotFoundError(
        f"'{name_or_path}' is neither an existing file nor a known checkpoint.\n"
        f"Known checkpoints: {', '.join(CHECKPOINTS)}"
    )


def resolve_dataset(name_or_path: str) -> Path:
    """Return a local path for a dataset name, local file, or $SENSOFORMER_DATA entry."""
    p = Path(name_or_path).expanduser()
    if p.exists():
        return p
    if name_or_path in DATASETS:
        fname = DATASETS[name_or_path]["filename"]
        local = LOCAL_DATA_DIR / fname
        if local.exists():
            return local
        return _hf_download(HF_DATA_REPO, fname, "dataset")
    raise FileNotFoundError(
        f"'{name_or_path}' is neither an existing file nor a known dataset.\n"
        f"Known datasets: {', '.join(DATASETS)}"
    )


def resolve_catalog(name_or_path: str = "yhs-socal") -> Path:
    """Return a local path for a catalog name, local file, or $SENSOFORMER_DATA entry.

    Catalogs are third-party products redistributed here for reproducibility;
    CATALOGS[name]["cite"] states the attribution each one requires.
    """
    p = Path(name_or_path).expanduser()
    if p.exists():
        return p
    if name_or_path in CATALOGS:
        fname = CATALOGS[name_or_path]["filename"]
        local = LOCAL_DATA_DIR / fname
        if local.exists():
            return local
        return _hf_download(HF_DATA_REPO, fname, "dataset")
    raise FileNotFoundError(
        f"'{name_or_path}' is neither an existing file nor a known catalog.\n"
        f"Known catalogs: {', '.join(CATALOGS)}"
    )


def build_model(arch: Dict[str, Any]):
    """Instantiate a model from an architecture dict (see _SENSOFORMER_ARCH)."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"model": dict(arch)})
    name = str(arch.get("name", "sensoformer")).lower()
    if name == "gnn":
        from .models.gnn import GNNSensoformer
        return GNNSensoformer(cfg)
    if name == "deeponet":
        from .models.deeponet import DeepONet
        return DeepONet(cfg)
    if name == "linear_baseline":
        from .models.simple_baselines import LinearBaseline
        return LinearBaseline(cfg)
    if name == "mlp_baseline":
        from .models.simple_baselines import MLPBaseline
        return MLPBaseline(cfg)
    from .models.network import Sensoformer
    return Sensoformer(cfg)


def load_pretrained(name_or_path: str = "sensoformer-v3-finetuned",
                    device: str = "cpu",
                    arch: Optional[Dict[str, Any]] = None,
                    strict: bool = True):
    """Load a ready-to-use model.

    Accepts both self-describing checkpoints (dicts carrying 'state_dict' and
    'arch', as written by scripts/package_checkpoints.py) and bare state dicts
    from older training runs. `DataParallel` 'module.' prefixes are stripped.

    Returns the model in eval() mode on `device`.
    """
    import torch

    path, entry = resolve_checkpoint(name_or_path)
    blob = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(blob, dict) and "state_dict" in blob:
        state = blob["state_dict"]
        arch = arch or blob.get("arch")
    else:
        state = blob
    if arch is None:
        arch = (entry or {}).get("arch") or _SENSOFORMER_ARCH
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}

    model = build_model(arch)
    result = model.load_state_dict(state, strict=strict)
    if result.missing_keys or result.unexpected_keys:
        import logging
        logging.getLogger(__name__).warning(
            "load_state_dict: %d missing, %d unexpected keys (strict=%s)",
            len(result.missing_keys), len(result.unexpected_keys), strict)
    model.to(device).eval()
    return model
