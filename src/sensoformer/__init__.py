"""
Sensoformer: set-attention inference of earthquake source parameters from
variable-geometry seismic networks.

Quick start
-----------
    from sensoformer import load_pretrained, SeismicDataset, collate_fn
    model = load_pretrained("sensoformer-v3-finetuned", device="cuda")

See docs/QUICKSTART.md for the end-to-end inference and fine-tuning paths.
"""
__version__ = "1.0.0"

from .hub import (  # noqa: F401
    CHECKPOINTS,
    DATASETS,
    build_model,
    load_pretrained,
    resolve_checkpoint,
    resolve_dataset,
)

__all__ = [
    "__version__",
    "load_pretrained",
    "build_model",
    "resolve_checkpoint",
    "resolve_dataset",
    "CHECKPOINTS",
    "DATASETS",
]


def __getattr__(name):  # lazy re-exports: keep `import sensoformer` torch-free
    if name == "Sensoformer":
        from .models.network import Sensoformer
        return Sensoformer
    if name in ("SeismicDataset", "collate_fn"):
        from . import data
        return getattr(data.dataset, name) if hasattr(data, "dataset") else \
            getattr(__import__("sensoformer.data.dataset", fromlist=[name]), name)
    raise AttributeError(f"module 'sensoformer' has no attribute '{name}'")
