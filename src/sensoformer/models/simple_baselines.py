"""
Trivial reference baselines: linear regression and a small MLP on
masked-mean-pooled raw inputs.

Both share the Sensoformer interface (forward(waveforms, features, mask) ->
(predictions, attn_weights)) so they run through the identical two-stage
training pipeline. Per-station input = flattened 12x101 waveform block (1212)
concatenated with the 20 scalar features (1232-D total); station dimension is
removed by masked MEAN pooling (permutation invariant), so neither model has
any cross-station interaction, attention, or convolutional structure.

  - LinearBaseline: one linear map, pooled input -> 6 targets. A true linear
    regression (no output squashing).
  - MLPBaseline: pooled input -> MLP (hidden 256, 128) -> 6 targets, tanh
    output (matching the target normalization, like Sensoformer's heads).
"""
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from omegaconf import DictConfig, OmegaConf


def _pool_inputs(waveforms: Tensor, features: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
    """(B,S,12,L) + (B,S,F) + (B,S) -> masked-mean pooled (B, 12*L+F)."""
    B, S = mask.shape
    x = torch.cat([waveforms.flatten(2), features], dim=-1)   # (B, S, D)
    m = (mask > 0).float().unsqueeze(-1)                      # (B, S, 1)
    pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    attn = m.squeeze(-1) / m.squeeze(-1).sum(dim=1, keepdim=True).clamp(min=1.0)
    return pooled, attn


class LinearBaseline(nn.Module):
    def __init__(self, cfg: Union[DictConfig, Dict[str, Any]]):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        m = cfg.model
        in_dim = 12 * int(m.get("waveform_len", 101)) + int(m.num_scalar_features)
        self.head = nn.Linear(in_dim, 6)

    def forward(self, waveforms: Tensor, features: Tensor, mask: Tensor):
        pooled, attn = _pool_inputs(waveforms, features, mask)
        return self.head(pooled), attn


class MLPBaseline(nn.Module):
    def __init__(self, cfg: Union[DictConfig, Dict[str, Any]]):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        m = cfg.model
        in_dim = 12 * int(m.get("waveform_len", 101)) + int(m.num_scalar_features)
        hidden = int(m.get("hidden_dim", 256))
        dropout = float(m.get("dropout", 0.1))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 6), nn.Tanh(),
        )

    def forward(self, waveforms: Tensor, features: Tensor, mask: Tensor):
        pooled, attn = _pool_inputs(waveforms, features, mask)
        return self.net(pooled), attn
