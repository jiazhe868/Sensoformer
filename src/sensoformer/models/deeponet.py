"""
DeepONet baseline for seismic source inversion.

Reviewer 2 requested a Neural Operator baseline that does not require a
uniform sensor grid, in contrast to FNO. DeepONet naturally handles
arbitrary station distributions because the Branch Net aggregates
observations from any set of sensors into a fixed-size vector.

Architecture
------------
Branch Net
    StationEncoder (reused from Sensoformer) encodes each station's
    waveforms and scalar metadata into an embed_dim-vector.  The
    per-station embeddings are then collapsed into a single event
    representation via masked attention pooling, followed by a small MLP
    that projects to the shared latent dimension p (branch_dim).

Trunk Net
    Each of the 6 output components (Magnitude, Mxx, Myy, Mxy, Mxz, Myz)
    is assigned a learnable categorical embedding of dimension p.  A
    two-layer MLP refines these embeddings into component-specific basis
    functions.

Output
    u_i = branch_out · trunk_out_i + b_i   for i = 0…5
    Passed through Tanh to match the [-1, 1] target normalisation used
    throughout Sensoformer.

Reference
---------
Lu et al. (2021) "Learning nonlinear operators via DeepONet based on the
universal approximation theorem of operators."  Nature Machine Intelligence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Union, Dict, Any
from omegaconf import DictConfig, OmegaConf

from .network import StationEncoder

# [Magnitude, Mxx, Myy, Mxy, Mxz, Myz] – must match Sensoformer output order
NUM_OUTPUTS = 6


class DeepONet(nn.Module):
    """
    DeepONet adapted for seismic source inversion with irregular station networks.

    Args:
        cfg: Hydra DictConfig (or plain dict) with a ``model`` sub-key containing:
            - in_channels (int): channels per wave type fed to each CNN tower (default 6).
            - num_scalar_features (int): scalar station feature dimension (default 20).
            - embed_dim (int): station embedding dimension output by StationEncoder.
            - branch_dim (int, optional): shared latent dimension *p* for the dot
              product; defaults to embed_dim.
            - dropout (float, optional): dropout rate (default 0.1).
    """

    def __init__(self, cfg: Union[DictConfig, Dict[str, Any]]):
        super().__init__()

        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)

        model_cfg = cfg.model

        in_channels = int(model_cfg.in_channels)
        num_scalar_features = int(model_cfg.num_scalar_features)
        self.embed_dim = int(model_cfg.embed_dim)
        self.branch_dim = int(model_cfg.get("branch_dim", self.embed_dim))
        dropout = float(model_cfg.get("dropout", 0.1))

        # ------------------------------------------------------------------ #
        # Branch Net                                                           #
        # ------------------------------------------------------------------ #

        # 1. Per-station encoder (shared with Sensoformer; dual P/S towers)
        self.station_encoder = StationEncoder(
            p_s_wave_in_channels=in_channels,
            num_scalar_features=num_scalar_features,
            feature_dim=self.embed_dim,
        )

        # 2. Attention-based pooling: collapses (B, S, embed_dim) → (B, embed_dim)
        #    Uses a 2-layer MLP to compute a scalar score per station, then
        #    softmax over valid stations (padding is masked with -inf).
        self.branch_attention = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # 3. Branch MLP: projects pooled event embedding to latent dimension p
        self.branch_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.branch_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.branch_dim, self.branch_dim),
        )

        # ------------------------------------------------------------------ #
        # Trunk Net                                                            #
        # ------------------------------------------------------------------ #

        # Learnable categorical embedding for each output component.
        # This is the "query point" in DeepONet terminology: instead of a
        # continuous spatial coordinate, we use a discrete component index
        # (0 = Magnitude, 1 = Mxx, …, 5 = Myz).
        self.trunk_embeddings = nn.Embedding(NUM_OUTPUTS, self.branch_dim)

        # Trunk MLP: maps raw embeddings to component-specific basis functions
        self.trunk_mlp = nn.Sequential(
            nn.Linear(self.branch_dim, self.branch_dim),
            nn.Tanh(),
            nn.Linear(self.branch_dim, self.branch_dim),
        )

        # Per-output bias term (standard in DeepONet to improve expressivity)
        self.output_bias = nn.Parameter(torch.zeros(NUM_OUTPUTS))

        self.final_activation = nn.Tanh()

    # ---------------------------------------------------------------------- #

    def forward(
        self, waveforms: Tensor, features: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            waveforms: ``(B, S, 12, L)`` – multi-channel station waveforms.
            features:  ``(B, S, num_scalar_features)`` – scalar station metadata.
            mask:      ``(B, S)`` – 1.0 for real stations, 0.0 for padding.

        Returns:
            predictions: ``(B, 6)`` – [Magnitude, Mxx, Myy, Mxy, Mxz, Myz],
                         each normalised to ``[-1, 1]`` via Tanh.
            weights:     ``(B, S)`` – attention weights used inside the Branch
                         Net pooling step (ones-of-valid-stations dummy for API
                         compatibility; the actual softmax weights are returned).
        """
        # ---- Branch Net -------------------------------------------------- #
        # Per-station embeddings
        station_emb = self.station_encoder(waveforms, features)  # (B, S, embed_dim)

        # Masked attention pooling
        padding_mask = mask == 0  # True where padded
        attn_logits = self.branch_attention(station_emb).squeeze(-1)  # (B, S)
        attn_logits = attn_logits.masked_fill(padding_mask, -1e9)
        attn_weights = F.softmax(attn_logits, dim=1)  # (B, S)

        event_emb = (station_emb * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, embed_dim)

        # Branch MLP
        branch_out = self.branch_mlp(event_emb)  # (B, branch_dim)

        # ---- Trunk Net --------------------------------------------------- #
        # Evaluate trunk for all 6 output components simultaneously
        component_ids = torch.arange(NUM_OUTPUTS, device=waveforms.device)  # (6,)
        trunk_out = self.trunk_mlp(self.trunk_embeddings(component_ids))    # (6, branch_dim)

        # ---- DeepONet dot product ---------------------------------------- #
        # Σ_k  branch_out[k] * trunk_out[i, k]  for each component i
        raw_preds = branch_out @ trunk_out.T + self.output_bias  # (B, 6)
        predictions = self.final_activation(raw_preds)

        return predictions, attn_weights
