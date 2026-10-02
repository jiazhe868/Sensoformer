import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Union, Dict, Any
from omegaconf import DictConfig, OmegaConf

from .components import ResidualBlock1D


class GeometryAttentionBias(nn.Module):
    """
    Continuous relative-geometry bias for set attention.

    For every ordered station pair (i, j) a small MLP maps pairwise relative
    geometry -- distance difference, sin/cos of the source-azimuth difference,
    lon/lat offsets and inter-station angular separation -- to one additive
    attention-logit bias per head. The result is injected as a float attn_mask
    into the standard nn.TransformerEncoder, so attention scores become
    QK^T/sqrt(d) + b_h(g_i, g_j) without modifying the attention internals.

    The final linear layer is zero-initialised, so at initialisation the model
    is exactly the unbiased baseline and the bias is a learned deviation.

    Feature-column semantics follow the dataset layout documented in
    sensoformer/data/dataset.py (0=distance km, 1=azimuth deg, 2=station lon,
    3=station lat).
    """

    NUM_PAIR_FEATURES = 6

    def __init__(
        self,
        num_heads: int,
        dist_index: int = 0,
        az_index: int = 1,
        lon_index: int = 2,
        lat_index: int = 3,
        hidden_dim: int = 32,
        dist_scale: float = 100.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dist_index = dist_index
        self.az_index = az_index
        self.lon_index = lon_index
        self.lat_index = lat_index
        self.dist_scale = dist_scale

        self.mlp = nn.Sequential(
            nn.Linear(self.NUM_PAIR_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, S, F) raw per-station scalar features.
        Returns:
            (B * num_heads, S, S) additive attention-logit bias, laid out to
            match nn.MultiheadAttention's 3D attn_mask convention.
        """
        B, S, _ = features.shape

        dist = features[..., self.dist_index] / self.dist_scale
        az = features[..., self.az_index] * (math.pi / 180.0)
        lon = features[..., self.lon_index]
        lat = features[..., self.lat_index]

        d_dist = dist.unsqueeze(2) - dist.unsqueeze(1)      # (B, S, S)
        d_az = az.unsqueeze(2) - az.unsqueeze(1)
        d_lon = lon.unsqueeze(2) - lon.unsqueeze(1)
        d_lat = lat.unsqueeze(2) - lat.unsqueeze(1)
        separation = torch.sqrt(d_lon ** 2 + d_lat ** 2 + 1e-12)

        pair = torch.stack(
            [d_dist, torch.sin(d_az), torch.cos(d_az), d_lon, d_lat, separation],
            dim=-1,
        )  # (B, S, S, 6)

        bias = self.mlp(pair)                                # (B, S, S, H)
        bias = bias.permute(0, 3, 1, 2)                      # (B, H, S, S)
        return bias.reshape(B * self.num_heads, S, S)


class BiasedTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    TransformerEncoderLayer that always runs the standard (non-fused) path.

    The native fast path taken in eval()/no_grad() only supports boolean mask
    semantics and silently treats any non-zero float attn_mask entry as fully
    masked, which turns an additive geometry bias into an all--inf mask and
    produces NaNs. Bypassing it restores correct additive float-mask handling.
    """

    def forward(self, src, src_mask=None, src_key_padding_mask=None,
                is_causal=False):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

class StationEncoder(nn.Module):
    """
    Encodes single-station waveforms (P & S) and scalar features into a latent vector.

    Args:
        p_s_wave_in_channels: Number of channels per wave type (typically 6).
        num_scalar_features: Number of scalar station features (0 to disable).
        feature_dim: Output embedding dimension.
        single_tower: If True, a single shared ResNet processes the concatenated
            P+S waveform (2 * p_s_wave_in_channels channels) instead of two
            separate towers.  Use this for ablation against the dual-tower design.
    """
    def __init__(
        self,
        p_s_wave_in_channels: int,
        num_scalar_features: int,
        feature_dim: int,
        single_tower: bool = False
    ):
        super().__init__()
        self.single_tower = single_tower

        if single_tower:
            # One shared tower receives P and S concatenated along the channel axis
            self.shared_cnn = self._build_tower(p_s_wave_in_channels * 2)
            fusion_dim = 128
        else:
            # Build identical towers for P and S waves
            self.p_wave_cnn = self._build_tower(p_s_wave_in_channels)
            self.s_wave_cnn = self._build_tower(p_s_wave_in_channels)
            fusion_dim = 128 + 128  # P + S

        # Tower for scalar features
        self.use_scalar = (num_scalar_features > 0)
        if self.use_scalar:
            self.scalar_mlp = nn.Sequential(
                nn.Linear(num_scalar_features, 64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )
            fusion_dim += 32

        # Fusion Layer
        self.final_fc = nn.Linear(fusion_dim, feature_dim)

    def _build_tower(self, in_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            ResidualBlock1D(32, 32),
            ResidualBlock1D(32, 64),
            ResidualBlock1D(64, 128),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, waveform: Tensor, features: Tensor) -> Tensor:
        B, S, C, L = waveform.shape

        # Re-order channels so P and S groups are contiguous regardless of mode
        p_in = torch.cat((waveform[:, :, 0:3, :], waveform[:, :, 6:9, :]), dim=2)  # (B,S,6,L)
        s_in = torch.cat((waveform[:, :, 3:6, :], waveform[:, :, 9:12, :]), dim=2)  # (B,S,6,L)

        if self.single_tower:
            # Concatenate P and S along channel dim → (B*S, 12, L)
            ps_data = torch.cat([p_in, s_in], dim=2).view(B * S, 12, L)
            emb = self.shared_cnn(ps_data).squeeze(-1)
            parts = [emb]
        else:
            p_emb = self.p_wave_cnn(p_in.view(B * S, 6, L)).squeeze(-1)
            s_emb = self.s_wave_cnn(s_in.view(B * S, 6, L)).squeeze(-1)
            parts = [p_emb, s_emb]

        if self.use_scalar:
            scalar_emb = self.scalar_mlp(features.view(B * S, -1))
            parts.append(scalar_emb)

        station_emb = self.final_fc(torch.cat(parts, dim=1))
        return station_emb.view(B, S, -1)

class Sensoformer(nn.Module):
    """
    Main Model Architecture.
    """
    def __init__(self, cfg: Union[DictConfig, Dict[str, Any]]):
        super().__init__()
        
        # Convert plain dict to DictConfig so we can use dot notation (cfg.model.layers) consistently.
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
            
        model_cfg = cfg.model

        self.in_channels = int(model_cfg.in_channels)
        self.scalar_dim = int(model_cfg.num_scalar_features)
        self.embed_dim = int(model_cfg.embed_dim)
        self.heads = int(model_cfg.heads)
        self.layers = int(model_cfg.layers)
        self.dropout = float(model_cfg.dropout)
        
        self.aggregator_type = model_cfg.get('aggregator', 'transformer')
        self.single_tower = bool(model_cfg.get('single_tower', False))

        # 1. Local Feature Extraction
        self.station_encoder = StationEncoder(
            p_s_wave_in_channels=self.in_channels,
            num_scalar_features=self.scalar_dim,
            feature_dim=self.embed_dim,
            single_tower=self.single_tower
        )
        
        # Optional continuous relative-geometry attention bias (see
        # GeometryAttentionBias). Off by default: with geometry_bias=false the
        # module is not created and the model is bitwise-identical to the
        # baseline (including state_dict keys).
        self.use_geometry_bias = bool(model_cfg.get('geometry_bias', False))

        # 2. Global Event Aggregation
        if self.aggregator_type == 'transformer':
            layer_kwargs = dict(
                d_model=self.embed_dim,
                nhead=self.heads,
                dim_feedforward=model_cfg.get('feedforward_dim', 2048),
                batch_first=True,
                dropout=self.dropout
            )
            if self.use_geometry_bias:
                gb_cfg = model_cfg.get('geometry_bias_params', None) or {}
                self.geometry_bias = GeometryAttentionBias(
                    num_heads=self.heads,
                    dist_index=int(gb_cfg.get('dist_index', 0)),
                    az_index=int(gb_cfg.get('az_index', 1)),
                    lon_index=int(gb_cfg.get('lon_index', 2)),
                    lat_index=int(gb_cfg.get('lat_index', 3)),
                    hidden_dim=int(gb_cfg.get('hidden_dim', 32)),
                    dist_scale=float(gb_cfg.get('dist_scale', 100.0)),
                )
                # BiasedTransformerEncoderLayer: the native fast path corrupts
                # additive float attn_masks (see class docstring).
                self.event_aggregator = nn.TransformerEncoder(
                    BiasedTransformerEncoderLayer(**layer_kwargs),
                    num_layers=self.layers,
                    enable_nested_tensor=False)
            else:
                self.event_aggregator = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(**layer_kwargs),
                    num_layers=self.layers)
        else:
            # "DeepSets" or "No-Interaction": Identity mapping
            # Station features go directly to pooling without interacting
            self.event_aggregator = nn.Identity()
            self.use_geometry_bias = False

        # 3. Attention Pooling
        self.attention_pooling = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # 4. Heads
        self.magnitude_head = nn.Sequential(
            nn.Linear(self.embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 1)
        )
        
        self.moment_tensor_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 5)
        )
        
        self.final_activation = nn.Tanh()

    def get_embedding(self, waveforms: Tensor, features: Tensor, mask: Tensor) -> Tensor:
        """
        Returns the pooled event-level embedding immediately before the
        regression heads (i.e. the input to magnitude_head / moment_tensor_head).
        """
        station_embeddings = self.station_encoder(waveforms, features)

        transformer_mask = (mask == 0)
        if self.aggregator_type == 'transformer':
            aggregated_features = self._aggregate(
                station_embeddings, features, transformer_mask)
        else:
            aggregated_features = self.event_aggregator(station_embeddings)

        attn_weights_raw = self.attention_pooling(aggregated_features).squeeze(-1)
        attn_weights_raw = attn_weights_raw.masked_fill(transformer_mask, -1e9)
        attn_weights = F.softmax(attn_weights_raw, dim=1)

        event_vector = (aggregated_features * attn_weights.unsqueeze(-1)).sum(dim=1)
        return event_vector

    def _aggregate(self, station_embeddings: Tensor, features: Tensor,
                   transformer_mask: Tensor) -> Tensor:
        """Run the Transformer aggregator, with the geometry bias merged into
        a single float attn_mask when enabled (padding folded in as -1e9)."""
        if not self.use_geometry_bias:
            return self.event_aggregator(
                station_embeddings, src_key_padding_mask=transformer_mask)

        bias = self.geometry_bias(features).to(station_embeddings.dtype)
        B, S = transformer_mask.shape
        pad = transformer_mask.view(B, 1, 1, S).expand(
            B, self.heads, S, S).reshape(B * self.heads, S, S)
        merged = bias.masked_fill(pad, -1e9)
        return self.event_aggregator(station_embeddings, mask=merged)

    def forward(self, waveforms: Tensor, features: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        # 1. Embed
        station_embeddings = self.station_encoder(waveforms, features) 
        
        # 2. Transformer
        transformer_mask = (mask == 0)
        if self.aggregator_type == 'transformer':
            aggregated_features = self._aggregate(
                station_embeddings, features, transformer_mask)
        else:
            # Identity passes through. 
            aggregated_features = self.event_aggregator(station_embeddings)
        
        # 3. Pool
        attn_weights_raw = self.attention_pooling(aggregated_features).squeeze(-1)
        attn_weights_raw = attn_weights_raw.masked_fill(transformer_mask, -1e9)
        attn_weights = F.softmax(attn_weights_raw, dim=1)
        
        event_vector = (aggregated_features * attn_weights.unsqueeze(-1)).sum(dim=1)
        
        # 4. Predict
        pred_mag = self.magnitude_head(event_vector)
        pred_mt = self.moment_tensor_head(event_vector)
        
        predictions = torch.cat([
            self.final_activation(pred_mag), 
            self.final_activation(pred_mt)
        ], dim=1)
        
        return predictions, attn_weights