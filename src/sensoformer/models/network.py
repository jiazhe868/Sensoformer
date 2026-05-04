import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Union, Dict, Any
from omegaconf import DictConfig, OmegaConf 

from .components import ResidualBlock1D

class StationEncoder(nn.Module):
    """
    Encodes single-station waveforms (P & S) and scalar features into a latent vector.
    """
    def __init__(
        self, 
        p_s_wave_in_channels: int, 
        num_scalar_features: int, 
        feature_dim: int
    ):
        super().__init__()
        
        # Build identical towers for P and S waves
        self.p_wave_cnn = self._build_tower(p_s_wave_in_channels)
        self.s_wave_cnn = self._build_tower(p_s_wave_in_channels)
        
        # Tower for scalar features
        self.use_scalar = (num_scalar_features > 0)
        fusion_dim = 128 + 128 # P + S

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
        
        # Slice Waveforms
        p_in = torch.cat((waveform[:, :, 0:3, :], waveform[:, :, 6:9, :]), dim=2)
        p_data = p_in.view(B * S, 6, L)

        s_in = torch.cat((waveform[:, :, 3:6, :], waveform[:, :, 9:12, :]), dim=2)
        s_data = s_in.view(B * S, 6, L)
        
        # Extract
        p_emb = self.p_wave_cnn(p_data).squeeze(-1)
        s_emb = self.s_wave_cnn(s_data).squeeze(-1)

        if self.use_scalar:
            scalar_emb = self.scalar_mlp(features.view(B * S, -1))
            combined = torch.cat([p_emb, s_emb, scalar_emb], dim=1)
        else:
            combined = torch.cat([p_emb, s_emb], dim=1)

        station_emb = self.final_fc(combined)
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

        # 1. Local Feature Extraction
        self.station_encoder = StationEncoder(
            p_s_wave_in_channels=self.in_channels,
            num_scalar_features=self.scalar_dim,
            feature_dim=self.embed_dim
        )
        
        # 2. Global Event Aggregation
        if self.aggregator_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=self.heads,
                dim_feedforward=model_cfg.get('feedforward_dim', 2048),
                batch_first=True,
                dropout=self.dropout
            )
            self.event_aggregator = nn.TransformerEncoder(encoder_layer, num_layers=self.layers)
        else:
            # "DeepSets" or "No-Interaction": Identity mapping
            # Station features go directly to pooling without interacting
            self.event_aggregator = nn.Identity()

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

    def forward(self, waveforms: Tensor, features: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        # 1. Embed
        station_embeddings = self.station_encoder(waveforms, features) 
        
        # 2. Transformer
        transformer_mask = (mask == 0) 
        if self.aggregator_type == 'transformer':
            aggregated_features = self.event_aggregator(
                station_embeddings, 
                src_key_padding_mask=transformer_mask
            )
        else:
            # Identity passes through. 
            aggregated_features = self.event_aggregator(station_embeddings)
        
        # 3. Pool
        attn_weights_raw = self.attention_pooling(aggregated_features).squeeze(-1)
        attn_weights_raw.masked_fill_(transformer_mask, -1e9)
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
