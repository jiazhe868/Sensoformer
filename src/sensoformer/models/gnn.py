import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Union, Dict, Any
from omegaconf import DictConfig, OmegaConf

from .network import StationEncoder


class DenseGATLayer(nn.Module):
    """
    Dense (batched) Graph Attention Layer.

    Operates on fully-materialized (B, S, S) adjacency masks — efficient for
    small graphs (≤ ~100 nodes). Does NOT require torch_geometric.

    Reference: Veličković et al., "Graph Attention Networks", ICLR 2018.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int, dropout: float):
        super().__init__()
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.heads = heads
        self.head_dim = out_dim // heads
        self.out_dim = out_dim

        # Linear projection shared across heads (weight matrix W)
        self.W = nn.Linear(in_dim, out_dim, bias=False)

        # Attention parameters: a^T [Wh_i || Wh_j] — one set per head
        self.a_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        Args:
            x:   (B, S, in_dim)  — node features
            adj: (B, S, S)       — adjacency mask; 1 = edge exists, 0 = masked out
                                   Padding nodes must have their rows/cols zeroed.
        Returns:
            (B, S, out_dim)
        """
        B, S, _ = x.shape

        # Project: (B, S, out_dim) -> (B, S, H, head_dim)
        Wh = self.W(x).view(B, S, self.heads, self.head_dim)

        # Attention logits: e_ij = LeakyReLU(a_src^T Wh_i + a_dst^T Wh_j)
        # a_src/a_dst: (H, head_dim)
        src_score = (Wh * self.a_src).sum(-1)  # (B, S, H)
        dst_score = (Wh * self.a_dst).sum(-1)  # (B, S, H)

        # Broadcast: e_ij = src_i + dst_j -> (B, S, S, H)
        e = self.leaky_relu(
            src_score.unsqueeze(2) + dst_score.unsqueeze(1)
        )

        # Mask non-edges with -inf before softmax
        # adj: (B, S, S) -> unsqueeze to (B, S, S, 1)
        mask = (adj == 0).unsqueeze(-1)          # (B, S, S, 1)
        e = e.masked_fill(mask, -1e9)

        # Softmax over neighbors (dim=2)
        alpha = F.softmax(e, dim=2)              # (B, S, S, H)
        alpha = self.dropout(alpha)

        # Aggregate: sum_j alpha_ij * Wh_j
        # Wh: (B, S, H, head_dim) -> (B, 1, S, H, head_dim)
        # alpha: (B, S, S, H) -> (B, S, S, H, 1)
        out = (alpha.unsqueeze(-1) * Wh.unsqueeze(1)).sum(2)  # (B, S, H, head_dim)
        out = out.view(B, S, self.out_dim)

        return self.layer_norm(F.elu(out))


class DenseGCNLayer(nn.Module):
    """
    Dense (batched) Graph Convolutional Layer (Kipf & Welling, ICLR 2017).

    Pure linear message passing with fixed (non-learned) symmetric degree
    normalization -- no attention mechanism, unlike DenseGATLayer. Used as
    the "MPNN" baseline so that any benefit from graph structure isn't
    confounded with a learned attention mechanism (Sensoformer's Transformer
    and attention pooling are both already attention-based).

    Reference: Kipf & Welling, "Semi-Supervised Classification with Graph
    Convolutional Networks", ICLR 2017.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        Args:
            x:   (B, S, in_dim)  — node features
            adj: (B, S, S)       — adjacency mask; 1 = edge exists, 0 = masked out
        Returns:
            (B, S, out_dim)
        """
        B, S, _ = x.shape

        # A_hat = A + I (add self-loops; idempotent if already present)
        eye = torch.eye(S, device=x.device, dtype=adj.dtype).unsqueeze(0).expand(B, -1, -1)
        adj_hat = ((adj + eye) > 0).float()

        # Symmetric normalization: D_hat^{-1/2} A_hat D_hat^{-1/2}
        deg = adj_hat.sum(dim=-1)                          # (B, S)
        deg_inv_sqrt = deg.clamp(min=1e-8).pow(-0.5)
        norm_adj = adj_hat * deg_inv_sqrt.unsqueeze(-1) * deg_inv_sqrt.unsqueeze(-2)

        # H' = sigma(norm_adj @ H @ W) -- aggregate neighbors, then a single
        # shared linear transform. No per-edge attention weights.
        agg = torch.bmm(norm_adj, x)          # (B, S, in_dim)
        out = self.dropout(self.W(agg))       # (B, S, out_dim)
        return self.layer_norm(F.relu(out))


class GNNSensoformer(nn.Module):
    """
    GNN baseline for Sensoformer.

    Uses the identical StationEncoder to obtain per-station embeddings, then
    replaces the Transformer aggregator with stacked GAT layers whose graph is
    constructed dynamically in the forward pass.

    Two graph construction modes (controlled by cfg.model.graph_type):
      - "full":  fully-connected graph among valid (non-padded) stations.
      - "knn":   K-nearest-neighbour graph built from station coordinates
                 (distance and azimuth) stored in features[:, :, 0:2].

    Two message-passing layer types (controlled by cfg.model.conv_type):
      - "gat":   DenseGATLayer, learned per-edge attention (default).
      - "gcn":   DenseGCNLayer, fixed symmetric-normalized aggregation, no
                 attention -- use this for a genuine "MPNN" baseline that
                 isn't confounded with an attention mechanism.

    Outputs are identical to Sensoformer: a 6-element tensor [Mw, Mxx, Myy, Mxy,
    Mxz, Myz] normalised to [-1, 1].
    """

    def __init__(self, cfg: Union[DictConfig, Dict[str, Any]]):
        super().__init__()

        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)

        model_cfg = cfg.model

        self.embed_dim = int(model_cfg.embed_dim)
        self.heads = int(model_cfg.heads)
        self.layers = int(model_cfg.layers)
        self.dropout = float(model_cfg.dropout)
        self.graph_type = str(model_cfg.get("graph_type", "knn"))
        self.conv_type = str(model_cfg.get("conv_type", "gat"))
        self.k_neighbors = int(model_cfg.get("k_neighbors", 5))
        # Indices of the coordinate features used for KNN (dist=0, az=1 by default)
        self.coord_indices = list(model_cfg.get("coord_feature_indices", [0, 1]))

        # --- Modules ---
        # 1. Station encoder (identical to Sensoformer)
        self.station_encoder = StationEncoder(
            p_s_wave_in_channels=int(model_cfg.in_channels),
            num_scalar_features=int(model_cfg.num_scalar_features),
            feature_dim=self.embed_dim,
            single_tower=bool(model_cfg.get('single_tower', False))
        )

        # 2. Stacked message-passing layers (GAT or GCN, per conv_type)
        conv_layers = []
        for _ in range(self.layers):
            if self.conv_type == "gcn":
                conv_layers.append(
                    DenseGCNLayer(
                        in_dim=self.embed_dim,
                        out_dim=self.embed_dim,
                        dropout=self.dropout,
                    )
                )
            else:
                conv_layers.append(
                    DenseGATLayer(
                        in_dim=self.embed_dim,
                        out_dim=self.embed_dim,
                        heads=self.heads,
                        dropout=self.dropout,
                    )
                )
        self.conv_layers = nn.ModuleList(conv_layers)

        # 3. Attention pooling (same as Sensoformer)
        self.attention_pooling = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # 4. Output heads (same as Sensoformer)
        self.magnitude_head = nn.Sequential(
            nn.Linear(self.embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 1),
        )
        self.moment_tensor_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 5),
        )
        self.final_activation = nn.Tanh()

    # ------------------------------------------------------------------
    # Graph construction helpers
    # ------------------------------------------------------------------

    def _build_full_adj(self, mask: Tensor) -> Tensor:
        """
        Fully-connected graph restricted to valid (non-padded) stations.

        Args:
            mask: (B, S) — 1 for real stations, 0 for padding.
        Returns:
            adj: (B, S, S) — binary adjacency matrix (self-loops included).
        """
        # Outer product of mask: adj[b,i,j] = mask[b,i] * mask[b,j]
        adj = mask.unsqueeze(2) * mask.unsqueeze(1)
        return adj

    def _build_knn_adj(self, features: Tensor, mask: Tensor) -> Tensor:
        """
        K-nearest-neighbour graph from station coordinates (dist, az).

        Builds a symmetric KNN adjacency using (dist, az) directly as the
        2-D feature space. Self-loops are always included.

        Args:
            features: (B, S, F) — scalar station features.
            mask:     (B, S)    — 1 for real stations, 0 for padding.
        Returns:
            adj: (B, S, S)
        """
        B, S, _ = features.shape
        device = features.device

        dist = features[:, :, self.coord_indices[0]]             # (B, S)
        az   = features[:, :, self.coord_indices[1]]             # (B, S)

        # Normalise each axis using only valid (non-padded) stations so that
        # padding zeros don't skew the mean/std. Both axes are mapped to
        # zero mean and unit variance, making them commensurate.
        def _normalize(vals: Tensor, valid_mask: Tensor) -> Tensor:
            # vals, valid_mask: (B, S)
            vals = vals * valid_mask
            n = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)       # (B, 1)
            mean = vals.sum(dim=1, keepdim=True) / n                   # (B, 1)
            var  = ((vals - mean) ** 2 * valid_mask).sum(dim=1, keepdim=True) / n
            std  = var.sqrt().clamp(min=1e-6)                          # (B, 1)
            return (vals - mean) / std

        valid = mask  # 1 for real stations, 0 for padding
        dist_n = _normalize(dist, valid)
        az_n   = _normalize(az,   valid)
        coords = torch.stack([dist_n, az_n], dim=-1)                   # (B, S, 2)

        # Pairwise squared distances
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)       # (B, S, S, 2)
        sq_dist = (diff ** 2).sum(-1)                          # (B, S, S)

        # Mask padded nodes: give them huge distance so they're never chosen
        pad_mask = (mask == 0)  # (B, S)
        INF = 1e12
        sq_dist = sq_dist.masked_fill(pad_mask.unsqueeze(1), INF)
        sq_dist = sq_dist.masked_fill(pad_mask.unsqueeze(2), INF)

        # Find K nearest neighbours (including self at distance 0)
        k = min(self.k_neighbors + 1, S)         # +1 because self is always included
        _, topk_idx = sq_dist.topk(k, dim=2, largest=False)   # (B, S, k)

        # Build adjacency
        adj = torch.zeros(B, S, S, device=device)
        adj.scatter_(2, topk_idx, 1.0)

        # Symmetrise and restrict to valid stations
        adj = ((adj + adj.transpose(1, 2)) > 0).float()
        valid = mask.unsqueeze(2) * mask.unsqueeze(1)          # (B, S, S)
        adj = adj * valid

        return adj

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, waveforms: Tensor, features: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            waveforms: (B, S, 12, L)
            features:  (B, S, F)
            mask:      (B, S) — 1 = valid station, 0 = padding
        Returns:
            predictions: (B, 6)  — [Mw, Mxx, Myy, Mxy, Mxz, Myz] in [-1, 1]
            attn_weights: (B, S) — pooling attention weights
        """
        # 1. Per-station embeddings
        x = self.station_encoder(waveforms, features)          # (B, S, D)

        # 2. Dynamic graph construction
        if self.graph_type == "knn":
            adj = self._build_knn_adj(features, mask)
        else:  # "full"
            adj = self._build_full_adj(mask)

        # 3. Graph message passing (GAT or GCN, per conv_type)
        for conv in self.conv_layers:
            x = conv(x, adj)                                   # (B, S, D)

        # 4. Attention pooling (mask out padding)
        pad_mask = (mask == 0)
        attn_raw = self.attention_pooling(x).squeeze(-1)       # (B, S)
        attn_raw = attn_raw.masked_fill(pad_mask, -1e9)
        attn_weights = F.softmax(attn_raw, dim=1)              # (B, S)

        event_vector = (x * attn_weights.unsqueeze(-1)).sum(1) # (B, D)

        # 5. Predict
        pred_mag = self.magnitude_head(event_vector)
        pred_mt = self.moment_tensor_head(event_vector)

        predictions = torch.cat(
            [self.final_activation(pred_mag), self.final_activation(pred_mt)],
            dim=1,
        )

        return predictions, attn_weights
