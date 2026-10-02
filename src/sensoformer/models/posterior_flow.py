"""
Conditional normalizing flow for amortized posterior inversion (NPE head).

A RealNVP-style conditional coupling flow over the low-dimensional physical
target y (scaled magnitude + 5 normalized deviatoric moment-tensor
components), conditioned on the frozen Sensoformer event embedding. Trained
by exact maximum likelihood on (y, X) pairs drawn from the stochastic PSDR
forward model, the flow converges to the Bayesian posterior p(y | X) under
the simulator's implied likelihood -- point labels suffice, since posterior
spread arises from the many-to-one, noisy forward process rather than from
label error bars.

Provides exact log_prob (for NLL training) and fast sampling (single forward
pass per layer; no ODE solver).
"""
import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class ConditionalAffineCoupling(nn.Module):
    """One affine coupling layer: the masked half of y, together with the
    context vector, predicts scale and shift for the unmasked half."""

    def __init__(self, dim: int, context_dim: int, hidden_dim: int,
                 mask: Tensor, scale_clamp: float = 3.0):
        super().__init__()
        self.register_buffer("mask", mask.float())
        self.scale_clamp = scale_clamp
        self.net = nn.Sequential(
            nn.Linear(dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * dim),
        )
        # Zero-init the last layer: the flow starts as the identity map.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _scale_shift(self, y_masked: Tensor, context: Tensor):
        h = self.net(torch.cat([y_masked, context], dim=-1))
        raw_s, t = h.chunk(2, dim=-1)
        # tanh-bounded log-scale keeps the transform well-conditioned
        s = self.scale_clamp * torch.tanh(raw_s / self.scale_clamp)
        return s, t

    def forward(self, y: Tensor, context: Tensor):
        """y -> z direction. Returns (z, log_det)."""
        y_masked = y * self.mask
        s, t = self._scale_shift(y_masked, context)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)
        z = y_masked + (1 - self.mask) * (y * torch.exp(s) + t)
        log_det = s.sum(dim=-1)
        return z, log_det

    def inverse(self, z: Tensor, context: Tensor):
        """z -> y direction."""
        z_masked = z * self.mask
        s, t = self._scale_shift(z_masked, context)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)
        y = z_masked + (1 - self.mask) * ((z - t) * torch.exp(-s))
        return y


class ConditionalPosteriorFlow(nn.Module):
    """
    Stack of conditional affine coupling layers with alternating masks and
    fixed random permutations, standard-normal base distribution.

    Args:
        dim: dimensionality of the target y.
        context_dim: dimensionality of the conditioning embedding.
        num_layers: number of coupling layers.
        hidden_dim: width of each coupling MLP.
        seed: seed for the fixed inter-layer permutations.
    """

    def __init__(self, dim: int = 6, context_dim: int = 128,
                 num_layers: int = 8, hidden_dim: int = 128, seed: int = 0):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim

        gen = torch.Generator().manual_seed(seed)
        couplings = []
        perms = []
        for i in range(num_layers):
            mask = torch.zeros(dim)
            if i % 2 == 0:
                mask[: dim // 2] = 1.0
            else:
                mask[dim // 2:] = 1.0
            couplings.append(ConditionalAffineCoupling(
                dim, context_dim, hidden_dim, mask))
            perms.append(torch.randperm(dim, generator=gen))
        self.couplings = nn.ModuleList(couplings)
        for i, p in enumerate(perms):
            self.register_buffer(f"perm_{i}", p)
            self.register_buffer(f"perm_inv_{i}", torch.argsort(p))

    def _perm(self, i: int, inverse: bool = False) -> Tensor:
        return getattr(self, f"perm_inv_{i}" if inverse else f"perm_{i}")

    def forward_transform(self, y: Tensor, context: Tensor):
        """y -> z with total log|det J|."""
        z = y
        total_log_det = torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)
        for i, coupling in enumerate(self.couplings):
            z = z[:, self._perm(i)]
            z, log_det = coupling(z, context)
            total_log_det = total_log_det + log_det
        return z, total_log_det

    def inverse_transform(self, z: Tensor, context: Tensor):
        """z -> y."""
        y = z
        for i in reversed(range(len(self.couplings))):
            y = self.couplings[i].inverse(y, context)
            y = y[:, self._perm(i, inverse=True)]
        return y

    def log_prob(self, y: Tensor, context: Tensor) -> Tensor:
        z, log_det = self.forward_transform(y, context)
        log_base = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=-1)
        return log_base + log_det

    @torch.no_grad()
    def sample(self, context: Tensor, num_samples: int) -> Tensor:
        """Draw posterior samples.

        Args:
            context: (B, context_dim)
        Returns:
            (B, num_samples, dim)
        """
        B = context.shape[0]
        z = torch.randn(B * num_samples, self.dim,
                        device=context.device, dtype=context.dtype)
        ctx = context.repeat_interleave(num_samples, dim=0)
        y = self.inverse_transform(z, ctx)
        return y.view(B, num_samples, self.dim)
