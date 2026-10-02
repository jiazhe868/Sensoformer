#!/usr/bin/env python3
"""
Unit tests for ConditionalPosteriorFlow: invertibility, log-det correctness,
identity initialization, and recovery of an analytically known Gaussian
posterior on a linear-Gaussian toy problem.
"""
import math
import sys
from pathlib import Path

import torch
import pytest

PROJ_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJ_ROOT / "src"))

from sensoformer.models.posterior_flow import ConditionalPosteriorFlow


@pytest.fixture
def flow():
    torch.manual_seed(0)
    return ConditionalPosteriorFlow(dim=4, context_dim=8, num_layers=6,
                                    hidden_dim=32)


def test_identity_at_init(flow):
    """Zero-initialized couplings: at init each coupling is the identity, so
    the whole flow is just the composition of the fixed permutations --
    volume-preserving (log_det = 0) and z is a permutation of y. The base
    density is permutation-invariant, so the initial density is exactly
    standard normal."""
    y = torch.randn(16, 4)
    ctx = torch.randn(16, 8)
    z, log_det = flow.forward_transform(y, ctx)
    assert torch.allclose(log_det, torch.zeros(16), atol=1e-6)
    assert torch.allclose(z.sort(dim=-1).values, y.sort(dim=-1).values,
                          atol=1e-6)


def test_invertibility(flow):
    for p in flow.parameters():
        torch.nn.init.normal_(p, std=0.1)
    y = torch.randn(32, 4)
    ctx = torch.randn(32, 8)
    z, _ = flow.forward_transform(y, ctx)
    y_rec = flow.inverse_transform(z, ctx)
    assert torch.allclose(y, y_rec, atol=1e-5)


def test_log_det_matches_autograd_jacobian(flow):
    """log|det J| from the flow must match the autograd Jacobian."""
    for p in flow.parameters():
        torch.nn.init.normal_(p, std=0.1)
    y = torch.randn(3, 4, dtype=torch.float64)
    ctx = torch.randn(3, 8, dtype=torch.float64)
    flow = flow.double()

    _, log_det = flow.forward_transform(y, ctx)
    for b in range(3):
        J = torch.autograd.functional.jacobian(
            lambda yy: flow.forward_transform(yy.unsqueeze(0),
                                              ctx[b:b + 1])[0].squeeze(0),
            y[b])
        assert torch.slogdet(J).logabsdet.item() == pytest.approx(
            log_det[b].item(), abs=1e-6)


def test_log_prob_integrates_to_standard_normal_at_init(flow):
    y = torch.randn(1000, 4)
    ctx = torch.zeros(1000, 8)
    lp = flow.log_prob(y, ctx)
    expected = -0.5 * (y ** 2 + math.log(2 * math.pi)).sum(-1)
    assert torch.allclose(lp, expected, atol=1e-5)


def test_sample_shape(flow):
    ctx = torch.randn(5, 8)
    s = flow.sample(ctx, num_samples=64)
    assert s.shape == (5, 64, 4)


def test_recovers_gaussian_posterior():
    """Linear-Gaussian toy problem with known posterior:
        theta ~ N(0, 1);  x = theta + eps,  eps ~ N(0, sigma^2)
        =>  p(theta | x) = N(x / (1 + sigma^2), sigma^2 / (1 + sigma^2))
    Train the conditional flow by NLL on (theta, x) pairs and compare
    posterior mean/std against the analytic values."""
    torch.manual_seed(1)
    sigma = 0.5
    post_var = sigma ** 2 / (1 + sigma ** 2)

    n = 20000
    theta = torch.randn(n, 1)
    x = theta + sigma * torch.randn(n, 1)

    flow = ConditionalPosteriorFlow(dim=1, context_dim=1, num_layers=4,
                                    hidden_dim=32)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
    for epoch in range(300):
        idx = torch.randint(0, n, (512,))
        loss = -flow.log_prob(theta[idx], x[idx]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    for x0 in (-1.0, 0.0, 1.5):
        ctx = torch.tensor([[x0]])
        samples = flow.sample(ctx, 4000).squeeze(0)
        analytic_mean = x0 / (1 + sigma ** 2)
        assert samples.mean().item() == pytest.approx(analytic_mean, abs=0.08)
        assert samples.std().item() == pytest.approx(math.sqrt(post_var), abs=0.08)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
