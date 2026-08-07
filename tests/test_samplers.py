"""Tests for ODE / SDE samplers and classifier-free guidance.

The samplers are validated against cases where the marginal fields are known
in closed form:

* Dirac data (a single point z*): the conditional field is the marginal
  field. For the CondOT path, Euler is *exact at the endpoint* — every
  trajectory must land on z*.
* Gaussian data N(mu, s^2 I): the marginal path is N(alpha_t mu,
  (alpha_t^2 s^2 + beta_t^2) I) with an analytic score, so integrating the
  converted velocity field must reproduce the data mean and std.
"""

import math

import pytest
import torch

from eps.paths import CondOTScheduler, GaussianProbabilityPath, pad_dims_like
from eps.sampling import (
    GuidedModel,
    cfg_combine,
    integrate_ode,
    integrate_sde,
    make_sigma_fn,
)


@pytest.fixture
def path():
    return GaussianProbabilityPath(CondOTScheduler())


def test_euler_dirac_endpoint_exact(path):
    """CondOT + Dirac data: Euler hits z* exactly at t = 1 (the last step's
    velocity (z - x)/(1 - t) contracts everything onto z)."""
    torch.manual_seed(0)
    z_star = torch.tensor([[1.7, -0.3]], dtype=torch.float64)

    def u(x, t):
        return path.conditional_velocity(x, z_star.expand_as(x), t)

    x0 = torch.randn(512, 2, dtype=torch.float64)
    x1 = integrate_ode(u, x0, num_steps=100, solver="euler")
    assert torch.allclose(x1, z_star.expand_as(x1), atol=1e-8)


def test_ode_trajectory_matches_flow_map(path):
    """Against the analytic flow psi_t(x0|z) = t z + (1 - t) x0 (Eq. 21)."""
    torch.manual_seed(1)
    z_star = torch.tensor([[0.5, -1.0, 2.0]], dtype=torch.float64)

    def u(x, t):
        return path.conditional_velocity(x, z_star.expand_as(x), t)

    x0 = torch.randn(64, 3, dtype=torch.float64)
    traj = integrate_ode(u, x0, num_steps=200, return_trajectory=True)
    for k, t in [(50, 0.25), (100, 0.5), (150, 0.75)]:
        expected = t * z_star + (1 - t) * x0
        assert torch.allclose(traj[k], expected, atol=1e-6)


class _GaussianData:
    """Marginal fields for p_data = N(mu, s^2 I) under a Gaussian path."""

    def __init__(self, path, mu, s):
        self.path, self.mu, self.s = path, mu, s

    def marginal_std(self, t):
        a = self.path.scheduler.alpha(t)
        b = self.path.scheduler.beta(t)
        return torch.sqrt(a**2 * self.s**2 + b**2)

    def score(self, x, t):
        a = pad_dims_like(self.path.scheduler.alpha(t), x)
        var = pad_dims_like(self.marginal_std(t) ** 2, x)
        return -(x - a * self.mu) / var

    def velocity(self, x, t):
        return self.path.velocity_from_score(self.score(x, t), x, t)

    def sample_marginal(self, n, t, generator=None):
        a = self.path.scheduler.alpha(t)
        std = self.marginal_std(t)
        return a * self.mu + std * torch.randn(
            n, self.mu.shape[-1], dtype=torch.float64, generator=generator
        )


@pytest.mark.parametrize("solver,steps", [("euler", 400), ("heun", 100)])
def test_ode_gaussian_marginals(path, solver, steps):
    torch.manual_seed(2)
    mu = torch.tensor([[1.0, -0.5]], dtype=torch.float64)
    gd = _GaussianData(path, mu, s=0.5)
    t0 = 1.0e-3
    x0 = gd.sample_marginal(8000, torch.tensor(t0, dtype=torch.float64))
    x1 = integrate_ode(gd.velocity, x0, num_steps=steps, t_start=t0, t_end=1.0, solver=solver)
    assert torch.allclose(x1.mean(0), mu[0], atol=0.05)
    assert torch.allclose(x1.std(0), torch.full((2,), 0.5, dtype=torch.float64), atol=0.05)


def test_sde_preserves_gaussian_marginals(path):
    """Theorem 17: for any sigma, X_t ~ p_t. Check moments at t_end = 0.8."""
    torch.manual_seed(3)
    mu = torch.tensor([[1.0, -0.5]], dtype=torch.float64)
    gd = _GaussianData(path, mu, s=0.5)
    t0, t1 = 1.0e-3, 0.8

    def both(x, t):
        return gd.velocity(x, t), gd.score(x, t)

    x0 = gd.sample_marginal(8000, torch.tensor(t0, dtype=torch.float64))
    x1 = integrate_sde(
        both, x0, num_steps=400, sigma_fn=make_sigma_fn(1.0), t_start=t0, t_end=t1
    )
    t1_t = torch.tensor(t1, dtype=torch.float64)
    expected_mean = path.scheduler.alpha(t1_t) * mu[0]
    expected_std = gd.marginal_std(t1_t)
    assert torch.allclose(x1.mean(0), expected_mean, atol=0.05)
    assert torch.allclose(
        x1.std(0), torch.full((2,), expected_std.item(), dtype=torch.float64), atol=0.05
    )


def test_sde_with_zero_sigma_equals_ode(path):
    """sigma_t = 0 reduces the SDE (Eq. 44) to the flow ODE exactly."""
    torch.manual_seed(4)
    z_star = torch.tensor([[0.2, 1.1]], dtype=torch.float64)

    def u(x, t):
        return path.conditional_velocity(x, z_star.expand_as(x), t)

    def both(x, t):
        return u(x, t), path.conditional_score(x, z_star.expand_as(x), t)

    x0 = torch.randn(128, 2, dtype=torch.float64)
    kw = dict(num_steps=100, t_start=0.001, t_end=0.95)
    a = integrate_ode(u, x0, solver="euler", **kw)
    b = integrate_sde(both, x0, sigma_fn=make_sigma_fn(0.0), **kw)
    assert torch.allclose(a, b, atol=1e-12)


def test_heun_more_accurate_than_euler(path):
    """Heun at n steps should beat Euler at n steps on a smooth field."""
    mu = torch.tensor([[0.7, -0.2]], dtype=torch.float64)
    gd = _GaussianData(path, mu, s=0.6)
    t0 = 1.0e-3
    g = torch.Generator().manual_seed(5)
    x0 = gd.sample_marginal(4000, torch.tensor(t0, dtype=torch.float64), generator=g)

    def endpoint_err(solver):
        x1 = integrate_ode(gd.velocity, x0, num_steps=40, t_start=t0, t_end=1.0, solver=solver)
        return (x1.std(0) - 0.6).abs().max().item()

    assert endpoint_err("heun") < endpoint_err("euler")


def test_cfg_combine_identities():
    cond = torch.randn(8, 3)
    uncond = torch.randn(8, 3)
    assert torch.allclose(cfg_combine(cond, uncond, 1.0), cond)
    assert torch.allclose(cfg_combine(cond, uncond, 0.0), uncond)
    w = 3.0  # (1 - w) uncond + w cond == uncond + w (cond - uncond)
    assert torch.allclose(cfg_combine(cond, uncond, w), uncond + w * (cond - uncond), atol=1e-6)


class _LabelNet(torch.nn.Module):
    """Deterministic toy net whose output encodes the label, to observe CFG."""

    def __init__(self, null_index):
        super().__init__()
        self.null_index = null_index

    def forward(self, x, t, y):
        scale = torch.where(y == self.null_index, torch.zeros_like(y), torch.ones_like(y))
        return scale[:, None].float() + 0.1 * x


def test_guided_model_cfg(path):
    net = _LabelNet(null_index=10)
    x = torch.zeros(4, 3)
    t = torch.full((4,), 0.5)
    y = torch.zeros(4, dtype=torch.long)
    # w = 1: pure conditional -> 1.0 ; w = 2: (1-2)*0 + 2*1 = 2.0
    g1 = GuidedModel(net, path, "velocity", y=y, null_index=10, guidance_scale=1.0)
    g2 = GuidedModel(net, path, "velocity", y=y, null_index=10, guidance_scale=2.0)
    assert torch.allclose(g1.velocity(x, t), torch.ones(4, 3))
    assert torch.allclose(g2.velocity(x, t), 2.0 * torch.ones(4, 3))
    # y = None: unconditional -> 0
    g0 = GuidedModel(net, path, "velocity", y=None, null_index=10, guidance_scale=4.0)
    assert torch.allclose(g0.velocity(x, t), torch.zeros(4, 3))


def test_guided_model_prediction_conversion(path):
    """A score-prediction net driven as a velocity field must equal the
    closed-form conditional velocity (Prop. 1 in action)."""
    torch.manual_seed(6)
    z_star = torch.randn(1, 3, dtype=torch.float64)

    class ScoreNet(torch.nn.Module):
        def forward(self, x, t, y=None):
            return path.conditional_score(x, z_star.expand_as(x), t)

    g = GuidedModel(ScoreNet(), path, "score", y=None, null_index=None, guidance_scale=1.0)
    x = torch.randn(16, 3, dtype=torch.float64)
    t = torch.rand(16, dtype=torch.float64) * 0.8 + 0.1
    expected = path.conditional_velocity(x, z_star.expand_as(x), t)
    assert torch.allclose(g.velocity(x, t), expected, atol=1e-8)
    u, s = g.velocity_and_score(x, t)
    assert torch.allclose(u, expected, atol=1e-8)
    assert torch.allclose(s, path.conditional_score(x, z_star.expand_as(x), t), atol=1e-8)


def test_callbacks_and_trajectory_shapes(path):
    calls = []

    def u(x, t):
        return torch.zeros_like(x)

    x0 = torch.randn(2, 3)
    traj = integrate_ode(u, x0, num_steps=7, return_trajectory=True,
                         callback=lambda i, n, x: calls.append((i, n)))
    assert traj.shape == (8, 2, 3)
    assert calls == [(i, 7) for i in range(1, 8)]
