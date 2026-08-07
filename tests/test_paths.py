"""Tests for the mathematical core: schedulers, Gaussian path, conversions.

Every closed-form formula from the lecture notes is cross-checked against an
independent construction (autograd derivatives of the flow map / Gaussian
log-density), so a sign or factor error anywhere would be caught.
"""

import pytest
import torch

from eps.paths import (
    CondOTScheduler,
    CosineScheduler,
    GaussianProbabilityPath,
    build_scheduler,
)

SCHEDULERS = [CondOTScheduler(), CosineScheduler()]


def _rand_t(n: int, lo: float = 0.05, hi: float = 0.95) -> torch.Tensor:
    return torch.rand(n, dtype=torch.float64) * (hi - lo) + lo


@pytest.mark.parametrize("sched", SCHEDULERS)
def test_scheduler_boundary_conditions(sched):
    """alpha_0 = beta_1 = 0, alpha_1 = beta_0 = 1 (Eq. 15 requirements)."""
    zero = torch.tensor([0.0], dtype=torch.float64)
    one = torch.tensor([1.0], dtype=torch.float64)
    assert torch.allclose(sched.alpha(zero), torch.tensor([0.0], dtype=torch.float64), atol=1e-12)
    assert torch.allclose(sched.alpha(one), torch.tensor([1.0], dtype=torch.float64), atol=1e-12)
    assert torch.allclose(sched.beta(zero), torch.tensor([1.0], dtype=torch.float64), atol=1e-12)
    assert torch.allclose(sched.beta(one), torch.tensor([0.0], dtype=torch.float64), atol=1e-6)


@pytest.mark.parametrize("sched", SCHEDULERS)
def test_scheduler_derivatives_match_autograd(sched):
    t = _rand_t(64).requires_grad_(True)
    for fn, d_fn in [(sched.alpha, sched.d_alpha), (sched.beta, sched.d_beta)]:
        val = fn(t)
        (grad,) = torch.autograd.grad(val.sum(), t, create_graph=False)
        assert torch.allclose(grad, d_fn(t.detach()), atol=1e-10)


@pytest.mark.parametrize("sched", SCHEDULERS)
def test_path_sample_moments(sched):
    """x_t = alpha_t z + beta_t eps has mean alpha_t z and std beta_t (Eq. 16)."""
    torch.manual_seed(0)
    path = GaussianProbabilityPath(sched)
    z = torch.full((20000, 2), 1.5, dtype=torch.float64)
    t = torch.full((20000,), 0.3, dtype=torch.float64)
    x = path.sample(z, t)
    alpha = sched.alpha(t[:1]).item()
    beta = sched.beta(t[:1]).item()
    assert torch.allclose(x.mean(0), torch.full((2,), alpha * 1.5, dtype=torch.float64), atol=0.02)
    assert torch.allclose(x.std(0), torch.full((2,), beta, dtype=torch.float64), atol=0.02)


def test_condot_conditional_velocity_closed_form():
    """For alpha_t = t, beta_t = 1 - t: u_t(x|z) = (z - x) / (1 - t)."""
    torch.manual_seed(0)
    path = GaussianProbabilityPath(CondOTScheduler())
    z = torch.randn(32, 3, dtype=torch.float64)
    x = torch.randn(32, 3, dtype=torch.float64)
    t = _rand_t(32)
    expected = (z - x) / (1.0 - t)[:, None]
    assert torch.allclose(path.conditional_velocity(x, z, t), expected, atol=1e-12)


@pytest.mark.parametrize("sched", SCHEDULERS)
def test_conditional_velocity_is_flow_derivative(sched):
    """Eq. (20) via its defining property (proof of Example 10):

        d/dt psi_t(x0|z) = u_t(psi_t(x0|z)|z)  with psi_t(x0|z) = alpha_t z + beta_t x0.
    """
    torch.manual_seed(1)
    path = GaussianProbabilityPath(sched)
    z = torch.randn(16, 4, dtype=torch.float64)
    x0 = torch.randn(16, 4, dtype=torch.float64)
    t = _rand_t(16).requires_grad_(True)
    psi = sched.alpha(t)[:, None] * z + sched.beta(t)[:, None] * x0
    # psi_i depends only on t_i, so grad w.r.t. t with ones as grad_outputs
    # gives grads[i] = sum_j d psi_{ij} / dt_i; compare with sum_j u_{ij}.
    grads = torch.autograd.grad(psi, t, grad_outputs=torch.ones_like(psi))[0]
    u = path.conditional_velocity(psi.detach(), z, t.detach())
    assert torch.allclose(grads, u.sum(dim=1), atol=1e-9)


@pytest.mark.parametrize("sched", SCHEDULERS)
def test_velocity_target_equals_conditional_velocity(sched):
    """Eq. (31): alpha-dot z + beta-dot eps == u_t(alpha z + beta eps | z)."""
    torch.manual_seed(2)
    path = GaussianProbabilityPath(sched)
    z = torch.randn(64, 5, dtype=torch.float64)
    eps = torch.randn(64, 5, dtype=torch.float64)
    t = _rand_t(64)
    x_t = path.sample(z, t, eps)
    assert torch.allclose(
        path.velocity_target(z, eps, t), path.conditional_velocity(x_t, z, t), atol=1e-9
    )


@pytest.mark.parametrize("sched", SCHEDULERS)
def test_conditional_score_matches_gaussian_logpdf_gradient(sched):
    """Eq. (40) against autograd of log N(x; alpha_t z, beta_t^2 I)."""
    torch.manual_seed(3)
    path = GaussianProbabilityPath(sched)
    z = torch.randn(16, 3, dtype=torch.float64)
    t = _rand_t(16)
    x = path.sample(z, t).requires_grad_(True)
    alpha = sched.alpha(t)[:, None]
    beta = sched.beta(t)[:, None]
    log_p = (-((x - alpha * z) ** 2) / (2.0 * beta**2)).sum()
    (grad,) = torch.autograd.grad(log_p, x)
    assert torch.allclose(grad, path.conditional_score(x.detach(), z, t), atol=1e-9)


@pytest.mark.parametrize("sched", SCHEDULERS)
def test_score_target_equals_conditional_score(sched):
    """-eps/beta_t == grad log p_t(x_t|z) at x_t = alpha_t z + beta_t eps."""
    torch.manual_seed(4)
    path = GaussianProbabilityPath(sched)
    z = torch.randn(64, 2, dtype=torch.float64)
    eps = torch.randn(64, 2, dtype=torch.float64)
    t = _rand_t(64)
    x_t = path.sample(z, t, eps)
    assert torch.allclose(
        path.score_target(eps, t), path.conditional_score(x_t, z, t), atol=1e-9
    )


@pytest.mark.parametrize("sched", SCHEDULERS)
def test_proposition_1_conversion(sched):
    """Prop. 1 (Eq. 41): u_t(x|z) = a_t grad log p_t(x|z) + b_t x, and the
    velocity <-> score conversions are mutual inverses."""
    torch.manual_seed(5)
    path = GaussianProbabilityPath(sched)
    z = torch.randn(64, 3, dtype=torch.float64)
    x = torch.randn(64, 3, dtype=torch.float64)
    t = _rand_t(64)
    u = path.conditional_velocity(x, z, t)
    s = path.conditional_score(x, z, t)
    assert torch.allclose(path.velocity_from_score(s, x, t), u, atol=1e-8)
    assert torch.allclose(path.score_from_velocity(u, x, t), s, atol=1e-8)


@pytest.mark.parametrize("sched", SCHEDULERS)
def test_denoiser_recovers_data_point(sched):
    """Eq. (43): with the conditional field of a single data point z, the
    denoiser D_t is exactly z."""
    torch.manual_seed(6)
    path = GaussianProbabilityPath(sched)
    z = torch.randn(32, 4, dtype=torch.float64)
    t = _rand_t(32)
    x = path.sample(z, t)
    u = path.conditional_velocity(x, z, t)
    assert torch.allclose(path.denoiser_from_velocity(u, x, t), z, atol=1e-8)


def test_build_scheduler():
    assert isinstance(build_scheduler("condot"), CondOTScheduler)
    assert isinstance(build_scheduler("cosine"), CosineScheduler)
    with pytest.raises(ValueError):
        build_scheduler("nope")
