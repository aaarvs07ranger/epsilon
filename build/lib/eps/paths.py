"""Probability paths: noise schedulers, the Gaussian conditional path, and the
score <-> velocity conversion formulas.

Mathematical source of truth: Holderrieth & Erives, *An Introduction to Flow
Matching and Diffusion Models* (MIT 6.S184, 2026). Equation numbers below
refer to those lecture notes.

Conventions (identical to the notes — note they are OPPOSITE to much of the
DDPM literature):

* Time runs from **t = 0 (pure noise)** to **t = 1 (data)**.
* Gaussian conditional probability path (Eq. 15):
      p_t(x|z) = N(alpha_t z, beta_t^2 I_d)
  with noise schedulers alpha, beta: continuously differentiable, monotonic,
  alpha_0 = beta_1 = 0 and alpha_1 = beta_0 = 1.
* Default scheduler is the CondOT path: alpha_t = t, beta_t = 1 - t.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor


def pad_dims_like(t: Tensor, x: Tensor) -> Tensor:
    """Reshape a batch of scalars ``t`` of shape (B,) to broadcast against ``x``
    of shape (B, ...), i.e. to shape (B, 1, ..., 1)."""
    return t.reshape(t.shape[0], *([1] * (x.ndim - 1)))


# ---------------------------------------------------------------------------
# Noise schedulers alpha_t, beta_t
# ---------------------------------------------------------------------------


class NoiseScheduler(ABC):
    """Noise schedulers (alpha_t, beta_t) of a Gaussian probability path (Eq. 15).

    Subclasses must satisfy alpha_0 = beta_1 = 0 and alpha_1 = beta_0 = 1, with
    alpha increasing and beta decreasing on [0, 1].
    """

    @abstractmethod
    def alpha(self, t: Tensor) -> Tensor:
        """alpha_t."""

    @abstractmethod
    def beta(self, t: Tensor) -> Tensor:
        """beta_t."""

    @abstractmethod
    def d_alpha(self, t: Tensor) -> Tensor:
        """alpha-dot_t = d/dt alpha_t."""

    @abstractmethod
    def d_beta(self, t: Tensor) -> Tensor:
        """beta-dot_t = d/dt beta_t."""


class CondOTScheduler(NoiseScheduler):
    """CondOT path: alpha_t = t, beta_t = 1 - t (Eq. 15 / Algorithm 3).

    The scheduler used by e.g. Stable Diffusion 3 and Movie Gen (Sec. 6.3).
    """

    def alpha(self, t: Tensor) -> Tensor:
        return t

    def beta(self, t: Tensor) -> Tensor:
        return 1.0 - t

    def d_alpha(self, t: Tensor) -> Tensor:
        return torch.ones_like(t)

    def d_beta(self, t: Tensor) -> Tensor:
        return -torch.ones_like(t)


class CosineScheduler(NoiseScheduler):
    """Trigonometric path: alpha_t = sin(pi t / 2), beta_t = cos(pi t / 2).

    Satisfies alpha_t^2 + beta_t^2 = 1 ("variance preserving").
    """

    def alpha(self, t: Tensor) -> Tensor:
        return torch.sin(torch.pi / 2.0 * t)

    def beta(self, t: Tensor) -> Tensor:
        return torch.cos(torch.pi / 2.0 * t)

    def d_alpha(self, t: Tensor) -> Tensor:
        return torch.pi / 2.0 * torch.cos(torch.pi / 2.0 * t)

    def d_beta(self, t: Tensor) -> Tensor:
        return -torch.pi / 2.0 * torch.sin(torch.pi / 2.0 * t)


SCHEDULERS: dict[str, type[NoiseScheduler]] = {
    "condot": CondOTScheduler,
    "cosine": CosineScheduler,
}


def build_scheduler(name: str) -> NoiseScheduler:
    try:
        return SCHEDULERS[name]()
    except KeyError:
        raise ValueError(f"Unknown scheduler '{name}'. Available: {list(SCHEDULERS)}")


# ---------------------------------------------------------------------------
# Gaussian conditional probability path
# ---------------------------------------------------------------------------


class GaussianProbabilityPath:
    """The Gaussian conditional probability path p_t(x|z) = N(alpha_t z, beta_t^2 I)
    (Eq. 15) together with its conditional vector field (Eq. 20), conditional
    score (Eq. 40), and the score <-> velocity conversion of Proposition 1.
    """

    def __init__(self, scheduler: NoiseScheduler) -> None:
        self.scheduler = scheduler

    # -- sampling ----------------------------------------------------------

    def sample(self, z: Tensor, t: Tensor, eps: Optional[Tensor] = None) -> Tensor:
        """Sample x_t ~ p_t(.|z) via x_t = alpha_t z + beta_t eps (Eq. 16 / 28).

        Args:
            z: data batch, shape (B, ...).
            t: times in [0, 1], shape (B,).
            eps: optional pre-drawn noise eps ~ N(0, I) of the same shape as
                ``z``; drawn internally if omitted.
        """
        if eps is None:
            eps = torch.randn_like(z)
        alpha = pad_dims_like(self.scheduler.alpha(t), z)
        beta = pad_dims_like(self.scheduler.beta(t), z)
        return alpha * z + beta * eps

    # -- conditional targets ----------------------------------------------

    def conditional_velocity(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        """Conditional (target) vector field, Eq. (20):

            u_t(x|z) = (alpha-dot_t - beta-dot_t/beta_t * alpha_t) z
                       + beta-dot_t/beta_t * x

        Undefined at t = 1 where beta_t = 0; training never needs it there
        (use :meth:`velocity_target`, which is division-free).
        """
        alpha = pad_dims_like(self.scheduler.alpha(t), x)
        beta = pad_dims_like(self.scheduler.beta(t), x)
        d_alpha = pad_dims_like(self.scheduler.d_alpha(t), x)
        d_beta = pad_dims_like(self.scheduler.d_beta(t), x)
        return (d_alpha - d_beta / beta * alpha) * z + d_beta / beta * x

    def velocity_target(self, z: Tensor, eps: Tensor, t: Tensor) -> Tensor:
        """CFM regression target evaluated at x_t = alpha_t z + beta_t eps
        (Eq. 31):

            u_t(x_t|z) = alpha-dot_t z + beta-dot_t eps

        Algebraically equal to ``conditional_velocity(sample(z,t,eps), z, t)``
        but free of the division by beta_t, hence valid for all t in [0, 1].
        """
        d_alpha = pad_dims_like(self.scheduler.d_alpha(t), z)
        d_beta = pad_dims_like(self.scheduler.d_beta(t), z)
        return d_alpha * z + d_beta * eps

    def conditional_score(self, x: Tensor, z: Tensor, t: Tensor) -> Tensor:
        """Conditional score, Eq. (40):

            grad log p_t(x|z) = -(x - alpha_t z) / beta_t^2

        Undefined at t = 1 where beta_t = 0.
        """
        alpha = pad_dims_like(self.scheduler.alpha(t), x)
        beta = pad_dims_like(self.scheduler.beta(t), x)
        return -(x - alpha * z) / beta.pow(2)

    def score_target(self, eps: Tensor, t: Tensor) -> Tensor:
        """DSM regression target evaluated at x_t = alpha_t z + beta_t eps
        (Sec. 4.3):

            grad log p_t(x_t|z) = -eps / beta_t

        Diverges as t -> 1 (beta_t -> 0): sample t away from 1 when training a
        score model (the notes call this out below Example 23).
        """
        beta = pad_dims_like(self.scheduler.beta(t), eps)
        return -eps / beta

    # -- score <-> velocity conversion (Proposition 1) ---------------------

    def conversion_coefficients(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """The coefficients (a_t, b_t) of Proposition 1 (Eq. 41):

            u_t(x) = a_t * grad log p_t(x) + b_t * x,
            a_t = beta_t^2 alpha-dot_t / alpha_t - beta-dot_t beta_t,
            b_t = alpha-dot_t / alpha_t

        Both are singular at t = 0 where alpha_t = 0; samplers that rely on the
        conversion therefore integrate on [t_start, t_end] with t_start > 0.
        """
        alpha = self.scheduler.alpha(t)
        beta = self.scheduler.beta(t)
        d_alpha = self.scheduler.d_alpha(t)
        d_beta = self.scheduler.d_beta(t)
        a = beta.pow(2) * d_alpha / alpha - d_beta * beta
        b = d_alpha / alpha
        return a, b

    def velocity_from_score(self, score: Tensor, x: Tensor, t: Tensor) -> Tensor:
        """u_t(x) = a_t s_t(x) + b_t x (Eq. 41/42). Holds for both the
        conditional and the marginal fields."""
        a, b = self.conversion_coefficients(t)
        return pad_dims_like(a, x) * score + pad_dims_like(b, x) * x

    def score_from_velocity(self, velocity: Tensor, x: Tensor, t: Tensor) -> Tensor:
        """Inverse of Eq. (41): s_t(x) = (u_t(x) - b_t x) / a_t."""
        a, b = self.conversion_coefficients(t)
        return (velocity - pad_dims_like(b, x) * x) / pad_dims_like(a, x)

    def denoiser_from_velocity(self, velocity: Tensor, x: Tensor, t: Tensor) -> Tensor:
        """Posterior-mean denoiser D_t(x) = E[z | x_t = x], Eq. (43):

            D_t(x) = (beta_t u_t(x) - beta-dot_t x)
                     / (alpha-dot_t beta_t - alpha_t beta-dot_t)
        """
        alpha = pad_dims_like(self.scheduler.alpha(t), x)
        beta = pad_dims_like(self.scheduler.beta(t), x)
        d_alpha = pad_dims_like(self.scheduler.d_alpha(t), x)
        d_beta = pad_dims_like(self.scheduler.d_beta(t), x)
        return (beta * velocity - d_beta * x) / (d_alpha * beta - alpha * d_beta)

    # -- SDE extension (Theorem 17) ----------------------------------------

    def sde_drift(self, velocity: Tensor, score: Tensor, sigma_t: Tensor, x: Tensor) -> Tensor:
        """Drift of the SDE extension (Eq. 44):

            dX_t = [u_t(X_t) + sigma_t^2 / 2 * grad log p_t(X_t)] dt + sigma_t dW_t

        Valid for any diffusion coefficient sigma_t >= 0; sigma_t = 0 recovers
        the probability-flow ODE of the flow model.
        """
        return velocity + pad_dims_like(sigma_t, x).pow(2) / 2.0 * score
