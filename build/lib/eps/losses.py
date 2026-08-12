"""Training losses: Conditional Flow Matching (CFM) and denoising score
matching (CSM/DSM), plus the classifier-free-guidance label dropout.

Equation numbers refer to the MIT 6.S184 lecture notes.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from .paths import GaussianProbabilityPath, pad_dims_like

# A conditional network u_t^theta(x|y) or s_t^theta(x|y): (x, t, y) -> tensor
ConditionalNet = Callable[[Tensor, Tensor, Optional[Tensor]], Tensor]


def sample_time(
    batch_size: int,
    device: torch.device,
    t_min: float = 0.0,
    t_max: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """t ~ Unif[t_min, t_max]. The notes use Unif[0, 1]; a restricted range is
    exposed because the DSM target -eps/beta_t diverges at t = 1 (Sec. 4.3)."""
    return torch.rand(batch_size, device=device, dtype=dtype) * (t_max - t_min) + t_min


def cfg_label_dropout(y: Tensor, null_index: int, p: float) -> Tensor:
    """Classifier-free guidance training: with probability ``p`` (the notes'
    eta) replace the label with the null token (Eq. 63-64, Algorithm 5)."""
    if p <= 0.0:
        return y
    drop = torch.rand(y.shape[0], device=y.device) < p
    return torch.where(drop, torch.full_like(y, null_index), y)


class ConditionalFlowMatchingLoss:
    """The (guided) conditional flow matching loss, Eq. (30/31) and Eq. (58):

        L_CFM(theta) = E_{t ~ Unif, (z,y) ~ p_data, eps ~ N(0, I)}
            || u_t^theta(alpha_t z + beta_t eps | y)
               - (alpha-dot_t z + beta-dot_t eps) ||^2

    For the CondOT path (alpha_t = t, beta_t = 1 - t) the target reduces to
    z - eps (Algorithm 3).
    """

    def __init__(
        self, path: GaussianProbabilityPath, t_min: float = 0.0, t_max: float = 1.0
    ) -> None:
        self.path = path
        self.t_min = t_min
        self.t_max = t_max

    def __call__(self, net: ConditionalNet, z: Tensor, y: Optional[Tensor] = None) -> Tensor:
        t = sample_time(z.shape[0], z.device, self.t_min, self.t_max, z.dtype)
        eps = torch.randn_like(z)
        x_t = self.path.sample(z, t, eps)
        target = self.path.velocity_target(z, eps, t)
        pred = net(x_t, t, y)
        return torch.mean((pred - target) ** 2)


class DenoisingScoreMatchingLoss:
    """The (guided) denoising score matching loss, Eq. (54) / Example 23:

        L_CSM(theta) = E || s_t^theta(alpha_t z + beta_t eps | y) + eps/beta_t ||^2

    ``weighting``:
        * ``"uniform"``: exactly the loss above. Numerically unstable for
          beta_t ~ 0 (t -> 1); pair with ``t_max < 1``.
        * ``"ddpm"``: multiply by beta_t^2, i.e. the DDPM noise-prediction loss
          || eps^theta - eps ||^2 with eps^theta = -beta_t s_t^theta (Sec. 4.3).
          Same minimiser, stable for all t.
    """

    def __init__(
        self,
        path: GaussianProbabilityPath,
        t_min: float = 0.0,
        t_max: float = 1.0,
        weighting: str = "ddpm",
    ) -> None:
        if weighting not in ("uniform", "ddpm"):
            raise ValueError(f"Unknown weighting '{weighting}'")
        self.path = path
        self.t_min = t_min
        self.t_max = t_max
        self.weighting = weighting

    def __call__(self, net: ConditionalNet, z: Tensor, y: Optional[Tensor] = None) -> Tensor:
        t = sample_time(z.shape[0], z.device, self.t_min, self.t_max, z.dtype)
        eps = torch.randn_like(z)
        x_t = self.path.sample(z, t, eps)
        pred = net(x_t, t, y)
        if self.weighting == "uniform":
            target = self.path.score_target(eps, t)  # -eps / beta_t
            return torch.mean((pred - target) ** 2)
        # DDPM weighting: || beta_t s^theta + eps ||^2
        beta = pad_dims_like(self.path.scheduler.beta(t), z)
        return torch.mean((beta * pred + eps) ** 2)


def build_loss(
    prediction: str,
    path: GaussianProbabilityPath,
    t_min: float = 0.0,
    t_max: float = 1.0,
):
    """Factory: 'velocity' -> CFM (Eq. 31), 'score' -> DSM (Eq. 54)."""
    if prediction == "velocity":
        return ConditionalFlowMatchingLoss(path, t_min=t_min, t_max=t_max)
    if prediction == "score":
        # Default to the numerically stable DDPM weighting; note the notes'
        # exact uniform-weight CSM loss remains available via the class.
        return DenoisingScoreMatchingLoss(path, t_min=t_min, t_max=t_max, weighting="ddpm")
    raise ValueError(f"Unknown prediction type '{prediction}' (expected 'velocity' or 'score')")
