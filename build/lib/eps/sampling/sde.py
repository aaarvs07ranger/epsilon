"""SDE simulation: diffusion-model sampling via Euler-Maruyama (Algorithm 2).

Simulates the SDE extension of Theorem 17 (Eq. 44):

    dX_t = [u_t(X_t) + sigma_t^2 / 2 * grad log p_t(X_t)] dt + sigma_t dW_t

which follows the same probability path as the flow ODE for *any* diffusion
coefficient sigma_t >= 0 (sigma_t = 0 recovers the ODE). The Euler-Maruyama
update (Eq. 9) is

    X_{t+h} = X_t + h * drift_t(X_t) + sqrt(h) * sigma_t * eps,  eps ~ N(0, I).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from ..paths import NoiseScheduler, pad_dims_like
from .ode import StepCallback, _time_grid

VelocityScoreFn = Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]
SigmaFn = Callable[[Tensor], Tensor]


def make_sigma_fn(
    sigma: float, schedule: str = "constant", scheduler: Optional[NoiseScheduler] = None
) -> SigmaFn:
    """Diffusion-coefficient schedules sigma_t.

    * ``"constant"``: sigma_t = sigma. The notes' default; any sigma >= 0 is
      valid (Theorem 17), the best value is an empirical choice.
    * ``"beta"``: sigma_t = sigma * beta_t. Anneals the injected noise to zero
      as t -> 1, which tames the diverging score term near the data end.
    """
    if schedule == "constant":
        return lambda t: torch.full_like(t, sigma)
    if schedule == "beta":
        if scheduler is None:
            raise ValueError("sigma schedule 'beta' needs the path's noise scheduler")
        return lambda t: sigma * scheduler.beta(t)
    raise ValueError(f"Unknown sigma schedule '{schedule}'")


@torch.no_grad()
def integrate_sde(
    velocity_and_score_fn: VelocityScoreFn,
    x0: Tensor,
    num_steps: int,
    sigma_fn: SigmaFn,
    t_start: float = 1.0e-4,
    t_end: float = 0.9999,
    return_trajectory: bool = False,
    callback: Optional[StepCallback] = None,
) -> Tensor:
    """Euler-Maruyama simulation (Algorithm 2) of the SDE in Eq. (44).

    Args:
        velocity_and_score_fn: returns (u_t(x), grad log p_t(x)) for one
            network evaluation — see GuidedModel.velocity_and_score.
        x0: initial state ~ N(0, I), shape (B, ...).
        num_steps: number of Euler-Maruyama steps.
        sigma_fn: diffusion coefficient sigma_t as a function of (B,) times.
        t_start / t_end: integration endpoints. Defaults stay strictly inside
            (0, 1): the score <-> velocity conversion (Prop. 1) is singular at
            t = 0 and the marginal score stiffens as beta_t -> 0 near t = 1.
        return_trajectory: if True, return (num_steps + 1, B, ...) states.
        callback: called as callback(step_index, num_steps, x) after each step.
    """
    ts = _time_grid(num_steps, t_start, t_end, x0.device)
    x = x0
    trajectory = [x0] if return_trajectory else None
    batch = x0.shape[0]
    for i in range(num_steps):
        t, t_next = ts[i], ts[i + 1]
        h = t_next - t
        t_vec = t.expand(batch)
        u, s = velocity_and_score_fn(x, t_vec)
        sigma_t = sigma_fn(t_vec)
        sigma_x = pad_dims_like(sigma_t, x)
        drift = u + sigma_x.pow(2) / 2.0 * s  # Eq. (44)
        eps = torch.randn_like(x)
        x = x + h * drift + torch.sqrt(h) * sigma_x * eps  # Eq. (9)
        if trajectory is not None:
            trajectory.append(x)
        if callback is not None:
            callback(i + 1, num_steps, x)
    if trajectory is not None:
        return torch.stack(trajectory, dim=0)
    return x
