"""ODE simulation: the flow-model sampler (Algorithm 1).

Simulates dX_t = u_t(X_t) dt from t_start to t_end given any velocity field
callable ``(x, (B,) times) -> velocity``. With the marginal (or CFG-guided)
velocity field this is exactly probability-flow / flow-matching sampling:
X_0 ~ p_init = N(0, I) yields X_1 ~ p_data (Theorem 9).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

VelocityFn = Callable[[Tensor, Tensor], Tensor]
StepCallback = Callable[[int, int, Tensor], None]


def _time_grid(num_steps: int, t_start: float, t_end: float, device: torch.device) -> Tensor:
    return torch.linspace(t_start, t_end, num_steps + 1, device=device)


@torch.no_grad()
def integrate_ode(
    velocity_fn: VelocityFn,
    x0: Tensor,
    num_steps: int,
    t_start: float = 0.0,
    t_end: float = 1.0,
    solver: str = "euler",
    return_trajectory: bool = False,
    callback: Optional[StepCallback] = None,
) -> Tensor:
    """Simulate the ODE dX_t = u_t(X_t) dt.

    Args:
        velocity_fn: u_t(x); called with (x, t) where t has shape (B,).
        x0: initial state X_{t_start}, typically ~ N(0, I), shape (B, ...).
        num_steps: number of solver steps n (step size h = (t_end-t_start)/n).
        solver: "euler" (Eq. 4) or "heun" (Sec. 2.1).
        return_trajectory: if True, return all states stacked on dim 0 with
            shape (num_steps + 1, B, ...).
        callback: called as callback(step_index, num_steps, x) after each step.

    Returns:
        X_{t_end} of shape (B, ...), or the full trajectory.
    """
    if solver not in ("euler", "heun"):
        raise ValueError(f"Unknown ODE solver '{solver}'")
    ts = _time_grid(num_steps, t_start, t_end, x0.device)
    x = x0
    trajectory = [x0] if return_trajectory else None
    batch = x0.shape[0]
    for i in range(num_steps):
        t, t_next = ts[i], ts[i + 1]
        h = t_next - t
        t_vec = t.expand(batch)
        u = velocity_fn(x, t_vec)
        if solver == "euler":
            # Euler method (Eq. 4): X_{t+h} = X_t + h u_t(X_t)
            x = x + h * u
        else:
            # Heun's method: guess with Euler, correct with averaged velocity.
            x_guess = x + h * u
            u_next = velocity_fn(x_guess, t_next.expand(batch))
            x = x + h / 2.0 * (u + u_next)
        if trajectory is not None:
            trajectory.append(x)
        if callback is not None:
            callback(i + 1, num_steps, x)
    if trajectory is not None:
        return torch.stack(trajectory, dim=0)
    return x
