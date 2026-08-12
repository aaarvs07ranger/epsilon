"""Exponential moving average of model parameters.

Sampling and evaluation always use the EMA weights — a large, consistent FID
improvement for flow/diffusion models at zero training cost.
"""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor


class EMA:
    """Tracks theta_ema <- d * theta_ema + (1 - d) * theta after every step.

    Args:
        model: the live model (DDP-unwrapped).
        decay: asymptotic decay d (e.g. 0.9999).
        warmup: if True, the effective decay ramps up as
            min(decay, (1 + step) / (10 + step)) so early EMA weights track the
            fast-moving young model instead of the random init.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup: bool = True) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        self.decay = decay
        self.warmup = warmup
        self.step = 0
        self.shadow: dict[str, Tensor] = {
            name: p.detach().clone() for name, p in model.named_parameters()
        }
        self._backup: dict[str, Tensor] = {}

    def _named_params(self, model: nn.Module) -> Iterator[tuple[str, nn.Parameter]]:
        return model.named_parameters()

    def current_decay(self) -> float:
        if self.warmup:
            return min(self.decay, (1.0 + self.step) / (10.0 + self.step))
        return self.decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        d = self.current_decay()
        for name, p in self._named_params(model):
            self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Overwrite the model's parameters with the EMA weights."""
        for name, p in self._named_params(model):
            p.copy_(self.shadow[name])

    @torch.no_grad()
    def store(self, model: nn.Module) -> None:
        """Back up the live weights (before copy_to for evaluation)."""
        self._backup = {name: p.detach().clone() for name, p in self._named_params(model)}

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Restore the live weights saved by :meth:`store`."""
        if not self._backup:
            raise RuntimeError("EMA.restore() called without a preceding store()")
        for name, p in self._named_params(model):
            p.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self) -> dict:
        return {"decay": self.decay, "warmup": self.warmup, "step": self.step, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.warmup = state["warmup"]
        self.step = state["step"]
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}
