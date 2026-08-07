"""Classifier-free guidance and the unified model wrapper used by samplers.

CFG (Summary 27, Eq. 65): given the guided field u_t(x|y) and the unguided
field u_t(x|null), the guided sample uses

    u~_t(x|y) = (1 - w) u_t(x|null) + w u_t(x|y)

with guidance scale w (w = 1 recovers plain conditional sampling). Per
Remark 28, the same combination applies to score models and SDE sampling.

The :class:`GuidedModel` wrapper additionally hides the network's prediction
type: whichever of {velocity, score} the checkpoint was trained to predict,
both quantities are available via the conversion formulas of Proposition 1
(Eq. 41-42), so any sampler can drive any checkpoint.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ..losses import ConditionalNet
from ..paths import GaussianProbabilityPath


def cfg_combine(cond: Tensor, uncond: Tensor, guidance_scale: float) -> Tensor:
    """Eq. (65): (1 - w) * uncond + w * cond."""
    return (1.0 - guidance_scale) * uncond + guidance_scale * cond


def cfg_predict(
    net: ConditionalNet,
    x: Tensor,
    t: Tensor,
    y: Optional[Tensor],
    null_index: Optional[int],
    guidance_scale: float,
) -> Tensor:
    """One guided prediction u~_t(x|y) (or s~_t(x|y)) with CFG.

    * ``y is None``: unconditional sample — a single pass with null labels.
    * ``guidance_scale == 1``: plain conditional — a single pass.
    * Class labels (integer ``y``): conditional and unconditional branches are
      batched into one forward pass.
    * Sequence conditioning (float ``y``, e.g. text embeddings): two passes,
      with ``None`` signalling the network's learned null context.
    """
    if y is None:
        if null_index is not None:
            y_null = torch.full((x.shape[0],), null_index, device=x.device, dtype=torch.long)
            return net(x, t, y_null)
        return net(x, t, None)
    if guidance_scale == 1.0:
        return net(x, t, y)
    if y.dtype in (torch.int32, torch.int64):
        if null_index is None:
            raise ValueError("null_index is required for CFG with class labels")
        y_null = torch.full_like(y, null_index)
        both = net(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, y_null]))
        cond, uncond = both.chunk(2)
    else:
        cond = net(x, t, y)
        uncond = net(x, t, None)
    return cfg_combine(cond, uncond, guidance_scale)


class GuidedModel:
    """CFG-guided velocity and score fields backed by one trained network.

    Args:
        net: the trained network ``(x, t, y) -> prediction``.
        path: the Gaussian probability path used at training time.
        prediction: what ``net`` outputs, ``"velocity"`` or ``"score"``.
        y: conditioning (class labels or context tensor), or None for
            unconditional generation.
        null_index: index of the null class token (class-conditional models).
        guidance_scale: the CFG weight w.
    """

    def __init__(
        self,
        net: ConditionalNet,
        path: GaussianProbabilityPath,
        prediction: str,
        y: Optional[Tensor] = None,
        null_index: Optional[int] = None,
        guidance_scale: float = 1.0,
    ) -> None:
        if prediction not in ("velocity", "score"):
            raise ValueError(f"Unknown prediction type '{prediction}'")
        self.net = net
        self.path = path
        self.prediction = prediction
        self.y = y
        self.null_index = null_index
        self.guidance_scale = guidance_scale

    def _predict(self, x: Tensor, t: Tensor) -> Tensor:
        return cfg_predict(self.net, x, t, self.y, self.null_index, self.guidance_scale)

    def velocity(self, x: Tensor, t: Tensor) -> Tensor:
        """The (guided) marginal velocity field u~_t(x|y)."""
        out = self._predict(x, t)
        if self.prediction == "velocity":
            return out
        return self.path.velocity_from_score(out, x, t)  # Eq. (42)

    def score(self, x: Tensor, t: Tensor) -> Tensor:
        """The (guided) marginal score grad log p_t(x|y)."""
        out = self._predict(x, t)
        if self.prediction == "score":
            return out
        return self.path.score_from_velocity(out, x, t)  # Eq. (41) inverted

    def velocity_and_score(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Both fields from a single network evaluation (for SDE sampling)."""
        out = self._predict(x, t)
        if self.prediction == "velocity":
            return out, self.path.score_from_velocity(out, x, t)
        return self.path.velocity_from_score(out, x, t), out
