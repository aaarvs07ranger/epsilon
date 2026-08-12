"""Epsilon: a from-scratch flow matching / diffusion image generator.

Mathematics follows the MIT 6.S184 lecture notes (Holderrieth & Erives,
*An Introduction to Flow Matching and Diffusion Models*, 2026) exactly:
time runs from t = 0 (noise) to t = 1 (data), the Gaussian probability path is
p_t(x|z) = N(alpha_t z, beta_t^2 I), and training minimises the conditional
flow matching loss (Eq. 31) or the denoising score matching loss (Eq. 54).
"""

from .config import EpsilonConfig, load_config, save_config
from .losses import ConditionalFlowMatchingLoss, DenoisingScoreMatchingLoss, build_loss
from .paths import (
    CondOTScheduler,
    CosineScheduler,
    GaussianProbabilityPath,
    NoiseScheduler,
    build_scheduler,
)

__version__ = "0.1.0"

__all__ = [
    "EpsilonConfig",
    "load_config",
    "save_config",
    "GaussianProbabilityPath",
    "NoiseScheduler",
    "CondOTScheduler",
    "CosineScheduler",
    "build_scheduler",
    "ConditionalFlowMatchingLoss",
    "DenoisingScoreMatchingLoss",
    "build_loss",
]
