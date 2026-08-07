"""Model backbones and embeddings."""

from __future__ import annotations

import torch.nn as nn

from ..config import ModelConfig
from .dit import DiT
from .embeddings import FourierTimeEmbedding, LabelEmbedder, TimestepEmbedder
from .unet import UNet
from .vae import VAE, PretrainedVAE, vae_loss

__all__ = [
    "DiT",
    "UNet",
    "VAE",
    "PretrainedVAE",
    "vae_loss",
    "FourierTimeEmbedding",
    "TimestepEmbedder",
    "LabelEmbedder",
    "build_model",
]


def build_model(cfg: ModelConfig, image_size: int) -> nn.Module:
    """Instantiate the configured backbone (u_t^theta or s_t^theta)."""
    if cfg.name == "unet":
        return UNet.from_config(cfg, image_size)
    if cfg.name == "dit":
        return DiT.from_config(cfg, image_size)
    raise ValueError(f"Unknown model '{cfg.name}' (expected 'unet' or 'dit')")
