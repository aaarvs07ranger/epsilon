"""Conditioning embeddings: Fourier time features (Eq. 68-69), class labels,
and fixed 2D sin-cos positional embeddings for the DiT.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class FourierTimeEmbedding(nn.Module):
    """Fourier features for the scalar time t in [0, 1], exactly Eq. (68-69):

        TimeEmb(t) = sqrt(2/d) [cos(2 pi w_1 t), ..., cos(2 pi w_{d/2} t),
                                sin(2 pi w_1 t), ..., sin(2 pi w_{d/2} t)]

    with geometrically spaced frequencies

        w_i = w_min (w_max / w_min)^{(i-1)/(d/2-1)},  i = 1, ..., d/2,

    yielding a normed embedding, ||TimeEmb(t)|| = 1.
    """

    def __init__(self, dim: int, w_min: float = 0.02, w_max: float = 100.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Fourier embedding dim must be even, got {dim}")
        self.dim = dim
        half = dim // 2
        i = torch.arange(half, dtype=torch.float32)
        freqs = w_min * (w_max / w_min) ** (i / max(half - 1, 1))  # Eq. (69)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: Tensor) -> Tensor:
        """t: (B,) in [0, 1] -> (B, dim)."""
        angles = 2.0 * math.pi * t[:, None].float() * self.freqs[None, :]
        emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        return math.sqrt(2.0 / self.dim) * emb


class TimestepEmbedder(nn.Module):
    """Fourier features followed by a 2-layer MLP, the standard conditioning
    pathway for both the U-Net and the DiT (Sec. 6.1.1)."""

    def __init__(
        self, hidden_size: int, freq_dim: int = 256, w_min: float = 0.02, w_max: float = 100.0
    ) -> None:
        super().__init__()
        self.fourier = FourierTimeEmbedding(freq_dim, w_min=w_min, w_max=w_max)
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self.fourier(t))


class LabelEmbedder(nn.Module):
    """Learned embedding for class labels y in {0, ..., N-1}, plus one extra
    embedding at index N for the null token (absence of conditioning) used by
    classifier-free guidance (Sec. 5.2)."""

    def __init__(self, num_classes: int, hidden_size: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.null_index = num_classes
        self.embedding = nn.Embedding(num_classes + 1, hidden_size)

    def forward(self, y: Optional[Tensor], batch_size: int, device: torch.device) -> Tensor:
        """Embed labels; ``y = None`` means the null (unconditional) token."""
        if y is None:
            y = torch.full((batch_size,), self.null_index, device=device, dtype=torch.long)
        return self.embedding(y)


def sincos_pos_embed_2d(embed_dim: int, grid_size: int) -> Tensor:
    """Fixed 2D sine-cosine positional embedding of shape (grid_size^2, embed_dim),
    as used by the original DiT. Not learned."""
    if embed_dim % 4 != 0:
        raise ValueError("2D sin-cos positional embedding needs embed_dim % 4 == 0")
    coords = torch.arange(grid_size, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")

    def embed_1d(pos: Tensor, dim: int) -> Tensor:
        omega = torch.arange(dim // 2, dtype=torch.float32) / (dim // 2)
        omega = 1.0 / (10000.0**omega)
        out = pos.reshape(-1)[:, None] * omega[None, :]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    emb_y = embed_1d(grid_y, embed_dim // 2)
    emb_x = embed_1d(grid_x, embed_dim // 2)
    return torch.cat([emb_y, emb_x], dim=1)
