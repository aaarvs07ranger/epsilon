"""Diffusion Transformer (DiT) backbone, following Sec. 6.1.2 / Remark 29.

Pipeline: Patchify -> [DiT blocks with AdaLN-Zero time conditioning, optional
cross-attention to a text-token sequence] -> final AdaLN + linear ->
Depatchify. Class-conditional operation folds the class embedding into the
AdaLN conditioning vector (the notes: class-conditioned DiTs "eschew the cross
attention layer in favor of a time and class-based AdaNorm conditioning").

AdaLN-Zero: the modulation MLP that produces the per-block (shift, scale,
gate) parameters is zero-initialised, so every residual branch starts as the
identity — the standard DiT training stabiliser.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from .embeddings import LabelEmbedder, TimestepEmbedder, sincos_pos_embed_2d


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """AdaNorm modulation (Remark 29): (1 + scale) * x + shift, broadcast over
    the token dimension."""
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class MultiHeadAttention(nn.Module):
    """Multi-head scaled-dot-product attention (Remark 29). Self-attention when
    ``context`` is None, cross-attention to the prompt sequence otherwise."""

    def __init__(self, dim: int, num_heads: int, context_dim: Optional[int] = None) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        kv_dim = context_dim if context_dim is not None else dim
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(kv_dim, dim)
        self.v_proj = nn.Linear(kv_dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        src = x if context is None else context
        b, n, d = x.shape
        m = src.shape[1]
        q = self.q_proj(x).reshape(b, n, self.num_heads, -1).transpose(1, 2)
        k = self.k_proj(src).reshape(b, m, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(src).reshape(b, m, self.num_heads, -1).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).reshape(b, n, d))


class DiTBlock(nn.Module):
    """One DiT layer (Remark 29):

        x <- x + g_self  * SelfAttn(AdaNorm(x))
        x <- x + g_cross * CrossAttn(AdaNorm(x), y)     [if cross-attention]
        x <- x + g_mlp   * MLP(AdaNorm(x))

    All (shift, scale, gate) parameters come from the conditioning vector c
    (time + class embedding) through a zero-initialised MLP (AdaLN-Zero).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        cross_attention: bool = False,
        context_dim: int = 512,
    ) -> None:
        super().__init__()
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(dim, num_heads)
        if cross_attention:
            self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.cross_attn = MultiHeadAttention(dim, num_heads, context_dim=context_dim)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        n_mod = 9 if cross_attention else 6
        self.ada_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, n_mod * dim))
        nn.init.zeros_(self.ada_mlp[-1].weight)
        nn.init.zeros_(self.ada_mlp[-1].bias)

    def forward(self, x: Tensor, c: Tensor, context: Optional[Tensor] = None) -> Tensor:
        mod = self.ada_mlp(c)
        if self.cross_attention:
            if context is None:
                raise ValueError("cross-attention DiT block requires a context sequence")
            (sh1, s1, g1, sh2, s2, g2, sh3, s3, g3) = mod.chunk(9, dim=-1)
            x = x + g1[:, None, :] * self.attn(modulate(self.norm1(x), sh1, s1))
            x = x + g2[:, None, :] * self.cross_attn(
                modulate(self.norm_cross(x), sh2, s2), context
            )
            x = x + g3[:, None, :] * self.mlp(modulate(self.norm2(x), sh3, s3))
        else:
            (sh1, s1, g1, sh2, s2, g2) = mod.chunk(6, dim=-1)
            x = x + g1[:, None, :] * self.attn(modulate(self.norm1(x), sh1, s1))
            x = x + g2[:, None, :] * self.mlp(modulate(self.norm2(x), sh2, s2))
        return x


class FinalLayer(nn.Module):
    """Final AdaNorm + linear projection to patch pixels (zero-initialised)."""

    def __init__(self, dim: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.proj = nn.Linear(dim, patch_size * patch_size * out_channels)
        nn.init.zeros_(self.ada_mlp[-1].weight)
        nn.init.zeros_(self.ada_mlp[-1].bias)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.ada_mlp(c).chunk(2, dim=-1)
        return self.proj(modulate(self.norm(x), shift, scale))


class DiT(nn.Module):
    """Diffusion Transformer for u_t^theta(x|y) / s_t^theta(x|y).

    ``y`` may be integer class labels (class-conditional, AdaLN pathway) or —
    when ``cross_attention=True`` — a float tensor (B, S, context_dim) of text
    token embeddings (cross-attention pathway, Sec. 6.1.2). ``y=None`` is the
    null conditioning for classifier-free guidance in either mode.
    """

    def __init__(
        self,
        input_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_classes: int = 1000,
        cross_attention: bool = False,
        context_dim: int = 512,
        time_embed_wmin: float = 0.02,
        time_embed_wmax: float = 100.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if input_size % patch_size != 0:
            raise ValueError("input_size must be divisible by patch_size")
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.grid_size = input_size // patch_size
        self.cross_attention = cross_attention
        self.gradient_checkpointing = gradient_checkpointing

        # PatchEmb(x) = Patchify(x) W  (Sec. 6.1.2), realised as a conv with
        # kernel = stride = patch size.
        self.patch_embed = nn.Conv2d(in_channels, hidden_size, patch_size, stride=patch_size)
        self.register_buffer(
            "pos_embed", sincos_pos_embed_2d(hidden_size, self.grid_size), persistent=False
        )

        self.time_embed = TimestepEmbedder(
            hidden_size, freq_dim=256, w_min=time_embed_wmin, w_max=time_embed_wmax
        )
        self.label_embed = LabelEmbedder(num_classes, hidden_size)
        self.null_index = self.label_embed.null_index
        if cross_attention:
            # Learned null context token (absence of a prompt) and a pooled
            # projection folded into the AdaLN conditioning vector.
            self.null_context = nn.Parameter(torch.zeros(1, 1, context_dim))
            self.context_pool = nn.Linear(context_dim, hidden_size)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    cross_attention=cross_attention,
                    context_dim=context_dim,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)

    @classmethod
    def from_config(cls, cfg: ModelConfig, image_size: int) -> "DiT":
        d = cfg.dit
        return cls(
            input_size=image_size,
            patch_size=d.patch_size,
            in_channels=d.in_channels,
            hidden_size=d.hidden_size,
            depth=d.depth,
            num_heads=d.num_heads,
            mlp_ratio=d.mlp_ratio,
            dropout=d.dropout,
            num_classes=cfg.num_classes,
            cross_attention=d.cross_attention,
            context_dim=d.context_dim,
            time_embed_wmin=cfg.time_embed_wmin,
            time_embed_wmax=cfg.time_embed_wmax,
            gradient_checkpointing=d.gradient_checkpointing,
        )

    def unpatchify(self, x: Tensor) -> Tensor:
        """(B, N, p*p*C) -> (B, C, H, W)."""
        b = x.shape[0]
        p, g, c = self.patch_size, self.grid_size, self.in_channels
        x = x.reshape(b, g, g, p, p, c)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(b, c, g * p, g * p)

    def forward(self, x: Tensor, t: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """u_t^theta(x|y) (or s_t^theta(x|y)).

        Args:
            x: (B, C, H, W) noisy input x_t.
            t: (B,) times in [0, 1].
            y: (B,) class labels, or (B, S, context_dim) text embeddings when
               cross-attention is enabled, or None for null conditioning.
        """
        b = x.shape[0]
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, d)
        tokens = tokens + self.pos_embed[None, :, :]

        c = self.time_embed(t)
        context: Optional[Tensor] = None
        if self.cross_attention:
            if y is not None and y.dtype.is_floating_point:
                context = y
            else:
                context = self.null_context.expand(b, -1, -1)
            c = c + self.context_pool(context.mean(dim=1))
        else:
            labels = y if (y is not None and not y.dtype.is_floating_point) else None
            c = c + self.label_embed(labels, b, x.device)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                tokens = checkpoint(block, tokens, c, context, use_reentrant=False)
            else:
                tokens = block(tokens, c, context)

        return self.unpatchify(self.final_layer(tokens, c))
