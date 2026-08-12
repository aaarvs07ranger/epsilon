"""A modern ADM-style U-Net backbone for u_t^theta(x|y) / s_t^theta(x|y).

Design follows Sec. 6.1.3 of the lecture notes (encoders / midcoder / decoders
with residual connections), fleshed out with the standard ingredients of
large-scale diffusion U-Nets: residual blocks conditioned on (t, y) through
adaptive group normalisation (AdaGN), and multi-head self-attention at coarse
resolutions.

The same network can be trained to predict either the velocity field or the
score function — the output head is identical (an image-shaped vector field);
only the regression target differs (see eps.losses).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig
from .embeddings import LabelEmbedder, TimestepEmbedder


def _norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with as many of 32 groups as divide the channel count."""
    groups = 32
    while channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


def _zero_module(module: nn.Module) -> nn.Module:
    """Zero-initialise a module's parameters (stabilises residual branches)."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ResBlock(nn.Module):
    """Residual block with AdaGN conditioning: the (t, y) embedding produces a
    per-channel (scale, shift) that modulates the normalised activations."""

    def __init__(self, in_channels: int, out_channels: int, emb_dim: int, dropout: float) -> None:
        super().__init__()
        self.in_norm = _norm(in_channels)
        self.in_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, 2 * out_channels)
        self.out_norm = _norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.out_conv = _zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.in_conv(F.silu(self.in_norm(x)))
        scale, shift = self.emb_proj(F.silu(emb))[:, :, None, None].chunk(2, dim=1)
        h = self.out_norm(h) * (1.0 + scale) + shift  # AdaGN
        h = self.out_conv(self.dropout(F.silu(h)))
        return self.skip(x) + h


class AttentionBlock(nn.Module):
    """Multi-head self-attention over spatial positions (pre-norm, residual)."""

    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels {channels} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.norm = _norm(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj = _zero_module(nn.Conv2d(channels, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(b, 3, self.num_heads, c // self.num_heads, h * w).permute(
            1, 0, 2, 4, 3
        )
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class UNet(nn.Module):
    """Class-conditional U-Net for 2D images.

    Args mirror :class:`eps.config.UNetConfig`; ``image_size`` determines
    at which spatial resolutions attention is inserted.
    """

    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 3,
        model_channels: int = 192,
        channel_mult: tuple[int, ...] = (1, 2, 3, 4),
        num_res_blocks: int = 3,
        attention_resolutions: tuple[int, ...] = (16, 8),
        num_heads: int = 6,
        dropout: float = 0.1,
        num_classes: int = 1000,
        time_embed_wmin: float = 0.02,
        time_embed_wmax: float = 100.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.gradient_checkpointing = gradient_checkpointing

        emb_dim = model_channels * 4
        self.time_embed = TimestepEmbedder(
            emb_dim, freq_dim=model_channels, w_min=time_embed_wmin, w_max=time_embed_wmax
        )
        self.label_embed = LabelEmbedder(num_classes, emb_dim)
        self.null_index = self.label_embed.null_index

        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Encoder ----------------------------------------------------------
        self.down_blocks = nn.ModuleList()
        skip_channels = [model_channels]
        ch = model_channels
        res = image_size
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                block = nn.ModuleList(
                    [
                        ResBlock(ch, out_ch, emb_dim, dropout),
                        AttentionBlock(out_ch, num_heads)
                        if res in attention_resolutions
                        else nn.Identity(),
                    ]
                )
                self.down_blocks.append(block)
                ch = out_ch
                skip_channels.append(ch)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch), nn.Identity()]))
                skip_channels.append(ch)
                res //= 2

        # Midcoder ---------------------------------------------------------
        self.mid_block1 = ResBlock(ch, ch, emb_dim, dropout)
        self.mid_attn = AttentionBlock(ch, num_heads)
        self.mid_block2 = ResBlock(ch, ch, emb_dim, dropout)

        # Decoder ----------------------------------------------------------
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            for i in range(num_res_blocks + 1):
                block = nn.ModuleList(
                    [
                        ResBlock(ch + skip_channels.pop(), out_ch, emb_dim, dropout),
                        AttentionBlock(out_ch, num_heads)
                        if res in attention_resolutions
                        else nn.Identity(),
                        Upsample(out_ch)
                        if (level != 0 and i == num_res_blocks)
                        else nn.Identity(),
                    ]
                )
                self.up_blocks.append(block)
                ch = out_ch
            if level != 0:
                res *= 2

        self.out_norm = _norm(ch)
        self.out_conv = _zero_module(nn.Conv2d(ch, in_channels, 3, padding=1))

    @classmethod
    def from_config(cls, cfg: ModelConfig, image_size: int) -> "UNet":
        u = cfg.unet
        return cls(
            image_size=image_size,
            in_channels=u.in_channels,
            model_channels=u.model_channels,
            channel_mult=u.channel_mult,
            num_res_blocks=u.num_res_blocks,
            attention_resolutions=u.attention_resolutions,
            num_heads=u.num_heads,
            dropout=u.dropout,
            num_classes=cfg.num_classes,
            time_embed_wmin=cfg.time_embed_wmin,
            time_embed_wmax=cfg.time_embed_wmax,
            gradient_checkpointing=u.gradient_checkpointing,
        )

    def _apply_block(self, fn, *args):
        if self.gradient_checkpointing and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(self, x: Tensor, t: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """u_t^theta(x|y) (or s_t^theta(x|y)).

        Args:
            x: (B, C, H, W) noisy input x_t.
            t: (B,) times in [0, 1].
            y: (B,) class labels in {0..N-1}, N = null token; None = null.
        """
        emb = self.time_embed(t) + self.label_embed(y, x.shape[0], x.device)

        h = self.conv_in(x)
        skips = [h]
        for block in self.down_blocks:
            first, attn = block
            if isinstance(first, Downsample):
                h = first(h)
            else:
                h = self._apply_block(first, h, emb)
                if not isinstance(attn, nn.Identity):
                    h = self._apply_block(attn, h)
            skips.append(h)

        h = self._apply_block(self.mid_block1, h, emb)
        h = self._apply_block(self.mid_attn, h)
        h = self._apply_block(self.mid_block2, h, emb)

        for block in self.up_blocks:
            res_block, attn, upsample = block
            h = self._apply_block(res_block, torch.cat([h, skips.pop()], dim=1), emb)
            if not isinstance(attn, nn.Identity):
                h = self._apply_block(attn, h)
            h = upsample(h)

        return self.out_conv(F.silu(self.out_norm(h)))
