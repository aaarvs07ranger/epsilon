"""Variational autoencoder for latent diffusion (Sec. 6.2, Algorithm 6).

Implements the beta-VAE with a Gaussian encoder q_phi(z|x) = N(mu_phi(x),
diag(sigma_phi^2(x))) and a fixed-variance Gaussian decoder p_theta(x|z) =
N(mu_theta(z), sigma-bar^2 I) (Eq. 71). The training loss is Algorithm 6:

    L = 1/(2 sigma-bar^2) ||x - x_hat||^2
        + beta * 1/2 sum_j (mu_j^2 + sigma_j^2 - log sigma_j^2 - 1)

At generation time we decode with the mean, x = mu_theta(z) (Remark 32).

For 256/512-scale latent diffusion, :class:`PretrainedVAE` wraps a frozen
``diffusers`` AutoencoderKL behind the same interface, so the flow/diffusion
training loop is agnostic to which autoencoder produced its latents.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import VAEConfig
from .unet import Downsample, ResBlock, Upsample, _norm


class VAEOutput(NamedTuple):
    reconstruction: Tensor
    mean: Tensor
    logvar: Tensor


class _TimelessResBlock(nn.Module):
    """A ResBlock without (t, y) conditioning — the VAE is a plain autoencoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ResBlock(in_channels, out_channels, emb_dim=4, dropout=0.0)
        self.register_buffer("_zero_emb", torch.zeros(1, 4), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x, self._zero_emb.expand(x.shape[0], -1))


class VAE(nn.Module):
    """Convolutional beta-VAE. Downsamples by 2^(len(channel_mult) - 1)."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 128,
        channel_mult: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder: mu_phi(x), log sigma_phi^2(x)
        enc: list[nn.Module] = [nn.Conv2d(in_channels, base_channels, 3, padding=1)]
        ch = base_channels
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            enc += [_TimelessResBlock(ch, out_ch), _TimelessResBlock(out_ch, out_ch)]
            ch = out_ch
            if level != len(channel_mult) - 1:
                enc.append(Downsample(ch))
        enc += [_norm(ch), nn.SiLU(), nn.Conv2d(ch, 2 * latent_channels, 3, padding=1)]
        self.encoder = nn.Sequential(*enc)

        # Decoder: mu_theta(z)
        dec: list[nn.Module] = [nn.Conv2d(latent_channels, ch, 3, padding=1)]
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            dec += [_TimelessResBlock(ch, out_ch), _TimelessResBlock(out_ch, out_ch)]
            ch = out_ch
            if level != 0:
                dec.append(Upsample(ch))
        dec += [_norm(ch), nn.SiLU(), nn.Conv2d(ch, in_channels, 3, padding=1)]
        self.decoder = nn.Sequential(*dec)

    @classmethod
    def from_config(cls, cfg: VAEConfig, in_channels: int = 3) -> "VAE":
        return cls(
            in_channels=in_channels,
            latent_channels=cfg.latent_channels,
            base_channels=cfg.base_channels,
            channel_mult=cfg.channel_mult,
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """x -> (mu, logvar) of q_phi(z|x)."""
        mean, logvar = self.encoder(x).chunk(2, dim=1)
        return mean, torch.clamp(logvar, -30.0, 20.0)

    def reparameterize(self, mean: Tensor, logvar: Tensor) -> Tensor:
        """z = mu + sigma * eps, eps ~ N(0, I) (the reparameterization trick)."""
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> VAEOutput:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return VAEOutput(self.decode(z), mean, logvar)

    def sample_latent(self, x: Tensor) -> Tensor:
        """Draw z ~ q_phi(.|x) — the latent-diffusion training input (Remark 32)."""
        mean, logvar = self.encode(x)
        return self.reparameterize(mean, logvar)


def vae_loss(
    output: VAEOutput,
    x: Tensor,
    kl_weight: float,
    decoder_variance: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Algorithm 6, with a fixed decoder variance sigma-bar^2.

        L_recon = mean over batch of 1/(2 sigma-bar^2) ||x - x_hat||^2
        L_KL    = mean over batch of 1/2 sum_j (mu_j^2 + sigma_j^2 - log sigma_j^2 - 1)
        L       = L_recon + beta * L_KL
    """
    b = x.shape[0]
    recon = ((x - output.reconstruction) ** 2).reshape(b, -1).sum(dim=1)
    recon = recon.mean() / (2.0 * decoder_variance)
    var = torch.exp(output.logvar)
    kl = 0.5 * (output.mean**2 + var - output.logvar - 1.0).reshape(b, -1).sum(dim=1).mean()
    loss = recon + kl_weight * kl
    return loss, {"vae/recon": recon.item(), "vae/kl": kl.item(), "vae/loss": loss.item()}


class PretrainedVAE(nn.Module):
    """Frozen ``diffusers`` AutoencoderKL behind the same encode/decode API.

    Latents are scaled by the model's ``scaling_factor`` so that they are
    approximately unit-variance — the flow/diffusion model then sees a
    well-conditioned latent distribution.
    """

    def __init__(self, model_id: str = "stabilityai/sd-vae-ft-ema") -> None:
        super().__init__()
        try:
            from diffusers import AutoencoderKL
        except ImportError as e:
            raise ImportError(
                "PretrainedVAE requires diffusers: pip install diffusers"
            ) from e
        self.vae = AutoencoderKL.from_pretrained(model_id)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.scaling_factor: float = float(self.vae.config.scaling_factor)
        self.latent_channels: int = int(self.vae.config.latent_channels)

    @torch.no_grad()
    def sample_latent(self, x: Tensor) -> Tensor:
        return self.vae.encode(x).latent_dist.sample() * self.scaling_factor

    @torch.no_grad()
    def decode(self, z: Tensor) -> Tensor:
        return self.vae.decode(z / self.scaling_factor).sample
