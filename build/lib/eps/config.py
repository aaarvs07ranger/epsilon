"""Configuration system for Epsilon.

A deliberately small, dependency-light alternative to Hydra: typed nested
dataclasses, loaded from YAML, with dotted-path command-line overrides
(e.g. ``training.lr=2e-4 model.name=dit``). Every experiment is fully
described by one YAML file; the resolved config is saved into every
checkpoint so runs are exactly reproducible.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml


# ---------------------------------------------------------------------------
# Leaf configs
# ---------------------------------------------------------------------------


@dataclass
class PathConfig:
    """Gaussian probability path p_t(x|z) = N(alpha_t z, beta_t^2 I)  [Eq. 15]."""

    scheduler: str = "condot"  # "condot" (alpha_t = t, beta_t = 1 - t) | "cosine"


@dataclass
class UNetConfig:
    in_channels: int = 3
    model_channels: int = 192
    channel_mult: tuple[int, ...] = (1, 2, 3, 4)
    num_res_blocks: int = 3
    attention_resolutions: tuple[int, ...] = (16, 8)
    num_heads: int = 6
    dropout: float = 0.1
    gradient_checkpointing: bool = False


@dataclass
class DiTConfig:
    in_channels: int = 3
    input_size: int = 64
    patch_size: int = 4
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    cross_attention: bool = False  # enable text cross-attention (Remark 29)
    context_dim: int = 512  # dim of text token embeddings when cross_attention
    gradient_checkpointing: bool = False


@dataclass
class ModelConfig:
    name: str = "unet"  # "unet" | "dit"
    prediction: str = "velocity"  # "velocity" (CFM) | "score" (DSM)
    num_classes: int = 1000  # class-conditional; null/unconditional token appended
    time_embed_wmin: float = 0.02  # Fourier frequencies, Eq. (69)
    time_embed_wmax: float = 100.0
    unet: UNetConfig = field(default_factory=UNetConfig)
    dit: DiTConfig = field(default_factory=DiTConfig)


@dataclass
class VAEConfig:
    """beta-VAE (Algorithm 6). Only used when training.latent_space = true."""

    latent_channels: int = 4
    base_channels: int = 128
    channel_mult: tuple[int, ...] = (1, 2, 4)
    kl_weight: float = 1.0e-6  # beta << 1 (Sec. 6.2 practical remarks)
    decoder_variance: float = 1.0  # fixed sigma-bar^2 of the Gaussian decoder
    pretrained: Optional[str] = None  # HF id for diffusers AutoencoderKL, or None


@dataclass
class DataConfig:
    # "imagenet64": official npz batches, WITH labels -> class-conditional.
    # "imagefolder": root/<class>/*.png, WITH labels -> class-conditional.
    # "flat": a flat dir or .zip of images, NO labels -> unconditional only.
    name: str = "imagenet64"
    root: str = "data/imagenet64"
    image_size: int = 64
    num_workers: int = 8
    horizontal_flip: bool = True
    # For quick local smoke tests: cap the number of samples (None = all).
    max_samples: Optional[int] = None


@dataclass
class TrainingConfig:
    batch_size: int = 128  # per-GPU batch size
    total_steps: int = 400_000
    lr: float = 1.0e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_steps: int = 5_000
    lr_schedule: str = "constant"  # "constant" | "cosine"
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    ema_decay: float = 0.9999
    cfg_dropout: float = 0.1  # eta: prob. of replacing y with null token (Eq. 64)
    mixed_precision: str = "bf16"  # "bf16" | "fp16" | "none"
    # Time-sampling range for t ~ Unif[t_min, t_max]. For score prediction the
    # DSM target -eps/beta_t diverges as beta_t -> 0 (t -> 1), so use t_max < 1.
    t_min: float = 0.0
    t_max: float = 1.0
    latent_space: bool = False  # train in the latent space of a VAE (Sec. 6.2)
    seed: int = 0
    compile: bool = False  # torch.compile the model


@dataclass
class EvalConfig:
    sample_every: int = 5_000  # save a preview grid every N steps (0 = never)
    sample_classes: tuple[int, ...] = (207, 360, 387, 88, 979, 417, 279, 972)
    sample_guidance_scale: float = 4.0
    sample_steps: int = 100
    fid_every: int = 0  # in-training FID every N steps (0 = off; heavy)
    fid_num_samples: int = 10_000
    fid_reference_dir: Optional[str] = None  # folder of real images


@dataclass
class LoggingConfig:
    output_dir: str = "runs/default"
    log_every: int = 100
    ckpt_every: int = 10_000
    wandb: bool = False
    wandb_project: str = "epsilon"
    wandb_run_name: Optional[str] = None


@dataclass
class SamplingConfig:
    """Inference-time defaults (web demo, scripts/sample.py)."""

    method: str = "ode"  # "ode" (flow matching) | "sde" (diffusion)
    solver: str = "euler"  # "euler" | "heun" (ODE only)
    parameterization: str = "velocity"  # "velocity" | "score" (Prop. 1 conversion)
    num_steps: int = 100
    guidance_scale: float = 4.0
    sigma: float = 1.0  # SDE diffusion coefficient sigma_t = sigma (Thm. 17)
    sigma_schedule: str = "constant"  # "constant" | "beta" (sigma_t = sigma * beta_t)
    # Integration endpoints. Score <-> velocity conversion (Prop. 1) divides by
    # alpha_t (t=0) and the SDE score term by beta_t^2 (t=1): clamp both ends.
    t_start: float = 1.0e-4
    t_end: float = 0.9999


@dataclass
class EpsilonConfig:
    path: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


# ---------------------------------------------------------------------------
# YAML <-> dataclass machinery
# ---------------------------------------------------------------------------


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Recursively construct dataclass ``cls`` from a plain dict."""
    kwargs: dict[str, Any] = {}
    valid = {f.name: f for f in dataclasses.fields(cls)}
    for key, value in data.items():
        if key not in valid:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        f = valid[key]
        if dataclasses.is_dataclass(f.type) and isinstance(value, dict):
            kwargs[key] = _build(f.type, value)
        elif isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _resolve_field_types(cls: type) -> None:
    """Resolve string annotations (PEP 563) into real types, in place."""
    hints = dataclasses.fields(cls)
    import typing

    resolved = typing.get_type_hints(cls)
    for f in hints:
        f.type = resolved[f.name]
        if dataclasses.is_dataclass(f.type):
            _resolve_field_types(f.type)


_resolve_field_types(EpsilonConfig)


def to_dict(cfg: EpsilonConfig) -> dict[str, Any]:
    """Config -> plain dict (tuples become lists), for YAML/checkpoints."""

    def convert(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj):
            return {f.name: convert(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if isinstance(obj, tuple):
            return [convert(v) for v in obj]
        return obj

    return convert(cfg)


def from_dict(data: dict[str, Any]) -> EpsilonConfig:
    return _build(EpsilonConfig, data)


def apply_overrides(cfg: EpsilonConfig, overrides: list[str]) -> EpsilonConfig:
    """Apply dotted-path overrides like ``training.lr=2e-4``.

    Values are parsed with YAML so numbers, booleans, lists, and strings all
    work: ``model.unet.channel_mult=[1,2,4]``.
    """
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' is not of the form key.path=value")
        dotted, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        if isinstance(value, str):
            # YAML 1.1 parses "2e-4" as a string; users type it as a number.
            try:
                value = float(value)
            except ValueError:
                pass
        parts = dotted.strip().split(".")
        obj: Any = cfg
        for part in parts[:-1]:
            if not hasattr(obj, part):
                raise KeyError(f"Unknown config section '{part}' in override '{item}'")
            obj = getattr(obj, part)
        leaf = parts[-1]
        if not hasattr(obj, leaf):
            raise KeyError(f"Unknown config key '{leaf}' in override '{item}'")
        if isinstance(value, list):
            value = tuple(value)
        setattr(obj, leaf, value)
    return cfg


def load_config(
    path: Union[str, Path, None] = None, overrides: Optional[list[str]] = None
) -> EpsilonConfig:
    """Load a config from YAML (defaults if ``path`` is None), then apply overrides."""
    if path is None:
        cfg = EpsilonConfig()
    else:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        cfg = from_dict(data)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def save_config(cfg: EpsilonConfig, path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)
