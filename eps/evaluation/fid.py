"""Sample generation and generative metrics (FID, Inception Score,
precision/recall).

Metrics backends (both optional dependencies):
  * ``clean-fid`` — the community-standard FID implementation.
  * ``torch-fidelity`` — FID + IS + precision/recall in one call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from ..config import EpsilonConfig
from ..paths import GaussianProbabilityPath
from ..sampling import GuidedModel, integrate_ode, integrate_sde, make_sigma_fn
from ..utils import tensor_to_pil


@torch.no_grad()
def sample_batch(
    net: nn.Module,
    path: GaussianProbabilityPath,
    cfg: EpsilonConfig,
    y: Optional[torch.Tensor],
    batch_size: int,
    image_shape: tuple[int, int, int],
    device: torch.device,
    generator: Optional[torch.Generator] = None,
    guidance_scale: Optional[float] = None,
    num_steps: Optional[int] = None,
    method: Optional[str] = None,
    parameterization: Optional[str] = None,
    callback=None,
) -> torch.Tensor:
    """Draw one batch of samples with the configured sampler.

    All sampler knobs default to ``cfg.sampling`` but can be overridden
    (the web demo passes its per-request settings through here).
    """
    s = cfg.sampling
    method = method or s.method
    parameterization = parameterization or s.parameterization
    num_steps = num_steps or s.num_steps
    w = s.guidance_scale if guidance_scale is None else guidance_scale

    guided = GuidedModel(
        net,
        path,
        prediction=cfg.model.prediction,
        y=y,
        null_index=cfg.model.num_classes,
        guidance_scale=w,
    )
    x0 = torch.randn(batch_size, *image_shape, device=device, generator=generator)

    if method == "ode":
        # The "parameterization" toggle chooses which field drives the solver:
        # the velocity directly, or the velocity reconstructed from the score
        # via Prop. 1 — mathematically identical, numerically distinct paths.
        if parameterization == "velocity":
            fn = guided.velocity
            t_start, t_end = 0.0, 1.0
            if cfg.model.prediction == "score":
                t_start, t_end = s.t_start, s.t_end  # conversion singular at ends
        else:
            fn = lambda x, t: path.velocity_from_score(guided.score(x, t), x, t)
            t_start, t_end = s.t_start, s.t_end
        return integrate_ode(
            fn, x0, num_steps, t_start=t_start, t_end=t_end, solver=s.solver, callback=callback
        )
    if method == "sde":
        sigma_fn = make_sigma_fn(s.sigma, s.sigma_schedule, path.scheduler)
        return integrate_sde(
            guided.velocity_and_score,
            x0,
            num_steps,
            sigma_fn,
            t_start=s.t_start,
            t_end=s.t_end,
            callback=callback,
        )
    raise ValueError(f"Unknown sampling method '{method}'")


@torch.no_grad()
def generate_samples(
    net: nn.Module,
    path: GaussianProbabilityPath,
    cfg: EpsilonConfig,
    out_dir: Union[str, Path],
    num_samples: int,
    batch_size: int,
    device: torch.device,
    seed: int = 0,
    guidance_scale: Optional[float] = None,
    num_steps: Optional[int] = None,
) -> Path:
    """Generate ``num_samples`` PNGs (uniformly random classes) into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_shape = (cfg.model.unet.in_channels if cfg.model.name == "unet" else cfg.model.dit.in_channels,
                   cfg.data.image_size, cfg.data.image_size)
    generator = torch.Generator(device=device).manual_seed(seed)
    net.eval()
    written = 0
    while written < num_samples:
        b = min(batch_size, num_samples - written)
        y = torch.randint(
            0, cfg.model.num_classes, (b,), device=device, generator=generator
        )
        x = sample_batch(
            net, path, cfg, y, b, image_shape, device,
            generator=generator, guidance_scale=guidance_scale, num_steps=num_steps,
        )
        for i in range(b):
            tensor_to_pil(x[i]).save(out_dir / f"{written + i:06d}.png")
        written += b
    return out_dir


def compute_metrics(
    gen_dir: Union[str, Path],
    ref: Union[str, Path],
    backend: str = "torch-fidelity",
) -> dict[str, float]:
    """FID (+ IS, precision/recall with torch-fidelity) between generated
    images and a reference directory of real images."""
    gen_dir, ref = str(gen_dir), str(ref)
    if backend == "clean-fid":
        from cleanfid import fid as cleanfid

        return {"fid": float(cleanfid.compute_fid(gen_dir, ref))}
    if backend == "torch-fidelity":
        import torch_fidelity

        out = torch_fidelity.calculate_metrics(
            input1=gen_dir,
            input2=ref,
            fid=True,
            isc=True,
            prc=True,
            batch_size=64,
            cuda=torch.cuda.is_available(),
            verbose=False,
        )
        return {
            "fid": float(out["frechet_inception_distance"]),
            "inception_score": float(out["inception_score_mean"]),
            "precision": float(out["precision"]),
            "recall": float(out["recall"]),
        }
    raise ValueError(f"Unknown metrics backend '{backend}'")
