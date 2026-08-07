#!/usr/bin/env python
"""Generate a grid of samples from a trained checkpoint.

    python scripts/sample.py --ckpt runs/unet64/ckpt_latest.pt \
        --classes 207 88 979 417 --guidance 4.0 --steps 100 --method ode \
        --out samples.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eps.config import from_dict
from eps.evaluation.fid import sample_batch
from eps.models import build_model
from eps.paths import GaussianProbabilityPath, build_scheduler
from eps.training.ema import EMA
from eps.utils import save_image_grid


def load_model(ckpt_path: str, device: torch.device, use_ema: bool = True):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = from_dict(ckpt["config"])
    sample_size = cfg.data.image_size // 8 if cfg.training.latent_space else cfg.data.image_size
    net = build_model(cfg.model, sample_size).to(device)
    net.load_state_dict(ckpt["model"])
    if use_ema and "ema" in ckpt:
        ema = EMA(net)
        ema.load_state_dict(ckpt["ema"])
        ema.copy_to(net)
    net.eval()
    return net, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--classes", type=int, nargs="+", default=[207, 88, 979, 417])
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--method", type=str, default="ode", choices=["ode", "sde"])
    parser.add_argument(
        "--parameterization", type=str, default="velocity", choices=["velocity", "score"]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="samples.png")
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    net, cfg = load_model(args.ckpt, device, use_ema=not args.no_ema)
    path = GaussianProbabilityPath(build_scheduler(cfg.path.scheduler))

    y = torch.tensor(args.classes, device=device).repeat_interleave(args.per_class)
    torch.manual_seed(args.seed)
    in_ch = cfg.model.unet.in_channels if cfg.model.name == "unet" else cfg.model.dit.in_channels
    size = cfg.data.image_size // 8 if cfg.training.latent_space else cfg.data.image_size
    x = sample_batch(
        net, path, cfg, y, y.shape[0], (in_ch, size, size), device,
        guidance_scale=args.guidance, num_steps=args.steps,
        method=args.method, parameterization=args.parameterization,
    )
    save_image_grid(x, args.out, num_columns=args.per_class)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
