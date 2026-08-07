#!/usr/bin/env python
"""Evaluate a trained checkpoint: FID, Inception Score, precision/recall.

1) Export the reference set once (real images as PNGs):
    python scripts/evaluate_fid.py export-ref --data-root data/imagenet64 \
        --out data/fid_ref --num 50000

2) Generate samples and compute metrics:
    python scripts/evaluate_fid.py run --ckpt runs/unet64/ckpt_latest.pt \
        --ref data/fid_ref --num 50000 --guidance 1.5 --steps 100

Note: FID improves with more samples; report the 50k number. CFG guidance
trades diversity for fidelity — sweep guidance in [1.0, 2.0] for the best FID
(high w looks better per-image but hurts FID's diversity term).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eps.data.imagenet import ImageNet64
from eps.evaluation.fid import compute_metrics, generate_samples
from eps.paths import GaussianProbabilityPath, build_scheduler
from eps.utils import tensor_to_pil


def export_reference(data_root: str, out: str, num: int, split: str = "train", seed: int = 0) -> None:
    """Export ``num`` real images as PNGs to serve as the FID reference.

    The subset is a *seeded random sample*, never a prefix: some sources
    (notably the Hugging Face 64x64 repack) store images in ascending class
    order, and a prefix of 50k there would cover ~39 of the 1000 classes —
    a reference set with almost no diversity, against which FID is meaningless.
    """
    import numpy as np

    ds = ImageNet64(data_root, split=split, horizontal_flip=False)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(num, len(ds))
    idx = np.random.default_rng(seed).permutation(len(ds))[:n]
    labels = set()
    for j, i in enumerate(idx):
        x, y = ds[int(i)]
        labels.add(int(y))
        tensor_to_pil(x).save(out_dir / f"{j:06d}.png")
        if (j + 1) % 5000 == 0:
            print(f"exported {j + 1}/{n}")
    print(f"wrote {n} reference images to {out_dir} covering {len(labels)} classes")
    if len(labels) < 900:
        print(
            f"WARNING: only {len(labels)} distinct classes in the reference set. "
            "FID against a low-diversity reference is not comparable to published "
            "ImageNet-64 numbers — check that your dataset is complete."
        )


def run_eval(args: argparse.Namespace) -> None:
    from scripts.sample import load_model  # reuse checkpoint loading

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    net, cfg = load_model(args.ckpt, device, use_ema=not args.no_ema)
    path = GaussianProbabilityPath(build_scheduler(cfg.path.scheduler))

    gen_dir = Path(args.out or (Path(args.ckpt).parent / "fid_samples"))
    generate_samples(
        net, path, cfg, gen_dir,
        num_samples=args.num, batch_size=args.batch_size, device=device,
        seed=args.seed, guidance_scale=args.guidance, num_steps=args.steps,
    )
    print(f"generated {args.num} samples in {gen_dir}; computing metrics...")
    metrics = compute_metrics(gen_dir, args.ref, backend=args.backend)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ref = sub.add_parser("export-ref", help="export real images for the FID reference")
    p_ref.add_argument("--data-root", type=str, required=True)
    p_ref.add_argument("--out", type=str, required=True)
    p_ref.add_argument("--num", type=int, default=50_000)
    p_ref.add_argument("--split", type=str, default="train", choices=["train", "val"])
    p_ref.add_argument("--seed", type=int, default=0, help="seed for the random subset")

    p_run = sub.add_parser("run", help="generate samples and compute metrics")
    p_run.add_argument("--ckpt", type=str, required=True)
    p_run.add_argument("--ref", type=str, required=True)
    p_run.add_argument("--num", type=int, default=50_000)
    p_run.add_argument("--batch-size", type=int, default=128)
    p_run.add_argument("--guidance", type=float, default=1.5)
    p_run.add_argument("--steps", type=int, default=100)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--out", type=str, default=None)
    p_run.add_argument("--backend", type=str, default="torch-fidelity",
                       choices=["torch-fidelity", "clean-fid"])
    p_run.add_argument("--no-ema", action="store_true")

    args = parser.parse_args()
    if args.cmd == "export-ref":
        export_reference(args.data_root, args.out, args.num, split=args.split, seed=args.seed)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
