#!/usr/bin/env python
"""Strip a training checkpoint down to what inference actually needs.

A training checkpoint carries four copies of the network: the live weights, the
EMA shadow, and two Adam moments. Inference needs one. Dropping the rest and
casting to fp16 takes the 92.5M U-Net from 1.4 GB to ~185 MB — which is the
difference between "awkward to host" and "fits anywhere".

What this writes:
    config  the resolved EpsilonConfig, so the model rebuilds exactly
    model   the EMA weights folded into the parameter tensors, plus the
            trained buffers
    step    provenance

What it drops: `optimizer`, `scaler`, and the separate `ema` entry.

`eps.web.app._load_checkpoint` and `scripts/sample.py` both apply the EMA
weights only when an `ema` key is present, so folding EMA into `model` and
omitting the key gives identical sampling from a quarter of the bytes.

    python scripts/export_inference_ckpt.py \
        --ckpt runs/cloud_unet/ckpt_final.pt --out deploy/unet.pt

    # keep fp32 if you want bit-identical outputs to the training run
    python scripts/export_inference_ckpt.py --ckpt ... --out ... --no-fp16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eps.config import from_dict, to_dict
from eps.models import build_model
from eps.training.ema import EMA


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="training checkpoint to strip")
    p.add_argument("--out", required=True, help="destination for the slim checkpoint")
    p.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="keep fp32 weights (2x larger; use if you want bit-identical output)",
    )
    p.add_argument(
        "--no-ema",
        dest="use_ema",
        action="store_false",
        help="export the live weights instead of the EMA ones (you almost never want this)",
    )
    args = p.parse_args()

    src = Path(args.ckpt)
    if not src.exists():
        raise SystemExit(f"no such checkpoint: {src}")

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    cfg = from_dict(ckpt["config"])
    size = cfg.data.image_size // 8 if cfg.training.latent_space else cfg.data.image_size

    # Rebuild on CPU, load the trained weights, then overwrite the *parameters*
    # with the EMA shadow. Buffers are not tracked by EMA and must come from the
    # live state dict, which is exactly what this ordering gives.
    net = build_model(cfg.model, size)
    net.load_state_dict(ckpt["model"])
    if args.use_ema and "ema" in ckpt:
        ema = EMA(net)
        ema.load_state_dict(ckpt["ema"])
        ema.copy_to(net)
        source = "EMA weights"
    else:
        source = "live weights" + ("" if args.use_ema else " (--no-ema)")

    state = net.state_dict()
    if args.fp16:
        # load_state_dict casts on copy_, so an fp32 model loads these fine.
        state = {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": to_dict(cfg),
            "model": state,
            "step": ckpt.get("step"),
            "note": f"inference-only export of {src.name} ({source}, "
            f"{'fp16' if args.fp16 else 'fp32'}); optimizer/scaler/ema removed",
        },
        out,
    )

    before, after = src.stat().st_size, out.stat().st_size
    params = sum(v.numel() for v in state.values())
    print(f"{src}  ->  {out}")
    print(f"  source     : {source}, step {ckpt.get('step')}")
    print(f"  tensors    : {params / 1e6:.1f}M values, {'fp16' if args.fp16 else 'fp32'}")
    print(f"  size       : {before / 1e9:.2f} GB -> {after / 1e6:.0f} MB ({before / after:.1f}x smaller)")


if __name__ == "__main__":
    main()
