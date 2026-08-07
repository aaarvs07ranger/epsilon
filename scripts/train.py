#!/usr/bin/env python
"""Train an Epsilon flow-matching / diffusion model.

Single process (MacBook):
    python scripts/train.py --config configs/train_unet.yaml \
        training.batch_size=32 data.max_samples=10000

Multi-GPU (Hyak, one node):
    torchrun --nproc_per_node=8 scripts/train.py --config configs/train_unet.yaml

Resume:
    python scripts/train.py --config configs/train_unet.yaml \
        --resume runs/unet64/ckpt_latest.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eps.config import load_config
from eps.training import Trainer
from eps.training.distributed import cleanup_distributed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint to resume from")
    parser.add_argument(
        "overrides", nargs="*", help="dotted config overrides, e.g. training.lr=2e-4"
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    trainer = Trainer(cfg, resume=args.resume)
    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
