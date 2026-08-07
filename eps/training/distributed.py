"""Distributed training utilities (torchrun / SLURM compatible).

Single-process runs (MacBook, debugging) and multi-GPU DDP runs (Hyak) share
one code path: every helper degrades gracefully when torch.distributed is not
initialised.
"""

from __future__ import annotations

import os
import random
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_distributed() -> torch.device:
    """Initialise the process group if launched by torchrun; pick the device.

    Returns the device this process should use (cuda:LOCAL_RANK, mps, or cpu).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def wrap_ddp(model: nn.Module, device: torch.device) -> nn.Module:
    """Wrap in DistributedDataParallel when running distributed."""
    if not is_distributed():
        return model
    ids = [device.index] if device.type == "cuda" else None
    return DistributedDataParallel(model, device_ids=ids)


def unwrap_ddp(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def seed_everything(seed: int) -> None:
    """Deterministic-ish seeding, offset by rank so data noise decorrelates."""
    s = seed + get_rank()
    random.seed(s)
    np.random.seed(s % 2**32)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
