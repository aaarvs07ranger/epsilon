"""Small shared utilities: image conversion and grid saving (PIL only)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def to_uint8(x: Tensor) -> Tensor:
    """Map images from model space [-1, 1] to uint8 [0, 255], (B, C, H, W)."""
    x = ((x.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
    return x


def tensor_to_pil(x: Tensor) -> Image.Image:
    """(C, H, W) in [-1, 1] -> PIL image."""
    arr = to_uint8(x[None])[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def make_grid(x: Tensor, num_columns: int = 8, padding: int = 2) -> Image.Image:
    """(B, C, H, W) in [-1, 1] -> one tiled PIL image."""
    b, c, h, w = x.shape
    cols = min(num_columns, b)
    rows = math.ceil(b / cols)
    grid = np.full(
        (rows * (h + padding) - padding, cols * (w + padding) - padding, c), 255, dtype=np.uint8
    )
    imgs = to_uint8(x).permute(0, 2, 3, 1).cpu().numpy()
    for i in range(b):
        r, col = divmod(i, cols)
        grid[r * (h + padding) : r * (h + padding) + h, col * (w + padding) : col * (w + padding) + w] = imgs[i]
    return Image.fromarray(grid.squeeze())


def save_image_grid(x: Tensor, path: Union[str, Path], num_columns: int = 8) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    make_grid(x, num_columns=num_columns).save(path)
