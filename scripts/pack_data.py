#!/usr/bin/env python
"""Pack an image dataset into npz shards in the official Downsampled-ImageNet
layout: ``data`` (N, 3*S*S) uint8 channel-major, ``labels`` (N,) int64 1-based
(omitted entirely when the source carries no class annotations).

Why: a folder of ~1.3M individual PNGs is the worst case for a shared HPC
filesystem — every epoch pays 1.3M metadata lookups, and /gscratch enforces
inode quotas. A handful of npz shards turns that into a few large sequential
reads, and the training loop keeps the whole thing in RAM as uint8.

Sources:
  * a .zip of images (read in place — never extracted), flat = unlabeled
  * a flat directory of images = unlabeled
  * a directory of class subdirectories (root/<class>/*.png) = labeled

Examples:
    # the Kaggle mirror you downloaded (unlabeled -> unconditional training)
    python scripts/pack_data.py \
        --input eps/data/imagenet64/train_64x64.zip \
        --out data/imagenet64_packed --shard-size 128000

    # a labeled class-subdirectory tree
    python scripts/pack_data.py --input data/raw_imagenet --labeled \
        --out data/imagenet64_packed
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _center_crop_resize(img: Image.Image, size: int) -> Image.Image:
    if img.size == (size, size):
        return img
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def _to_row(img: Image.Image, size: int) -> np.ndarray:
    """PIL image -> (3*size*size,) uint8, channel-major (CHW flattened)."""
    arr = np.asarray(_center_crop_resize(img.convert("RGB"), size), dtype=np.uint8)
    return arr.transpose(2, 0, 1).reshape(-1)


def iter_zip(path: Path, size: int):
    with zipfile.ZipFile(path) as zf:
        names = sorted(
            n for n in zf.namelist()
            if Path(n).suffix.lower() in EXTENSIONS and not n.startswith("__")
        )
        print(f"[pack] {len(names):,} images inside {path.name}")
        for n in names:
            yield _to_row(Image.open(io.BytesIO(zf.read(n))), size), None


def iter_dir(root: Path, size: int, labeled: bool):
    if labeled:
        class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
        if not class_dirs:
            raise SystemExit(f"--labeled given but no class subdirectories under {root}")
        print(f"[pack] {len(class_dirs)} classes under {root}")
        (root / "classes.txt").write_text("\n".join(d.name for d in class_dirs))
        for label, d in enumerate(class_dirs):
            for f in sorted(d.rglob("*")):
                if f.suffix.lower() in EXTENSIONS:
                    yield _to_row(Image.open(f), size), label + 1  # 1-based
    else:
        files = sorted(f for f in root.rglob("*") if f.suffix.lower() in EXTENSIONS)
        print(f"[pack] {len(files):,} images under {root}")
        for f in files:
            yield _to_row(Image.open(f), size), None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help=".zip, flat dir, or class-subdir tree")
    p.add_argument("--out", required=True, help="output directory for npz shards")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--shard-size", type=int, default=128_000, help="images per npz shard")
    p.add_argument("--labeled", action="store_true",
                   help="treat --input as root/<class>/*.png and emit labels")
    p.add_argument("--limit", type=int, default=None, help="stop after N images (testing)")
    args = p.parse_args()

    src = Path(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if src.suffix == ".zip":
        if args.labeled:
            raise SystemExit("--labeled is not supported for .zip input (no class structure)")
        stream = iter_zip(src, args.image_size)
    else:
        stream = iter_dir(src, args.image_size, args.labeled)

    row_dim = 3 * args.image_size * args.image_size
    shard, labels, shard_idx, total = [], [], 1, 0

    def flush() -> None:
        nonlocal shard, labels, shard_idx
        if not shard:
            return
        path = out / f"train_data_batch_{shard_idx}.npz"
        payload = {"data": np.stack(shard).astype(np.uint8)}
        if any(l is not None for l in labels):
            payload["labels"] = np.asarray(labels, dtype=np.int64)
        np.savez(path, **payload)
        print(f"[pack] wrote {path.name}  ({len(shard):,} images"
              f"{', labeled' if 'labels' in payload else ', UNLABELED'})")
        shard, labels = [], []
        shard_idx += 1

    for row, label in stream:
        if row.shape[0] != row_dim:
            raise SystemExit(f"unexpected row size {row.shape[0]}, expected {row_dim}")
        shard.append(row)
        labels.append(label)
        total += 1
        if total % 20_000 == 0:
            print(f"[pack] {total:,} images processed")
        if len(shard) >= args.shard_size:
            flush()
        if args.limit and total >= args.limit:
            break
    flush()

    print(f"[pack] done: {total:,} images -> {shard_idx - 1} shard(s) in {out}")
    if not args.labeled and src.suffix == ".zip":
        print("[pack] NOTE: no labels were available. These shards support "
              "UNCONDITIONAL training only (data.name=imagenet64 will read "
              "them and mark every sample UNLABELED).")


if __name__ == "__main__":
    main()
