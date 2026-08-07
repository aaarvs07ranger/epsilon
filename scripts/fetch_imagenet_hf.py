#!/usr/bin/env python
"""Fetch **labeled** ImageNet-1K 64x64 from the Hugging Face Hub and write it in
the official Downsampled-ImageNet npz layout that :class:`eps.data.ImageNet64`
already reads.

Why this exists
---------------
Class-conditional training, classifier-free guidance, and the demo's class
picker all need (image, label) pairs. The Kaggle mirror
(``train_64x64.zip``) is 1,281,149 flat PNGs with **no labels**, and the
official npz release from image-net.org is behind a manual approval queue.

``benjamin-paine/imagenet-1k-64x64`` is an ungated repack of the same 1000
classes with integer labels, in Parquet, at ~1.8 GB for the training split.
Label indices follow the standard sorted-synset order, so they are already
index-aligned with ``eps/data/imagenet_classes.txt`` (0 = tench,
207 = golden retriever).

Output layout (identical to image-net.org's release):
    train_data_batch_1.npz ... train_data_batch_N.npz   (and val_data.npz)
each with ``data`` (N, 3*S*S) uint8 channel-major and ``labels`` (N,) int64
in **1..1000** — the loader shifts these to 0-based.

Examples
--------
    # smoke test: 20k labeled images, ~1 minute
    python scripts/fetch_imagenet_hf.py --split train --limit 20000 \
        --out data/imagenet64

    # the real thing: full 1.28M training split
    python scripts/fetch_imagenet_hf.py --split train --out data/imagenet64

    # the validation split, used to export the FID reference
    python scripts/fetch_imagenet_hf.py --split validation --out data/imagenet64

Ordering
--------
The repo's **train** split is stored in ascending class order, unlike the
official release which is shuffled. Left as-is, the first N images are the
first N/1281 classes — which silently breaks anything that takes a prefix
(``data.max_samples``, a FID reference export). ``--shuffle`` (the default for
``train``) therefore does a true global permutation via a temporary
disk-backed memmap, so the shards are a drop-in for the official layout.
The **validation** split is already shuffled upstream.

Caveat worth recording in your writeup: this repack was resized with Lanczos
and stored as JPEG, whereas the official npz release is raw uint8 from a box
filter. FID against your own exported reference is self-consistent either way,
but absolute numbers are not directly comparable to published ImageNet-64 FIDs
computed on the official release.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_REPO = "benjamin-paine/imagenet-1k-64x64"


def _shard_files(repo: str, split: str) -> list[str]:
    """The parquet files for ``split``, in order."""
    from huggingface_hub import list_repo_files

    files = [
        f
        for f in list_repo_files(repo, repo_type="dataset")
        if f.startswith("data/") and f.endswith(".parquet") and Path(f).name.startswith(split)
    ]
    if not files:
        raise SystemExit(f"No parquet files for split '{split}' in {repo}")
    return sorted(files)


def _to_row(blob: bytes, size: int) -> np.ndarray:
    """Encoded image bytes -> (3*size*size,) uint8, channel-major (CHW flat)."""
    img = Image.open(io.BytesIO(blob)).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8).transpose(2, 0, 1).reshape(-1)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo", default=DEFAULT_REPO, help="HF dataset repo id")
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--out", required=True, help="output directory for npz shards")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--shard-size", type=int, default=128_000, help="images per npz shard")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after N images. Because the source is class-sorted this "
        "takes a *class prefix*, not a random subset — it proves the pipeline "
        "runs, it is not a miniature ImageNet. Omit it for the real dataset.",
    )
    p.add_argument(
        "--keep-parquet",
        action="store_true",
        help="keep the downloaded parquet files (default: delete each after conversion)",
    )
    p.add_argument(
        "--shuffle",
        dest="shuffle",
        action="store_true",
        default=None,
        help="globally permute before sharding (default: on for train, off otherwise)",
    )
    p.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    p.add_argument("--seed", type=int, default=0, help="shuffle seed")
    p.add_argument(
        "--tmp-dir",
        default=None,
        help="where the shuffle scratch memmap lives (default: alongside --out). "
        "Needs ~12 GB for the full training split.",
    )
    args = p.parse_args()
    if args.shuffle is None:
        args.shuffle = args.split == "train"

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = _shard_files(args.repo, args.split)
    print(f"[fetch] {args.repo}:{args.split} -> {len(files)} parquet shard(s)")

    row_dim = 3 * args.image_size * args.image_size

    def shard_name(idx: int) -> str:
        if args.split == "validation":
            # ImageNet64 globs "val_data*.npz", so extra shards are still found
            # if --shard-size is small enough to split the 50k validation set.
            return "val_data.npz" if idx == 1 else f"val_data_{idx}.npz"
        return f"{args.split}_data_batch_{idx}.npz"

    def write_shard(idx: int, data: np.ndarray, labels: np.ndarray) -> None:
        name = shard_name(idx)
        np.savez(
            out / name,
            data=data,
            # Official release labels 1..1000; ImageNet64 shifts to 0-based.
            labels=labels.astype(np.int64) + 1,
        )
        print(f"[fetch] wrote {name}  ({data.shape[0]:,} images, labeled)")

    # -- pass 1: decode every image into a flat scratch file --------------------
    # Rows go to disk unshuffled and are read back in permuted order below, so
    # peak memory is one shard (shard_size * row_dim bytes), not the dataset.
    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else out
    tmp_dir.mkdir(parents=True, exist_ok=True)
    scratch = tmp_dir / f".fetch_{args.split}_scratch.u8"

    labels_acc: list[int] = []
    total = 0
    stop = False
    try:
        with open(scratch, "wb") as sink:
            for f in files:
                if stop:
                    break
                print(f"[fetch] downloading {f} ...")
                local = Path(hf_hub_download(args.repo, f, repo_type="dataset"))
                pf = pq.ParquetFile(local)
                for batch in pf.iter_batches(batch_size=1024, columns=["image", "label"]):
                    for img_struct, label in zip(
                        batch.column("image").to_pylist(), batch.column("label").to_pylist()
                    ):
                        row = _to_row(img_struct["bytes"], args.image_size)
                        if row.shape[0] != row_dim:
                            raise SystemExit(
                                f"unexpected row size {row.shape[0]}, expected {row_dim}"
                            )
                        sink.write(row.tobytes())
                        labels_acc.append(int(label))
                        total += 1
                        if total % 20_000 == 0:
                            print(f"[fetch] {total:,} images decoded")
                        if args.limit and total >= args.limit:
                            stop = True
                            break
                    if stop:
                        break
                if not args.keep_parquet:
                    # The HF cache keeps the blob; unlink the cached file to
                    # reclaim space as we go (matters on /gscratch and a laptop).
                    try:
                        local.resolve().unlink()
                    except OSError:
                        pass

        if total == 0:
            raise SystemExit("[fetch] no images decoded")

        # -- pass 2: permute, then write shards ----------------------------------
        all_labels = np.asarray(labels_acc, dtype=np.int64)
        order = np.arange(total)
        if args.shuffle:
            np.random.default_rng(args.seed).shuffle(order)
            print(f"[fetch] globally shuffling {total:,} rows (seed {args.seed})")

        mm = np.memmap(scratch, dtype=np.uint8, mode="r", shape=(total, row_dim))
        rng = np.random.default_rng(args.seed + 1)
        n_shards = 0
        for start in range(0, total, args.shard_size):
            idx = np.sort(order[start : start + args.shard_size])
            # Read in ascending file order — one forward scan instead of random
            # I/O across the scratch file — then permute the block in memory, so
            # rows are shuffled *within* the shard too. Without this second step
            # each shard is a random subset but still class-ordered inside, and
            # anything taking a prefix (data.max_samples, the FID reference
            # export) would still see only the lowest class ids.
            block, block_labels = np.asarray(mm[idx]), all_labels[idx]
            inner = rng.permutation(block.shape[0]) if args.shuffle else np.arange(block.shape[0])
            n_shards += 1
            write_shard(n_shards, block[inner], block_labels[inner])
        del mm
    finally:
        scratch.unlink(missing_ok=True)

    print(f"[fetch] done: {total:,} labeled images -> {n_shards} shard(s) in {out}")
    print(f"[fetch] classes present: {len(np.unique(all_labels))}/1000")
    if args.limit:
        print(
            "[fetch] NOTE: --limit was set, this is a partial dataset "
            "(fine for smoke tests, not for the real run)."
        )


if __name__ == "__main__":
    main()
