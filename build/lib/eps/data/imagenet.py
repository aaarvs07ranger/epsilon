"""Dataset classes for Epsilon.

Three sources, distinguished by whether they carry class labels:

===============  =======================================  ========
``data.name``    layout                                   labels?
===============  =======================================  ========
``imagenet64``   official Downsampled-ImageNet npz shards  yes
``imagefolder``  ``root/<class>/*.png``                    yes
``flat``         a flat directory, or a ``.zip`` read      no
                 in place without extracting
===============  =======================================  ========

Only the labeled sources can train the class-conditional model with
classifier-free guidance; ``flat`` exists so that the unlabeled Kaggle mirror
of ImageNet-64 is still usable for unconditional throughput smoke tests.

Conventions that matter
-----------------------
* Images are returned in **model space, [-1, 1]**, shaped ``(C, H, W)`` float32.
  This is the exact inverse of :func:`eps.utils.to_uint8`.
* The npz layout stores ``labels`` **1-based (1..1000)**, matching the official
  image-net.org release; this module shifts them to 0-based on load. A shard
  with no ``labels`` member at all is unlabeled, and every label becomes
  :data:`UNLABELED`.
* :data:`UNLABELED` is ``-1``, never ``num_classes``. The trainer maps negative
  labels to the null token *before* CFG dropout, so the unconditional branch is
  trained either way — but keeping the sentinel distinct from the null index
  means an unlabeled dataset cannot be silently mistaken for a labeled one.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib import format as npformat
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

# Sentinel label for sources that carry no class annotations. The trainer
# routes anything negative to the null token (see trainer.py).
UNLABELED = -1

_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

_CLASS_FILE = Path(__file__).resolve().parent / "imagenet_classes.txt"


def imagenet_class_names() -> list[str]:
    """The 1000 ImageNet-1K class names, index-aligned with the label ids.

    Index 0 is ``tench``, 207 is ``golden retriever``, 979 is ``valley``. The
    file deliberately has no trailing newline, so ``wc -l`` reports 999 while
    this returns 1000 entries.
    """
    names = _CLASS_FILE.read_text().strip().split("\n")
    if len(names) != 1000:
        raise ValueError(f"{_CLASS_FILE} has {len(names)} entries, expected 1000")
    return names


def _to_model_space(u8: np.ndarray) -> Tensor:
    """(C, H, W) uint8 -> (C, H, W) float32 in [-1, 1]."""
    return torch.from_numpy(np.ascontiguousarray(u8)).float().div_(127.5).sub_(1.0)


def _resize(x: Tensor, size: int) -> Tensor:
    """Resize ``(C, H, W)`` to ``size`` only if it is not already that size.

    ``area`` for downscaling is a box filter, which is what the official
    Downsampled-ImageNet release used; bilinear for the (unusual) upscale case.
    This is what makes ``data.image_size=32`` a one-line config change.
    """
    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    mode = "area" if x.shape[-1] > size else "bilinear"
    kwargs = {} if mode == "area" else {"align_corners": False}
    return F.interpolate(x[None], size=(size, size), mode=mode, **kwargs)[0]


def _npz_member_shape(path: Path, member: str) -> Optional[tuple]:
    """Read an npz member's shape from its header without decoding the array.

    ``np.savez`` stores uncompressed ``.npy`` members, so the shape is readable
    from a ~128-byte header. This lets :class:`ImageNet64` preallocate one
    contiguous array and fill it shard by shard. Concatenating instead would
    momentarily hold two copies of the split — 31 GB for the full training set,
    which does not fit in 36 GB of unified memory shared with the GPU.

    Returns ``None`` if the header cannot be read, in which case the caller
    falls back to loading shards and concatenating.
    """
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            name = f"{member}.npy"
            if name not in names:
                return None
            with z.open(name) as fh:
                version = npformat.read_magic(fh)
                if version == (1, 0):
                    shape, _, _ = npformat.read_array_header_1_0(fh)
                elif version == (2, 0):
                    shape, _, _ = npformat.read_array_header_2_0(fh)
                else:
                    return None
        return tuple(shape)
    except Exception:
        return None


class ImageNet64(Dataset):
    """Downsampled ImageNet from npz shards, held in RAM as uint8.

    ``root`` holds ``train_data_batch_*.npz`` (and/or ``val_data*.npz``), each
    with ``data`` of shape ``(N, 3*S*S)`` uint8 channel-major and, if labeled,
    ``labels`` of shape ``(N,)`` 1-based.

    The whole split lives in memory as uint8 — ~16 GB for the full 1.28M-image
    training set. That is deliberate (it makes the dataloader a non-bottleneck)
    but it is also why ``max_samples`` exists for local work.
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        image_size: int = 64,
        horizontal_flip: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.horizontal_flip = horizontal_flip

        shards = self._find_shards(split)
        if not shards:
            raise FileNotFoundError(
                f"No npz shards for split '{split}' under {self.root}. Expected "
                f"{self._patterns(split)[0]}. Fetch the data first:\n"
                f"  python scripts/fetch_imagenet_hf.py --split "
                f"{'train' if split == 'train' else 'validation'} --out {self.root}"
            )

        self.data, labels = self._load(shards, max_samples)
        self.labels = labels
        self.stored_size = int(round((self.data.shape[1] // 3) ** 0.5))
        n = self.data.shape[0]
        self.data = self.data.reshape(n, 3, self.stored_size, self.stored_size)

        labeled = "unlabeled" if self.labels is None else "labeled"
        print(
            f"[data] ImageNet64 {split}: {n:,} images at {self.stored_size}x"
            f"{self.stored_size} ({labeled}) from {len(shards)} shard(s), "
            f"{self.data.nbytes / 1e9:.1f} GB uint8"
        )

    # -- discovery ---------------------------------------------------------
    @staticmethod
    def _patterns(split: str) -> list[str]:
        if split in ("val", "valid", "validation"):
            return ["val_data*.npz"]
        return ["train_data_batch_*.npz"]

    def _find_shards(self, split: str) -> list[Path]:
        found: list[Path] = []
        for pat in self._patterns(split):
            found.extend(sorted(self.root.glob(pat)))
        return found

    # -- loading -----------------------------------------------------------
    def _load(
        self, shards: list[Path], max_samples: Optional[int]
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        shapes = [_npz_member_shape(p, "data") for p in shards]
        if any(s is None or len(s) != 2 for s in shapes):
            return self._load_by_concat(shards, max_samples)

        total = sum(s[0] for s in shapes)
        row_dim = shapes[0][1]
        if any(s[1] != row_dim for s in shapes):
            raise ValueError(f"shards under {self.root} disagree on row size: {shapes}")
        keep = total if max_samples is None else min(total, max_samples)

        data = np.empty((keep, row_dim), dtype=np.uint8)
        labels = np.empty(keep, dtype=np.int64)
        any_labels = False
        filled = 0
        for path, shape in zip(shards, shapes):
            if filled >= keep:
                break
            take = min(shape[0], keep - filled)
            with np.load(path) as z:
                data[filled : filled + take] = z["data"][:take]
                if "labels" in z.files:
                    any_labels = True
                    labels[filled : filled + take] = self._shift(z["labels"][:take], path)
                else:
                    labels[filled : filled + take] = UNLABELED
            filled += take
        return data, (labels if any_labels else None)

    def _load_by_concat(
        self, shards: list[Path], max_samples: Optional[int]
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Fallback for shards whose headers could not be peeked."""
        chunks, label_chunks, any_labels, total = [], [], False, 0
        for path in shards:
            with np.load(path) as z:
                d = z["data"]
                if max_samples is not None and total + d.shape[0] > max_samples:
                    d = d[: max_samples - total]
                chunks.append(np.asarray(d, dtype=np.uint8))
                if "labels" in z.files:
                    any_labels = True
                    label_chunks.append(self._shift(z["labels"][: d.shape[0]], path))
                else:
                    label_chunks.append(np.full(d.shape[0], UNLABELED, dtype=np.int64))
            total += chunks[-1].shape[0]
            if max_samples is not None and total >= max_samples:
                break
        data = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        labels = np.concatenate(label_chunks) if len(label_chunks) > 1 else label_chunks[0]
        return data, (labels if any_labels else None)

    @staticmethod
    def _shift(raw: np.ndarray, path: Path) -> np.ndarray:
        """1-based (1..1000) on disk -> 0-based (0..999) in memory.

        Loud on out-of-range values rather than silently producing -1, which
        would masquerade as UNLABELED and quietly train those images on the
        unconditional branch.
        """
        out = np.asarray(raw, dtype=np.int64) - 1
        lo, hi = int(out.min()), int(out.max())
        if lo < 0 or hi > 999:
            raise ValueError(
                f"{path.name}: labels map to [{lo}, {hi}] after the 1-based "
                "shift, expected [0, 999]. The npz layout stores labels 1..1000 "
                "(see scripts/fetch_imagenet_hf.py); this file appears to be "
                "0-based or corrupt."
            )
        return out

    # -- Dataset -----------------------------------------------------------
    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        u8 = self.data[i]
        if self.horizontal_flip and np.random.rand() < 0.5:
            u8 = u8[:, :, ::-1]
        x = _resize(_to_model_space(u8), self.image_size)
        y = UNLABELED if self.labels is None else int(self.labels[i])
        return x, torch.tensor(y, dtype=torch.long)


class ImageFolder64(Dataset):
    """``root/<class>/*.png`` on disk, labeled by sorted subdirectory name.

    Images are loaded lazily, resized short-side then centre-cropped — so
    non-square sources are handled without distorting the aspect ratio.
    """

    def __init__(
        self,
        root: Union[str, Path],
        image_size: int = 64,
        horizontal_flip: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.horizontal_flip = horizontal_flip

        class_dirs = sorted(d for d in self.root.iterdir() if d.is_dir())
        if not class_dirs:
            raise FileNotFoundError(f"No class subdirectories under {self.root}")
        self.classes = [d.name for d in class_dirs]

        self.samples: list[tuple[Path, int]] = []
        for label, d in enumerate(class_dirs):
            for f in sorted(d.rglob("*")):
                if f.suffix.lower() in _EXTENSIONS:
                    self.samples.append((f, label))
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise FileNotFoundError(f"No images under {self.root}")
        print(
            f"[data] ImageFolder64: {len(self.samples):,} images, "
            f"{len(self.classes)} classes from {self.root}"
        )

    def _load(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        s = self.image_size
        if img.size != (s, s):
            w, h = img.size
            scale = s / min(w, h)
            img = img.resize((max(s, round(w * scale)), max(s, round(h * scale))), Image.BICUBIC)
            w, h = img.size
            left, top = (w - s) // 2, (h - s) // 2
            img = img.crop((left, top, left + s, top + s))
        return np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        path, label = self.samples[i]
        u8 = self._load(path)
        if self.horizontal_flip and np.random.rand() < 0.5:
            u8 = u8[:, :, ::-1]
        return _to_model_space(u8), torch.tensor(label, dtype=torch.long)


class FlatImageDataset(Dataset):
    """A flat directory of images, or a ``.zip`` read in place.

    Every label is :data:`UNLABELED`, so this supports unconditional training
    only. Reading straight out of the zip avoids materialising 1.28M files.
    """

    def __init__(
        self,
        root: Union[str, Path],
        image_size: int = 64,
        horizontal_flip: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.horizontal_flip = horizontal_flip
        self.is_zip = self.root.suffix == ".zip"
        # A ZipFile handle is not safe to share across forked dataloader
        # workers; each process opens its own on first use, keyed by pid.
        self._zip: Optional[zipfile.ZipFile] = None
        self._zip_pid: Optional[int] = None

        if self.is_zip:
            with zipfile.ZipFile(self.root) as z:
                self.names = sorted(
                    n for n in z.namelist() if Path(n).suffix.lower() in _EXTENSIONS
                )
        else:
            self.names = [
                str(f.relative_to(self.root))
                for f in sorted(self.root.rglob("*"))
                if f.suffix.lower() in _EXTENSIONS
            ]
        if max_samples is not None:
            self.names = self.names[:max_samples]
        if not self.names:
            raise FileNotFoundError(f"No images found in {self.root}")
        print(
            f"[data] FlatImageDataset: {len(self.names):,} images from {self.root} "
            "(UNLABELED — unconditional training only)"
        )

    def _handle(self) -> zipfile.ZipFile:
        pid = os.getpid()
        if self._zip is None or self._zip_pid != pid:
            self._zip = zipfile.ZipFile(self.root)
            self._zip_pid = pid
        return self._zip

    def _load(self, name: str) -> np.ndarray:
        if self.is_zip:
            with self._handle().open(name) as fh:
                img = Image.open(fh).convert("RGB")
                img.load()
        else:
            img = Image.open(self.root / name).convert("RGB")
        s = self.image_size
        if img.size != (s, s):
            w, h = img.size
            scale = s / min(w, h)
            img = img.resize((max(s, round(w * scale)), max(s, round(h * scale))), Image.BICUBIC)
            w, h = img.size
            left, top = (w - s) // 2, (h - s) // 2
            img = img.crop((left, top, left + s, top + s))
        return np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)

    def __getstate__(self) -> dict:
        # Never pickle the zip handle into a spawned worker.
        state = self.__dict__.copy()
        state["_zip"] = None
        state["_zip_pid"] = None
        return state

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        u8 = self._load(self.names[i])
        if self.horizontal_flip and np.random.rand() < 0.5:
            u8 = u8[:, :, ::-1]
        return _to_model_space(u8), torch.tensor(UNLABELED, dtype=torch.long)


def build_dataset(cfg) -> Dataset:
    """Build the training dataset from a :class:`eps.config.DataConfig`."""
    name = cfg.name.lower()
    common = dict(
        image_size=cfg.image_size,
        horizontal_flip=cfg.horizontal_flip,
        max_samples=cfg.max_samples,
    )
    if name == "imagenet64":
        return ImageNet64(cfg.root, split="train", **common)
    if name == "imagefolder":
        return ImageFolder64(cfg.root, **common)
    if name == "flat":
        return FlatImageDataset(cfg.root, **common)
    raise ValueError(
        f"Unknown data.name '{cfg.name}'. Expected one of: "
        "'imagenet64', 'imagefolder', 'flat'."
    )
