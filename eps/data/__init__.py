"""Dataset loading for Epsilon.

See :mod:`eps.data.imagenet` for the layouts each class expects. The important
convention: labels are 0-based (0..999) in memory, ``UNLABELED = -1`` marks
sources with no class annotations, and the null/unconditional token is
``num_classes`` (1000) — assigned by the trainer, never by a dataset.
"""

from .imagenet import (
    UNLABELED,
    FlatImageDataset,
    ImageFolder64,
    ImageNet64,
    build_dataset,
    imagenet_class_names,
)

__all__ = [
    "UNLABELED",
    "ImageNet64",
    "ImageFolder64",
    "FlatImageDataset",
    "build_dataset",
    "imagenet_class_names",
]
