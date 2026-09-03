"""Group sampled training rows by image aspect."""

from __future__ import annotations

import math
from torch.utils.data import Sampler
from typing import TYPE_CHECKING, Protocol
from src.domains.images.augmentation import aspect_bucket

if TYPE_CHECKING:
    from collections.abc import Iterator
    from src.state.training import Sample


class IndexSampler(Protocol):
    """Index iteration and size consumed by aspect batching."""

    def __iter__(self) -> Iterator[int]:
        """Iterate sampled dataset indexes."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the sampled index count."""
        raise NotImplementedError


class AspectBatchSampler(Sampler[list[int]]):
    """Batch sampled rows by aspect bucket."""

    def __init__(self, samples: list[Sample], sampler: IndexSampler, batch_size: int) -> None:
        """Create an aspect batch sampler."""
        self.samples = samples
        self.sampler = sampler
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate over aspect-bucketed batches."""
        buckets: dict[str, list[int]] = {"portrait": [], "square": [], "landscape": []}
        for index in self.sampler:
            bucket = aspect_bucket(self.samples[index])
            buckets[bucket].append(index)
            if len(buckets[bucket]) == self.batch_size:
                yield buckets[bucket]
                buckets[bucket] = []

        leftovers: list[int] = []
        for bucket in ("portrait", "square", "landscape"):
            leftovers.extend(buckets[bucket])
        for start in range(0, len(leftovers), self.batch_size):
            yield leftovers[start : start + self.batch_size]

    def __len__(self) -> int:
        """Return the batch count."""
        return math.ceil(len(self.sampler) / self.batch_size)
