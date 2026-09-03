"""Training framework type bounds."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    import torch
    from torch import nn
    from torch.utils.data import Dataset
    from src.state.contracts import DatasetSplit


@dataclass(frozen=True)
class SampleMeta:
    """Batch metadata for one sample."""

    split: DatasetSplit
    tar_path: str
    image_member: str
    width: int
    height: int


type NormalizeStats = tuple[float, float, float]
type TrainItem = tuple[torch.Tensor, torch.Tensor, torch.Tensor, SampleMeta]
type TrainDataset = Dataset[TrainItem | None]
type TrainModel = nn.Module
