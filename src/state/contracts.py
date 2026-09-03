"""Shared training state aliases."""

from __future__ import annotations

from typing import Literal
from dataclasses import dataclass

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
type ExportFormat = Literal["fp16", "fp32"]
type DatasetSplit = Literal["train", "val", "test"]
type DatasetSafety = Literal["safe", "hot", "nsfw", "forbidden"]
type TrainingDevice = Literal["cpu", "cuda", "mps"]
type ExportCheckpoint = Literal["best", "latest"]


@dataclass(frozen=True)
class ImageItem:
    """Image path with its category."""

    image_path: str
    category: str
