"""Shared training state aliases."""

from __future__ import annotations

from typing import Literal

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
type ExportFormat = Literal["fp16", "fp32"]
type DatasetSplit = Literal["train", "val", "test"]
type TrainingDevice = Literal["cpu", "cuda", "mps"]
type ExportCheckpoint = Literal["best", "latest"]
