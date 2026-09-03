"""Read and validate staged dataset summary data."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast
from src.state.dataset import DatasetSummary
from src.config.dataset import DATASET_CONFIG

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Never
    from src.state.contracts import DatasetSafety, DatasetSplit

DATASET_INTEGRITY_ERROR = "Dataset output failed integrity check"
SUMMARY_FIELDS = {
    "rejected_images",
    "safety",
    "safety_counts",
    "screen",
    "screen_counts",
    "shard_counts",
    "total_images",
}


def _error(detail: str, *, cause: Exception | None = None) -> Never:
    """Raise one stable dataset-integrity error."""
    message = f"{DATASET_INTEGRITY_ERROR}: {detail}"
    raise ValueError(message) from cause


def _parse_record(value: object, name: str) -> dict[str, object]:
    """Require a JSON object with string keys."""
    if not isinstance(value, dict):
        _error(f"dataset_info.json {name} must be a JSON object.")
    raw = cast("dict[object, object]", value)
    if any(not isinstance(key, str) for key in raw):
        _error(f"dataset_info.json {name} keys must be strings.")
    return {key: item for key, item in raw.items() if isinstance(key, str)}


def _parse_string_array(value: object, name: str) -> list[str]:
    """Require a non-empty unique string array."""
    if not isinstance(value, list):
        _error(f"dataset_info.json {name} must be a non-empty string array.")
    items = cast("list[object]", value)
    if not items or any(not isinstance(item, str) or not item for item in items):
        _error(f"dataset_info.json {name} must be a non-empty string array.")
    strings = cast("list[str]", items)
    if len(set(strings)) != len(strings):
        _error(f"dataset_info.json {name} must not contain duplicates.")
    return strings


def _parse_count(value: object, name: str, *, is_positive: bool = False) -> int:
    """Require a nonnegative or positive integer count."""
    minimum = 1 if is_positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if is_positive else "non-negative"
        _error(f"dataset_info.json {name} must be a {qualifier} integer.")
    return value


def _parse_count_record(value: object, name: str) -> dict[str, int]:
    """Require an object containing only nonnegative integer counts."""
    counts = _parse_record(value, name)
    return {key: _parse_count(item, f"{name}.{key}") for key, item in counts.items()}


def _parse_summary(value: object) -> DatasetSummary:
    """Validate and return the complete dataset summary schema."""
    summary = _parse_record(value, "root")
    if set(summary) != SUMMARY_FIELDS:
        _error("dataset_info.json fields do not match the supported summary schema.")
    total_images = _parse_count(summary["total_images"], "total_images", is_positive=True)
    rejected_images = _parse_count(summary["rejected_images"], "rejected_images")
    screen = _parse_string_array(summary["screen"], "screen")
    safety = _parse_string_array(summary["safety"], "safety")
    shard_counts = _parse_count_record(summary["shard_counts"], "shard_counts")
    screen_counts = _parse_count_record(summary["screen_counts"], "screen_counts")
    safety_counts = _parse_count_record(summary["safety_counts"], "safety_counts")
    if set(shard_counts) != set(DATASET_CONFIG["split"]["names"]):
        _error("dataset_info.json shard_counts keys must match configured splits.")
    if set(screen_counts) != set(screen):
        _error("dataset_info.json screen_counts keys must match screen.")
    if set(safety_counts) != set(safety):
        _error("dataset_info.json safety_counts keys must match safety.")
    if tuple(safety) != DATASET_CONFIG["source"]["safety_values"]:
        _error("dataset_info.json safety labels must match configured safety values.")
    if sum(screen_counts.values()) != total_images or sum(safety_counts.values()) != total_images:
        _error("dataset_info.json label counts must sum to total_images.")
    return DatasetSummary(
        total_images=total_images,
        rejected_images=rejected_images,
        shard_counts=cast("dict[DatasetSplit, int]", shard_counts),
        screen=screen,
        safety=cast("list[DatasetSafety]", safety),
        screen_counts=screen_counts,
        safety_counts=cast("dict[DatasetSafety, int]", safety_counts),
    )


def read_dataset_summary(target_dir: Path) -> DatasetSummary:
    """Read and validate dataset_info.json from a confined staging directory."""
    path = target_dir / DATASET_CONFIG["output"]["metadata_file"]
    if not path.is_file():
        _error("missing dataset_info.json.")
    try:
        return _parse_summary(json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        _error("dataset_info.json is not valid JSON.", cause=error)
