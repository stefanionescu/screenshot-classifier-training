"""Dataset state."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from src.state.contracts import DatasetSafety, DatasetSplit


class DatasetRepo(TypedDict):
    """Dataset repository configuration."""

    id: str


class DatasetSource(TypedDict):
    """Dataset source configuration."""

    path: str
    safety_values: tuple[DatasetSafety, ...]


class DatasetSplitConfig(TypedDict):
    """Dataset split configuration."""

    names: tuple[DatasetSplit, ...]
    default_val_percent: float
    default_test_percent: float
    max_held_out_percent: int
    min_bucket_size: int


class DatasetOutput(TypedDict):
    """Dataset artifact configuration."""

    default_dir: str
    readme_file: str
    metadata_file: str
    shards_dir: str
    manifest_jsonl_file: str
    rejected_jsonl_file: str
    manifest_parquet_dir: str
    manifest_parquet_part_file: str
    sample_prefix_width: int
    shard_extension: str
    shard_index_width: int
    shard_target_bytes: int


class DatasetCard(TypedDict):
    """Dataset card configuration."""

    pretty_name: str


class DatasetConfig(TypedDict):
    """Dataset build configuration."""

    repo: DatasetRepo
    source: DatasetSource
    split: DatasetSplitConfig
    output: DatasetOutput
    card: DatasetCard
    collection_concurrency: int


@dataclass(frozen=True)
class DatasetCommandArgs:
    """Parsed dataset command arguments."""

    output_dir: str
    val_percent: float
    test_percent: float


@dataclass(frozen=True)
class DatasetBuildOptions:
    """Dataset build settings."""

    output_dir: str
    val_percent: float
    test_percent: float


@dataclass(frozen=True)
class DatasetCandidate:
    """Validated source image before split and output IDs are assigned."""

    image_path: str
    source_path: str
    screen: str
    safety: DatasetSafety
    width: int
    height: int
    byte_size: int
    content_sha256: str


@dataclass(frozen=True)
class DatasetEntry:
    """One image included in the generated dataset."""

    prefix: str
    image_path: str
    source_path: str
    screen: str
    split: DatasetSplit
    safety: DatasetSafety
    width: int
    height: int
    byte_size: int
    content_sha256: str


@dataclass(frozen=True)
class DatasetRejectedImage:
    """Source image excluded from a dataset build."""

    source_path: str
    reason: str


@dataclass(frozen=True)
class DatasetCollection:
    """Validated entries and explicit source rejections."""

    entries: list[DatasetEntry]
    rejected: list[DatasetRejectedImage]


@dataclass(frozen=True)
class DatasetManifestRow:
    """Manifest row for a dataset shard member."""

    split: DatasetSplit
    tar_path: str
    image_member: str
    json_member: str
    screen: str
    safety: DatasetSafety
    width: int
    height: int


class DatasetSampleMetadata(TypedDict):
    """JSON metadata stored beside one image in a dataset shard."""

    screen: str
    safety: DatasetSafety


@dataclass(frozen=True)
class DatasetShard:
    """Dataset shard grouped by split."""

    index: int
    split: DatasetSplit
    entries: list[DatasetEntry]


@dataclass(frozen=True)
class DatasetShardWriteResult:
    """Shard generation result."""

    shard_counts: dict[DatasetSplit, int]
    manifest_rows: list[DatasetManifestRow]


@dataclass(frozen=True)
class DatasetBuildResult:
    """Dataset build result."""

    output_dir: str
    summary: DatasetSummary


class DatasetSummary(TypedDict):
    """Dataset counts persisted in dataset_info.json."""

    total_images: int
    rejected_images: int
    shard_counts: dict[DatasetSplit, int]
    screen: list[str]
    safety: list[DatasetSafety]
    screen_counts: dict[str, int]
    safety_counts: dict[DatasetSafety, int]


@dataclass(frozen=True)
class DatasetValidationResult:
    """Validated dataset output and parsed summary."""

    output_dir: str
    summary: DatasetSummary


@dataclass(frozen=True)
class DatasetProgressEvent:
    """Progress update for one dataset build phase."""

    phase: Literal["entries", "shards", "manifest"]
    completed: int
    total: int


type DatasetProgressCallback = Callable[[DatasetProgressEvent], None]
