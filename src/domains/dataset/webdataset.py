"""Write deterministic WebDataset shards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from webdataset.writer import TarWriter
from src.config.dataset import DATASET_CONFIG
from src.shared.paths import PROJECT_ROOT, confined_path
from src.state.dataset import (
    DatasetEntry,
    DatasetShard,
    DatasetManifestRow,
    DatasetProgressEvent,
    DatasetSampleMetadata,
    DatasetShardWriteResult,
)

if TYPE_CHECKING:
    from src.state.contracts import DatasetSplit
    from src.state.dataset import DatasetProgressCallback

TAR_BLOCK_BYTES = 512


def get_shard_name(index: int) -> str:
    """Return the deterministic tar shard filename for a split-local index."""
    shard_index = str(index).zfill(DATASET_CONFIG["output"]["shard_index_width"])
    return f"{shard_index}{DATASET_CONFIG['output']['shard_extension']}"


def _tar_member_bytes(payload_bytes: int) -> int:
    """Return tar header and padded payload bytes for one member."""
    padded_payload = ((payload_bytes + TAR_BLOCK_BYTES - 1) // TAR_BLOCK_BYTES) * TAR_BLOCK_BYTES
    return TAR_BLOCK_BYTES + padded_payload


def _create_shards(split: DatasetSplit, entries: list[DatasetEntry]) -> list[DatasetShard]:
    """Group split entries into size-bounded shards including tar overhead."""
    shards: list[DatasetShard] = []
    current: list[DatasetEntry] = []
    current_bytes = TAR_BLOCK_BYTES * 2
    target_bytes = DATASET_CONFIG["output"]["shard_target_bytes"]
    for entry in entries:
        metadata = json.dumps(DatasetSampleMetadata(screen=entry.screen, safety=entry.safety)).encode("utf-8")
        entry_bytes = _tar_member_bytes(entry.byte_size) + _tar_member_bytes(len(metadata))
        if current and current_bytes + entry_bytes > target_bytes:
            shards.append(DatasetShard(index=len(shards), split=split, entries=current))
            current = []
            current_bytes = TAR_BLOCK_BYTES * 2
        current.append(entry)
        current_bytes += entry_bytes
    if current:
        shards.append(DatasetShard(index=len(shards), split=split, entries=current))
    return shards


def source_image_bytes(image_path: str) -> bytes:
    """Read source bytes after confining the path to the dataset source tree."""
    source_root = (PROJECT_ROOT / DATASET_CONFIG["source"]["path"]).resolve(strict=True)
    target = confined_path(source_root, image_path, is_existing_required=True)
    if not target.is_file():
        msg = "dataset source image must be a regular file."
        raise ValueError(msg)
    return target.read_bytes()


def _write_shard(shard: DatasetShard, output_dir: Path) -> list[DatasetManifestRow]:
    """Write one shard and return rows describing every member."""
    split_dir = output_dir / shard.split
    split_dir.mkdir(parents=True, exist_ok=True)
    shard_name = get_shard_name(shard.index)
    shard_path = split_dir / shard_name
    rows: list[DatasetManifestRow] = []
    with TarWriter(str(shard_path), encoder=False) as writer:
        for entry in shard.entries:
            extension = Path(entry.source_path).suffix.lower().lstrip(".")
            image_name = f"{entry.prefix}.{extension}"
            metadata_name = f"{entry.prefix}.json"
            writer.write(
                {
                    "__key__": entry.prefix,
                    extension: source_image_bytes(entry.image_path),
                    "json": json.dumps(DatasetSampleMetadata(screen=entry.screen, safety=entry.safety)).encode("utf-8"),
                },
            )
            rows.append(
                DatasetManifestRow(
                    split=shard.split,
                    tar_path=f"{DATASET_CONFIG['output']['shards_dir']}/{shard.split}/{shard_name}",
                    image_member=image_name,
                    json_member=metadata_name,
                    screen=entry.screen,
                    safety=entry.safety,
                    width=entry.width,
                    height=entry.height,
                ),
            )
    return rows


def write_dataset_shards(
    entries: list[DatasetEntry],
    output_dir: Path,
    on_progress: DatasetProgressCallback | None = None,
) -> DatasetShardWriteResult:
    """Write entries into deterministic tar shards."""
    shard_counts: dict[DatasetSplit, int] = {"train": 0, "val": 0, "test": 0}
    shards: list[DatasetShard] = []
    for split in DATASET_CONFIG["split"]["names"]:
        split_entries = [entry for entry in entries if entry.split == split]
        split_shards = _create_shards(split, split_entries)
        shard_counts[split] = len(split_shards)
        shards.extend(split_shards)
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    if on_progress is not None and shards:
        on_progress(DatasetProgressEvent("shards", 0, len(shards)))
    rows: list[DatasetManifestRow] = []
    for index, shard in enumerate(shards):
        rows.extend(_write_shard(shard, output_dir))
        if on_progress is not None:
            on_progress(DatasetProgressEvent("shards", index + 1, len(shards)))
    return DatasetShardWriteResult(shard_counts=shard_counts, manifest_rows=rows)
