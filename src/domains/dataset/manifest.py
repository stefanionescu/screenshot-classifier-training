"""Write dataset JSONL and Parquet manifests."""

from __future__ import annotations

import json
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import asdict
from typing import TYPE_CHECKING, cast
from src.config.dataset import DATASET_CONFIG
from src.state.dataset import DatasetProgressEvent

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable
    from src.state.dataset import DatasetManifestRow, DatasetProgressCallback

MANIFEST_PARQUET_SCHEMA: pa.Schema = pa.schema(
    [
        ("split", pa.string()),
        ("tar_path", pa.string()),
        ("image_member", pa.string()),
        ("json_member", pa.string()),
        ("screen", pa.string()),
        ("safety", pa.string()),
        ("width", pa.int64()),
        ("height", pa.int64()),
    ],
)


def _row_sort_key(row: DatasetManifestRow) -> tuple[int, str, str]:
    """Return deterministic manifest row ordering fields."""
    split_order = DATASET_CONFIG["split"]["names"].index(row.split)
    return split_order, row.tar_path, row.image_member


def _write_parquet_manifest(rows: list[DatasetManifestRow], output_dir: Path) -> None:
    """Write one Parquet manifest partition for each split."""
    manifest_dir = output_dir / DATASET_CONFIG["output"]["manifest_parquet_dir"]
    manifest_dir.mkdir(parents=True, exist_ok=False)
    write_table: Callable[..., None] = cast("Callable[..., None]", pq.write_table)
    ordered = sorted(rows, key=_row_sort_key)
    for split in DATASET_CONFIG["split"]["names"]:
        split_rows = [asdict(row) for row in ordered if row.split == split]
        split_dir = manifest_dir / f"split={split}"
        split_dir.mkdir()
        table = pa.Table.from_pylist(split_rows, schema=MANIFEST_PARQUET_SCHEMA)
        write_table(
            table,
            split_dir / DATASET_CONFIG["output"]["manifest_parquet_part_file"],
            compression="zstd",
        )


def write_dataset_manifest(
    rows: list[DatasetManifestRow],
    output_dir: Path,
    on_progress: DatasetProgressCallback | None = None,
) -> None:
    """Write JSONL and Parquet manifests next to generated shards."""
    if on_progress is not None:
        on_progress(DatasetProgressEvent("manifest", 0, 2))
    content = "".join(f"{json.dumps(asdict(row))}\n" for row in sorted(rows, key=_row_sort_key))
    (output_dir / DATASET_CONFIG["output"]["manifest_jsonl_file"]).write_text(content, encoding="utf-8")
    if on_progress is not None:
        on_progress(DatasetProgressEvent("manifest", 1, 2))
    _write_parquet_manifest(rows, output_dir)
    if on_progress is not None:
        on_progress(DatasetProgressEvent("manifest", 2, 2))
