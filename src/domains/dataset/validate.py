"""Validate a complete staged dataset artifact."""

from __future__ import annotations

import io
import json
import tarfile
import pyarrow as pa
from PIL import Image
import pyarrow.parquet as pq
from collections import Counter
from typing import TYPE_CHECKING, cast
from src.config.dataset import DATASET_CONFIG
from src.shared.paths import confined_path, output_path
from src.domains.dataset.webdataset import get_shard_name
from src.state.dataset import DatasetManifestRow, DatasetValidationResult
from src.domains.dataset.summary import DATASET_INTEGRITY_ERROR, read_dataset_summary

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Never
    from collections.abc import Callable
    from src.state.contracts import DatasetSplit
    from src.state.dataset import DatasetSummary

MANIFEST_FIELDS = {"height", "image_member", "json_member", "safety", "screen", "split", "tar_path", "width"}
REJECTION_FIELDS = {"reason", "source_path"}


def _error(detail: str, *, cause: Exception | None = None) -> Never:
    """Raise one stable dataset-integrity error."""
    message = f"{DATASET_INTEGRITY_ERROR}: {detail}"
    raise ValueError(message) from cause


def _require(path: Path, label: str, *, is_directory: bool = False) -> None:
    """Require one file or directory in the staged artifact."""
    if is_directory and not path.is_dir():
        _error(f"missing or invalid {label} directory.")
    if not is_directory and not path.is_file():
        _error(f"missing or invalid {label} file.")


def _validate_output_root(target_dir: Path) -> None:
    """Reject undeclared top-level artifact entries."""
    allowed = {
        DATASET_CONFIG["output"]["readme_file"],
        DATASET_CONFIG["output"]["metadata_file"],
        DATASET_CONFIG["output"]["manifest_jsonl_file"],
        DATASET_CONFIG["output"]["rejected_jsonl_file"],
        DATASET_CONFIG["output"]["manifest_parquet_dir"],
        DATASET_CONFIG["output"]["shards_dir"],
    }
    unexpected = sorted(entry.name for entry in target_dir.iterdir() if entry.name not in allowed)
    if unexpected:
        _error(f"dataset output contains unexpected entries: {', '.join(unexpected)}.")


def _json_mapping(line: str, label: str) -> dict[str, object]:
    """Decode one JSONL object without coercing keys or values."""
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        _error(f"{label} contains invalid JSON.", cause=error)
    if not isinstance(value, dict):
        _error(f"{label} entries must be JSON objects with string keys.")
    items = cast("dict[object, object]", value)
    if any(not isinstance(key, str) for key in items):
        _error(f"{label} entries must be JSON objects with string keys.")
    return cast("dict[str, object]", items)


def _required_string(record: dict[str, object], name: str, label: str) -> str:
    """Read one required non-empty string field."""
    value = record.get(name)
    if not isinstance(value, str) or not value:
        _error(f"{label}.{name} must be a non-empty string.")
    return value


def _required_dimension(record: dict[str, object], name: str, label: str) -> int:
    """Read one required positive integer dimension."""
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _error(f"{label}.{name} must be a positive integer.")
    return value


def _parse_manifest_row(record: dict[str, object], line_number: int) -> DatasetManifestRow:
    """Validate one manifest JSONL row."""
    label = f"manifest.jsonl line {line_number}"
    if set(record) != MANIFEST_FIELDS:
        _error(f"{label} fields do not match the manifest schema.")
    split = _required_string(record, "split", label)
    if split not in DATASET_CONFIG["split"]["names"]:
        _error(f"{label}.split is not supported.")
    safety = _required_string(record, "safety", label)
    if safety not in DATASET_CONFIG["source"]["safety_values"]:
        _error(f"{label}.safety is not supported.")
    return DatasetManifestRow(
        split=split,
        tar_path=_required_string(record, "tar_path", label),
        image_member=_required_string(record, "image_member", label),
        json_member=_required_string(record, "json_member", label),
        screen=_required_string(record, "screen", label),
        safety=safety,
        width=_required_dimension(record, "width", label),
        height=_required_dimension(record, "height", label),
    )


def _read_manifest(target_dir: Path) -> list[DatasetManifestRow]:
    """Read and validate every JSONL manifest row."""
    path = target_dir / DATASET_CONFIG["output"]["manifest_jsonl_file"]
    rows = [
        _parse_manifest_row(_json_mapping(line, "manifest.jsonl"), line_number)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if line
    ]
    if len({(row.tar_path, row.image_member) for row in rows}) != len(rows):
        _error("manifest.jsonl contains duplicate image members.")
    return rows


def _validate_rejections(target_dir: Path, expected_count: int) -> None:
    """Validate the source-rejection journal against the summary."""
    path = target_dir / DATASET_CONFIG["output"]["rejected_jsonl_file"]
    records = [_json_mapping(line, "rejected.jsonl") for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != expected_count:
        _error("rejected.jsonl count does not match dataset_info.json.")
    for record in records:
        if set(record) != REJECTION_FIELDS:
            _error("rejected.jsonl fields do not match the rejection schema.")
        _required_string(record, "source_path", "rejected.jsonl")
        _required_string(record, "reason", "rejected.jsonl")


def _validate_shard_names(shards_root: Path, shard_counts: dict[DatasetSplit, int]) -> None:
    """Validate split directories and contiguous shard names."""
    splits = DATASET_CONFIG["split"]["names"]
    unexpected = sorted(entry.name for entry in shards_root.iterdir() if entry.name not in splits)
    if unexpected:
        _error(f"data directory contains unexpected entries: {', '.join(unexpected)}.")
    for split in splits:
        split_dir = shards_root / split
        _require(split_dir, f"data/{split}", is_directory=True)
        names = sorted(entry.name for entry in split_dir.iterdir() if entry.is_file())
        expected = [get_shard_name(index) for index in range(shard_counts[split])]
        if names != expected:
            _error(f"{split} split does not contain the expected contiguous tar shards.")
        if any(entry.is_dir() for entry in split_dir.iterdir()):
            _error(f"{split} split contains an unexpected directory.")


def _validate_tar_members(target_dir: Path, rows: list[DatasetManifestRow]) -> None:
    """Cross-check tar members, metadata, and decoded dimensions."""
    rows_by_tar: dict[str, list[DatasetManifestRow]] = {}
    for row in rows:
        rows_by_tar.setdefault(row.tar_path, []).append(row)
    for tar_path, tar_rows in rows_by_tar.items():
        archive_path = confined_path(target_dir, tar_path, is_existing_required=True)
        with tarfile.open(archive_path, mode="r") as archive:
            expected_names = sorted(member for row in tar_rows for member in (row.image_member, row.json_member))
            actual_names = sorted(member.name for member in archive.getmembers() if member.isfile())
            if actual_names != expected_names:
                _error(f"{tar_path} members do not match manifest.jsonl.")
            for row in tar_rows:
                metadata_file = archive.extractfile(row.json_member)
                image_file = archive.extractfile(row.image_member)
                if metadata_file is None or image_file is None:
                    _error(f"{tar_path} is missing a declared member.")
                metadata = _json_mapping(metadata_file.read().decode("utf-8"), row.json_member)
                if metadata != {"screen": row.screen, "safety": row.safety}:
                    _error(f"{row.json_member} does not match manifest labels.")
                with Image.open(io.BytesIO(image_file.read())) as image:
                    if image.size != (row.width, row.height):
                        _error(f"{row.image_member} dimensions do not match manifest.jsonl.")
                    image.verify()


def _validate_parquet(target_dir: Path, rows: list[DatasetManifestRow]) -> None:
    """Cross-check Parquet partitions against JSONL manifest rows."""
    manifest_root = target_dir / DATASET_CONFIG["output"]["manifest_parquet_dir"]
    expected_records = [row.__dict__ for row in rows]
    parquet_records: list[dict[str, object]] = []
    expected_dirs = {f"split={split}" for split in DATASET_CONFIG["split"]["names"]}
    actual_dirs = {entry.name for entry in manifest_root.iterdir()}
    if actual_dirs != expected_dirs:
        _error("Parquet partition directories do not match configured splits.")
    for split in DATASET_CONFIG["split"]["names"]:
        split_dir = manifest_root / f"split={split}"
        part_path = split_dir / DATASET_CONFIG["output"]["manifest_parquet_part_file"]
        _require(part_path, f"parquet/split={split} manifest")
        if {entry.name for entry in split_dir.iterdir()} != {part_path.name}:
            _error(f"parquet/split={split} contains unexpected entries.")
        read_table: Callable[..., pa.Table] = cast("Callable[..., pa.Table]", pq.read_table)
        table = read_table(part_path)
        parquet_records.extend(cast("list[dict[str, object]]", table.to_pylist()))
    if parquet_records != expected_records:
        _error("Parquet manifests do not match manifest.jsonl.")


def _validate_counts(rows: list[DatasetManifestRow], summary: DatasetSummary) -> None:
    """Cross-check row and label counts against summary data."""
    if len(rows) != summary["total_images"]:
        _error("manifest row count does not match dataset_info.json.")
    screen_counts = dict(Counter(row.screen for row in rows))
    safety_counts = dict(Counter(row.safety for row in rows))
    if screen_counts != summary["screen_counts"] or safety_counts != summary["safety_counts"]:
        _error("manifest label counts do not match dataset_info.json.")


def validate_dataset_output(output_dir: str | Path) -> DatasetValidationResult:
    """Validate every declared component of a staged dataset."""
    target_dir = output_path(output_dir, is_root_allowed=False, is_existing_required=True)
    _require(target_dir, "dataset output", is_directory=True)
    for name in (
        DATASET_CONFIG["output"]["readme_file"],
        DATASET_CONFIG["output"]["metadata_file"],
        DATASET_CONFIG["output"]["manifest_jsonl_file"],
        DATASET_CONFIG["output"]["rejected_jsonl_file"],
    ):
        _require(target_dir / name, name)
    shards_root = target_dir / DATASET_CONFIG["output"]["shards_dir"]
    manifest_root = target_dir / DATASET_CONFIG["output"]["manifest_parquet_dir"]
    _require(shards_root, "data", is_directory=True)
    _require(manifest_root, "parquet", is_directory=True)
    _validate_output_root(target_dir)
    summary = read_dataset_summary(target_dir)
    rows = _read_manifest(target_dir)
    _validate_rejections(target_dir, summary["rejected_images"])
    _validate_shard_names(shards_root, summary["shard_counts"])
    _validate_tar_members(target_dir, rows)
    _validate_parquet(target_dir, rows)
    _validate_counts(rows, summary)
    return DatasetValidationResult(output_dir=str(target_dir), summary=summary)
