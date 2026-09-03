"""Read validated WebDataset manifests and decode training samples."""

from __future__ import annotations

import io
import torch
import tarfile
import pyarrow as pa
from PIL import Image
import pyarrow.parquet as pq
from src.errors import TrainingError
from torch.utils.data import Dataset
from src.state.training import Sample
from src.state.types import SampleMeta
from typing import TYPE_CHECKING, cast
from src.domains.labels import IGNORED_LABEL_ID
from src.domains.images.skipped.journal import SkippedImageJournal
from src.config.model import MANIFEST_COLUMNS, TRAINABLE_SAFETY_LABELS
from src.domains.images.augmentation import augment_image, preprocess_image, to_training_rgb

if TYPE_CHECKING:
    from pathlib import Path
    from src.state.inputs import DatasetSpec
    from src.state.contracts import DatasetSplit
    from collections.abc import Callable, Iterable

TrainItem = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    SampleMeta,
]


class MultiTaskDataset(Dataset[TrainItem | None]):
    """Training samples decoded from validated dataset shards.

    Attributes:
        samples: Ordered training sample records.
        labels: Label mappings for both prediction heads.
        image_size: Longest image side used during preprocessing.
        preprocess: Normalization settings for model input.
        augment: Whether training augmentation is enabled.
        skipped_path: Journal path for unreadable samples.

    """

    def __init__(self, spec: DatasetSpec) -> None:
        """Create a dataset from a training dataset spec."""
        self.samples = spec.samples
        self.labels = spec.labels
        self.image_size = spec.image_size
        self.preprocess = spec.preprocess
        self.augment = spec.augment
        self.skipped_path = spec.skipped_path
        self._skipped_journal = SkippedImageJournal(spec.skipped_path)

    def __len__(self) -> int:
        """Return the number of available samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> TrainItem | None:
        """Read and transform one training sample by index."""
        sample = self.samples[index]
        try:
            with tarfile.TarFile.open(sample.tar_path, "r") as archive:
                member = archive.getmember(sample.image_name)
                if not member.isfile():
                    msg = "image member must be a regular file."
                    self._skipped_journal.record(sample, RuntimeError(msg))
                    return None
                extracted = archive.extractfile(member)
                if extracted is None:
                    msg = "image member is missing from its dataset shard."
                    self._skipped_journal.record(sample, RuntimeError(msg))
                    return None
                image_bytes = extracted.read()

            with Image.open(io.BytesIO(image_bytes)) as opened:
                image = to_training_rgb(opened)
            if self.augment:
                image = augment_image(image)

            tensor = preprocess_image(image, self.image_size, self.preprocess.mean, self.preprocess.std)
            return (
                tensor,
                torch.tensor(self.labels.screen_to_id[sample.screen_label], dtype=torch.long),
                torch.tensor(
                    self.labels.safety_to_id.get(sample.safety_label, IGNORED_LABEL_ID),
                    dtype=torch.long,
                ),
                SampleMeta(
                    split=sample.split,
                    tar_path=str(sample.tar_path),
                    image_member=sample.image_name,
                    width=sample.width,
                    height=sample.height,
                ),
            )
        except (OSError, RuntimeError, tarfile.TarError, ValueError) as exc:
            self._skipped_journal.record(sample, exc)
            return None


def parquet_dir(dataset_path: Path, split: DatasetSplit) -> Path:
    """Resolve a required Parquet partition within a training dataset."""
    directory = dataset_path / "parquet" / f"split={split}"
    if not directory.is_dir():
        msg = "dataset Parquet split is missing. Rebuild the dataset artifacts."
        raise TrainingError(msg)
    return directory


def read_split_samples(dataset_path: Path, split: DatasetSplit) -> list[Sample]:
    """Read split samples."""
    samples: list[Sample] = []
    parquet_paths = sorted(parquet_dir(dataset_path, split).glob("*.parquet"))
    if not parquet_paths:
        msg = f"dataset parquet split has no files: {split}"
        raise TrainingError(msg)
    for parquet_path in parquet_paths:
        samples.extend(read_parquet_samples(dataset_path, parquet_path, split))
    if not samples:
        msg = f"dataset split has no samples: {split}"
        raise TrainingError(msg)
    return samples


def _parquet_text(value: object, field: str) -> str:
    """Require one non-empty Parquet string field."""
    if isinstance(value, str) and value:
        return value
    msg = f"Parquet row is missing the required {field} value."
    raise TrainingError(msg)


def _parquet_dimension(value: object) -> int:
    """Require one positive Parquet image dimension."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    msg = "Parquet row has invalid image dimensions."
    raise TrainingError(msg)


def _parquet_tar_path(dataset_path: Path, value: object) -> Path:
    """Resolve one existing tar path confined to the dataset."""
    tar_value = _parquet_text(value, "tar_path")
    tar_path = (dataset_path / tar_value).resolve()
    try:
        tar_path.relative_to(dataset_path.resolve())
    except ValueError as error:
        msg = "Parquet archive reference must remain inside the dataset directory."
        raise TrainingError(msg) from error
    if not tar_path.is_file():
        msg = "Parquet archive reference does not exist. Rebuild the dataset artifacts."
        raise TrainingError(msg)
    return tar_path


def read_parquet_samples(dataset_path: Path, parquet_path: Path, split: DatasetSplit) -> list[Sample]:
    """Read parquet samples."""
    read_table: Callable[..., pa.Table] = cast("Callable[..., pa.Table]", pq.read_table)
    columns = cast("dict[str, list[object]]", read_table(parquet_path, columns=list(MANIFEST_COLUMNS)).to_pydict())
    samples: list[Sample] = []
    for tar_value, image_name, screen_label, safety_label, width, height in zip(
        columns["tar_path"],
        columns["image_member"],
        columns["screen"],
        columns["safety"],
        columns["width"],
        columns["height"],
        strict=True,
    ):
        samples.append(
            Sample(
                tar_path=_parquet_tar_path(dataset_path, tar_value),
                image_name=_parquet_text(image_name, "image_member"),
                split=split,
                screen_label=_parquet_text(screen_label, "string screen"),
                safety_label=_parquet_text(safety_label, "string safety value"),
                width=_parquet_dimension(width),
                height=_parquet_dimension(height),
            ),
        )
    return samples


def select_screen_labels(samples: list[Sample], requested: list[str] | None) -> list[str]:
    """Select screen labels."""
    available = sorted({sample.screen_label for sample in samples})
    if requested is None or len(requested) == 0:
        return available
    missing = sorted(set(requested) - set(available))
    if missing:
        msg = f"unknown --screens value: {', '.join(missing)}. Valid labels: {', '.join(available)}"
        raise TrainingError(msg)
    return list(dict.fromkeys(requested))


def select_safety_labels(samples: list[Sample], min_count: int) -> list[str]:
    """Select represented labels from the fixed trainable safety taxonomy."""
    available: list[str] = list(TRAINABLE_SAFETY_LABELS)
    counts = count_values((sample.safety_label for sample in samples), available)
    labels = [label for label in available if counts[label] >= min_count]
    if not labels:
        eligible = ", ".join(TRAINABLE_SAFETY_LABELS)
        msg = f"no trainable safety labels ({eligible}) have at least {min_count} train samples after screen filtering."
        raise TrainingError(msg)
    return labels


def filter_screen_samples(samples: list[Sample], labels: set[str]) -> list[Sample]:
    """Filter screen samples."""
    return [sample for sample in samples if sample.screen_label in labels]


def remap_screen_samples(samples: list[Sample], mapping: dict[str, str]) -> list[Sample]:
    """Return samples with configured sparse screen labels folded."""
    return [
        Sample(
            tar_path=sample.tar_path,
            image_name=sample.image_name,
            split=sample.split,
            screen_label=mapping.get(sample.screen_label, sample.screen_label),
            safety_label=sample.safety_label,
            width=sample.width,
            height=sample.height,
        )
        for sample in samples
    ]


def count_values(values: Iterable[str], labels: list[str]) -> dict[str, int]:
    """Count only values belonging to the ordered active-label set."""
    counts = dict.fromkeys(labels, 0)
    for value in values:
        if value in counts:
            counts[value] += 1
    return counts


def assert_split_coverage(name: str, counts: dict[str, int]) -> None:
    """Require every selected label to occur in a dataset split."""
    missing = [label for label, count in counts.items() if count == 0]
    if missing:
        msg = f"selected labels have no {name} samples: {', '.join(missing)}"
        raise TrainingError(msg)
