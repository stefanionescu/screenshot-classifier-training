"""Balance label sampling while preserving resumable deterministic state."""

from __future__ import annotations

import math
import torch
from src.errors import TrainingError
from typing import TYPE_CHECKING, cast
from src.domains.labels import IGNORED_LABEL_ID
from src.state.sampler.snapshot import SamplerSnapshot
from src.domains.samples.aspect import AspectBatchSampler
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from src.domains.images.augmentation import BatchItem, collate_batch
from src.state.sampler.adaptive import SamplerCoverage, SamplerLabelCoverage

if TYPE_CHECKING:
    from collections.abc import Iterator
    from src.state.inputs import IteratorConfig
    from src.domains.samples.dataset import MultiTaskDataset, TrainItem


QUEUE_SAMPLER_SCHEMA = 1


class QueueLabelSampler(Sampler[int]):
    """Represent queue label sampler."""

    def __init__(
        self,
        labels: list[int],
        target_effective_ratio: float,
        max_repeat_factor: int,
        seed: int,
    ) -> None:
        """Create a sampler for repeated label-balanced epochs."""
        self.target_effective_ratio = target_effective_ratio
        self.max_repeat_factor = max_repeat_factor
        self.seed = seed
        self.epoch = 0
        self.indices_by_label = group_indices_by_label(labels)
        self.epoch_counts = build_epoch_counts(
            {label: len(indices) for label, indices in self.indices_by_label.items()},
            target_effective_ratio,
            max_repeat_factor,
        )
        self.epoch_size = sum(self.epoch_counts.values())
        self.orders_by_label = {
            label: shuffled_indices(indices, self.seed + label) for label, indices in self.indices_by_label.items()
        }
        self.cursors_by_label = dict.fromkeys(self.indices_by_label, 0)
        self.cycles_by_label = dict.fromkeys(self.indices_by_label, 0)

    def __iter__(self) -> Iterator[int]:
        """Iterate over the values."""
        selected: list[int] = []
        for label in sorted(self.epoch_counts):
            selected.extend(self.pop_index(label) for _ in range(self.epoch_counts[label]))
        return iter(shuffled_indices(selected, self.seed + self.epoch))

    def __len__(self) -> int:
        """Return the configured epoch size."""
        return self.epoch_size

    def pop_index(self, label: int) -> int:
        """Draw the next index for one label and reshuffle exhausted cycles."""
        order = self.orders_by_label[label]
        cursor = self.cursors_by_label[label]
        if cursor >= len(order):
            self.cycles_by_label[label] += 1
            order = shuffled_indices(
                self.indices_by_label[label],
                self.seed + label + self.cycles_by_label[label],
            )
            self.orders_by_label[label] = order
            cursor = 0
        value = order[cursor]
        self.cursors_by_label[label] = cursor + 1
        return value

    def state_dict(self) -> dict[str, object]:
        """Serialize positions and immutable settings for a checkpoint."""
        return {
            "schema": QUEUE_SAMPLER_SCHEMA,
            "epoch": self.epoch,
            "seed": self.seed,
            "target_effective_ratio": self.target_effective_ratio,
            "max_repeat_factor": self.max_repeat_factor,
            "orders_by_label": self.orders_by_label,
            "cursors_by_label": self.cursors_by_label,
            "cycles_by_label": self.cycles_by_label,
        }

    def restore_state(self, state: dict[str, object]) -> None:
        """Restore a strictly validated sampler snapshot."""
        snapshot = self.validated_snapshot(state)
        self.epoch = snapshot.epoch
        self.orders_by_label = snapshot.orders
        self.cursors_by_label = snapshot.cursors
        self.cycles_by_label = snapshot.cycles

    def validated_snapshot(self, state: dict[str, object]) -> SamplerSnapshot:
        """Return one snapshot after validating immutable settings and positions."""
        validate_sampler_header(state, self.seed, self.target_effective_ratio, self.max_repeat_factor)
        snapshot = sampler_snapshot(state)
        validate_sampler_snapshot(snapshot, self.indices_by_label)
        return snapshot

    def coverage_stats(self) -> SamplerCoverage:
        """Return coverage counters by label ID."""
        stats: SamplerCoverage = {}
        for label, indices in sorted(self.indices_by_label.items()):
            count = len(indices)
            cursor = self.cursors_by_label[label]
            cycles = self.cycles_by_label[label]
            total_draws = cycles * count + cursor
            stats[str(label)] = SamplerLabelCoverage(
                dataset_count=count,
                current_cycle_seen=cursor,
                current_cycle_fraction=cursor / count if count > 0 else 0.0,
                cycle_count=cycles,
                total_draws=total_draws,
                epoch_target_count=self.epoch_counts[label],
            )
        return stats


def build_adaptive_train_iterator(train: IteratorConfig) -> tuple[DataLoader[BatchItem | None], QueueLabelSampler]:
    """Build adaptive train iterator."""
    sampler = QueueLabelSampler(train.labels, train.target_ratio, train.max_repeat, train.seed)
    dataset: MultiTaskDataset = cast("MultiTaskDataset", train.dataset)
    iterator = cast(
        "DataLoader[BatchItem | None]",
        DataLoader(
            dataset,
            batch_sampler=AspectBatchSampler(dataset.samples, sampler, train.batch_size),
            num_workers=train.workers,
            pin_memory=train.run_device.type == "cuda",
            collate_fn=collate_batch,
        ),
    )
    return iterator, sampler


def validate_sampler_header(state: dict[str, object], seed: int, ratio: float, max_repeat: int) -> None:
    """Validate immutable sampler settings and the exact state schema."""
    expected_fields = {
        "schema",
        "epoch",
        "seed",
        "target_effective_ratio",
        "max_repeat_factor",
        "orders_by_label",
        "cursors_by_label",
        "cycles_by_label",
    }
    checks = (
        (set(state) != expected_fields, "Checkpoint sampler fields do not match the supported schema."),
        (state.get("schema") != QUEUE_SAMPLER_SCHEMA, "Checkpoint sampler state has an unsupported schema."),
        (state.get("seed") != seed, "Checkpoint sampler seed does not match the current run."),
        (state.get("target_effective_ratio") != ratio, "Checkpoint sampler target ratio does not match the run."),
        (state.get("max_repeat_factor") != max_repeat, "Checkpoint sampler repeat limit does not match the run."),
    )
    for invalid, message in checks:
        if invalid:
            raise TrainingError(message)


def sampler_snapshot(state: dict[str, object]) -> SamplerSnapshot:
    """Parse sampler collections and epoch into a named record."""
    epoch = state.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        msg = "checkpoint sampler epoch is invalid."
        raise TrainingError(msg)
    return SamplerSnapshot(
        epoch=epoch,
        orders=int_keyed_lists(state.get("orders_by_label"), "orders_by_label"),
        cursors=int_keyed_ints(state.get("cursors_by_label"), "cursors_by_label"),
        cycles=int_keyed_ints(state.get("cycles_by_label"), "cycles_by_label"),
    )


def validate_sampler_snapshot(snapshot: SamplerSnapshot, indices_by_label: dict[int, list[int]]) -> None:
    """Prove restored positions belong to the selected dataset."""
    expected = set(indices_by_label)
    if set(snapshot.orders) != expected or set(snapshot.cursors) != expected or set(snapshot.cycles) != expected:
        msg = "checkpoint sampler labels do not match the selected labels."
        raise TrainingError(msg)
    for label, order in snapshot.orders.items():
        if sorted(order) != sorted(indices_by_label[label]):
            msg = "checkpoint sampler order does not match the selected dataset."
            raise TrainingError(msg)
        if not 0 <= snapshot.cursors[label] <= len(order):
            msg = "checkpoint sampler cursor is out of range."
            raise TrainingError(msg)
        if snapshot.cycles[label] < 0:
            msg = "checkpoint sampler cycle count is out of range."
            raise TrainingError(msg)


def build_eval_iterator(
    dataset: Dataset[TrainItem | None],
    batch_size: int,
    workers: int,
    run_device: torch.device,
) -> DataLoader[BatchItem | None]:
    """Build eval iterator."""
    return cast(
        "DataLoader[BatchItem | None]",
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=run_device.type == "cuda",
            collate_fn=collate_batch,
        ),
    )


def build_balanced_subset(
    dataset: Dataset[TrainItem | None],
    labels: list[int],
    max_per_class: int,
    seed: int,
) -> Subset[TrainItem | None]:
    """Build balanced subset."""
    grouped: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        if label == IGNORED_LABEL_ID:
            continue
        grouped.setdefault(label, []).append(index)
    selected: list[int] = []
    for label, indices in grouped.items():
        count = min(len(indices), max_per_class)
        selected.extend(shuffled_indices(indices, seed + label)[:count])
    if not selected:
        msg = "balanced evaluation subset is empty."
        raise TrainingError(msg)
    selected.sort()
    return Subset(dataset, selected)


def group_indices_by_label(labels: list[int]) -> dict[int, list[int]]:
    """Group active dataset positions by encoded label."""
    grouped: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        if label == IGNORED_LABEL_ID:
            continue
        grouped.setdefault(label, []).append(index)
    if not grouped:
        msg = "adaptive sampler labels are empty."
        raise TrainingError(msg)
    return grouped


def shuffled_indices(indices: list[int], seed: int) -> list[int]:
    """Return a deterministic torch permutation of dataset positions."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    positions = torch.randperm(len(indices), generator=generator)
    return [indices[int(position.item())] for position in positions]


def auto_alpha(counts: list[int], target_effective_ratio: float) -> float:
    """Choose the power transform that meets the target class ratio."""
    positive = [count for count in counts if count > 0]
    raw_ratio = max(positive) / min(positive)
    if raw_ratio <= target_effective_ratio:
        return 1.0
    alpha = math.log(target_effective_ratio) / math.log(raw_ratio)
    return max(0.0, min(1.0, alpha))


def build_epoch_counts(
    counts_by_label: dict[int, int],
    target_effective_ratio: float,
    max_repeat_factor: int,
) -> dict[int, int]:
    """Build epoch counts."""
    alpha = auto_alpha(list(counts_by_label.values()), target_effective_ratio)
    total = sum(counts_by_label.values())
    scores = {label: count**alpha for label, count in counts_by_label.items()}
    score_total = sum(scores.values())
    epoch_counts: dict[int, int] = {}
    for label, count in counts_by_label.items():
        desired = max(1, round(total * (scores[label] / score_total)))
        epoch_counts[label] = min(desired, count * max_repeat_factor)
    return epoch_counts


def int_keyed_lists(value: object, name: str) -> dict[int, list[int]]:
    """Validate a checkpoint mapping from numeric labels to index lists."""
    if not isinstance(value, dict):
        msg = f"checkpoint sampler state is missing {name}."
        raise TrainingError(msg)
    parsed: dict[int, list[int]] = {}
    raw = cast("dict[object, object]", value)
    for key, item in raw.items():
        values = cast("list[object]", item) if isinstance(item, list) else []
        if not values or any(isinstance(index, bool) or not isinstance(index, int) for index in values):
            msg = f"checkpoint sampler state has invalid {name}."
            raise TrainingError(msg)
        parsed[parse_int_key(key, name)] = cast("list[int]", values)
    return parsed


def int_keyed_ints(value: object, name: str) -> dict[int, int]:
    """Validate a checkpoint mapping from numeric labels to counters."""
    if not isinstance(value, dict):
        msg = f"checkpoint sampler state is missing {name}."
        raise TrainingError(msg)
    parsed: dict[int, int] = {}
    raw = cast("dict[object, object]", value)
    for key, item in raw.items():
        if isinstance(item, bool) or not isinstance(item, int):
            msg = f"checkpoint sampler state has invalid {name}."
            raise TrainingError(msg)
        parsed[parse_int_key(key, name)] = item
    return parsed


def parse_int_key(value: object, name: str) -> int:
    """Parse an integer or numeric string key with a domain error."""
    if isinstance(value, bool):
        msg = f"checkpoint sampler state has invalid {name}."
        raise TrainingError(msg)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value and value.lstrip("-").isdigit():
        return int(value)
    msg = f"checkpoint sampler state has invalid {name}."
    raise TrainingError(msg)
