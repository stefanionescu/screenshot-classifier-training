"""Profile label intersections, image shapes, and sparse evaluation groups."""

from __future__ import annotations

from typing import TYPE_CHECKING
from src.state.metrics import TrainingProfile
from src.domains.samples.dataset import count_values
from src.config.model import PROFILE_LARGE_SIDE_MIN, PROFILE_MEDIUM_SIDE_MIN, PROFILE_MIN_LABEL_COUNT

if TYPE_CHECKING:
    from src.state.training import Sample


def profile_samples(samples_by_split: dict[str, list[Sample]], screen_labels: list[str]) -> TrainingProfile:
    """Build dataset counts and metric-reliability warnings."""
    profile = TrainingProfile(
        split_counts={},
        screen_counts={},
        safety_counts={},
        screen_by_safety={},
        size_buckets={},
        warnings=[],
    )
    for split, samples in samples_by_split.items():
        profile["split_counts"][split] = len(samples)
        profile["screen_counts"][split] = count_values((sample.screen_label for sample in samples), screen_labels)
        safety_values = sorted({sample.safety_label for sample in samples})
        profile["safety_counts"][split] = count_values((sample.safety_label for sample in samples), safety_values)
        profile["screen_by_safety"][split] = count_screen_by_safety(samples, screen_labels, safety_values)
        profile["size_buckets"][split] = count_size_buckets(samples, screen_labels)
    add_profile_warnings(profile, screen_labels)
    return profile


def count_screen_by_safety(
    samples: list[Sample],
    screen_labels: list[str],
    safety_labels: list[str],
) -> dict[str, dict[str, int]]:
    """Count the safety distribution within each screen class."""
    matrix = {screen: dict.fromkeys(safety_labels, 0) for screen in screen_labels}
    for sample in samples:
        if sample.screen_label in matrix and sample.safety_label in matrix[sample.screen_label]:
            matrix[sample.screen_label][sample.safety_label] += 1
    return matrix


def count_size_buckets(samples: list[Sample], screen_labels: list[str]) -> dict[str, dict[str, int]]:
    """Count size and orientation buckets within each screen class."""
    buckets = {
        label: {
            "small": 0,
            "medium": 0,
            "large": 0,
            "portrait": 0,
            "landscape": 0,
            "square": 0,
        }
        for label in screen_labels
    }
    for sample in samples:
        bucket = buckets.get(sample.screen_label)
        if bucket is None:
            continue
        long_side = max(sample.width, sample.height)
        if long_side < PROFILE_MEDIUM_SIDE_MIN:
            bucket["small"] += 1
        elif long_side < PROFILE_LARGE_SIDE_MIN:
            bucket["medium"] += 1
        else:
            bucket["large"] += 1
        if sample.width == sample.height:
            bucket["square"] += 1
        elif sample.width > sample.height:
            bucket["landscape"] += 1
        else:
            bucket["portrait"] += 1
    return buckets


def add_profile_warnings(profile: TrainingProfile, screen_labels: list[str]) -> None:
    """Add warnings for small validation and test label groups."""
    warnings = profile["warnings"]
    for split in ("val", "test"):
        counts = profile["screen_counts"].get(split, {})
        for label in screen_labels:
            count = counts.get(label, 0)
            if count < PROFILE_MIN_LABEL_COUNT:
                warnings.append(f"{label} has only {count} {split} examples; do not trust that per-class metric.")
