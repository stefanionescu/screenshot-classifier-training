"""Build label indexes and encode multitask training targets."""

from __future__ import annotations

from src.errors import TrainingError
from src.state.train.training import LabelState
from src.config.train.model import TRAINABLE_SAFETY_LABELS

IGNORED_LABEL_ID = -100


def build_label_state(screen: list[str], safety: list[str]) -> LabelState:
    """Build label state."""
    if not screen:
        msg = "training needs at least one screen label."
        raise TrainingError(msg)
    if not safety:
        msg = "training needs at least one safety label."
        raise TrainingError(msg)
    unsupported = sorted(set(safety) - set(TRAINABLE_SAFETY_LABELS))
    if unsupported:
        msg = f"training does not support safety labels: {', '.join(unsupported)}."
        raise TrainingError(msg)
    return LabelState(
        screen=screen,
        safety=safety,
        screen_to_id={label: index for index, label in enumerate(screen)},
        safety_to_id={label: index for index, label in enumerate(safety)},
    )
