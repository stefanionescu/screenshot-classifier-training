"""Adaptive sampler state contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict

if TYPE_CHECKING:
    from src.state.train.sampler.snapshot import SamplerSnapshot


class SamplerLabelCoverage(TypedDict):
    """Coverage counters for one adaptive-sampler label."""

    dataset_count: int
    current_cycle_seen: int
    current_cycle_fraction: float
    cycle_count: int
    total_draws: int
    epoch_target_count: int


type SamplerCoverage = dict[str, SamplerLabelCoverage]


class AdaptiveSampler(Protocol):
    """Adaptive sampler shape stored in training state."""

    epoch: int

    def state_dict(self) -> dict[str, object]:
        """Return serializable sampler state."""
        raise NotImplementedError

    def restore_state(self, state: dict[str, object]) -> None:
        """Load sampler state."""
        raise NotImplementedError(state)

    def validated_snapshot(self, state: dict[str, object]) -> SamplerSnapshot:
        """Validate and parse sampler state without mutation."""
        raise NotImplementedError(state)

    def coverage_stats(self) -> SamplerCoverage:
        """Return sampler coverage metrics."""
        raise NotImplementedError
