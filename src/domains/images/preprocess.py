"""Derive exported normalization settings from a timm backbone."""

from __future__ import annotations

import timm
from typing import TYPE_CHECKING, cast
from src.state.inputs import PreprocessSpec
from src.domains.model.network import normalize_stats

if TYPE_CHECKING:
    from typing import Protocol
    from collections.abc import Mapping

    class TimmModel(Protocol):
        """Model config fields used for preprocessing."""

        pretrained_cfg: Mapping[str, object]


def build_preprocess_spec(model_id: str) -> PreprocessSpec:
    """Build preprocess spec."""
    model = cast(
        "TimmModel",
        timm.create_model(
            f"hf_hub:{model_id}",
            pretrained=False,
            num_classes=0,
            global_pool="",
        ),
    )
    return PreprocessSpec(
        mean=normalize_stats(model.pretrained_cfg.get("mean")),
        std=normalize_stats(model.pretrained_cfg.get("std")),
    )
