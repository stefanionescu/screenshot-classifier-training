"""Construct the shared-backbone screen and safety classifier."""

from __future__ import annotations

import math
import timm
import torch
from torch import nn
from collections.abc import Sequence
from src.errors import TrainingError
from typing import TYPE_CHECKING, cast
from src.domains.images.augmentation import round_up
from src.config.model import (
    BACKBONE_LR_DIVISOR,
    BACKBONE_PROBE_SIZE,
    SPATIAL_FEATURE_DIMS,
    SAFETY_POOL_MAX_CELLS,
    SAFETY_POOL_MIN_CELLS,
    SAFETY_POOL_CELL_RATIO,
)
from src.config.artifacts import (
    ONNX_MIN_DIMENSION,
    ONNX_DUMMY_CHANNELS,
    ONNX_DUMMY_BATCH_SIZE,
    ONNX_SAFETY_OUTPUT_NAME,
    ONNX_SCREEN_OUTPUT_NAME,
    ONNX_SCREENSHOT_WIDTH_NUMERATOR,
    ONNX_SCREENSHOT_WIDTH_DENOMINATOR,
)

if TYPE_CHECKING:
    from src.state.types import NormalizeStats


class MultiTaskClassifier(nn.Module):
    """Represent multi task classifier."""

    def __init__(
        self,
        model_id: str,
        screen_count: int,
        safety_count: int,
    ) -> None:
        """Create the shared backbone and task heads."""
        super().__init__()
        self.backbone = timm.create_model(
            f"hf_hub:{model_id}",
            pretrained=True,
            num_classes=0,
            global_pool="",
        )
        feature_count = backbone_channels(self.backbone)
        self.screen_pool = nn.AdaptiveAvgPool2d(1)
        self.screen = nn.Linear(feature_count, screen_count)
        self.safety = nn.Conv2d(feature_count, safety_count, kernel_size=1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Produce screen logits and spatially pooled safety logits."""
        feature_map = self.backbone(images)
        screen_features = self.screen_pool(feature_map).flatten(1)
        safety_logits = self.safety(feature_map).flatten(2)
        safety_topk = safety_logits.topk(k=safety_pool_count(safety_logits.shape[-1]), dim=2).values
        return {
            "screen": self.screen(screen_features),
            "safety": safety_topk.mean(dim=2),
        }


def device() -> torch.device:
    """Select the best available torch accelerator, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safety_pool_count(cell_count: int) -> int:
    """Choose the bounded number of spatial safety cells to pool."""
    count = max(SAFETY_POOL_MIN_CELLS, safety_pool_ratio_count(cell_count))
    return min(count, SAFETY_POOL_MAX_CELLS, cell_count)


def safety_pool_ratio_count(cell_count: int) -> int:
    """Round half up after applying the configured spatial-cell ratio."""
    if cell_count < 1:
        msg = "safety pooling requires at least one spatial cell."
        raise TrainingError(msg)
    return max(1, math.floor(cell_count * SAFETY_POOL_CELL_RATIO + 0.5))


def backbone_channels(backbone: nn.Module) -> int:
    """Probe and return the channel count of a spatial backbone output."""
    backbone.eval()
    with torch.no_grad():
        features = backbone(
            torch.zeros(
                (
                    ONNX_DUMMY_BATCH_SIZE,
                    ONNX_DUMMY_CHANNELS,
                    BACKBONE_PROBE_SIZE,
                    BACKBONE_PROBE_SIZE,
                ),
            ),
        )
    if features.ndim == SPATIAL_FEATURE_DIMS:
        return int(features.shape[1])
    msg = f"backbone must return spatial feature map, got shape {tuple(features.shape)}"
    raise TrainingError(msg)


def normalize_stats(value: object) -> NormalizeStats:
    """Validate three finite numeric channel-normalization values."""
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        stats = cast("Sequence[object]", value)
        if len(stats) == ONNX_DUMMY_CHANNELS and all(
            not isinstance(item, bool) and isinstance(item, int | float) and math.isfinite(float(item))
            for item in stats
        ):
            values = [float(item) for item in cast("Sequence[int | float]", stats)]
            return values[0], values[1], values[2]
    msg = "model preprocessing config must include three-channel normalization stats."
    raise TrainingError(msg)


def build_optimizer(model: MultiTaskClassifier, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Build optimizer."""
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": lr / BACKBONE_LR_DIVISOR},
            {"params": model.screen.parameters(), "lr": lr},
            {"params": model.safety.parameters(), "lr": lr},
        ],
        weight_decay=weight_decay,
    )


def check_backbone_features(model: MultiTaskClassifier, image_size: int, run_device: torch.device) -> None:
    """Check backbone features."""
    model.eval()
    width = round_up(
        max(
            ONNX_MIN_DIMENSION,
            round(image_size * ONNX_SCREENSHOT_WIDTH_NUMERATOR / ONNX_SCREENSHOT_WIDTH_DENOMINATOR),
        ),
    )
    height = round_up(image_size)
    with torch.no_grad():
        features = model.backbone(
            torch.zeros((ONNX_DUMMY_BATCH_SIZE, ONNX_DUMMY_CHANNELS, height, width), device=run_device),
        )
    if features.ndim != SPATIAL_FEATURE_DIMS:
        msg = f"backbone must return spatial feature map, got shape {tuple(features.shape)}"
        raise TrainingError(msg)


def public_logits(output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map internal task heads to stable exported ONNX output names."""
    return {
        ONNX_SCREEN_OUTPUT_NAME: output["screen"],
        ONNX_SAFETY_OUTPUT_NAME: output["safety"],
    }
