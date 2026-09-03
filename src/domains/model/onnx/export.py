"""Export and validate ONNX training models."""

from __future__ import annotations

import io
import copy
import torch
import inspect
import contextlib
import numpy as np
from torch import nn
from src.errors import TrainingError
from typing import TYPE_CHECKING, cast
from src.domains.model.network import public_logits
from src.domains.images.augmentation import round_up
from src.domains.model.onnx.runtime import create_onnx_session
from src.config.model import (
    TRAIN_DEVICE_CPU,
    TORCH_DYNAMIC_BATCH_DIM,
    TORCH_DYNAMIC_WIDTH_DIM,
    TORCH_DYNAMIC_HEIGHT_DIM,
)
from src.config.artifacts import (
    ONNX_INPUT_NAME,
    ONNX_OUTPUT_NAMES,
    ONNX_CPU_PROVIDERS,
    ONNX_MIN_DIMENSION,
    ONNX_OPSET_VERSION,
    ONNX_DUMMY_CHANNELS,
    ONNX_DUMMY_BATCH_SIZE,
    ONNX_SAFETY_OUTPUT_NAME,
    ONNX_SCREEN_OUTPUT_NAME,
    ONNX_DUMMY_WIDTH_DIVISOR,
    ONNX_SCREENSHOT_WIDTH_NUMERATOR,
    ONNX_SCREENSHOT_WIDTH_DENOMINATOR,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable
    from src.state.types import TrainModel


def export_onnx(
    model: TrainModel,
    output_path: Path,
    image_size: int,
) -> None:
    """Export an ONNX model."""

    class OnnxModel(nn.Module):
        """Expose classifier outputs as ONNX tuple outputs."""

        def __init__(self, classifier: TrainModel) -> None:
            super().__init__()
            self.classifier = classifier

        def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Run classifier and return screen and safety logits."""
            output = self.classifier(image)
            public = public_logits(output)
            return public[ONNX_SCREEN_OUTPUT_NAME], public[ONNX_SAFETY_OUTPUT_NAME]

    onnx_model = OnnxModel(copy.deepcopy(model).to(TRAIN_DEVICE_CPU).eval()).eval()
    dummy_width = round_up(max(ONNX_MIN_DIMENSION, image_size // ONNX_DUMMY_WIDTH_DIVISOR))
    dummy = torch.zeros(
        (ONNX_DUMMY_BATCH_SIZE, ONNX_DUMMY_CHANNELS, round_up(image_size), dummy_width),
        device=TRAIN_DEVICE_CPU,
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        export_model: Callable[..., None] = cast("Callable[..., None]", torch.onnx.export)
        export_model(
            onnx_model,
            (dummy,),
            output_path,
            input_names=[ONNX_INPUT_NAME],
            output_names=list(ONNX_OUTPUT_NAMES),
            opset_version=ONNX_OPSET_VERSION,
            **onnx_dynamic_export_options(),
        )


def onnx_dynamic_export_options() -> dict[str, object]:
    """Return ONNX dynamic export options for the installed Torch API."""
    export_model: Callable[..., object] = cast("Callable[..., object]", torch.onnx.export)
    parameters = inspect.signature(export_model).parameters
    if "dynamic_shapes" in parameters:
        return {
            "dynamic_shapes": {
                ONNX_INPUT_NAME: {
                    0: torch.export.Dim(TORCH_DYNAMIC_BATCH_DIM),
                    2: torch.export.Dim(TORCH_DYNAMIC_HEIGHT_DIM),
                    3: torch.export.Dim(TORCH_DYNAMIC_WIDTH_DIM),
                },
            },
            "dynamo": True,
        }
    return {
        "dynamic_axes": {
            ONNX_INPUT_NAME: {
                0: TORCH_DYNAMIC_BATCH_DIM,
                2: TORCH_DYNAMIC_HEIGHT_DIM,
                3: TORCH_DYNAMIC_WIDTH_DIM,
            },
            ONNX_SCREEN_OUTPUT_NAME: {
                0: TORCH_DYNAMIC_BATCH_DIM,
            },
            ONNX_SAFETY_OUTPUT_NAME: {
                0: TORCH_DYNAMIC_BATCH_DIM,
            },
        },
        "dynamo": False,
    }


def test_onnx_export(
    output_path: Path,
    screen_count: int,
    safety_count: int,
    image_size: int,
) -> None:
    """Validate an exported ONNX model."""
    width = round_up(
        max(
            ONNX_MIN_DIMENSION,
            round(image_size * ONNX_SCREENSHOT_WIDTH_NUMERATOR / ONNX_SCREENSHOT_WIDTH_DENOMINATOR),
        ),
    )
    image = np.zeros(
        (ONNX_DUMMY_BATCH_SIZE, ONNX_DUMMY_CHANNELS, round_up(image_size), width),
        dtype=np.float32,
    )
    session = create_onnx_session(output_path, ONNX_CPU_PROVIDERS)
    input_names = [value.name for value in session.get_inputs()]
    if input_names != [ONNX_INPUT_NAME]:
        msg = f"ONNX input names must be {[ONNX_INPUT_NAME]}, got {input_names}"
        raise TrainingError(msg)
    output_names = [output.name for output in session.get_outputs()]
    if output_names != list(ONNX_OUTPUT_NAMES):
        msg = f"ONNX output names must be {list(ONNX_OUTPUT_NAMES)}, got {output_names}"
        raise TrainingError(msg)
    outputs = session.run(None, {ONNX_INPUT_NAME: image})
    screen_output = outputs[0]
    safety_output = outputs[1]
    if tuple(screen_output.shape) != (ONNX_DUMMY_BATCH_SIZE, screen_count):
        msg = f"ONNX screen output has wrong shape: {tuple(screen_output.shape)}"
        raise TrainingError(msg)
    if tuple(safety_output.shape) != (ONNX_DUMMY_BATCH_SIZE, safety_count):
        msg = f"ONNX safety output has wrong shape: {tuple(safety_output.shape)}"
        raise TrainingError(msg)
