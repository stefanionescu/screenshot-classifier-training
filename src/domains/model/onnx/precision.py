"""Write and validate reduced-precision ONNX model artifacts."""

from __future__ import annotations

import onnx
import warnings
from src.errors import TrainingError
from typing import TYPE_CHECKING, cast
import onnxconverter_common.float16 as onnx_float16  # pyright: ignore[reportMissingTypeStubs] -- reason: package ships no stubs
from src.config.artifacts import (
    ONNX_MODEL_FILENAME,
    ONNX_HALF_MODEL_FILENAME,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable


def write_half_precision_model(onnx_dir: Path) -> None:
    """Convert the full-precision ONNX artifact to FP16."""
    source = onnx_dir / ONNX_MODEL_FILENAME
    if not source.is_file():
        msg = "cannot write FP16 candidate because the source ONNX model is missing."
        raise TrainingError(msg)
    read_model: Callable[[str], object] = cast("Callable[[str], object]", onnx.load)
    save_model: Callable[[object, str], None] = cast("Callable[[object, str], None]", onnx.save)
    convert_model: Callable[..., object] = cast(
        "Callable[..., object]",
        onnx_float16.convert_float_to_float16,
    )
    model = read_model(str(source))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        converted = convert_model(model, keep_io_types=True)
    save_model(converted, str(onnx_dir / ONNX_HALF_MODEL_FILENAME))
