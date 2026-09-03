"""Convert ONNX tensor storage to float16 through a typed adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
import onnxconverter_common.float16 as onnx_float16  # pyright: ignore[reportMissingTypeStubs] -- reason: package ships no stubs

if TYPE_CHECKING:
    from collections.abc import Callable


def convert_to_fp16(model: object, *, is_io_type_preserved: bool) -> object:
    """Convert an ONNX model to FP16."""
    convert: Callable[..., object] = cast("Callable[..., object]", onnx_float16.convert_float_to_float16)
    return convert(model, keep_io_types=is_io_type_preserved)
