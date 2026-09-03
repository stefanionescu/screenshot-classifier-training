"""Create typed ONNX Runtime sessions for exported artifacts."""

from __future__ import annotations

import onnxruntime as onnx_runtime  # pyright: ignore[reportMissingTypeStubs] -- reason: package ships no stubs
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path
    from src.state.eval import OnnxSession
    from collections.abc import Callable, Sequence


def create_onnx_session(model_path: Path, providers: Sequence[str]) -> OnnxSession:
    """Create an ONNX Runtime session."""
    create_session: Callable[..., object] = cast("Callable[..., object]", onnx_runtime.InferenceSession)
    return cast("OnnxSession", create_session(str(model_path), providers=list(providers)))
