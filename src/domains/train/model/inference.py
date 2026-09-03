"""Run exported ONNX classifiers without the training application."""

from __future__ import annotations

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import onnxruntime as onnx_runtime  # pyright: ignore[reportMissingTypeStubs] -- reason: package ships no stubs
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


MODEL_DIR = Path(__file__).resolve().parent.parent
PADDING_MULTIPLE = 32
ModelFormat = Literal["fp32", "fp16"]
MODEL_FILENAMES: dict[ModelFormat, str] = {
    "fp32": "model.onnx",
    "fp16": "model.fp16.onnx",
}


class Preprocess(TypedDict):
    """Represent preprocess."""

    resize_longest_side_px: int
    mean: list[float]
    std: list[float]


class LabelSet(TypedDict):
    """Represent label set."""

    labels: list[str]


class Labels(TypedDict):
    """Represent labels."""

    screen: LabelSet
    safety: LabelSet


class Prediction(TypedDict):
    """Represent prediction."""

    screen: str
    safety: str


class OnnxSession(Protocol):
    """ONNX Runtime operation required by the standalone classifier."""

    def run(self, _output_names: None, _inputs: dict[str, np.ndarray]) -> Sequence[np.ndarray]:
        """Execute one model batch."""
        raise NotImplementedError


class Classifier:
    """Standalone classifier backed by one exported ONNX model."""

    def __init__(
        self,
        model_dir: str | Path = MODEL_DIR,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        model_format: ModelFormat = "fp32",
    ) -> None:
        """Create an ONNX classifier from an exported model directory."""
        self._model_dir = Path(model_dir)
        self._session = create_onnx_session(select_model_path(self._model_dir, model_format), providers)
        self._preprocess = cast(
            "Preprocess",
            json.loads((self._model_dir / "preprocess.json").read_text(encoding="utf-8")),
        )
        self._labels = cast(
            "Labels",
            json.loads((self._model_dir / "inference" / "labels.json").read_text(encoding="utf-8")),
        )

    def classify_image(self, image_path: str | Path) -> Prediction:
        """Classify one image."""
        image = preprocess_image(Path(image_path), self._preprocess)
        outputs = self._session.run(None, {"image": collate_images((image,))})
        return decode_predictions(outputs[0], outputs[1], self._labels)[0]

    def classify_images(self, image_paths: Sequence[str | Path]) -> list[Prediction]:
        """Classify an ordered image collection."""
        if not image_paths:
            return []
        images = [preprocess_image(Path(image_path), self._preprocess) for image_path in image_paths]
        outputs = self._session.run(None, {"image": collate_images(images)})
        return decode_predictions(outputs[0], outputs[1], self._labels)


def create_onnx_session(model_path: Path, providers: Sequence[str]) -> OnnxSession:
    """Create an ONNX Runtime session."""
    create_session: Callable[..., object] = cast("Callable[..., object]", onnx_runtime.InferenceSession)
    return cast("OnnxSession", create_session(str(model_path), providers=list(providers)))


def select_model_path(model_dir: Path, model_format: ModelFormat) -> Path:
    """Resolve the requested ONNX artifact or report that it is absent."""
    path = model_dir / "onnx" / MODEL_FILENAMES[model_format]
    if not path.is_file():
        msg = f"missing {model_format} ONNX model. Export that model format before classification."
        raise FileNotFoundError(msg)
    return path


def preprocess_image(image_path: Path, preprocess: Preprocess) -> np.ndarray:
    """Decode and normalize an image into a channel-first model tensor."""
    with Image.open(image_path) as opened:
        image = to_training_rgb(opened)
    resized = resize_image(image, preprocess["resize_longest_side_px"])
    array = np.asarray(resized).astype("float32") / 255.0
    mean = np.asarray(preprocess["mean"], dtype="float32")
    std = np.asarray(preprocess["std"], dtype="float32")
    array = (array - mean) / std
    return np.transpose(array, (2, 0, 1))


def to_training_rgb(image: Image.Image) -> Image.Image:
    """Apply training-time orientation and alpha-flattening semantics."""
    image = ImageOps.exif_transpose(image)
    if image.mode == "P" and isinstance(image.info.get("transparency"), bytes):
        image = image.convert("RGBA")
    if image.mode in ("RGBA", "LA", "PA"):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)
    return image.convert("RGB")


def collate_images(images: Sequence[np.ndarray]) -> np.ndarray:
    """Pad channel-first images into one aligned float32 batch."""
    height = round_up(max(int(image.shape[1]) for image in images))
    width = round_up(max(int(image.shape[2]) for image in images))
    batch = np.zeros((len(images), 3, height, width), dtype="float32")
    for index, image in enumerate(images):
        image_height = int(image.shape[1])
        image_width = int(image.shape[2])
        batch[index, :, :image_height, :image_width] = image
    return batch


def resize_image(image: Image.Image, image_size: int) -> Image.Image:
    """Resize image."""
    scale = image_size / max(image.width, image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    return image.resize((width, height), Image.Resampling.BICUBIC)


def round_up(value: int, multiple: int = PADDING_MULTIPLE) -> int:
    """Round a tensor dimension up to the required padding multiple."""
    return ((value + multiple - 1) // multiple) * multiple


def decode_predictions(screen_logits: np.ndarray, safety_logits: np.ndarray, labels: Labels) -> list[Prediction]:
    """Map flat ONNX logits to their highest-scoring label names."""
    screen_indices = top_indices(screen_logits)
    safety_indices = top_indices(safety_logits)
    return [
        Prediction(
            screen=labels["screen"]["labels"][screen_index],
            safety=labels["safety"]["labels"][safety_index],
        )
        for screen_index, safety_index in zip(screen_indices, safety_indices, strict=True)
    ]


def top_indices(logits: np.ndarray) -> list[int]:
    """Return the highest-scoring class index for each batch row."""
    return [int(index) for index in np.argmax(logits, axis=1)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify images with the exported ONNX model.")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--model-format", choices=tuple(MODEL_FILENAMES), default="fp32")
    args = parser.parse_args()
    classifier = Classifier(cast("str", args.model_dir), model_format=cast("ModelFormat", args.model_format))
    predictions = classifier.classify_images(cast("list[str]", args.images))
    value: Prediction | list[Prediction] = predictions[0] if len(predictions) == 1 else predictions
    sys.stdout.write(f"{json.dumps(value, indent=2)}\n")
