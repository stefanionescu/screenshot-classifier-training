"""Normalize, collate, and augment images for model training."""

from __future__ import annotations

import io
import torch
import numpy as np
from typing import TYPE_CHECKING
from src.state.types import SampleMeta
from PIL import Image, ImageEnhance, ImageOps, features
from src.config.model import (
    NORMALIZE_STD,
    NORMALIZE_MEAN,
    AUG_CONTRAST_MAX,
    AUG_CONTRAST_MIN,
    PADDING_MULTIPLE,
    AUG_BRIGHTNESS_MAX,
    AUG_BRIGHTNESS_MIN,
    ARTIFACT_AUG_CODECS,
    PORTRAIT_ASPECT_MAX,
    LANDSCAPE_ASPECT_MIN,
    ARTIFACT_AUG_PROBABILITY,
    ARTIFACT_AUG_QUALITY_MAX,
    ARTIFACT_AUG_QUALITY_MIN,
    AUG_CONTRAST_PROBABILITY,
    AUG_BRIGHTNESS_PROBABILITY,
)

if TYPE_CHECKING:
    from src.state.training import Sample

BatchItem = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[SampleMeta, ...],
]

DatasetItem = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    SampleMeta,
]
GRAYSCALE_DIMENSIONS = 2


def aspect_bucket(sample: Sample) -> str:
    """Assign a sample to its portrait, square, or landscape batch."""
    ratio = sample.width / max(sample.height, 1)
    if ratio < PORTRAIT_ASPECT_MAX:
        return "portrait"
    if ratio > LANDSCAPE_ASPECT_MIN:
        return "landscape"
    return "square"


def round_up(value: int, multiple: int = PADDING_MULTIPLE) -> int:
    """Round a tensor dimension up to the required padding multiple."""
    return ((value + multiple - 1) // multiple) * multiple


def preprocess_image(
    image: Image.Image,
    image_size: int,
    normalize_mean: tuple[float, float, float] = NORMALIZE_MEAN,
    normalize_std: tuple[float, float, float] = NORMALIZE_STD,
) -> torch.Tensor:
    """Resize and normalize an RGB image into a model tensor."""
    long_side = max(image.width, image.height)
    scale = image_size / long_side
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    resized = image.resize((width, height), Image.Resampling.BICUBIC)
    tensor = image_to_tensor(resized)
    return normalize_tensor(tensor, normalize_mean, normalize_std)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a channel-first float tensor."""
    array = np.array(image, dtype=np.float32, copy=True) / np.float32(255.0)
    if array.ndim == GRAYSCALE_DIMENSIONS:
        array = array[:, :, None]
    return torch.tensor(np.transpose(array, (2, 0, 1)), dtype=torch.float32)


def normalize_tensor(
    tensor: torch.Tensor,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    """Normalize an image tensor channel-wise."""
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)[:, None, None]
    std_tensor = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)[:, None, None]
    return (tensor - mean_tensor) / std_tensor


def to_training_rgb(image: Image.Image) -> Image.Image:
    """Apply orientation and white-background alpha flattening."""
    image = ImageOps.exif_transpose(image)
    if image.mode == "P" and isinstance(image.info.get("transparency"), bytes):
        image = image.convert("RGBA")
    if image.mode in ("RGBA", "LA", "PA"):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)
    return image.convert("RGB")


def collate_batch(
    batch: list[DatasetItem | None],
) -> BatchItem | None:
    """Drop rejected samples and collate the remaining multitask batch."""
    samples = [sample for sample in batch if sample is not None]
    if not samples:
        return None
    images, screen, safety, metas = zip(*samples, strict=True)
    return (
        collate_image_inputs(list(images)),
        torch.stack(screen),
        torch.stack(safety),
        metas,
    )


def collate_image_inputs(images: list[torch.Tensor]) -> torch.Tensor:
    """Pad variable-size tensors to one aligned batch shape."""
    height = round_up(max(image.shape[1] for image in images))
    width = round_up(max(image.shape[2] for image in images))
    padded_images = [
        torch.nn.functional.pad(image, (0, width - image.shape[2], 0, height - image.shape[1])) for image in images
    ]
    return torch.stack(padded_images)


def augment_image(image: Image.Image) -> Image.Image:
    """Apply independently sampled photometric and codec augmentation."""
    if float(torch.rand(()).item()) < AUG_BRIGHTNESS_PROBABILITY:
        image = apply_brightness(image)
    if float(torch.rand(()).item()) < AUG_CONTRAST_PROBABILITY:
        image = apply_contrast(image)
    if float(torch.rand(()).item()) < ARTIFACT_AUG_PROBABILITY:
        image = apply_artifact_aug(image)
    return image


def apply_brightness(image: Image.Image) -> Image.Image:
    """Apply brightness."""
    factor = float(torch.empty(()).uniform_(AUG_BRIGHTNESS_MIN, AUG_BRIGHTNESS_MAX).item())
    return ImageEnhance.Brightness(image).enhance(factor)


def apply_contrast(image: Image.Image) -> Image.Image:
    """Apply contrast."""
    factor = float(torch.empty(()).uniform_(AUG_CONTRAST_MIN, AUG_CONTRAST_MAX).item())
    return ImageEnhance.Contrast(image).enhance(factor)


def apply_artifact_aug(image: Image.Image) -> Image.Image:
    """Apply artifact aug."""
    buffer = io.BytesIO()
    codecs = artifact_codecs()
    codec_index = int(torch.randint(len(codecs), ()).item())
    image.save(buffer, format=codecs[codec_index], quality=sample_artifact_quality())
    buffer.seek(0)
    with Image.open(buffer) as reloaded:
        return to_training_rgb(reloaded)


def artifact_codecs() -> tuple[str, ...]:
    """Return artifact codecs supported by the installed Pillow build."""
    if features.check("webp"):
        return ARTIFACT_AUG_CODECS
    return ("JPEG",)


def sample_artifact_quality() -> int:
    """Sample an inclusive codec-quality value from the configured range."""
    return int(torch.randint(ARTIFACT_AUG_QUALITY_MIN, ARTIFACT_AUG_QUALITY_MAX + 1, ()).item())
