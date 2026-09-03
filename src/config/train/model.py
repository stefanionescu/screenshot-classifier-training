"""Model, sampling, and optimization settings for training runs."""

from __future__ import annotations

ORG_NAME = "yapwithai"
DATASET_REPO = "yapwithai/phone-screenshots"
DEFAULT_DATASET = "dataset/phone-screenshots"
DEFAULT_MODEL = "timm/mobilenetv4_conv_medium.e250_r384_in12k"
SUPPORTED_MODELS = (
    "timm/mobilenetv4_conv_medium.e250_r384_in12k",
    "timm/mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k",
    "timm/mobilenetv4_conv_medium.e500_r256_in1k",
    "timm/mobilenetv4_conv_small.e2400_r224_in1k",
    "timm/mobilenetv4_conv_large.e500_r256_in1k",
    "timm/mobilenetv4_conv_large.e600_r384_in1k",
    "timm/mobilenetv4_conv_aa_large.e230_r384_in12k",
)
DEFAULT_OUTPUT_DIR = "phone-screen-classifier"
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_MIN_TRAIN_COUNT = 300
PROFILE_MEDIUM_SIDE_MIN = 512
PROFILE_LARGE_SIDE_MIN = 1280
PROFILE_MIN_LABEL_COUNT = 25
PORTRAIT_ASPECT_MAX = 0.85
LANDSCAPE_ASPECT_MIN = 1.15
SCREEN_TARGET_RATIO = 5.0
SAFETY_TARGET_RATIO = 2.0
SCREEN_MAX_REPEAT = 6
SAFETY_MAX_REPEAT = 8
SAFETY_LOSS_WEIGHT = 0.5
SAFETY_BATCH_PROBABILITY = 0.35
SCREEN_OTHER_LABEL = "other"
SCREEN_FALLBACK_LABEL = "generic"
TRAINABLE_SAFETY_LABELS = ("safe", "nsfw", "hot")
TARGET_EFFECTIVE_BATCH_SIZE = 64
BACKBONE_LR_DIVISOR = 5
SAFETY_POOL_MIN_CELLS = 8
SAFETY_POOL_MAX_CELLS = 64
SAFETY_POOL_CELL_RATIO = 0.01
BACKBONE_PROBE_SIZE = 224
SPATIAL_FEATURE_DIMS = 4
PADDING_MULTIPLE = 32
MANIFEST_COLUMNS = ("tar_path", "image_member", "screen", "safety", "width", "height")
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)
AUG_BRIGHTNESS_PROBABILITY = 0.30
AUG_BRIGHTNESS_MIN = 0.90
AUG_BRIGHTNESS_MAX = 1.10
AUG_CONTRAST_PROBABILITY = 0.30
AUG_CONTRAST_MIN = 0.90
AUG_CONTRAST_MAX = 1.10
ARTIFACT_AUG_PROBABILITY = 0.15
ARTIFACT_AUG_QUALITY_MIN = 88
ARTIFACT_AUG_QUALITY_MAX = 98
ARTIFACT_AUG_CODECS = ("JPEG", "WEBP")
TORCH_DYNAMIC_BATCH_DIM = "batch"
TORCH_DYNAMIC_HEIGHT_DIM = "height"
TORCH_DYNAMIC_WIDTH_DIM = "width"

TRAIN_DEVICE_CPU = "cpu"
TRAIN_DEVICE_CUDA = "cuda"
TRAIN_DEVICE_MPS = "mps"
