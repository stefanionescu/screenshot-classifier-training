"""Names and size constraints for persisted training artifacts."""

from __future__ import annotations

TRAIN_ONNX_DIR = "onnx"
TRAIN_EXPORT_DIR = "export"
TRAIN_INFERENCE_DIR = "inference"
TRAIN_TEXT_ENCODING = "utf-8"
TRAIN_CHECKPOINTS_DIR = "checkpoints"
TRAIN_LOCAL_DIR = "train"
TRAIN_LOCAL_CONFIG_DIR = "config"
TRAIN_LOCAL_METRICS_DIR = "metrics"
TRAIN_LOCAL_PREDICTIONS_DIR = "predictions"
TRAIN_LOCAL_FAILURES_DIR = "failures"
TRAIN_LOCAL_ANALYSIS_DIR = "analysis"
TRAIN_LOCAL_LOGS_DIR = "logs"
TRAIN_README_FILENAME = "README.md"
TRAIN_CONFIG_FILENAME = "train.json"
TRAIN_PROFILE_FILENAME = "dataset_profile.json"
TRAIN_SKIPPED_IMAGES_FILENAME = "skipped_images.jsonl"
TRAIN_REPORT_FILENAME = "report.json"
TRAIN_REPORT_MARKDOWN_FILENAME = "report.md"
TRAIN_EPOCHS_FILENAME = "epochs.jsonl"
TRAIN_CALIBRATION_FILENAME = "calibration.json"
VAL_ERRORS_FILENAME = "val_full_errors.jsonl"
TEST_ERRORS_FILENAME = "test_full_errors.jsonl"
TOP_TWO_RESCUES_FILENAME = "top2_rescues.jsonl"
CONFIDENT_WRONG_EIGHTY_FILENAME = "confident_wrong_80.jsonl"
CONFIDENT_WRONG_NINETY_FILENAME = "confident_wrong_90.jsonl"
TRAIN_UNCONFIDENT_WRONG_FILENAME = "unconfident_wrong.jsonl"
TRAIN_MODEL_CONFIG_FILENAME = "config.json"
TRAIN_MODEL_WEIGHTS_FILENAME = "model.safetensors"
TRAIN_PREPROCESS_FILENAME = "preprocess.json"
TRAIN_PYTHON_INFERENCE_FILENAME = "python.py"
TRAIN_LABELS_FILENAME = "labels.json"
TRAIN_BEST_CHECKPOINT_FILENAME = "best.pt"
CURRENT_CHECKPOINT_FILENAME = "latest.pt"
TRAIN_EXPORT_CHECKPOINT_FILENAME = "checkpoint.pt"

ONNX_MODEL_FILENAME = "model.onnx"
ONNX_HALF_MODEL_FILENAME = "model.fp16.onnx"
ONNX_INPUT_NAME = "image"
ONNX_SCREEN_OUTPUT_NAME = "screen"
ONNX_SAFETY_OUTPUT_NAME = "safety"
ONNX_OUTPUT_NAMES = ("screen", "safety")
ONNX_CPU_PROVIDERS = ("CPUExecutionProvider",)
ONNX_TIMING_PROVIDER = "onnxruntime:CPUExecutionProvider"
ONNX_OPSET_VERSION = 18
ONNX_DUMMY_BATCH_SIZE = 1
ONNX_DUMMY_CHANNELS = 3
ONNX_MIN_DIMENSION = 32
ONNX_DUMMY_WIDTH_DIVISOR = 2
ONNX_SCREENSHOT_WIDTH_NUMERATOR = 9
ONNX_SCREENSHOT_WIDTH_DENOMINATOR = 16

TRAIN_REQUIRED_EXPORT_PATHS = (
    "README.md",
    "config.json",
    "model.safetensors",
    "onnx/model.onnx",
    "onnx/model.onnx.data",
    "preprocess.json",
    "train.json",
    "checkpoints/checkpoint.pt",
    "inference/python.py",
    "inference/labels.json",
)

TRAIN_MODEL_NAME = "Screenshot Classifier"
TRAIN_MODEL_ARCHITECTURE = "timm-spatial-flat-multitask"
TRAIN_MODEL_LIBRARY = "timm"
