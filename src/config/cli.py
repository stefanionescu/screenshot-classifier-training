"""Defaults and display text owned by the training command line."""

from __future__ import annotations

TRAIN_CLI_PROGRAM = "python -m src.cli.train"
TRAIN_CLI_DESCRIPTION = "Train the local phone screen classifier."

TRAIN_ENV_HF_HOME = "HF_HOME"
TRAIN_ENV_HF_PROGRESS = "HF_HUB_DISABLE_PROGRESS_BARS"
TRAIN_ENV_TQDM_DISABLE = "TQDM_DISABLE"

DEFAULT_EPOCHS = 12
DEFAULT_BATCH_SIZE = 8
DEFAULT_WORKERS = 4
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 0.05
DEFAULT_GRAD_ACCUM_STEPS = 0
DEFAULT_SEED = 1337
EVAL_CLASS_LIMIT = 1000

TRAIN_RESUME_OVERRIDES = {
    "workers",
    "repo",
    "push",
    "push_only",
    "public",
    "resume",
}
