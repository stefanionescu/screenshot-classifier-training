"""CLI environment setup."""

from __future__ import annotations

import os
from dotenv import load_dotenv
from src.shared.paths import PROJECT_ROOT
from src.config.train.paths import TRAIN_ENV_FILE_NAME, TRAIN_HUGGINGFACE_CACHE_DIR
from src.config.train.cli import TRAIN_ENV_HF_HOME, TRAIN_ENV_HF_PROGRESS, TRAIN_ENV_TQDM_DISABLE


def configure_cli_environment() -> None:
    """Load the stable project environment once at a command entrypoint."""
    load_dotenv(PROJECT_ROOT / TRAIN_ENV_FILE_NAME)
    os.environ.setdefault(TRAIN_ENV_HF_HOME, str(PROJECT_ROOT / TRAIN_HUGGINGFACE_CACHE_DIR))
    os.environ.setdefault(TRAIN_ENV_HF_PROGRESS, "1")
    os.environ.setdefault(TRAIN_ENV_TQDM_DISABLE, "1")
