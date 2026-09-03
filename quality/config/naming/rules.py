"""Naming rule runner policy."""

from quality.config.repository.paths import PYTHON_SOURCE_DIRS

SINGLE_FILE_SOURCE_ROOTS = {
    "src",
    "quality",
}

SCOPE_SOURCE_DIRS = {
    "all": PYTHON_SOURCE_DIRS,
    "python": PYTHON_SOURCE_DIRS,
    "quality": ("quality",),
    "src": ("src",),
}

PYTHON_SCOPES = {
    "all",
    "python",
    "quality",
    "src",
    "staged",
}

SHELL_SCOPES = {
    "all",
    "shell",
    "hooks",
    "mise",
    "quality",
    "staged",
}

DIGIT_CHECK_CATEGORIES = {
    "directories",
    "files",
}

VAGUE_SCRIPT_NAMES = {
    "common",
    "helper",
    "helpers",
    "utils",
    "util",
}
