"""Folder ownership policy."""

PYTHON_PREFIX_COLLISION_ALLOWLIST: tuple[str, ...] = ()

SINGLE_FILE_PACKAGE_ALLOWLIST: tuple[str, ...] = ()

DISALLOWED_AMBIGUOUS_FOLDER_NAMES = (
    "bash",
    "common",
    "core",
    "helper",
    "helpers",
    "javascript",
    "python",
    "support",
    "util",
    "utils",
)

AMBIGUOUS_FOLDER_ALLOWLIST = ("quality/python",)

AMBIGUOUS_FOLDER_EXCLUDED_PREFIXES = ("quality/config/",)
