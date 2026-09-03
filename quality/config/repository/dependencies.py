"""Dependency ownership policy."""

PROJECT_METADATA_FILE = "pyproject.toml"
LOCK_FILE = "uv.lock"
MISE_CONFIG_FILE = "mise.toml"

PROJECT_SECTION = "project"
PROJECT_DEPENDENCIES_FIELD = "dependencies"
PROJECT_PYTHON_FIELD = "requires-python"
DEPENDENCY_GROUPS_SECTION = "dependency-groups"
TOOLS_SECTION = "tools"
UV_TOOL_NAME = "uv"

REQUIRED_DEPENDENCY_GROUPS = ("dev", "quality")
ROOT_REQUIREMENTS_GLOB = "requirements*.txt"
APPLICATION_TASKS = {
    "dataset": "src.cli.dataset",
    "train": "src.cli.train",
}
POLICY_SCAN_ROOTS = (
    ".mise/tasks",
    "quality",
    "rules",
    "README.md",
    "mise.toml",
    "pyproject.toml",
)
POLICY_FILE_SUFFIXES = {".md", ".py", ".sh", ".toml"}
POLICY_FILE_IGNORED_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "dataset",
    "models",
    "node_modules",
    "output",
}
POLICY_FILE_IGNORED_PATHS = (
    "quality/config/repository/dependencies.py",
    "quality/repository/dependencies.py",
)
RETIRED_WORKFLOW_MARKERS = (
    ".venv/bin/",
    "easy_install ",
    "pip install ",
    "requirements.txt",
    "setup.py develop",
    "setup.py install",
    "uv pip install ",
)
