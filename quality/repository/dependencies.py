"""Validate uv dependency ownership and command workflow."""

from __future__ import annotations

import re
import tomllib
from typing import cast
from pathlib import Path
from quality.lib.output import write_error
from collections.abc import Mapping, Sequence
from quality.config.repository.dependencies import (
    LOCK_FILE,
    UV_TOOL_NAME,
    TOOLS_SECTION,
    PROJECT_SECTION,
    MISE_CONFIG_FILE,
    APPLICATION_TASKS,
    POLICY_SCAN_ROOTS,
    POLICY_FILE_SUFFIXES,
    PROJECT_PYTHON_FIELD,
    PROJECT_METADATA_FILE,
    ROOT_REQUIREMENTS_GLOB,
    RETIRED_WORKFLOW_MARKERS,
    DEPENDENCY_GROUPS_SECTION,
    POLICY_FILE_IGNORED_PARTS,
    POLICY_FILE_IGNORED_PATHS,
    PROJECT_DEPENDENCIES_FIELD,
    REQUIRED_DEPENDENCY_GROUPS,
)

TOOL_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.-]+)?$")
APPLICATION_TASK_TEMPLATE = """#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${MISE_PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
cd "${REPO_ROOT}"

<dispatch>
"""


def _mapping(value: object) -> Mapping[str, object] | None:
    """Return a validated string-keyed mapping."""
    if not isinstance(value, dict):
        return None
    items = cast("dict[object, object]", value)
    if any(not isinstance(key, str) for key in items):
        return None
    return cast("Mapping[str, object]", items)


def _nonempty_strings(value: object) -> bool:
    """Return whether a value is a non-empty sequence of non-empty strings or tables."""
    if not isinstance(value, Sequence) or isinstance(value, str) or not value:
        return False
    items = cast("Sequence[object]", value)
    return all((isinstance(item, str) and bool(item.strip())) or _mapping(item) is not None for item in items)


def _project_errors(config: Mapping[str, object]) -> list[str]:
    """Validate project metadata and dependency groups."""
    errors: list[str] = []
    project = _mapping(config.get(PROJECT_SECTION))
    if project is None:
        errors.append(f"{PROJECT_METADATA_FILE} [{PROJECT_SECTION}] is required")
    else:
        if not _nonempty_strings(project.get(PROJECT_DEPENDENCIES_FIELD)):
            errors.append(f"[{PROJECT_SECTION}].{PROJECT_DEPENDENCIES_FIELD} must own runtime dependencies")
        if project.get(PROJECT_PYTHON_FIELD) != "==3.12.*":
            errors.append(f"[{PROJECT_SECTION}].{PROJECT_PYTHON_FIELD} must be ==3.12.*")
    groups = _mapping(config.get(DEPENDENCY_GROUPS_SECTION))
    errors.extend(
        f"[{DEPENDENCY_GROUPS_SECTION}].{group} must be a non-empty dependency group"
        for group in REQUIRED_DEPENDENCY_GROUPS
        if groups is None or not _nonempty_strings(groups.get(group))
    )
    return errors


def _mise_errors(root: Path) -> list[str]:
    """Validate the exact uv tool pin in Mise metadata."""
    mise_path = root / MISE_CONFIG_FILE
    mise = _mapping(tomllib.loads(mise_path.read_text(encoding="utf-8"))) if mise_path.is_file() else None
    tools = _mapping(mise.get(TOOLS_SECTION)) if mise is not None else None
    uv_version = tools.get(UV_TOOL_NAME) if tools is not None else None
    if isinstance(uv_version, str) and TOOL_VERSION_RE.fullmatch(uv_version):
        return []
    return [f"{MISE_CONFIG_FILE} must pin [{TOOLS_SECTION}].{UV_TOOL_NAME} to an exact version"]


def collect_metadata_violations(root: Path) -> list[str]:
    """Return diagnostics for project metadata, lock, and uv tool pinning."""
    errors: list[str] = []
    pyproject_path = root / PROJECT_METADATA_FILE
    if not pyproject_path.is_file():
        return [f"{PROJECT_METADATA_FILE} is required"]
    if not (root / LOCK_FILE).is_file():
        errors.append(f"{LOCK_FILE} is required and must be generated from {PROJECT_METADATA_FILE}")

    config = _mapping(tomllib.loads(pyproject_path.read_text(encoding="utf-8")))
    if config is None:
        return [*errors, f"{PROJECT_METADATA_FILE} root must be a TOML table"]
    errors.extend(_project_errors(config))
    errors.extend(_mise_errors(root))
    return errors


def collect_task_violations(root: Path) -> list[str]:
    """Return diagnostics for application and dependency mise tasks."""
    errors: list[str] = []
    for task, module in APPLICATION_TASKS.items():
        path = root / ".mise" / "tasks" / task
        dispatch = f'uv run python -m {module} "$@"'
        expected = APPLICATION_TASK_TEMPLATE.replace("<dispatch>", dispatch)
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            errors.append(f"{path.relative_to(root)} must use the application mise task contract")
    deps_task = root / ".mise" / "tasks" / "deps" / "_default"
    if not deps_task.is_file() or "uv sync --all-groups --locked" not in deps_task.read_text(encoding="utf-8"):
        errors.append(".mise/tasks/deps/_default must install the locked uv environment")
    return errors


def iter_policy_files(root: Path) -> list[Path]:
    """Return relevant workflow and documentation files without entering data roots."""
    files: list[Path] = []
    for relative_scope in POLICY_SCAN_ROOTS:
        scope = root / relative_scope
        if scope.is_file():
            files.append(scope)
            continue
        if not scope.is_dir():
            continue
        for directory, names, filenames in scope.walk():
            names[:] = [name for name in names if name not in POLICY_FILE_IGNORED_PARTS]
            files.extend(directory / name for name in filenames if (directory / name).suffix in POLICY_FILE_SUFFIXES)
    return sorted(set(files))


def collect_workflow_violations(root: Path) -> list[str]:
    """Return diagnostics for retired environment and requirements workflow references."""
    errors = [
        f"{path.name} must not be committed; declare dependencies in {PROJECT_METADATA_FILE}"
        for path in sorted(root.glob(ROOT_REQUIREMENTS_GLOB))
        if path.is_file()
    ]
    for path in iter_policy_files(root):
        relative = path.relative_to(root).as_posix()
        if relative in POLICY_FILE_IGNORED_PATHS:
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            errors.extend(
                f"{relative}:{line_number} must not reference retired workflow marker {marker}"
                for marker in RETIRED_WORKFLOW_MARKERS
                if marker in line
            )
    return errors


def main() -> int:
    """Run dependency ownership checks."""
    root = Path.cwd()
    violations = [
        *collect_metadata_violations(root),
        *collect_task_violations(root),
        *collect_workflow_violations(root),
    ]
    if not violations:
        return 0
    write_error("Dependency policy violations:")
    for violation in violations:
        write_error(f"- {violation}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
