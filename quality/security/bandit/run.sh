#!/usr/bin/env bash
#
# Run Bandit security checks.
# Runtime: Bash 3.2+, macOS and Linux.
set -euo pipefail

REPO_ROOT="${MISE_PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
cd "${REPO_ROOT}" || exit 1

# main - Scans repository Python sources with Bandit.
main() {
  uv run bandit -c pyproject.toml -r src quality
}

main "$@"
