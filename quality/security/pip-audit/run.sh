#!/usr/bin/env bash
#
# Run pip-audit against the locked Python environment.
# Runtime: Bash 3.2+, macOS and Linux.
set -euo pipefail

REPO_ROOT="${MISE_PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
cd "${REPO_ROOT}" || exit 1

# main - Audits the locked Python environment for known vulnerabilities.
main() {
  uv run pip-audit \
    --strict \
    --progress-spinner off
}

main "$@"
