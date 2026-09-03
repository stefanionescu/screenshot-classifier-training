#!/usr/bin/env bash
#
# Run OSV Scanner against repository dependency state.
# Runtime: Bash 3.2+, macOS and Linux.
set -euo pipefail

REPO_ROOT="${MISE_PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
cd "${REPO_ROOT}" || exit 1

# shellcheck source=../../config/security/osv/environment.sh
source "${REPO_ROOT}/quality/config/security/osv/environment.sh"

# main - Scans locked repository dependencies with OSV Scanner.
main() {
  local lockfile
  local -a scanner_args

  scanner_args=(scan source --config quality/config/security/osv/config.toml)
  for lockfile in "${OSV_LOCKFILES[@]}"; do
    scanner_args+=(--lockfile "${lockfile}")
  done
  osv-scanner "${scanner_args[@]}"
}

main "$@"
