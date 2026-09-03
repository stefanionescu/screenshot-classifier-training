#!/usr/bin/env bash
#
# Run Bearer security checks as a required baseline gate.
# Runtime: Bash 3.2+, macOS and Linux.
set -euo pipefail

REPO_ROOT="${MISE_PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
cd "${REPO_ROOT}" || exit 1

# shellcheck source=../../config/security/bearer/environment.sh
source "${REPO_ROOT}/quality/config/security/bearer/environment.sh"

# main - Scans repository sources with Bearer security policies.
main() {
  if ! bearer scan --help >/dev/null 2>&1; then
    printf '%s\n' 'error: bearer is not available. Run: mise run setup' >&2
    return 1
  fi

  bearer scan "${BEARER_FLAGS[@]}" "${BEARER_SCAN_PATHS[@]}"
}

main "$@"
