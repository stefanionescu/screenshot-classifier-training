#!/usr/bin/env bash
#
# Generate a CycloneDX SBOM for the active Python virtual environment.
# Runtime: Bash 3.2+, macOS and Linux.
set -euo pipefail

REPO_ROOT="${MISE_PROJECT_ROOT:-$(git rev-parse --show-toplevel)}"
cd "${REPO_ROOT}" || exit 1

# main - Generates the Python environment software bill of materials.
main() {
  mkdir -p .artifacts/sbom
  uv run cyclonedx-py environment --output-file .artifacts/sbom/python.cdx.json
}

main "$@"
