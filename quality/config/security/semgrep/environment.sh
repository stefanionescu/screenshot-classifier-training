#!/usr/bin/env bash
#
# Configure repository-owned Semgrep policies and scan paths.
# Runtime: Bash 3.2+, macOS and Linux.
# shellcheck shell=bash
# lint:justify -- reason: Semgrep policy is sourced by the scanner entrypoint -- ticket: quality-security
# shellcheck disable=SC2034

[[ -n ${_CFG_SEMGREP_ENVIRONMENT_READY:-} ]] && return 0
readonly _CFG_SEMGREP_ENVIRONMENT_READY=1

SEMGREP_FLAGS=(
  --error
  --no-rewrite-rule-ids
  --metrics=off
  --disable-version-check
)
SEMGREP_LOCAL_CONFIGS=(
  p/python
  p/bandit
  p/owasp-top-ten
  p/secrets
  r/bash
  quality/config/security/semgrep/python.yml
  quality/config/security/semgrep/secrets.yml
  quality/config/security/semgrep/markers.yml
)
SEMGREP_SCAN_PATHS=(
  src
  quality
)
readonly -a SEMGREP_FLAGS
readonly -a SEMGREP_LOCAL_CONFIGS
readonly -a SEMGREP_SCAN_PATHS
