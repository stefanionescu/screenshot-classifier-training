#!/usr/bin/env bash
#
# Configure repository-owned Bearer scan arguments.
# Runtime: Bash 3.2+, macOS and Linux.
# shellcheck shell=bash
# lint:justify -- reason: Bearer policy is sourced by scanner scripts -- ticket: quality-security
# shellcheck disable=SC2034

[[ -n ${_CFG_BEARER_ENVIRONMENT_READY:-} ]] && return 0
readonly _CFG_BEARER_ENVIRONMENT_READY=1

BEARER_SCAN_PATHS=(src quality)
BEARER_FLAGS=(
  --severity "critical,high,medium"
  --format json
  --quiet
  --exit-code 1
)
readonly BEARER_SCAN_PATHS
readonly -a BEARER_FLAGS
