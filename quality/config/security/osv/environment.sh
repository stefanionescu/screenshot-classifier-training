#!/usr/bin/env bash
#
# Configure the dependency lockfiles scanned by OSV Scanner.
# Runtime: Bash 3.2+, macOS and Linux.
# shellcheck shell=bash
# lint:justify -- reason: OSV policy is sourced by the scanner entrypoint -- ticket: quality-security
# shellcheck disable=SC2034

[[ -n ${_CFG_OSV_ENVIRONMENT_READY:-} ]] && return 0
readonly _CFG_OSV_ENVIRONMENT_READY=1

OSV_LOCKFILES=("uv.lock" "bun.lock")
