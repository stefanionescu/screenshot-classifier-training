#!/usr/bin/env bash
#
# Provide shared hook-step execution with environment-controlled skipping.
# Runtime: Bash 3.2+, macOS and Linux.

# hook_run_step - Invokes one hook command unless its named skip variable is enabled.
# Globals:
#   Reads the skip variable named by the second argument.
# Arguments:
#   $1: Human-readable step name.
#   $2: Optional environment variable that skips the step when set to 1.
#   Remaining arguments: Command and arguments to invoke.
# Outputs:
#   Writes the step name to standard output when the command runs.
# Returns:
#   Returns the invoked command status, or zero when the step is skipped.
hook_run_step() {
  local name="$1"
  local skip_var="$2"
  local skip_value="0"
  shift 2

  if [[ -n ${skip_var} ]] && [[ -n ${!skip_var+x} ]]; then
    skip_value="${!skip_var}"
  fi
  if [[ -n ${skip_var} && ${skip_value} == "1" ]]; then
    return 0
  fi

  printf '[hook] %s\n' "${name}"
  "$@"
}
