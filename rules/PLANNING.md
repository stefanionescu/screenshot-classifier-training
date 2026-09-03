# Creating Plans

These rules apply whenever an agent creates any implementation plan, refactor
plan, documentation plan, data change plan, model-artifact plan, script plan, or
other planned change sequence.

## Contents

- [Complete Change Content](#complete-change-content)
- [No Unrequested Testing or Linting](#no-unrequested-testing-or-linting)
- [Implementation Order](#implementation-order)
- [Plan Detail Level](#plan-detail-level)

## Complete Change Content

A plan must contain the complete content of every change it proposes.

Rules:

- Include a complete code diff or text diff for every code, configuration,
  documentation, data, script, and text change.
- Do not summarize a change when the exact diff can be shown.
- Do not describe a future edit without including the exact patch that makes the
  edit.
- Include every new file's full contents.
- Include every deleted file's full removed contents or the full deletion diff.
- Include every command needed to create, transform, move, rename, resize,
  regenerate, or delete an artifact.
- Include changes to generated files when the plan expects generated files to
  change.
- Include changes to model checkpoints, generated evaluation results, binary
  assets, and other non-text artifacts by listing the exact source path, output
  path, operation, dimensions or metadata changes, and command or tool invocation
  needed to reproduce the result.
- If a binary diff cannot be represented as text, include enough exact
  reproduction detail that the asset change is part of the plan rather than an
  implied follow-up.

Bad:

```text
Change runtime warmup behavior and regenerate the documented warmup artifact.
```

Good:

```diff
diff --git a/src/runtime/settings.py b/src/runtime/settings.py
--- a/src/runtime/settings.py
+++ b/src/runtime/settings.py
@@
-warmup_timeout_s = timeout_s
+warmup_timeout_s = min(timeout_s, max_warmup_timeout_s)
```

```bash
bash scripts/warmup.sh --model Qwen/Qwen3-14B
```

```text
Artifact change:
- Path: .artifacts/warmup/
- Operation: generate a new warmup result JSON file
- Command: bash scripts/warmup.sh --model Qwen/Qwen3-14B
```

## No Unrequested Testing or Linting

Plans must not include testing or linting unless the user directly and
explicitly asks for it.

Rules:

- Do not include commands to run tests unless the user asks to run tests.
- Do not include commands to create, update, or broaden tests unless the user
  asks for test work.
- Do not include lint, typecheck, formatting, security scan, build verification,
  or broad validation commands unless the user asks for that verification.
- Do not add a testing, linting, verification, or quality-gate step by default.
- Do not mention testing or linting as an optional final step when the user did
  not request it.
- If the user asks for testing or linting, include only the exact requested
  checks and keep them scoped to the planned change.

Bad:

```text
4. Run lint and tests.
```

Good when the user did not ask for verification:

```text
4. Review the planned diff for consistency with the requested behavior.
```

Good when the user explicitly asked for unit tests:

```text
4. Add the unit test shown in the diff above.
5. Run the requested unit test command.
```

## Implementation Order

Plans must define the exact order of implementation.

Rules:

- Break the work into sequential steps.
- Put dependency discovery before edits that depend on that discovery.
- Put shared contract or type changes before callers that use them.
- Put data shape changes before runtime, quantization, engine-build, or
  publishing surfaces that consume the data.
- Put ownership moves before import or call-site updates.
- Put generated output after the source change that produces it.
- Put cleanup after all call sites have moved.
- Keep each step concrete enough that another agent can execute it without
  inventing missing decisions.
- State which files, symbols, assets, commands, and diffs belong to each step.
- Do not hide multiple unrelated edits inside one broad step.

Bad:

```text
1. Refactor the loader.
2. Update the runtime command.
```

Good:

```text
1. Add the new `ModelSettings` field shown in the diff.
2. Update `build_settings` to populate the field using the exact diff.
3. Replace runtime and warmup call sites using the exact diff.
4. Remove the old derived-value helper using the exact deletion diff.
```

## Plan Detail Level

Plans must be extensive and detailed enough to be directly executable.

Rules:

- Include the reasoning needed to understand why the steps are ordered that way.
- Include file paths for every planned edit.
- Include exact names for new files, functions, types, commands, assets, and
  configuration keys.
- Include expected intermediate states when a sequence temporarily changes
  contracts, generated outputs, or asset files.
- Include all constraints, assumptions, and dependencies that affect execution.
- Include edge cases or failure modes only when they are part of the real
  requested work.
- Do not leave placeholders such as "update as needed", "adjust imports", or
  "fix any errors".
- Do not rely on the implementer to infer omitted code, omitted commands, or
  omitted asset operations.
