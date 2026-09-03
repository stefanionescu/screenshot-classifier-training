# How to Work in This Repository

These rules apply to all files in this repository.

This repository is a Python CLI project for training and exporting the screenshot
classifier. Application code lives under `src/`. The `dataset/` tree is reserved
for local training data and is excluded from version control. All user-facing
commands are mise tasks that call
`uv run python -m src.cli.<command>`.

TS and JS are not application source languages in this repository. Bun and Node
may only be used for Markdown, Prettier, and duplicate-code quality tools.

## Contents

- [Thinking Before Coding](#thinking-before-coding)
- [Scope Discipline](#scope-discipline)
- [No Defensive Logic](#no-defensive-logic)
- [Managing Sprawl](#managing-sprawl)
- [Naming](#naming)
- [Abstractions](#abstractions)
    - [Prefer Duplication Over the Wrong Abstraction](#prefer-duplication-over-the-wrong-abstraction)
    - [Do Not Abstract for One Caller](#do-not-abstract-for-one-caller)
    - [Watch for Boxing](#watch-for-boxing)
    - [Inline the Wrong Abstraction](#inline-the-wrong-abstraction)
    - [Make the Change Easy](#make-the-change-easy)
    - [Keep Granularity Continuous](#keep-granularity-continuous)
    - [Use Inversion of Control Deliberately](#use-inversion-of-control-deliberately)
    - [Prefer Data Flow and Data Structures](#prefer-data-flow-and-data-structures)
- [Secrets and Sensitive Data](#secrets-and-sensitive-data)
- [Error Messages](#error-messages)
- [Testing Philosophy](#testing-philosophy)
    - [Test Behavior, Not Values](#test-behavior-not-values)
- [Verification Commands](#verification-commands)
- [Code Style](#code-style)
    - [Present State Only](#present-state-only)
    - [Avoid Referencing Specific File Paths](#avoid-referencing-specific-file-paths)
    - [Punctuation](#punctuation)
    - [Table of Contents](#table-of-contents)
    - [Comments](#comments)
    - [Documentation Requirements](#documentation-requirements)
    - [Documentation Maintenance](#documentation-maintenance)
- [Protected Files](#protected-files)
- [Language Discipline](#language-discipline)
- [No Backward Compatibility](#no-backward-compatibility)
- [Working With Uncommitted Changes](#working-with-uncommitted-changes)

## Thinking Before Coding

Read the relevant code before touching it. Understand the contracts, data flow,
and ownership boundaries. Then think through your approach:

- What is the simplest change that solves the problem correctly?
- What are the failure modes? What happens with bad input, missing data, concurrent access, network failures?
- Does this change affect other parts of the system? Trace the call chain.
- Will someone reading this code in six months understand what it does and why?

When you rename or move a file, audit the entire codebase for references and update them immediately. Stale imports and broken references are worse than the original problem.

Before adding a new module, directory, or helper, identify the correct owner for
the behavior. Do not create parallel implementations for a concept that has an
owner. If a new file is needed, be ready to explain why the owning module should
grow that way.

## Scope Discipline

- Do what was asked. Do not expand scope.
- If you discover something unrelated that needs fixing, mention it to the user. Do not silently fix it unless it is trivial and in a file you are editing.
- Do not add features, refactor surrounding code, or "improve" things that were not requested.

## No Defensive Logic

Do not invent defensive logic for scenarios that are not part of the real contract.

- Do not add guards, fallbacks, retries, optional handling, defaults, or wrappers for states that cannot occur under the real contract.
- Handle known failure modes at real boundaries: user input, network calls, persistence, permissions, and external services.
- Trust internal invariants after they are established. If an invariant is unclear, trace the code and clarify the contract instead of adding speculative protection.
- Do not pad the codebase with logic meant to protect against hypothetical future failures.

## Managing Sprawl

Keep one clear implementation for each concept.

- Do not create multiple functions, services, types, or wrappers that do nearly the same thing.
- Do not wrap a helper with another helper unless the wrapper owns a real boundary, policy, or transformation.
- Do not add an abstraction for one call site or one concept.
- Before adding a new helper, find the owner of the behavior and put the logic
  there.
- When touching duplicated logic in the same area, collapse it into the owner
  instead of adding another layer.

## Naming

All naming rules live in [`NAMING.md`](NAMING.md). Follow that file for
identifiers, files, directories, functions, booleans, role suffixes, language
case conventions, boundary names, and test names.

The local linting tools under `quality/` and `mise run lint:quality` are also
authoritative for enforced naming policy.

## Abstractions

Abstractions are useful only when they remove real complexity. They are harmful
when they hide ownership, combine unrelated behavior, or predict reuse before
the code proves it.

### Prefer Duplication Over the Wrong Abstraction

- Duplication is cheaper than the wrong abstraction.
- Prefer duplication until there are at least two real examples that prove the
  same concept exists.
- Do not build reusable code before the code is usable.
- Do not preserve an abstraction because of sunk cost.

### Do Not Abstract for One Caller

- Do not introduce an abstraction for one caller.
- Do not introduce an abstraction for hypothetical future reuse.

### Watch for Boxing

- If a shared abstraction starts gaining flags, modes, optional branches, or
  caller-specific conditionals, treat that as evidence the abstraction is wrong.
- "Boxing" is forbidden: do not stuff loosely related behavior into one
  function/class/module with parameters deciding which behavior runs.

### Inline the Wrong Abstraction

- When an abstraction is wrong, inline it back into each caller, delete the
  branches each caller does not need, then extract only the common behavior that
  remains.

### Make the Change Easy

- Preparatory refactoring is allowed when it makes the requested change easier:
  first preserve behavior, then make the behavior change.
- Keep refactoring and behavior changes separate when practical.

### Keep Granularity Continuous

- Higher-level helpers must be replaceable by a small number of lower-level
  operations. Do not create API granularity gaps.

### Use Inversion of Control Deliberately

- Use inversion of command when it prevents option explosion across multiple
  real use cases.
- Do not add inversion of command for a single use case if it makes the call
  site harder without reducing complexity.

### Prefer Data Flow and Data Structures

- Prefer plain functions and explicit data flow before classes, interfaces,
  factories, strategies, inheritance, or framework patterns.
- Prefer data structures and their relationships over code-pattern taxonomies.
- Push state and I/O outward where practical; keep core logic pure or close to
  pure when that reduces moving parts.

## Secrets and Sensitive Data

- Never hardcode API keys, tokens, passwords, or secrets anywhere in the codebase.
- Never log sensitive data (tokens, passwords, PII, full request bodies with auth headers).
- Use environment variables for all secrets. Reference them through config modules, not directly in business logic.

## Error Messages

Error messages visible to users, command-line callers, generated model cards, or
logs must not leak internal system details.

**Never include in error messages:**

- Schema names, table names, column names, or function names
- Internal identifiers (row IDs, user IDs, session tokens)
- Stack traces or file paths
- Implementation details (trigger names, policy names, internal state like "deleted" flags)

**Always:**

- Start error messages with an uppercase letter (sentence case)
- Make messages actionable: tell the user what went wrong, not how the system works
- Use generic messages for configuration and infrastructure failures
- Keep messages consistent in tone and casing across the repository

## Testing Philosophy

Tests exist to catch bugs. A test that cannot fail when someone introduces a bug is wasted code.

### What Makes a Good Test

- It tests behavior, not implementation. Assert on what the code does, not how it does it.
- It breaks when a real bug is introduced. If you can delete a line of production code and every test still passes, the tests are insufficient.
- It uses mocks only for external boundaries such as Hugging Face, file systems, subprocesses, and network calls. Mock the boundary, test the logic.
- It tests edge cases that matter: empty inputs, null values, boundary conditions, error paths.
- Its name follows [`NAMING.md`](NAMING.md) and identifies the scenario and expected outcome.

### What Makes a Bad Test

- Testing that a mock returns what you told it to return.
- Testing implementation details (internal method calls, private state, call order) that change during refactors.
- Tests that always pass regardless of the production code's correctness.
- Tests with no assertions or with assertions that verify nothing useful.
- Snapshot tests that nobody reviews when they change.

### Test Behavior, Not Values

Never write a test that asserts a parameter, config value, or return value equals a specific hardcoded literal. These tests break the moment the value changes and catch zero bugs. They test configuration, not whether the system works correctly.

**What to test:**

- **Constraints and invariants.** If a value must fall within a range, test the boundaries. If invalid input must be rejected, test the rejection.
- **Access control.** Permissions, ownership rules, and role-specific behavior.
- **Interactions.** Multi-entity workflows, admin actions, and cross-role behavior.
- **Side effects.** Functions, triggers, scheduled work, computed values, and external boundary calls.
- **Command and boundary behavior.** Contracts, error handling, caching, authentication flows, and edge cases.
- **State transitions.** What happens when valid input is given, what happens when invalid input is given, what happens at the boundaries.

**What not to test:**

- That a specific parameter is set to a specific value (e.g., `expect(config.temperature).toBe(0.7)`).
- That a function returns an exact hardcoded item when the item is just configuration.
- That an artifact field has a specific default value by reading it back and comparing.

**The distinction:** if the value can change freely without breaking anything, do not pin it in a test. If the value has constraints (must be between 0 and 2, must not be null, must be one of an enum set), test those constraints.

### When Tests Are Requested

- Create or update tests only when the user asks for tests.
- Keep requested tests focused on the behavior under change.
- If implementation work reveals that tests need updates, report that follow-up instead of editing tests unasked.

### When Tests Are Not Requested

- Do not create, update, or broaden tests.
- Do not refactor production code to make unrequested tests easier to write.
- Do not change snapshots or fixtures unless the requested task is test work.

## Verification Commands

Do not run lint, tests, formatting, security scans, or other verification commands unless the user asks for them.

When verification is requested:

- Source lint: `mise run lint:python`.
- Quality lint: `mise run lint:quality`.
- Shell lint: `mise run lint:shell`.
- Hook lint: `mise run lint:hooks`.
- Mise task lint: `mise run lint:mise`.
- Typecheck: `mise run type`.
- Formatting: `mise run format:check`.
- Security: `mise run security`.
- Dependencies: `mise run deps:verify`.
- Licenses: `mise run licenses`.
- Run only the command needed for the specific project or files in scope.
- Report failures with the relevant command and the failing area.

## Code Style

### Present State Only

Comments and documentation describe what the code does right now. Never mention what was removed, deleted, renamed, refactored, or how the code used to work. No changelogs in comments.

Bad: `# Removed the old checkpoint loader.`
Good: `# Loads model checkpoints from the configured artifact directory.`

### Avoid Referencing Specific File Paths

Comments and documentation must not reference specific file names or paths. File names change; hardcoding them creates stale references.

Exceptions: well-known configuration files such as `pyproject.toml`,
`uv.lock`, and `dev dependency group` may be mentioned by name when
genuinely useful.

### Punctuation

Do not use em dashes or double hyphens. Use a space, comma, or colon instead.

Bad: `The server handles requests - including retries - before responding.`
Good: `The server handles requests, including retries, before responding.`

### Table of Contents

Long documentation files use a `## Contents` section with accurate anchor links to the sections readers need most. Keep it up to date whenever headings change.

### Comments

Keep comments concise and focused on intent ("why"), not narration ("what"). Do not embed default values in comments; they drift when code changes. Reference concept names, not file paths.

Comments and doc comments must never contain:

- **Code change history.** No "changed X to Y", "replaced old Z", "updated to use W", "refactored from". Git tracks history.
- **What was done to variables or code.** No "added this field", "moved this constant", "renamed from oldName". Describe the present purpose.
- **File or variable locations.** Do not say "defined in X.ts" or "see the value in config.Y" unless the reference is essential for understanding. Code is searchable; stale path references are not.

Good doc comments describe what a function does, what its parameters mean, and what it returns. They do not narrate how the function came to exist or what it replaced.

### Documentation Requirements

Doc comments are required where they clarify public behavior, non-obvious
contracts, or example and model boundaries. Linters enforce the mandatory level where a
local rule exists; human review covers the rest.

#### Linter-Enforced (Mandatory)

Follow the local Python and structural lint rules configured in `pyproject.toml`
and `quality/`. Do not add comments only to satisfy a generic style preference
when the code is already obvious.

#### Always Comment

Regardless of language or visibility, add a comment when a function:

- Handles edge cases or non-obvious failure modes.
- Has concurrency, cancellation, or isolation requirements.
- Makes security or privacy decisions.
- Encodes domain invariants ("must be monotonic", "idempotent", "retry-safe").
- Sits on a performance-sensitive hot path.
- Would take a reader more than ten seconds to understand from the signature and body alone.

These are enforced during code review, not by linters. The comment explains why, not what.

### Documentation Maintenance

When editing any file, check that comments and doc comments are still accurate. Stale comments are worse than no comments because they actively mislead.

Specific expectations:

- If you change a function's behavior, update its doc comment to match.
- If you change a function's parameters, update `@param` tags.
- If you change what a function returns, update `@returns`.
- If a comment references behavior that no longer exists, rewrite or remove it.
- If a comment describes the "why" of a decision you are undoing, remove it.

This applies to Python docstrings, Bash comments, Markdown documentation, and
generated model-card text.

## Protected Files

Do not modify `AGENTS.md`, `CLAUDE.md`, or files under `rules/` unless the user
explicitly asks for rule changes.

`dataset/` contains local training input. Add or rewrite its data only as part
of an approved migration. `models/` and `output/` are generated runtime
artifacts.

## Language Discipline

Python is the only application implementation language. Shell is allowed for
mise tasks, hooks, and small orchestration scripts. Markdown, JSON, TOML, and
YAML are configuration or documentation formats only.

Do not add TS, JS, JSX, TSX, MJS, CJS, or generated JS wrappers. Do not
preserve old Bun command compatibility.

Use direct, concrete language in code, comments, filenames, and documentation.

- Follow [`NAMING.md`](NAMING.md) for test data, adapter, and wrapper names.
- Describe optional conditions explicitly.
- State the actual probability or condition instead of using vague hedging.
- Describe redundancy directly instead of using idioms.

If code uses vague language, improve it when touching that code.

## No Backward Compatibility

Never introduce compatibility layers, wrapper functions, re-exports for renamed symbols, deprecated-but-kept code, or any other form of backward-compatible scaffolding.

When something is replaced or renamed:

- Delete the old implementation entirely.
- Update every call site to use the new version.
- Remove unused files, functions, types, and variables.

The codebase must always reflect only the latest implementation.

## Working With Uncommitted Changes

When `git status` or the worktree shows changes you did not make, do not panic. Other agents or contributors may be working in parallel.

- Do not revert, stash, clean, or overwrite changes you did not make.
- Continue working if your changes do not conflict with uncommitted changes.
- Only stop and ask the user if a change you are about to make directly contradicts or overwrites an uncommitted worktree change.
- Build on top of uncommitted changes, or commit your own changes alongside them.
- If another agent is known to be committing those changes separately, leave them alone.
