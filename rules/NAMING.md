# Naming

This is the single source of truth for naming in this repository. It covers
general naming principles plus Python, Bash, prompt modules, image assets,
dataset artifacts, model artifacts, Hugging Face, and test naming rules.

Use this file together with the automated checks. This document explains how to
choose names. The local tooling enforces the concrete policy.

## Contents

- [Authority and Local Enforcement](#authority-and-local-enforcement)
- [General Naming Rules](#general-naming-rules)
- [Vocabulary and Role Words](#vocabulary-and-role-words)
- [Functions and Methods](#functions-and-methods)
- [Booleans and Predicates](#booleans-and-predicates)
- [Files and Directories](#files-and-directories)
- [Python](#python)
- [Bash](#bash)
- [Data, Models, and External Boundaries](#data-models-and-external-boundaries)
- [Tests](#tests)
- [Review Checklist](#review-checklist)

## Authority and Local Enforcement

Naming decisions must satisfy both this guide and the local quality tooling.

Rules:

- Follow this file when choosing names for files, directories, modules,
  packages, classes, dataclasses, functions, methods, parameters, variables,
  constants, CLI flags, environment variables, test helpers, data examples, and
  documentation examples.
- Also follow `ruff` naming checks configured in `pyproject.toml`.
- Also follow the structural naming checks under `quality/python/rules/`,
  especially `quality/python/rules/prefix_collisions.py` and
  `quality/python/rules/single_file_folders.py`.
- Also follow the repository-wide naming index under
  `quality/repository/naming/` and the policy in
  `quality/config/naming/policy.json`.
- Treat local lint failures as authoritative. If this guide and tooling disagree,
  fix the disagreement instead of working around it locally.
- Do not bypass naming policy by hiding bad names in string keys, aliases,
  generated wrappers, or CLI flags.
- Generated model or provider-owned names may keep provider spelling, but
  hand-written wrappers around them must follow this guide.

## General Naming Rules

Names are a design tool. A name should let a reader understand the concept,
scope, role, and expected value without reading the implementation first.

Rules:

- Name by role, responsibility, and domain meaning.
- Do not name by storage type, library type, collection shape, or implementation
  accident.
- Use English unless representing an external identifier that must keep another
  spelling.
- Prefer the shortest name that is still clear at the use site.
- Add qualifiers only when the unqualified name is genuinely ambiguous.
- Avoid private shorthand that only the original author understands.
- Avoid contractions created by deleting letters from a word.
- Do not duplicate context already supplied by the enclosing module, directory,
  class, or package.
- Do not encode every implementation detail in a name.
- Use the same vocabulary for the same concept across the repository.
- Use singular names for single values and plural names for collections.
- Name collections by their contents, not by the collection type.
- Use role words when primitive or weak types do not carry enough meaning.
- Preserve required external names at boundaries, but translate them into domain
  names before they move inward.
- Do not use banned generic verbs or role words in local names. The banned list is
  enforced by `quality/config/naming/policy.json` and includes `load`, `loader`,
  `loaded`, `loading`, `resolve`, `resolving`, `resolution`, `manager`, `handler`,
  `helper`, `util`, `utils`, `processor`, `service`, `common`, `core`, and
  `data`. Use precise verbs such as `read`, `choose`, `derive`, `build`, `create`,
  `parse`, `validate`, `sanitize`, or `format` only when the verb describes the
  real operation.

Bad:

```python
string = "answerdotai/ModernBERT-base"
array = examples
data = build_examples()
tmp = choose_max_length(tokenizer, config, requested)
```

Good:

```python
base_model_name = "Qwen/Qwen3-8B"
quantization_jobs = jobs
request_messages = parse_messages(payload)
effective_max_length = choose_max_length(tokenizer, config, requested_max_length)
```

### Role Instead of Type

Names should explain what the value means in the domain.

Bad:

```python
data = tokenizer(text)
dict_value = compute_latency_stats(examples)
bool_value = torch.cuda.is_available()
```

Good:

```python
encoded_input = tokenizer(text)
latency_stats = compute_latency_stats(examples)
is_cuda_available = torch.cuda.is_available()
```

### Avoid Redundant Context

Let the owner provide context. Add context only when the name would otherwise be
ambiguous outside the owner.

Bad:

```python
@dataclass
class PromptMessage:
    prompt_message_text: str
    prompt_message_role: str
```

Good:

```python
@dataclass
class PromptMessage:
    text: str
    role: str
```

### Avoid Type and Shape Duplication

Do not repeat information already expressed by the type system or declaration.

Bad:

```python
name_string: str = variant.name
messages_list: list[Message] = parse_messages(payload)
config_dict: dict[str, object] = build_config()
```

Good:

```python
name: str = variant.name
messages: list[Message] = parse_messages(payload)
runtime_config: dict[str, object] = build_config()
```

### Avoid Vague and Inflated Words

Do not use vague words to avoid naming the real responsibility.

Bad:

```text
model_utils.py
data_helper.py
runtime_manager.py
common.py
base_processor.py
```

Good:

```text
engines.py
quantization.py
runtime.py
paths.py
latency_stats.py
```

Avoid `Helper`, `Helpers`, `Utility`, `Utilities`, `Util`, `Utils`, `Common`,
`Shared`, `Base`, `Core`, `Manager`, and `Processor` unless the name is a
platform term or the role is truly exact.

## Vocabulary and Role Words

Choose role words deterministically. A deterministic suffix tells a reader what
kind of boundary or owner they are looking at.

Use these meanings consistently:

| Role word        | Use when                                                                            |
| ---------------- | ----------------------------------------------------------------------------------- |
| `Config`         | A cohesive configuration value or module.                                           |
| `Prompt`         | Text or builder logic sent to a vision/model provider.                              |
| `Provider`       | A model, API, or service boundary such as OpenAI, Grok, OpenRouter, or Sightengine. |
| `Client`         | A concrete SDK, HTTP, Hugging Face, filesystem, or subprocess boundary.             |
| `Image`          | A screenshot/image file or decoded image value.                                     |
| `Category`       | A classification taxonomy label or path component.                                  |
| `Classification` | A predicted class/category result or classification workflow concept.               |
| `Analysis`       | A detailed screen analysis result or workflow concept.                              |
| `Safety`         | A safety rating/result or safety workflow concept.                                  |
| `Sanitize`       | Watermark/template detection and rewrite behavior.                                  |
| `Dataset`        | Dataset artifacts, split assignment, manifest rows, shards, or push workflow.       |
| `Optimizer`      | Lossless image optimization workflow modules.                                       |
| `Manifest`       | A structured inventory of files or dataset rows.                                    |
| `Summary`        | Aggregated command or workflow output.                                              |
| `Result`         | A completed operation's structured output.                                          |
| `Stats`          | Aggregated measurements or counters.                                                |
| `Parser`         | Converts raw CLI, model, or file input into structured values.                      |
| `Validator`      | Checks a value and returns or raises validation failure.                            |
| `Formatter`      | Converts values into user-facing text, logs, cards, or wire content.                |
| `Runner`         | Owns a top-level command workflow.                                                  |

Do not use a suffix just because a name feels too short. If the role is not
real, rename the symbol to the concrete domain concept.

Bad:

```python
class RuntimeManager:
    ...

def process(data):
    ...
```

Good:

```python
class ClassificationProvider:
    ...

def build_dataset_entries(source_items):
    ...
```

## Functions and Methods

Function and method names should describe the action and the domain item being
acted on without repeating context already supplied by the owner.

Rules:

- Start with the action unless a framework convention requires another shape.
- Include enough domain context to read clearly at the call site.
- Do not use generic names such as `process`, `handle`, `run`, `execute`,
  `manage`, `perform`, or `do_work` when the action can be named.
- Use `handle` only for callbacks, framework/event boundaries, or signal
  handlers.
- Use `read` for reading values from a source into memory.
- Use `get` for immediate access that does not imply work, mutation, or I/O.
- Use `set` only for assigning a value directly.
- Use `reset` only for returning to an initial state.
- Use `build` for constructing a value from existing values.
- Use `create` when making a new independent durable or domain value.
- Use `choose` when selecting the final value from inputs, defaults, and
  constraints.
- Use `parse` for raw input to structured data.
- Use `decode` for encoded bytes or serialized payloads into typed values.
- Use `encode` for typed values into bytes or serialized payloads.
- Use `validate` for checking and reporting invalidity.
- Use `sanitize` only when the function actually transforms input into a safe
  external representation.
- Avoid positional boolean parameters. Use options, enums, or explicit function
  names when a boolean would be ambiguous.

Bad:

```python
def process(value):
    ...

def handle_data(data):
    ...

def run(model_name, output_dir):
    ...
```

Good:

```python
def truncate_conversation_text(text, tokenizer, max_length):
    ...

def build_examples(source_items):
    ...

def run_quantization(model_name, output_dir):
    ...
```

### One Concept per Function Name

If the function name needs `and`, `or`, `with`, `plus`, or a vague umbrella verb,
the function may own too many concepts.

Bad:

```python
def validate_and_push_model(model_dir, repo_id):
    ...
```

Good:

```python
def validate_model_dir(model_dir):
    ...

def push_model(model_dir, repo_id):
    ...
```

### Boundary Names

At boundaries, name the conversion explicitly.

Bad:

```python
def transform(item):
    ...
```

Good:

```python
def build_example(raw_item):
    ...

def sanitize_base_model(candidate, default):
    ...
```

## Booleans and Predicates

Boolean names must read as assertions at the use site.

Rules:

- Use `is_` for state or characteristics.
- Use `has_` for possession or presence.
- Use `can_` for capability.
- Do not introduce `should_` in new local names unless an external framework owns
  that spelling.
- Avoid negative names such as `is_not_ready` when the positive form is clearer.
- Do not name booleans like nouns that could be non-boolean values.
- Prefer the boolean name that matches the branch without double negation.

Bad:

```python
remote = args.remote
token = check_hf_token()
not_ready = status != "ready"
```

Good:

```python
is_remote = args.remote
has_hf_token = check_hf_token()
is_ready = status == "ready"
```

Bad:

```bash
ready='false'
if [[ "${ready}" != 'true' ]]; then
  fail 'not ready'
fi
```

Good:

```bash
is_ready='false'
if [[ "${is_ready}" != 'true' ]]; then
  fail 'not ready'
fi
```

## Files and Directories

Files and directories define ownership. Name them for the behavior or entity
they own, not for reuse intent.

Rules:

- Python source filenames use `snake_case.py`, except Python-owned files such as
  `__init__.py` and `__main__.py`.
- Shell files use kebab case or snake case.
- Source modules under `src/` are named for cohesive training capabilities:
  configuration, command boundaries, state, dataset sampling, evaluation,
  model export, and artifact handling.
- Test modules under `tests/` are named for the evaluated scenario or runner
  responsibility.
- Quality modules under `quality/python/rules/`, `quality/repository/`, and
  `quality/security/` are named for the rule or workflow they enforce.
- Do not create catch-all files or directories for unrelated code.
- Do not move code into shared locations just because a future caller might
  appear.
- Promote shared code only when there is a repeated concept and a stable owner.
- A directory named by a broad layer is acceptable only when the project
  architecture explicitly owns that layer.

Bad:

```text
src/helpers.py
src/utils.py
src/models.py
src/misc.py
quality/lib/misc.sh
```

Good:

```text
src/domains/classification/provider.py
src/domains/dataset/manifest.py
src/cli/workflows/optimize/progress.py
src/prompts/analysis/text/blocks/message/chat.txt
```

## Python

Python naming follows PEP 8 where it fits this repository, with local rules from
`pyproject.toml` and `quality/config/naming/policy.json`.

### Python Case Rules

Rules:

- Modules and packages use `snake_case`.
- Functions, methods, variables, and parameters use `snake_case`.
- Classes, dataclasses, and exception types use `PascalCase`.
- Constants use `UPPER_SNAKE_CASE`.
- CLI flags use lowercase kebab case, such as `--model-name` and
  `--output-dir`.
- Environment variables use `UPPER_SNAKE_CASE`.
- Avoid one-letter names except tiny conventional scopes such as `i` in a short
  loop.
- Preserve provider capitalization in external names such as `HF_TOKEN` and
  Hugging Face repository IDs.

Bad:

```python
class tool_dataset:
    ...

MAXLEN = 512

def BuildExamples(data):
    ...
```

Good:

```python
class RuntimeConfig:
    ...

MAX_LENGTH = 512

def build_examples(source_items):
    ...
```

### Python Modules and Imports

Rules:

- Keep import aliases rare. Use aliases only for standard, widely understood
  conventions or real collision avoidance.
- Named imports keep their exported name unless a collision forces an alias.
- If aliasing is required, use a domain or module component that explains the
  collision.
- Do not create package-level re-export layers only to preserve old names.
- Keep `__all__` names accurate and ordered according to local lint rules.

Bad:

```python
from src.state.settings import ModelSettings as Thing
```

Good:

```python
from src.state.settings import ModelSettings
```

### Python Types and Dataclasses

Rules:

- Dataclass names describe the domain value they represent.
- Field names describe the value inside the owning type without repeating the
  type name.
- Use `Path` variables with names that reveal whether they point to a directory,
  file, model, result, or repository root.
- Use `*_path` for filesystem paths and `*_dir` only for directories.
- Use `*_id` only for real identifiers, not arbitrary names or labels.
- Use `*_name` for display or provider names.
- Use `*_key` for dictionary keys and supported variant keys.

Bad:

```python
@dataclass
class EvalExample:
    eval_example_text: str
    eval_example_result: str
```

Good:

```python
@dataclass
class EvalExample:
    text: str
    prediction: str
```

### Python Boundary Names

Rules:

- Keep raw provider or CLI names at the boundary.
- Translate external names into domain names before passing values inward when
  the external name is not the domain concept.
- Keep Hugging Face token lookup inside the Hugging Face boundary.
- Keep path construction in config/path owners rather than rebuilding paths in
  business logic.
- Name functions that cross boundaries for the operation they perform.

Bad:

```python
def data(value):
    ...

token = os.environ["HF_TOKEN"]
```

Good:

```python
def build_readme(repo_id, base_model, variant_key):
    ...

token = read_token(cli_token)
```

## Bash

Bash naming follows Google shell guidance where it fits this repository, with
local overrides for existing pipeline step files.

### Bash Case Rules

Rules:

- Shell source file stems use lowercase words separated by hyphens or
  underscores.
- Numbered pipeline step scripts use `NN_description.sh`.
- Executable scripts use `.sh` when invoked through documented commands.
- Sourced libraries use `.sh` and are not executable.
- Functions and mutable variables use `lower_snake_case`.
- Function-local variables use `lower_snake_case`.
- Constants, readonly values, exported environment variables, and externally
  configured values use `UPPER_SNAKE_CASE`.
- Do not use the `function` keyword for new functions. Use `name() { ...; }`.

Bad:

```text
TrainScript.sh
build-script.sh
helpers.sh
```

Good:

```text
main.sh
lint.sh
05_start_server.sh
common.sh
```

Bad:

```bash
function Train() {
  local TMP="$1"
}
```

Good:

```bash
run_quantization_step() {
  local output_dir="$1"
}
```

### Bash Variables

Rules:

- Loop variables describe the item being iterated.
- Use `tmp_dir` or `tmp_file` only for actual temporary filesystem paths.
- Avoid vague names when a domain name is available.
- Avoid shell-reserved and shell-special names for unrelated values.
- Initialize variables before use.
- Prefer explicit empty strings or arrays over relying on unset variables.
- Declare function-specific variables with `local`.
- Separate `local`, `declare`, `readonly`, and `export` from command
  substitutions when the command status matters.

Bad:

```bash
X=/tmp/a
for i in "${things[@]}"; do
  do_it "${i}"
done

local output="$(generate_results)"
```

Good:

```bash
readonly MODELS_DIR="${ROOT_DIR}/models"

for model_dir in "${model_dirs[@]}"; do
  validate_model "${model_dir}"
done

local result_output
result_output="$(generate_results)" || return 1
```

### Bash Functions

Rules:

- Function names use verb phrases when the function has side effects.
- Functions that print data to STDOUT should be named for the data printed.
- Functions that validate should return status and log errors deliberately.
- Do not name scripts or functions after shell builtins or common commands.
- Do not make function names so generic that logs and stack traces lose context.

Bad:

```bash
test() {
  ...
}

run() {
  ...
}

process() {
  ...
}
```

Good:

```bash
venv_python() {
  ...
}

run_quantization_step() {
  ...
}

validate_model_dir() {
  ...
}
```

### Bash Environment Names

Rules:

- Environment variables are `UPPER_SNAKE_CASE`.
- Export only variables child processes need.
- Do not overwrite important shell environment names casually.
- Validate configured environment variable names before using indirect
  expansion.
- Name required environment values by the external contract when the runtime or
  provider platform owns the name.

Bad:

```bash
export token="${TOKEN}"
name="$1"
printf '%s\n' "${!name}"
```

Good:

```bash
export HF_TOKEN="${HF_TOKEN}"

env_name="$1"
if [[ ! "${env_name}" =~ ^[A-Z_][A-Z0-9_]*$ ]]; then
  printf 'error: invalid environment variable name\n' >&2
  return 1
fi
printf '%s\n' "${!env_name}"
```

## Data, Models, and External Boundaries

Model names, provider names, dataset labels, prompt names, asset path names, and
Hugging Face names are repository contracts. Rename them deliberately.

Rules:

- Internal keys use lowercase words separated by underscores.
- Hugging Face repository IDs and base model names preserve provider spelling.
- Prompt names describe the screen or instruction scenario, not the file that
  created them.
- Classification labels, safety labels, dataset split names, and asset path
  components use the stable external taxonomy.
- Metrics and latency fields use stable operational names such as `p50`, `p90`,
  `p95`, `total_latency`, `prediction`, and `expected`.
- Do not put raw user text, tokens, local absolute paths, or private identifiers
  into artifact names intended for publishing.
- If a name is part of an external contract, treat renaming it as a contract
  change.

Bad:

```python
provider_key = "myNewThing"
prompt_name = "test_1"
result_field = "thing"
```

Good:

```python
engine_key = "vllm_awq"
conversation_name = "multi_turn_warmup"
result_field = "prediction"
```

## Tests

Test names and test data names should describe observable behavior, not private
implementation details.

Rules:

- Name tests for the behavior and expected outcome.
- Name test and warmup scenarios for the user behavior or domain being
  evaluated.
- Avoid names tied to private helper names.
- Avoid test data names that hide the scenario.
- Test helpers should be named for the behavior they create.
- Test result files may include timestamps when they are generated artifacts.

Bad:

```text
test1
helper_case
thing_should_work
```

Good:

```text
stream_response_preserves_order
idle_runner_stops_after_timeout
vllm_awq_benchmark_result
```

## Review Checklist

Before accepting a new name, ask:

- Does the name describe the role or domain concept instead of the type shape?
- Is the name clear at the call site?
- Is context supplied by the owner omitted from the local name?
- Does the name avoid vague role words unless the role is real?
- Does the name use the correct Python or Bash case rule?
- Does the file or directory name describe ownership?
- Does the function name name the action and domain item?
- Does each boolean read as a positive assertion?
- Are external names isolated to boundary modules?
- Are model, data, and result names treated as contracts?
- Does the name satisfy `pyproject.toml` and the local quality rules under `quality/`?
