# Screenshot Classifier

Screenshot Classifier is a Python command-line application that trains, evaluates,
and exports a multi-task classifier for mobile screenshots. The classifier
predicts both a screen category and a content-safety category.

The repository contains the dataset builder, training application, and supporting
quality tooling. It does not include the labeled screenshot corpus. Raw labeled
screenshots belong directly under `dataset/`.

## Contents

- [How it works](#how-it-works)
- [Set up the project](#set-up-the-project)
- [Prepare a dataset](#prepare-a-dataset)
- [Train the classifier](#train-the-classifier)
- [Publish an export](#publish-an-export)
- [Inspect results](#inspect-results)
- [Develop the project](#develop-the-project)

## How it works

The dataset command validates and deduplicates the images, assigns reproducible
train, validation, and test splits within each screen-and-safety label group,
and writes a checked WebDataset artifact below `output/dataset`. The training
command reads that artifact, balances both prediction tasks, and fits a supported
image backbone.

After training, the command evaluates the selected checkpoint and exports a
validated ONNX model. The export also includes model weights, preprocessing
settings, labels, the training recipe, evaluation results, and a standalone
Python inference module.

## Set up the project

Run these commands from the repository root:

```shell
mise trust
mise run setup
```

The setup task installs the pinned tools, synchronizes the locked Python and Bun
dependencies, and configures the repository Git hooks.

The default public model does not require a Hugging Face token. If a model
download or upload requires authentication, copy the environment template:

```shell
cp .env.example .env
```

Set `HF_TOKEN` in `.env`. The local file is ignored by Git.

## Prepare a dataset

Place labeled screenshots directly under `dataset/`, grouped by screen and
safety label. For example:

```text
dataset/
├── app-store/
│   └── safe/
│       └── screenshot.png
└── message/
    └── chat/
        ├── hot/
        │   └── screenshot.png
        └── safe/
            └── screenshot.png
```

The tracked `dataset/.gitkeep` preserves the directory while its images remain
excluded from version control. The dataset command creates the train,
validation, and test splits:

```shell
mise run dataset
```

## Train the classifier

Train, evaluate, and export the classifier:

```shell
mise run train
```

Training records checkpoints and metrics below `output/models`. Run the live
command help before changing model, label, sampling, optimization, resume, or
export settings:

```shell
mise run train -- --help
```

Resume the latest run for the selected output and model:

```shell
mise run train -- --resume
```

Resume mode validates the saved training recipe. Only the command's supported
resume overrides can differ from the original run.

## Publish an export

Publishing changes a remote Hugging Face model repository. Confirm the target
repository and visibility before running either command.

Train and publish the completed export:

```shell
mise run train -- --push --repo screenshot-classifier
```

Publish an existing validated export without training:

```shell
mise run train -- --push-only --repo screenshot-classifier
```

Publishing requires `HF_TOKEN`. Repositories are private unless `--public` is
present.

## Inspect results

Each model and backbone combination has its own directory below
`output/models`. Private training state contains the resolved recipe,
checkpoints, metrics, predictions, diagnostic failure sets, skipped-image
records, and reports.

The `export` directory is the deployable artifact. Treat its preprocessing
settings and label files as the inference contract instead of copying values
from application source.

## Develop the project

Use the repository tasks for dependency and quality work:

```shell
mise run deps:verify
mise run format:check
mise run lint:quality
mise run lint:python
mise run lint:shell
mise run type
```

`mise run check` runs the complete repository check, including dependency,
policy, hook, task, Python, shell, security, and license checks.
