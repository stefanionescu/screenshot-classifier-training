# Screenshot Classifier

Screenshot Classifier is a Python command-line application that trains, evaluates,
and exports a multi-task classifier for mobile screenshots. The classifier
predicts both a screen category and a content-safety category.

The repository contains the training application and its supporting quality
tooling. It does not include the labeled screenshot corpus or a dataset builder.
Training requires a previously built dataset artifact under `dataset/`.

## Contents

- [How it works](#how-it-works)
- [Set up the project](#set-up-the-project)
- [Prepare a dataset](#prepare-a-dataset)
- [Train the classifier](#train-the-classifier)
- [Publish an export](#publish-an-export)
- [Inspect results](#inspect-results)
- [Develop the project](#develop-the-project)

## How it works

The training command reads a checked WebDataset artifact with train, validation,
and test splits. It selects eligible screen and safety labels, balances samples
for both prediction tasks, and fits a supported image backbone.

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

Place a complete dataset artifact at
`dataset/phone-screenshots` to use the default command. The artifact
must contain the data shards and Parquet manifests for the train, validation,
and test splits.

To use another dataset directory, keep it below `dataset` and pass its
repository-relative path:

```shell
mise run train -- --dataset dataset/<DATASET_NAME>
```

Replace `<DATASET_NAME>` with the dataset directory name.

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
mise run train -- --push --repo yapwithai/<MODEL_REPOSITORY>
```

Publish an existing validated export without training:

```shell
mise run train -- --push-only --repo yapwithai/<MODEL_REPOSITORY>
```

Replace `<MODEL_REPOSITORY>` with the model repository name. Publishing
requires `HF_TOKEN`. Repositories are private unless `--public` is present.

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
