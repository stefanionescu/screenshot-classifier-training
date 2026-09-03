---
model_name: '{{model_name}}'
library_name: '{{library_name}}'
base_model: '{{model_id}}'
tags:
    - image-classification
    - mobile-screenshots
    - phone-screenshots
    - screenshot-classifier
    - content-safety
    - timm
---

<!-- Generated during model export. Local edits are overwritten. -->

# {{model_name}}

{{model_name}} predicts a screen category and a content-safety category for a
mobile screenshot.

## Contents

- [Intended use](#intended-use)
- [Files](#files)
- [Model outputs](#model-outputs)
- [Evaluation](#evaluation)
- [Preprocessing](#preprocessing)
- [Inference](#inference)
- [Training data](#training-data)
- [Citation](#citation)

## Intended use

Use the model to route or filter mobile screenshot processing before more
expensive analysis. The exported label map defines the screen and safety
categories available in this release.

## Files

| Artifact | Purpose |
| -------- | ------- |

<!-- markdownlint-disable MD055 MD056 -->

{{artifact_table}}

<!-- markdownlint-enable MD055 MD056 -->

The model settings, preprocessing settings, training settings, and label file
contain the exact values used by this export.

## Model outputs

The ordered output names and their label arrays are recorded in the exported
model configuration. The inference helper returns the highest-scoring labels
for {{output_names}}.

## Evaluation

### Test results

{{test_results}}

### CPU timing

{{test_timing}}

Timing was measured with ONNX Runtime CPU execution on
`{{benchmark_hardware}}`. Total latency includes image loading, preprocessing,
model inference, and label decoding.

## Preprocessing

Use the exported preprocessing settings rather than copying resize or
normalization values from this card. The inference helper applies EXIF
orientation, converts images to RGB, preserves aspect ratio during resizing,
normalizes pixels with the exported statistics, and pads images for batching.

The exported model accepts dynamic batch, height, and width dimensions.

## Inference

Install the runtime dependencies in a `uv` project:

```shell
uv add huggingface-hub numpy onnxruntime pillow
```

Replace the example image path, then run this complete Python example:

```python
from pathlib import Path

from huggingface_hub import snapshot_download

model_dir = Path("exported_model")
snapshot_download("{{repo_id}}", local_dir=model_dir)

from exported_model.inference.python import Classifier

image_path = Path("path/to/screenshot.png")
classifier = Classifier(model_dir)
print(classifier.classify_image(image_path))
```

The downloaded helper can also run directly:

```shell
python exported_model/inference/python.py path/to/screenshot.png
```

Keep the downloaded directory intact because the helper loads its model,
preprocessing settings, and labels from that directory.

## Training data

The saved training settings record the training input and sampling values.
Screen labels below the selected minimum sample count are combined with the
fallback label chosen for that run. Safety labels that do not qualify are left
out of safety training and evaluation.

## Citation

```bibtex
@misc{phone-screen-classifier,
  title={{{model_name}}},
  year={{{citation_year}}},
  publisher={Yap With AI},
  url={https://huggingface.co/{{repo_id}}}
}
```
