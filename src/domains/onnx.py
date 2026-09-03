"""Collect ONNX evaluation metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING
from src.errors import TrainingError
from src.state.export import OnnxModelMetrics
from src.state.eval import EvalArtifacts, OnnxEval
from src.domains.evaluation.metrics import evaluate_onnx
from src.state.metrics import EvaluationMetrics, ModelEvaluationMetrics
from src.config.artifacts import ONNX_HALF_MODEL_FILENAME, ONNX_MODEL_FILENAME, TRAIN_ONNX_DIR

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from src.state.export import OnnxMetrics
    from src.state.training import BatchData, TrainArgs

BASE_EVAL_ARTIFACT_STAGES = (
    "val_full",
    "val_screen_balanced",
    "val_safety_balanced",
    "test_full",
    "test_screen_balanced",
    "test_safety_balanced",
)
MODEL_FP16 = "fp16"
MODEL_FP32 = "fp32"
TEST_FULL = "test_full"
TEST_SAFETY_BALANCED = "test_safety_balanced"
TEST_SCREEN_BALANCED = "test_screen_balanced"
TEST_FULL_STAGE = "test full"
TEST_SAFETY_BALANCED_STAGE = "test safety balanced"
TEST_SCREEN_BALANCED_STAGE = "test screen balanced"


def collect_onnx_metrics(metrics: OnnxMetrics) -> tuple[dict[str, ModelEvaluationMetrics], ModelEvaluationMetrics]:
    """Collect ONNX metrics."""
    models: dict[str, tuple[str, str]] = {
        MODEL_FP32: (ONNX_MODEL_FILENAME, MODEL_FP32),
    }
    if metrics.args.export == "fp16":
        models[MODEL_FP16] = (ONNX_HALF_MODEL_FILENAME, MODEL_FP16)
    onnx_metrics: dict[str, ModelEvaluationMetrics] = {}
    for model_key, (filename, model_format) in models.items():
        model_path = metrics.export_dir / TRAIN_ONNX_DIR / filename
        onnx_metrics[model_key] = collect_single_onnx_metrics(
            OnnxModelMetrics(
                model_path=model_path,
                model_format=model_format,
                model_key=model_key,
                metrics=metrics,
            ),
        )
    return onnx_metrics, onnx_metrics[MODEL_FP32]


def collect_single_onnx_metrics(model: OnnxModelMetrics) -> ModelEvaluationMetrics:
    """Collect metrics for one ONNX model."""
    metrics = model.metrics
    if not model.model_path.is_file():
        msg = "ONNX model does not exist. Export the model before evaluation."
        raise TrainingError(msg)
    tests = (
        (TEST_FULL, TEST_FULL_STAGE, metrics.iterators.test),
        (TEST_SCREEN_BALANCED, TEST_SCREEN_BALANCED_STAGE, metrics.iterators.test_screen),
        (TEST_SAFETY_BALANCED, TEST_SAFETY_BALANCED_STAGE, metrics.iterators.test_safety),
    )
    results = {key: eval_onnx_test(model, key, stage, iterator) for key, stage, iterator in tests}
    return ModelEvaluationMetrics(
        test_full=results[TEST_FULL],
        test_screen_balanced=results[TEST_SCREEN_BALANCED],
        test_safety_balanced=results[TEST_SAFETY_BALANCED],
    )


def eval_onnx_test(
    model: OnnxModelMetrics,
    report_key: str,
    stage: str,
    iterator: DataLoader[BatchData | None],
) -> EvaluationMetrics:
    """Evaluate one ONNX test iterator."""
    metrics = model.metrics
    return evaluate_onnx(
        OnnxEval(
            model_path=model.model_path,
            iterator=iterator,
            screen_labels=metrics.labels.screen,
            safety_labels=metrics.labels.safety,
            image_size=metrics.args.image_size,
            batch_size=metrics.args.micro_batch_size,
            model_format=model.model_format,
            dashboard=metrics.dashboard,
            stage=f"{model.model_format} {stage}",
            artifacts=EvalArtifacts(
                metrics.run_dir,
                onnx_artifact_stage(model.model_key, report_key),
                "test",
            ),
        ),
    )


def onnx_artifact_stage(model_key: str, test_key: str) -> str:
    """Return the artifact stage key."""
    if model_key == MODEL_FP32:
        return test_key
    return f"{model_key}_{test_key}"


def eval_artifact_stages(args: TrainArgs) -> tuple[str, ...]:
    """Return evaluation artifact stages."""
    if args.export == "fp32":
        return BASE_EVAL_ARTIFACT_STAGES
    fp16_stages = tuple(
        onnx_artifact_stage(model_key, test_key)
        for model_key in (MODEL_FP16,)
        for test_key in (
            TEST_FULL,
            TEST_SCREEN_BALANCED,
            TEST_SAFETY_BALANCED,
        )
    )
    return (*BASE_EVAL_ARTIFACT_STAGES, *fp16_stages)
