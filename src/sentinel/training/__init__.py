"""Training stage: trainer, evaluation gate, artifacts, experiment tracking."""

from sentinel.training.artifacts import TrainingMetadata, load_metadata, load_pipeline, run_dir
from sentinel.training.evaluate import (
    GateResult,
    MetricsResult,
    evaluate_pipeline,
    evaluate_predictions,
    run_quality_gate,
)
from sentinel.training.tracking import ExperimentTracker, LocalTracker, MLflowTracker, get_tracker
from sentinel.training.train import TrainingResult, train_model

__all__ = [
    "TrainingMetadata",
    "TrainingResult",
    "load_metadata",
    "load_pipeline",
    "run_dir",
    "GateResult",
    "MetricsResult",
    "evaluate_pipeline",
    "evaluate_predictions",
    "run_quality_gate",
    "ExperimentTracker",
    "LocalTracker",
    "MLflowTracker",
    "get_tracker",
    "train_model",
]
