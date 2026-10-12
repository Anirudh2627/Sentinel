"""Data pipeline: ingestion, validation, preprocessing, splitting."""

from sentinel.data.ingestion import (
    DatasetCard,
    apply_drift,
    build_dataset,
    generate_synthetic_churn,
    load_dataset_card,
    load_dataset_frame,
    persist_dataset,
)
from sentinel.data.preprocessing import build_pipeline, model_feature_names
from sentinel.data.splitting import SplitReport, persist_splits, split_dataset
from sentinel.data.validation import DataValidationError, DataValidator, ValidationReport

__all__ = [
    "DatasetCard",
    "apply_drift",
    "build_dataset",
    "generate_synthetic_churn",
    "load_dataset_card",
    "load_dataset_frame",
    "persist_dataset",
    "build_pipeline",
    "model_feature_names",
    "SplitReport",
    "persist_splits",
    "split_dataset",
    "DataValidationError",
    "DataValidator",
    "ValidationReport",
]
