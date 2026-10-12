"""Reproducible training pipeline.

``train_model`` executes: ingest -> validate -> split -> fit -> evaluate ->
persist artifacts -> track experiment. It returns a :class:`TrainingResult`
pointing at a run directory that the registry can promote to a model version.

Nothing is registered or deployed here — stage boundaries are explicit so the
quality gate and deployment controller stay in charge of promotion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from sentinel.config import SentinelSettings, get_logger, log_event
from sentinel.data.ingestion import build_dataset, persist_dataset
from sentinel.data.preprocessing import build_pipeline, model_feature_names
from sentinel.data.splitting import SplitReport, persist_splits, split_dataset
from sentinel.data.validation import DataValidator, ValidationReport
from sentinel.training.artifacts import (
    TrainingMetadata,
    build_reference_stats,
    git_info,
    new_run_id,
    save_run_artifacts,
)
from sentinel.training.evaluate import MetricsResult, evaluate_pipeline
from sentinel.training.tracking import ExperimentTracker, get_tracker

logger = get_logger(__name__)


class TrainingResult(BaseModel):
    run_id: str
    run_dir: str
    scenario: str
    dataset_version: str
    algorithm: str
    val_metrics: MetricsResult
    test_metrics: MetricsResult
    validation_passed: bool

    @property
    def path(self) -> Path:
        return Path(self.run_dir)


def _corrupt_for_scenario(
    settings: SentinelSettings, train_df: pd.DataFrame, scenario: str
) -> pd.DataFrame:
    """Deliberately degrade the *training* signal (failure-injection tooling).

    ``degraded`` flips a configured fraction of training labels and subsamples
    rows, producing a genuinely bad model — offline metrics are really poor;
    nothing is faked.
    """
    if scenario != "degraded":
        return train_df
    cfg = settings.model.degraded
    rng = np.random.default_rng(settings.model.random_state + 1)
    out = train_df.sample(frac=cfg.subsample_frac, random_state=settings.model.random_state).copy()
    flip = rng.random(len(out)) < cfg.label_flip_prob
    target = settings.features.target
    out.loc[flip, target] = 1 - out.loc[flip, target]
    log_event(
        logger,
        logging.WARNING,
        "degraded scenario: training signal corrupted",
        label_flip_prob=cfg.label_flip_prob,
        subsample_frac=cfg.subsample_frac,
        rows=len(out),
    )
    return out


def train_model(
    settings: SentinelSettings,
    scenario: str = "default",
    n_samples: int | None = None,
    tracker: ExperimentTracker | None = None,
) -> TrainingResult:
    """Run the full training stage and persist a run directory."""
    if scenario not in {"default", "degraded"}:
        raise ValueError(f"unknown training scenario: {scenario}")

    tracker = tracker or get_tracker(settings)
    run_id = new_run_id()

    # 1. Ingest --------------------------------------------------------------
    raw_df = build_dataset(settings, n_samples=n_samples)
    validator = DataValidator(settings)
    report: ValidationReport = validator.validate(raw_df)
    csv_path, card = persist_dataset(
        settings,
        raw_df,
        source=settings.data.source,
        seed=settings.data.seed,
        generator_params={"n_samples": len(raw_df), "scenario": scenario},
    )

    # 2. Split ----------------------------------------------------------------
    train_df, val_df, test_df, split_report = split_dataset(settings, raw_df, card.dataset_version)

    # 3. Train -----------------------------------------------------------------
    numeric, categorical = model_feature_names(settings)
    feature_cols = [*settings.features.numeric, *settings.features.categorical]

    X_train = train_df[feature_cols]
    y_train = train_df[settings.features.target].to_numpy()
    train_df_eff = _corrupt_for_scenario(settings, train_df, scenario)
    X_train = train_df_eff[feature_cols]
    y_train = train_df_eff[settings.features.target].to_numpy()

    pipeline = build_pipeline(settings)
    pipeline.fit(X_train, y_train)

    # 4. Evaluate ---------------------------------------------------------------
    val_metrics = evaluate_pipeline(
        pipeline, val_df[feature_cols], val_df[settings.features.target].to_numpy()
    )
    test_metrics = evaluate_pipeline(
        pipeline, test_df[feature_cols], test_df[settings.features.target].to_numpy()
    )
    persist_splits(settings, card.dataset_version, train_df, val_df, test_df, split_report)

    # 5. Reference distributions for drift detection (post-engineering space)
    engineered_train = pipeline.named_steps["feature_engineer"].transform(
        train_df_eff[feature_cols]
    )
    reference_stats = build_reference_stats(
        engineered_train,
        dataset_version=card.dataset_version,
        numeric_features=list(numeric),
        categorical_features=list(categorical),
    )

    # 6. Metadata + artifacts -----------------------------------------------------
    git = git_info()
    metadata = TrainingMetadata(
        run_id=run_id,
        model_name=settings.model.name,
        scenario=scenario,
        algorithm=settings.model.algorithm,
        params=settings.model.params_for(),
        dataset_version=card.dataset_version,
        git_commit=git.commit,
        git_dirty=git.dirty,
        metrics={
            "val": val_metrics.as_flat_dict(),
            "test": test_metrics.as_flat_dict(),
        },
        numeric_features=list(numeric),
        categorical_features=list(categorical),
        n_train_rows=int(len(train_df_eff)),
        extra={
            "validation_passed": report.passed,
            "val_confusion_matrix": val_metrics.confusion_matrix,
            "test_confusion_matrix": test_metrics.confusion_matrix,
        },
    )
    out_dir = save_run_artifacts(
        settings,
        run_id=run_id,
        pipeline=pipeline,
        metadata=metadata,
        reference_stats=reference_stats,
        split_report=split_report,
    )

    # 7. Experiment tracking --------------------------------------------------------
    tags = {"scenario": scenario, "dataset_version": card.dataset_version, "env": settings.env}
    tracker.start_run(run_id, tags)
    tracker.log_params(
        {
            "algorithm": settings.model.algorithm,
            "scenario": scenario,
            "dataset_version": card.dataset_version,
            "n_train_rows": len(train_df_eff),
            "seed": settings.data.seed,
            **{f"param_{k}": v for k, v in settings.model.params_for().items()},
        }
    )
    tracker.log_metrics(val_metrics.as_flat_dict(prefix="val_"))
    tracker.log_metrics(test_metrics.as_flat_dict(prefix="test_"))
    tracker.log_artifact_dir(out_dir)
    tracker.end_run()

    log_event(
        logger,
        logging.INFO,
        "training complete",
        run_id=run_id,
        scenario=scenario,
        val_f1=round(val_metrics.f1, 4),
        val_auc=round(val_metrics.roc_auc, 4),
        test_f1=round(test_metrics.f1, 4),
        test_auc=round(test_metrics.roc_auc, 4),
    )
    return TrainingResult(
        run_id=run_id,
        run_dir=str(out_dir),
        scenario=scenario,
        dataset_version=card.dataset_version,
        algorithm=settings.model.algorithm,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        validation_passed=report.passed,
    )


def load_pipeline_from_run(path: str | Path) -> Any:
    """Convenience loader used by tests/scripts."""
    import joblib

    return joblib.load(Path(path) / "model.joblib")


def split_report_of(result: TrainingResult) -> SplitReport | None:
    import json

    p = Path(result.run_dir) / "split_report.json"
    if not p.exists():
        return None
    return SplitReport.model_validate(json.loads(p.read_text()))


def feature_frame(df: pd.DataFrame, settings: SentinelSettings) -> pd.DataFrame:
    """Project a raw dataframe onto the model's input feature columns."""
    cols = [*settings.features.numeric, *settings.features.categorical]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    return df[cols]
