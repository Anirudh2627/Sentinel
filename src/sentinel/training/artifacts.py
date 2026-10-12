"""Training run artifacts: persistence + provenance metadata.

A *run* is a directory under ``artifacts/runs/<run_id>/`` containing:

    model.joblib          fitted sklearn Pipeline (engineering + model)
    metadata.json         TrainingMetadata (versions, params, metrics, git)
    reference_stats.json  training feature distributions (drift baseline)
    split_report.json     dataset split provenance
    metrics.json          flat metrics for quick inspection/CI summaries

The registry later copies an approved run into ``models/v<n>/``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from pydantic import BaseModel, Field

from sentinel import __version__
from sentinel.config import SentinelSettings, get_logger, log_event

logger = get_logger(__name__)

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
REFERENCE_FILE = "reference_stats.json"
SPLIT_FILE = "split_report.json"
METRICS_FILE = "metrics.json"


class GitInfo(BaseModel):
    commit: str | None = None
    dirty: bool | None = None


def git_info() -> GitInfo:
    """Best-effort git provenance. Never faked — ``None`` when unavailable."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        return GitInfo(commit=commit or None, dirty=bool(status))
    except (OSError, subprocess.SubprocessError):
        return GitInfo(commit=None, dirty=None)


class TrainingMetadata(BaseModel):
    """Everything needed to trace a model version back to its inputs."""

    run_id: str
    model_name: str
    scenario: str = "default"  # default | degraded
    algorithm: str
    params: dict[str, Any] = Field(default_factory=dict)
    dataset_version: str
    git_commit: str | None = None
    git_dirty: bool | None = None
    code_version: str = __version__
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    metrics: dict[str, dict[str, float]] = Field(
        default_factory=dict
    )  # {"val": {...}, "test": {...}}
    numeric_features: list[str] = Field(default_factory=list)
    categorical_features: list[str] = Field(default_factory=list)
    n_train_rows: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


def new_run_id() -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{ts}-{uuid.uuid4().hex[:6]}"


def run_dir(settings: SentinelSettings, run_id: str) -> Path:
    return settings.resolve(settings.paths.artifacts_dir) / "runs" / run_id


def save_run_artifacts(
    settings: SentinelSettings,
    run_id: str,
    pipeline: Any,
    metadata: TrainingMetadata,
    reference_stats: dict[str, Any],
    split_report: Any | None = None,
) -> Path:
    """Persist all artifacts for a training run; returns the run directory."""
    out = run_dir(settings, run_id)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, out / MODEL_FILE)
    (out / METADATA_FILE).write_text(metadata.model_dump_json(indent=2))
    (out / REFERENCE_FILE).write_text(json.dumps(reference_stats, indent=2))
    if split_report is not None:
        dump = split_report.model_dump() if hasattr(split_report, "model_dump") else split_report
        (out / SPLIT_FILE).write_text(json.dumps(dump, indent=2))
    flat = {
        **metadata.metrics.get("val", {}),
        **{f"test_{k}": v for k, v in metadata.metrics.get("test", {}).items()},
    }
    (out / METRICS_FILE).write_text(json.dumps(flat, indent=2))

    log_event(
        logger,
        logging.INFO,
        "run artifacts saved",
        run_id=run_id,
        path=str(out),
        dataset_version=metadata.dataset_version,
    )
    return out


def load_metadata(path: Path) -> TrainingMetadata:
    return TrainingMetadata.model_validate(json.loads((path / METADATA_FILE).read_text()))


def load_pipeline(path: Path) -> Any:
    return joblib.load(path / MODEL_FILE)


def load_reference_stats(path: Path) -> dict[str, Any]:
    return json.loads((path / REFERENCE_FILE).read_text())


def build_reference_stats(
    df: pd.DataFrame,
    dataset_version: str,
    numeric_features: list[str],
    categorical_features: list[str],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute the training distribution baseline used by drift detection."""
    from sentinel.monitoring.drift import fit_reference_stats  # local import: avoids cycle

    return fit_reference_stats(
        df,
        dataset_version=dataset_version,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        n_bins=n_bins,
    )
