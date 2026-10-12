"""Experiment tracking abstraction.

``MLflowTracker`` is the primary backend (parameters, metrics, artifacts,
model registry sync). ``LocalTracker`` is a dependency-free JSON fallback so
that training, CI and tests never hard-require an MLflow installation or
server. Backend selection: ``experiment_tracking.backend`` (auto|mlflow|local|none).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from sentinel.config import SentinelSettings, get_logger, log_event

logger = get_logger(__name__)


class ExperimentTracker(Protocol):
    """Minimal tracking contract used by the training pipeline."""

    name: str

    def start_run(self, run_id: str, tags: dict[str, str]) -> None: ...
    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    def log_artifact_dir(self, path: Path) -> None: ...
    def register_model_version(self, run_id: str, version: str, status: str) -> str | None: ...
    def end_run(self) -> None: ...


class NullTracker:
    """No-op tracker (backend=none)."""

    name = "none"

    def start_run(self, run_id: str, tags: dict[str, str]) -> None:
        pass

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def log_artifact_dir(self, path: Path) -> None:
        pass

    def register_model_version(self, run_id: str, version: str, status: str) -> str | None:
        return None

    def end_run(self) -> None:
        pass


class LocalTracker:
    """JSON-lines tracker under ``artifacts/experiments/``."""

    name = "local"

    def __init__(self, settings: SentinelSettings):
        self.dir = settings.resolve(settings.experiment_tracking.local_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "runs.jsonl"
        self._record: dict[str, Any] = {}

    def start_run(self, run_id: str, tags: dict[str, str]) -> None:
        self._record = {
            "run_id": run_id,
            "backend": "local",
            "started_at": datetime.now(UTC).isoformat(),
            "tags": tags,
            "params": {},
            "metrics": {},
            "artifacts": [],
        }

    def log_params(self, params: dict[str, Any]) -> None:
        self._record.setdefault("params", {}).update(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._record.setdefault("metrics", {}).update(metrics)

    def log_artifact_dir(self, path: Path) -> None:
        self._record.setdefault("artifacts", []).append(str(path))

    def register_model_version(self, run_id: str, version: str, status: str) -> str | None:
        self._record["registered_version"] = {"version": version, "status": status}
        return version

    def end_run(self) -> None:
        self._record["ended_at"] = datetime.now(UTC).isoformat()
        with self.file.open("a") as fh:
            fh.write(json.dumps(self._record, default=str) + "\n")
        self._record = {}


class MLflowTracker:
    """MLflow-backed tracker. Imported lazily so mlflow stays optional."""

    name = "mlflow"

    def __init__(self, settings: SentinelSettings):
        import mlflow  # lazy: keeps the rest of Sentinel importable without mlflow

        self._mlflow = mlflow
        cfg = settings.experiment_tracking.mlflow
        self.settings = settings
        self._registered_model_name = cfg.registered_model_name
        # Default local backend: SQLite (MLflow 3.x put the plain file store in
        # maintenance mode). Point tracking_uri at a server (http://...) in docker.
        tracking_uri = cfg.tracking_uri or f"sqlite:///{settings.resolve('mlflow.db')}"
        mlflow.set_tracking_uri(tracking_uri)
        self._experiment_id = mlflow.set_experiment(cfg.experiment_name).experiment_id
        self._run: Any = None
        log_event(logger, logging.INFO, "mlflow tracker ready", tracking_uri=tracking_uri)

    def start_run(self, run_id: str, tags: dict[str, str]) -> None:
        self._run = self._mlflow.start_run(run_name=run_id, tags=tags)

    def log_params(self, params: dict[str, Any]) -> None:
        flat = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
        self._mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v is not None and v == v}  # drop NaN
        self._mlflow.log_metrics(clean, step=step)

    def log_artifact_dir(self, path: Path) -> None:
        self._mlflow.log_artifacts(str(path))

    def register_model_version(self, run_id: str, version: str, status: str) -> str | None:
        """Mirror a Sentinel registry version into the MLflow Model Registry."""
        if self._run is None:
            return None
        try:
            model_uri = f"runs:/{self._run.info.run_id}/model"
            mv = self._mlflow.register_model(model_uri, self._registered_model_name)
            client = self._mlflow.tracking.MlflowClient()
            client.set_model_version_tag(
                self._registered_model_name, mv.version, "sentinel_version", version
            )
            client.set_model_version_tag(
                self._registered_model_name, mv.version, "sentinel_status", status
            )
            return f"{self._registered_model_name}:{mv.version}"
        except Exception as exc:  # best-effort mirror; never blocks the pipeline
            log_event(logger, logging.WARNING, "mlflow model registration skipped", error=str(exc))
            return None

    def end_run(self) -> None:
        if self._run is not None:
            self._mlflow.end_run()
            self._run = None


def get_tracker(settings: SentinelSettings) -> ExperimentTracker:
    """Factory honouring ``experiment_tracking.backend`` with graceful fallback."""
    backend = settings.experiment_tracking.backend
    if backend == "none":
        return NullTracker()
    if backend == "local":
        return LocalTracker(settings)
    if backend == "mlflow":
        return MLflowTracker(settings)
    # auto
    try:
        import mlflow  # noqa: F401

        if settings.experiment_tracking.mlflow.enabled:
            return MLflowTracker(settings)
    except ImportError:
        pass
    except Exception as exc:  # e.g. unreachable tracking server
        log_event(
            logger,
            logging.WARNING,
            "mlflow unavailable, falling back to local tracker",
            error=str(exc),
        )
    return LocalTracker(settings)
