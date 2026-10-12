"""Shared pytest fixtures.

All fixtures are local and cheap: tiny synthetic data, fast models
(logistic regression), tmp-path settings, and in-process API clients.
No network, no docker, no MLflow server required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.config.settings import SentinelSettings  # noqa: E402


@pytest.fixture()
def settings(tmp_path: Path) -> SentinelSettings:
    """A development settings object rooted entirely in tmp_path, tuned for speed."""
    s = SentinelSettings.load("development")
    s.paths.root_dir = tmp_path

    # small + fast
    s.data.n_samples = 1200
    s.data.seed = 11
    s.validation.min_rows = 100
    s.model.algorithm = "logistic_regression"
    s.experiment_tracking.backend = "local"

    # monitoring tuned so tests need only a handful of requests
    s.monitoring.window_seconds = 3600
    s.monitoring.min_samples_for_decision = 5
    s.monitoring.drift.min_samples = 20
    s.monitoring.performance.min_samples = 10
    s.canary.min_healthy_windows_per_step = 1
    s.rollback.cooldown_seconds = 0

    s._ensure_dirs()
    return s


@pytest.fixture()
def raw_df(settings: SentinelSettings) -> pd.DataFrame:
    from sentinel.data.ingestion import generate_synthetic_churn

    return generate_synthetic_churn(n_samples=settings.data.n_samples, seed=settings.data.seed)


@pytest.fixture()
def trained_run(settings: SentinelSettings):
    """Run the real training stage once (fast config) and return the result."""
    from sentinel.training.tracking import NullTracker
    from sentinel.training.train import train_model

    return train_model(settings, scenario="default", tracker=NullTracker())


@pytest.fixture()
def registered_version(settings: SentinelSettings, trained_run):
    """Register the trained run and pass the quality gate (best effort)."""
    from sentinel.deployment.promotion import PromotionManager
    from sentinel.registry.manager import ModelRegistry

    registry = ModelRegistry(settings)
    mv = registry.register_run(Path(trained_run.run_dir))
    gate = PromotionManager(settings, registry=registry).run_gate(
        mv.version, force=True, actor="test"
    )
    return mv, gate


from tests.fixtures.helpers import write_feedback_log, write_prediction_log  # noqa: E402,F401
