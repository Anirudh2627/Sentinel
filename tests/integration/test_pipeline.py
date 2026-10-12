"""Integration test: train → evaluate → register → gate pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from sentinel.pipeline.orchestration import run_pipeline
from sentinel.registry.manager import ModelRegistry
from sentinel.registry.metadata import ModelStatus
from sentinel.training.tracking import LocalTracker


def test_pipeline_registers_and_gates(settings):
    result = run_pipeline(settings, scenario="default", tracker=LocalTracker(settings))
    assert result.ok
    assert result.model_version == "v1"
    assert result.gate_passed in (True, False)  # outcome is real, not assumed

    registry = ModelRegistry(settings)
    mv = registry.get("v1")
    assert mv.dataset_version == result.dataset_version
    if result.gate_passed:
        assert mv.status == ModelStatus.CANDIDATE
    else:
        assert mv.status == ModelStatus.TRAINED  # blocked models stay un-promoted

    # metrics report persisted
    metrics_path = Path(result.reports["metrics"])
    assert metrics_path.is_file()
    payload = json.loads(metrics_path.read_text())
    assert payload["run_id"] == result.run_id
    assert 0.0 <= payload["val"]["f1"] <= 1.0


def test_pipeline_artifacts_complete(settings):
    result = run_pipeline(settings, tracker=LocalTracker(settings))
    registry = ModelRegistry(settings)
    artifact = registry.artifact_dir(result.model_version)
    for name in (
        "model.joblib",
        "metadata.json",
        "reference_stats.json",
        "split_report.json",
        "metrics.json",
    ):
        assert (artifact / name).is_file(), f"missing artifact: {name}"

    metadata = json.loads((artifact / "metadata.json").read_text())
    assert metadata["dataset_version"] == result.dataset_version
    assert metadata["algorithm"] == settings.model.algorithm
    # git_commit is populated dynamically or None — never faked
    assert metadata["git_commit"] is None or len(metadata["git_commit"]) == 40


def test_pipeline_gate_report_written(settings):
    result = run_pipeline(settings, tracker=LocalTracker(settings))
    gate_path = Path(result.reports["gate"])
    assert gate_path.is_file()
    content = gate_path.read_text()
    assert "Evaluation gate" in content
    assert "min_f1" in content


def test_pipeline_degraded_scenario_fails_gate(settings):
    """A deliberately corrupted model must be blocked by the gate (real metrics)."""
    result = run_pipeline(settings, scenario="degraded", tracker=LocalTracker(settings))
    assert result.gate_passed is False
    assert result.val_metrics["f1"] < settings.evaluation.min_f1
    registry = ModelRegistry(settings)
    assert registry.get(result.model_version).status == ModelStatus.TRAINED


def test_two_runs_are_reproducible(settings):
    """Same seed + same config ⇒ identical dataset version (content-addressed)."""
    r1 = run_pipeline(
        settings, n_samples=600, register=False, gate=False, tracker=LocalTracker(settings)
    )
    r2 = run_pipeline(
        settings, n_samples=600, register=False, gate=False, tracker=LocalTracker(settings)
    )
    assert r1.dataset_version == r2.dataset_version


def test_local_tracker_records_runs(settings):
    tracker = LocalTracker(settings)
    run_pipeline(settings, n_samples=600, register=False, gate=False, tracker=tracker)
    runs_file = settings.resolve(settings.experiment_tracking.local_dir) / "runs.jsonl"
    assert runs_file.is_file()
    record = json.loads(runs_file.read_text().strip().splitlines()[-1])
    assert "val_f1" in record["metrics"]
    assert record["params"]["algorithm"] == settings.model.algorithm
