"""Integration tests: monitoring collector over realistic serving logs."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from sentinel.monitoring.collector import MetricsCollector, tail_jsonl
from sentinel.simulation.scenarios import get_scenario
from sentinel.simulation.traffic import _row_to_features, generate_scenario_samples


def _write_realistic_log(
    settings,
    model_version: str,
    n: int,
    scenario: str = "normal",
    seed: int = 3,
    auc_good: bool = True,
    error_rate: float = 0.0,
) -> list[tuple[str, int]]:
    """Write prediction records with real generated features + consistent labels."""
    samples = generate_scenario_samples(get_scenario(scenario), n, seed)
    rng = np.random.default_rng(seed + 100)
    path = settings.resolve(settings.serving.prediction_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    feedback: list[tuple[str, int]] = []
    now = datetime.now(UTC) - timedelta(seconds=5)
    with path.open("a") as fh:
        for i in range(n):
            row = samples.iloc[i]
            truth = int(row["churned"])
            request_id = f"mon-{model_version}-{i}"
            is_error = rng.random() < error_rate
            # probability correlates with truth when the model is 'good'
            base = 0.85 if auc_good else 0.15
            prob = float(np.clip(base if truth == 1 else 1 - base, 0.01, 0.99))
            prob = float(np.clip(prob + rng.normal(0, 0.1), 0.01, 0.99))
            rec = {
                "ts": (now + timedelta(milliseconds=10 * i)).isoformat(),
                "request_id": request_id,
                "model_version": model_version,
                "served_by": "production",
                "status": "error" if is_error else "ok",
                "latency_ms": float(8 + rng.normal(0, 1)),
                "prediction": None if is_error else int(prob >= 0.5),
                "probability": None if is_error else prob,
                "error_type": "injected" if is_error else None,
                "features": _row_to_features(row),
            }
            fh.write(json.dumps(rec) + "\n")
            if not is_error:
                feedback.append((request_id, truth))
    return feedback


def _write_feedback(settings, entries):
    path = settings.resolve(settings.serving.feedback_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for rid, truth in entries:
            fh.write(
                json.dumps(
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "request_id": rid,
                        "ground_truth": truth,
                    }
                )
                + "\n"
            )


@pytest.fixture()
def version(settings, registered_version):
    mv, _ = registered_version
    return mv.version


def test_tail_jsonl_filters_by_time(settings, tmp_path):
    path = tmp_path / "log.jsonl"
    old = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    new = datetime.now(UTC).isoformat()
    path.write_text(json.dumps({"ts": old, "v": 1}) + "\n" + json.dumps({"ts": new, "v": 2}) + "\n")
    since = datetime.now(UTC) - timedelta(minutes=5)
    records = tail_jsonl(path, since=since)
    assert len(records) == 1 and records[0]["v"] == 2
    assert len(tail_jsonl(path)) == 2


def test_collector_window_stats(settings, version):
    _write_realistic_log(settings, version, 60)
    collector = MetricsCollector(settings)
    snapshot = collector.collect()
    stats = snapshot.stats[version]
    assert stats.n_requests == 60
    assert stats.error_rate == 0.0
    assert stats.latency_ms is not None and stats.latency_ms.p50 > 0
    assert stats.positive_rate is not None


def test_collector_performance_from_feedback(settings, version):
    entries = _write_realistic_log(settings, version, 60, auc_good=True)
    _write_feedback(settings, entries)
    snapshot = MetricsCollector(settings).collect()
    perf = snapshot.performance[version]
    assert perf.n_labels >= 50
    assert perf.roc_auc is not None and perf.roc_auc > 0.8  # labels correlate with probs


def test_collector_performance_degradation_detected(settings, version):
    entries = _write_realistic_log(settings, version, 60, auc_good=False)
    _write_feedback(settings, entries)
    snapshot = MetricsCollector(settings).collect()
    perf = snapshot.performance[version]
    assert perf.roc_auc < 0.3
    kinds = {v.kind for v in snapshot.violations}
    assert "performance" in kinds


def test_collector_no_drift_on_in_distribution_traffic(settings, version):
    _write_realistic_log(settings, version, 80, scenario="normal")
    snapshot = MetricsCollector(settings).collect()
    report = snapshot.drift[version]
    assert not report.is_drifted
    assert report.n_samples == 80


def test_collector_detects_drift_on_shifted_traffic(settings, version):
    _write_realistic_log(settings, version, 120, scenario="distribution_shift", seed=5)
    snapshot = MetricsCollector(settings).collect()
    report = snapshot.drift[version]
    assert report.is_drifted
    drifted_features = {f.feature for f in report.features if f.drifted}
    assert drifted_features & {"monthly_charges", "tenure_months", "contract_type"}
    # default severity for drift is warning: alert-only, NOT a rollback trigger
    drift_violations = [v for v in snapshot.violations if v.kind == "drift"]
    assert drift_violations and all(v.severity == "warning" for v in drift_violations)


def test_collector_counts_errors(settings, version):
    _write_realistic_log(settings, version, 60, error_rate=0.5, seed=9)
    snapshot = MetricsCollector(settings).collect()
    stats = snapshot.stats[version]
    assert stats.error_rate > 0.3
    assert any(v.kind == "error_rate" and v.severity == "critical" for v in snapshot.violations)


def test_snapshot_persisted_and_readable(settings, version):
    _write_realistic_log(settings, version, 40)
    collector = MetricsCollector(settings)
    snapshot = collector.collect()
    path = settings.resolve(settings.monitoring.snapshot_path)
    assert path.is_file()
    latest = collector.latest_snapshot()
    assert latest is not None
    assert latest.ts == snapshot.ts


def test_registry_lookup_for_unknown_version_is_skipped(settings):
    """Records referencing an unknown model must not crash the collector."""
    _write_realistic_log(settings, "v999", 40)
    snapshot = MetricsCollector(settings).collect()
    assert snapshot.stats["v999"].n_requests == 40
    assert "v999" not in snapshot.drift  # no reference stats -> drift skipped
