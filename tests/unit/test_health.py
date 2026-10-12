"""Unit tests: health evaluation policy (thresholds -> violations)."""

from __future__ import annotations

from sentinel.monitoring.drift import DriftReport, FeatureDrift
from sentinel.monitoring.health import HealthEvaluator
from sentinel.monitoring.metrics import LatencyStats, PerformanceStats, WindowStats


def _stats(version, n=100, errors=0, p95=50.0, p99=80.0, positive_rate=0.4) -> WindowStats:
    return WindowStats(
        model_version=version,
        n_requests=n,
        n_errors=errors,
        error_rate=errors / n,
        throughput_rps=10.0,
        latency_ms=LatencyStats(p50=20.0, p95=p95, p99=p99, mean=25.0, max=p99 + 10),
        positive_rate=positive_rate,
        mean_probability=0.5,
        n_predictions=n - errors,
    )


def _drift_report(version, score=0.5, drifted=True) -> DriftReport:
    return DriftReport(
        method="psi",
        reference_dataset_version="ds-x",
        n_samples=100,
        is_drifted=drifted,
        max_score=score,
        n_drifted_features=1 if drifted else 0,
        features=[
            FeatureDrift(
                feature="monthly_charges",
                kind="numeric",
                score=score,
                threshold=0.25,
                drifted=drifted,
            )
        ],
    )


def test_healthy_system_no_violations(settings):
    ev = HealthEvaluator(settings.monitoring, settings.rollback)
    v = ev.evaluate({"v1": _stats("v1")}, production_version="v1")
    assert v == []


def test_error_rate_violation(settings):
    settings.monitoring.max_error_rate = 0.05
    ev = HealthEvaluator(settings.monitoring, settings.rollback)
    violations = ev.evaluate({"v1": _stats("v1", errors=10)}, production_version="v1")
    kinds = {v.kind for v in violations}
    assert "error_rate" in kinds
    err = next(v for v in violations if v.kind == "error_rate")
    assert err.severity == "critical"  # rollback.on_error_rate defaults true


def test_error_rate_violation_warning_when_rollback_disabled_for_kind(settings):
    settings.rollback.on_error_rate = False
    ev = HealthEvaluator(settings.monitoring, settings.rollback)
    violations = ev.evaluate({"v1": _stats("v1", errors=10)}, production_version="v1")
    err = next(v for v in violations if v.kind == "error_rate")
    assert err.severity == "warning"


def test_latency_violations(settings):
    settings.monitoring.max_p95_latency_ms = 100.0
    settings.monitoring.max_p99_latency_ms = 200.0
    ev = HealthEvaluator(settings.monitoring, settings.rollback)
    violations = ev.evaluate({"v1": _stats("v1", p95=150.0, p99=300.0)})
    kinds = {v.kind for v in violations}
    assert {"latency_p95", "latency_p99"} <= kinds


def test_insufficient_samples_skips_slo_checks(settings):
    settings.monitoring.min_samples_for_decision = 50
    ev = HealthEvaluator(settings.monitoring, settings.rollback)
    violations = ev.evaluate({"v1": _stats("v1", n=10, errors=10)})
    assert violations == []


def test_prediction_shift_between_canary_and_production(settings):
    settings.monitoring.prediction_shift.max_delta_positive_rate = 0.15
    ev = HealthEvaluator(settings.monitoring, settings.rollback)
    stats = {
        "v1": _stats("v1", positive_rate=0.30),
        "v2": _stats("v2", positive_rate=0.75),
    }
    violations = ev.evaluate(stats, production_version="v1", canary_version="v2")
    shift = [v for v in violations if v.kind == "prediction_shift"]
    assert len(shift) == 1
    assert shift[0].model_version == "v2"
    assert shift[0].severity == "critical"


def test_prediction_shift_disabled(settings):
    settings.monitoring.prediction_shift.enabled = False
    ev = HealthEvaluator(settings.monitoring, settings.rollback)
    stats = {"v1": _stats("v1", positive_rate=0.1), "v2": _stats("v2", positive_rate=0.9)}
    violations = ev.evaluate(stats, production_version="v1", canary_version="v2")
    assert not [v for v in violations if v.kind == "prediction_shift"]


def test_drift_warning_by_default(settings):
    settings.monitoring.drift.severity = "warning"
    ev = HealthEvaluator(settings.monitoring, settings.rollback)
    violations = ev.evaluate({"v1": _stats("v1")}, drift_reports={"v1": _drift_report("v1")})
    drift = [v for v in violations if v.kind == "drift"]
    assert len(drift) == 1
    assert drift[0].severity == "warning"  # alert-only: drift != model failure


def test_drift_critical_only_when_configured(settings):
    settings.monitoring.drift.severity = "critical"
    settings.rollback.on_drift = True
    ev = HealthEvaluator(settings.monitoring, settings.rollback)
    violations = ev.evaluate({"v1": _stats("v1")}, drift_reports={"v1": _drift_report("v1")})
    drift = [v for v in violations if v.kind == "drift"]
    assert drift[0].severity == "critical"


def test_no_drift_no_violation(settings):
    ev = HealthEvaluator(settings.monitoring, settings.rollback)
    violations = ev.evaluate(
        {"v1": _stats("v1")}, drift_reports={"v1": _drift_report("v1", drifted=False, score=0.01)}
    )
    assert violations == []


def test_performance_degradation(settings):
    settings.monitoring.performance.min_auc = 0.6
    settings.monitoring.performance.min_samples = 20
    ev = HealthEvaluator(settings.monitoring, settings.rollback)
    perf = {"v2": PerformanceStats(model_version="v2", n_labels=50, roc_auc=0.52)}
    violations = ev.evaluate({"v2": _stats("v2")}, performance=perf)
    kinds = {v.kind for v in violations}
    assert "performance" in kinds
    p = next(v for v in violations if v.kind == "performance")
    assert p.severity == "critical"


def test_performance_needs_min_labels(settings):
    settings.monitoring.performance.min_samples = 100
    ev = HealthEvaluator(settings.monitoring, settings.rollback)
    perf = {"v2": PerformanceStats(model_version="v2", n_labels=10, roc_auc=0.3)}
    assert ev.evaluate({"v2": _stats("v2")}, performance=perf) == []
