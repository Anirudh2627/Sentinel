"""Unit tests: deployment controller decision logic (hand-built snapshots).

These tests drive the controller with synthetic monitoring snapshots so the
decision matrix (rollback / step_up / promote / alert / none) is verified
without a server. End-to-end behaviour with real traffic lives in
tests/integration/test_deployment_flow.py.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from sentinel.deployment.canary import CanaryManager, RoutingState, write_routing_state
from sentinel.deployment.controller import DeploymentController
from sentinel.monitoring.collector import MonitoringSnapshot, RoutingSnapshot
from sentinel.monitoring.health import Violation
from sentinel.monitoring.metrics import LatencyStats, WindowStats
from sentinel.registry.manager import ModelRegistry
from sentinel.registry.metadata import ModelStatus


def _stats(version: str, n: int = 100, errors: int = 0, positive_rate: float = 0.4) -> WindowStats:
    return WindowStats(
        model_version=version,
        n_requests=n,
        n_errors=errors,
        error_rate=errors / n if n else 0.0,
        throughput_rps=10.0,
        latency_ms=LatencyStats(p50=20.0, p95=40.0, p99=60.0, mean=22.0, max=70.0),
        positive_rate=positive_rate,
        mean_probability=0.5,
        n_predictions=n - errors,
    )


def _snapshot(
    production: str | None,
    canary: str | None,
    percent: int,
    stats: dict[str, WindowStats],
    violations: list[Violation],
) -> MonitoringSnapshot:
    return MonitoringSnapshot(
        ts=datetime.now(UTC).isoformat(),
        window_seconds=60,
        routing=RoutingSnapshot(production=production, canary=canary, canary_percent=percent),
        stats=stats,
        violations=violations,
    )


def _critical(version: str, kind: str = "error_rate") -> Violation:
    return Violation(
        kind=kind,
        model_version=version,
        value=0.5,
        threshold=0.05,
        severity="critical",
        message=f"{kind} on {version}",
    )


def _warning(version: str) -> Violation:
    return Violation(
        kind="drift",
        model_version=version,
        value=0.4,
        threshold=0.25,
        severity="warning",
        message="drift warning",
        feature="monthly_charges",
    )


@pytest.fixture()
def deployed(settings, trained_run):
    """v1 production, v2 canary @ 20%."""
    registry = ModelRegistry(settings)
    v1 = registry.register_run(Path(trained_run.run_dir))
    v2 = registry.register_run(Path(trained_run.run_dir))
    registry.mark_validated(v1.version)
    registry.mark_validated(v2.version)
    manager = CanaryManager(settings, registry)
    manager.deploy_production(v1.version)
    manager.start_canary(v2.version, percent=20)
    controller = DeploymentController(settings, registry=registry)
    return settings, registry, manager, controller, v1.version, v2.version


def test_healthy_window_counts_toward_promotion(deployed):
    settings, registry, manager, controller, v1, v2 = deployed
    settings.canary.min_healthy_windows_per_step = 3
    snap = _snapshot(v1, v2, 20, {v1: _stats(v1), v2: _stats(v2)}, [])
    decision = controller.step(snap)
    assert decision.action == "none"
    state = controller.load_state()
    assert state.healthy_windows == 1
    assert state.phase == "canary"


def test_window_without_traffic_is_not_healthy(deployed):
    settings, registry, manager, controller, v1, v2 = deployed
    snap = _snapshot(v1, v2, 20, {v1: _stats(v1)}, [])  # no canary traffic evidence
    decision = controller.step(snap)
    assert decision.action == "none"
    assert controller.load_state().healthy_windows == 0


def test_critical_violation_on_canary_rolls_back(deployed):
    settings, registry, manager, controller, v1, v2 = deployed
    snap = _snapshot(v1, v2, 20, {v1: _stats(v1), v2: _stats(v2, errors=50)}, [_critical(v2)])
    decision = controller.step(snap)
    assert decision.action == "rollback"
    assert decision.rollback.success
    assert registry.get(v2).status == ModelStatus.ROLLED_BACK
    assert manager.current_state().canary is None
    assert controller.load_state().phase == "production_monitor"


def test_warning_on_canary_does_not_roll_back(deployed):
    settings, registry, manager, controller, v1, v2 = deployed
    settings.canary.min_healthy_windows_per_step = 3
    snap = _snapshot(v1, v2, 20, {v1: _stats(v1), v2: _stats(v2)}, [_warning(v2)])
    decision = controller.step(snap)
    assert decision.action == "alert"
    assert decision.violations and decision.violations[0].severity == "warning"
    assert registry.get(v2).status == ModelStatus.CANARY  # still in traffic


def test_critical_on_production_does_not_roll_back_canary(deployed):
    settings, registry, manager, controller, v1, v2 = deployed
    snap = _snapshot(v1, v2, 20, {v1: _stats(v1), v2: _stats(v2)}, [_critical(v1)])
    decision = controller.step(snap)
    assert decision.action != "rollback"  # canary is not responsible for production's errors
    assert registry.get(v2).status == ModelStatus.CANARY


def test_step_up_then_promote(deployed):
    settings, registry, manager, controller, v1, v2 = deployed
    settings.canary.promotion_steps = [50, 100]
    settings.canary.min_healthy_windows_per_step = 2

    snap = _snapshot(v1, v2, 20, {v1: _stats(v1), v2: _stats(v2)}, [])
    assert controller.step(snap).action == "none"  # 1/2 healthy windows
    decision = controller.step(snap)
    assert decision.action == "step_up"
    assert decision.new_percent == 50
    assert manager.current_state().canary_percent == 50

    snap50 = _snapshot(v1, v2, 50, {v1: _stats(v1), v2: _stats(v2)}, [])
    controller.step(snap50)
    decision = controller.step(snap50)
    assert decision.action == "promote"
    assert registry.get(v2).status == ModelStatus.PRODUCTION
    assert registry.get(v1).status == ModelStatus.ARCHIVED
    state = manager.current_state()
    assert state.production == v2 and state.canary is None


def test_idle_when_nothing_deployed(settings):
    controller = DeploymentController(settings)
    decision = controller.step(_snapshot(None, None, 0, {}, []))
    assert decision.action == "none"
    assert controller.load_state().phase == "idle"


def test_production_monitor_alert(settings, trained_run):
    registry = ModelRegistry(settings)
    v1 = registry.register_run(Path(trained_run.run_dir))
    registry.mark_validated(v1.version)
    CanaryManager(settings, registry).deploy_production(v1.version)
    controller = DeploymentController(settings, registry=registry)

    snap = _snapshot(v1.version, None, 0, {v1.version: _stats(v1.version)}, [_warning(v1.version)])
    decision = controller.step(snap)
    assert decision.action == "alert"
    assert controller.load_state().phase == "production_monitor"


def test_production_restore_when_enabled(settings, trained_run):
    settings.rollback.auto_restore_previous_production = True
    registry = ModelRegistry(settings)
    v1 = registry.register_run(Path(trained_run.run_dir))
    v2 = registry.register_run(Path(trained_run.run_dir))
    registry.mark_validated(v1.version)
    registry.mark_validated(v2.version)
    registry.promote_to_production(v1.version)
    registry.promote_to_production(v2.version)
    write_routing_state(
        settings, RoutingState(production=v2.version, canary=None, canary_percent=0)
    )
    controller = DeploymentController(settings, registry=registry)
    snap = _snapshot(
        v2.version, None, 0, {v2.version: _stats(v2.version, errors=80)}, [_critical(v2.version)]
    )
    decision = controller.step(snap)
    assert decision.action == "restore_previous"
    assert registry.production().version == v1.version


def test_state_persists_between_steps(deployed):
    settings, registry, manager, controller, v1, v2 = deployed
    settings.canary.min_healthy_windows_per_step = 5
    snap = _snapshot(v1, v2, 20, {v1: _stats(v1), v2: _stats(v2)}, [])
    controller.step(snap)
    controller.step(snap)
    state = controller.load_state()
    assert state.windows_evaluated == 2
    assert state.healthy_windows == 2
    assert state.canary_version == v2
