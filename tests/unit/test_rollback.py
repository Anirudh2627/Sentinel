"""Unit tests: rollback manager + audit trail."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentinel.deployment.audit import AuditEntry, AuditLog
from sentinel.deployment.canary import CanaryManager
from sentinel.deployment.rollback import RollbackManager
from sentinel.monitoring.health import Violation
from sentinel.registry.manager import ModelRegistry
from sentinel.registry.metadata import ModelStatus


@pytest.fixture()
def canary_setup(settings, trained_run):
    """v1 in production, v2 in canary at 30%."""
    registry = ModelRegistry(settings)
    v1 = registry.register_run(Path(trained_run.run_dir))
    v2 = registry.register_run(Path(trained_run.run_dir))
    registry.mark_validated(v1.version)
    registry.mark_validated(v2.version)
    manager = CanaryManager(settings, registry)
    manager.deploy_production(v1.version)
    manager.start_canary(v2.version, percent=30)
    return settings, registry, manager, v1.version, v2.version


VIOLATION = Violation(
    kind="error_rate",
    model_version="v2",
    value=0.4,
    threshold=0.05,
    severity="critical",
    message="error rate 0.4 > 0.05",
)


def test_rollback_canary_restores_production(canary_setup):
    settings, registry, manager, v1, v2 = canary_setup
    rm = RollbackManager(settings, registry, manager)
    result = rm.rollback_canary(reason="error rate spike", violations=[VIOLATION])
    assert result.success
    assert result.action == "canary_rolled_back"
    assert result.rolled_back_version == v2
    assert result.restored_version == v1
    state = manager.current_state()
    assert state.canary is None and state.production == v1 and state.canary_percent == 0
    assert registry.get(v2).status == ModelStatus.ROLLED_BACK


def test_rollback_is_audited_with_metric_details(canary_setup):
    settings, registry, manager, v1, v2 = canary_setup
    rm = RollbackManager(settings, registry, manager)
    rm.rollback_canary(reason="error rate spike", violations=[VIOLATION], actor="controller")

    entries = AuditLog(settings).read()
    rollback_entries = [e for e in entries if e.action == "canary_rolled_back"]
    assert len(rollback_entries) == 1
    entry = rollback_entries[0]
    assert entry.model_version == v2
    assert entry.previous_version == v1
    assert entry.restored_version == v1
    assert entry.reason == "error rate spike"
    assert entry.triggered_metric == "error_rate"
    assert entry.observed_value == 0.4
    assert entry.threshold == 0.05
    assert entry.canary_percent == 30
    assert entry.actor == "controller"
    assert entry.timestamp


def test_rollback_disabled_by_config(canary_setup):
    settings, registry, manager, v1, v2 = canary_setup
    settings.rollback.enabled = False
    rm = RollbackManager(settings, registry, manager)
    result = rm.rollback_canary(reason="whatever")
    assert not result.success
    assert manager.current_state().canary == v2  # untouched


def test_rollback_without_canary_is_noop(canary_setup):
    settings, registry, manager, v1, v2 = canary_setup
    manager.rollback_canary(reason="first rollback")
    rm = RollbackManager(settings, registry, manager)
    result = rm.rollback_canary(reason="second rollback")
    assert not result.success
    assert "no active canary" in result.reason


def test_restore_previous_production(canary_setup):
    settings, registry, manager, v1, v2 = canary_setup
    # promote v2 so v1 becomes previous production, then restore v1
    manager.promote_canary()
    rm = RollbackManager(settings, registry, manager)
    result = rm.restore_previous_production(reason="prod degraded")
    assert result.success
    assert result.restored_version == v1
    assert registry.production().version == v1
    assert manager.current_state().production == v1


def test_restore_previous_without_history_fails(canary_setup):
    settings, registry, manager, v1, v2 = canary_setup
    rm = RollbackManager(settings, registry, manager)
    result = rm.restore_previous_production(reason="nothing to restore")
    assert not result.success


def test_should_rollback_filters_critical(canary_setup):
    settings, registry, manager, v1, v2 = canary_setup
    rm = RollbackManager(settings, registry, manager)
    warning = Violation(
        kind="drift", model_version="v2", value=0.3, threshold=0.25, severity="warning", message="d"
    )
    assert rm.should_rollback([warning]) == []
    assert rm.should_rollback([warning, VIOLATION]) == [VIOLATION]


def test_audit_log_append_and_read(settings):
    log = AuditLog(settings)
    log.record(AuditEntry(action="alert", reason="one"))
    log.record(AuditEntry(action="traffic_changed", canary_percent=25))
    entries = log.read()
    assert len(entries) == 2
    assert entries[-1].canary_percent == 25
    assert len(log.read(last_n=1)) == 1


def test_audit_last_rollback_ts(settings):
    log = AuditLog(settings)
    assert log.last_rollback_ts() is None
    log.record(AuditEntry(action="alert", reason="x"))
    assert log.last_rollback_ts() is None
    log.record(AuditEntry(action="canary_rolled_back", reason="y"))
    ts = log.last_rollback_ts()
    assert ts is not None
    import time

    assert abs(ts - time.time()) < 5


def test_audit_file_is_jsonl(settings):
    log = AuditLog(settings)
    log.record(AuditEntry(action="alert", reason="one"))
    log.record(AuditEntry(action="alert", reason="two"))
    lines = settings.resolve(settings.rollback.audit_path).read_text().strip().splitlines()
    assert len(lines) == 2
    assert all(json.loads(ln) for ln in lines)
