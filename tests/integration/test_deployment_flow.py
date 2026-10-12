"""Integration test: the FULL deployment lifecycle in one process.

    train v1 → gate → deploy production
    train v2 → gate → canary → healthy traffic → step up → promote
    train v3 (degraded) → forced gate → canary → real traffic with feedback
    → controller detects live performance collapse → automatic rollback

Traffic is driven through the real FastAPI app via an in-process transport
(fastapi TestClient), so latency, errors and predictions are all real.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sentinel.deployment.audit import AuditLog
from sentinel.deployment.canary import CanaryManager
from sentinel.deployment.controller import DeploymentController
from sentinel.deployment.promotion import PromotionManager
from sentinel.pipeline.orchestration import run_pipeline
from sentinel.registry.manager import ModelRegistry
from sentinel.registry.metadata import ModelStatus
from sentinel.serving.app import create_app
from sentinel.simulation.traffic import TestClientTransport, TrafficSimulator
from sentinel.training.tracking import NullTracker


def _simulate(settings, client, scenario="normal", n=160, feedback_rate=1.0, seed=5):
    return TrafficSimulator(
        settings,
        scenario=scenario,
        transport=TestClientTransport(client),
        requests_per_second=1000,  # in-process: no real pacing needed
        duration_seconds=1,
        concurrency=4,
        feedback_rate=feedback_rate,
        seed=seed,
        max_requests=n,
        progress_every=10_000,
    ).run()


@pytest.fixture()
def fast_settings(settings):
    """Tune decision thresholds so the flow completes in a few windows."""
    settings.canary.min_healthy_windows_per_step = 1
    settings.canary.promotion_steps = [25, 50, 100]
    settings.monitoring.min_samples_for_decision = 10
    settings.monitoring.performance.min_samples = 15
    settings.monitoring.drift.min_samples = 20
    settings.monitoring.prediction_shift.min_samples = 30
    settings.serving.routing_reload_interval_seconds = 0.0
    return settings


def _train_and_register(
    settings, seed: int, scenario: str = "default", force_gate: bool = False
) -> str:
    settings.data.seed = seed
    result = run_pipeline(
        settings, scenario=scenario, register=True, gate=False, tracker=NullTracker()
    )
    promotion = PromotionManager(settings)
    gate = promotion.run_gate(result.model_version, force=force_gate, actor="test")
    assert gate.passed or force_gate
    return result.model_version


def test_full_lifecycle_promotion(fast_settings):
    settings = fast_settings

    # --- v1: train, gate, direct production deploy -------------------------
    v1 = _train_and_register(settings, seed=21, force_gate=True)
    promotion = PromotionManager(settings)
    decision = promotion.deploy_candidate(v1, via_canary=False, actor="test")
    assert decision.deployed and decision.mode == "production"

    registry = ModelRegistry(settings)
    assert registry.production().version == v1

    app = create_app(settings)
    with TestClient(app) as client:
        # --- v2: train, gate, canary at 10% --------------------------------
        v2 = _train_and_register(settings, seed=22, force_gate=True)
        decision = promotion.deploy_candidate(v2, via_canary=True, percent=30, actor="test")
        assert decision.deployed and decision.mode == "canary"
        assert registry.canary().version == v2

        # --- healthy traffic ------------------------------------------------
        report = _simulate(settings, client, scenario="normal", n=400)
        assert report.n_ok > 350
        assert report.n_errors == 0
        assert set(report.served_by_version) == {v1, v2}  # both versions served
        # sticky hash routing approximates the 30% split
        share = report.served_by_version[v2] / report.n_sent
        assert 0.15 < share < 0.45

        # --- controller ramps traffic then promotes --------------------------
        controller = DeploymentController(settings)
        d1 = controller.step()
        assert d1.action == "step_up" and d1.new_percent == 50
        d2 = controller.step()
        assert d2.action == "promote"

        assert registry.production().version == v2
        assert registry.get(v1).status == ModelStatus.ARCHIVED
        assert registry.previous_production().version == v1
        assert CanaryManager(settings).current_state().canary is None

        # --- traffic now flows 100% to v2 ------------------------------------
        report2 = _simulate(settings, client, scenario="normal", n=60, seed=77)
        assert set(report2.served_by_version) == {v2}

    # --- audit trail tells the whole story ---------------------------------
    actions = [e.action for e in AuditLog(settings).read()]
    assert "gate_decision" in actions
    assert "production_deployed" in actions
    assert "canary_started" in actions
    assert actions.count("traffic_changed") >= 1
    assert "canary_promoted" in actions


def test_full_lifecycle_rollback_of_bad_canary(fast_settings):
    settings = fast_settings

    # --- healthy production model -------------------------------------------
    v1 = _train_and_register(settings, seed=31, force_gate=True)
    promotion = PromotionManager(settings)
    promotion.deploy_candidate(v1, via_canary=False, actor="test")

    # --- deliberately degraded candidate, gate FORCED (audited override) -----
    v_bad = _train_and_register(settings, seed=31, scenario="degraded", force_gate=True)
    registry = ModelRegistry(settings)
    assert registry.get(v_bad).scenario == "degraded"
    overrides = [e for e in AuditLog(settings).read() if e.action == "manual_override"]
    assert overrides, "forced gate must leave an audit trail"

    decision = promotion.deploy_candidate(v_bad, via_canary=True, percent=50, actor="test")
    assert decision.deployed

    app = create_app(settings)
    with TestClient(app) as client:
        # --- normal traffic WITH delayed ground truth -------------------------
        report = _simulate(settings, client, scenario="normal", n=240, feedback_rate=1.0, seed=9)
        assert report.n_errors == 0
        assert report.served_by_version.get(v_bad, 0) >= 50

        # --- controller must detect degradation and roll back -----------------
        controller = DeploymentController(settings)
        decision_obj = controller.step()
        assert decision_obj.action == "rollback", (
            f"expected rollback, got {decision_obj.action}: {decision_obj.reason}"
        )
        assert decision_obj.rollback.success
        assert decision_obj.rollback.rolled_back_version == v_bad
        assert decision_obj.rollback.restored_version == v1

        # --- state after rollback ----------------------------------------------
        assert registry.get(v_bad).status == ModelStatus.ROLLED_BACK
        assert registry.production().version == v1
        state = CanaryManager(settings).current_state()
        assert state.canary is None and state.canary_percent == 0

        # --- all traffic back on production ------------------------------------
        report2 = _simulate(settings, client, scenario="normal", n=40, seed=123)
        assert set(report2.served_by_version) == {v1}

    # --- rollback audit entry carries the forensic details --------------------
    rollbacks = [e for e in AuditLog(settings).read() if e.action == "canary_rolled_back"]
    assert len(rollbacks) == 1
    entry = rollbacks[0]
    assert entry.model_version == v_bad
    assert entry.restored_version == v1
    assert entry.triggered_metric in {"performance", "prediction_shift", "error_rate"}
    assert entry.observed_value is not None and entry.threshold is not None
    assert entry.timestamp


def test_rollback_cooldown_blocks_new_canary(fast_settings):
    settings = fast_settings
    settings.rollback.cooldown_seconds = 3600

    v1 = _train_and_register(settings, seed=41, force_gate=True)
    promotion = PromotionManager(settings)
    promotion.deploy_candidate(v1, via_canary=False)
    v2 = _train_and_register(settings, seed=42, force_gate=True)

    promotion.deploy_candidate(v2, via_canary=True, percent=20)
    from sentinel.deployment.rollback import RollbackManager

    RollbackManager(settings).rollback_canary(reason="test cooldown")

    v3 = _train_and_register(settings, seed=43, force_gate=True)
    decision = promotion.deploy_candidate(v3, via_canary=True, percent=20)
    assert not decision.deployed
    assert "cooldown" in decision.reason


def test_gate_blocks_degraded_model_without_force(fast_settings):
    """The honest path: a bad model never becomes a candidate without an override."""
    settings = fast_settings
    v1 = _train_and_register(settings, seed=51, force_gate=True)
    PromotionManager(settings).deploy_candidate(v1, via_canary=False)

    settings.data.seed = 51
    result = run_pipeline(
        settings, scenario="degraded", register=True, gate=True, tracker=NullTracker()
    )
    assert result.gate_passed is False
    registry = ModelRegistry(settings)
    assert registry.get(result.model_version).status == ModelStatus.TRAINED
    assert registry.candidate() is None or registry.candidate().version != result.model_version
    # cannot be deployed: not a candidate
    decision = PromotionManager(settings).deploy_candidate(result.model_version, via_canary=True)
    assert not decision.deployed
