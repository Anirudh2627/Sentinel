"""Unit tests: canary routing, traffic allocation, manager operations."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.deployment.canary import (
    CanaryError,
    CanaryManager,
    CanaryRouter,
    RoutingState,
    cooldown_active,
    read_routing_state,
    write_routing_state,
)
from sentinel.registry.metadata import ModelStatus


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
def test_router_no_canary_goes_production():
    router = CanaryRouter(sticky=True)
    state = RoutingState(production="v1", canary=None, canary_percent=0)
    assert router.route(state, "req-1") == ("v1", "production")


def test_router_empty_state():
    router = CanaryRouter()
    assert router.route(RoutingState(), "req-1") == (None, "production")


def test_router_sticky_is_deterministic():
    router = CanaryRouter(sticky=True)
    state = RoutingState(production="v1", canary="v2", canary_percent=50)
    assert router.route(state, "req-xyz") == router.route(state, "req-xyz")


def test_router_bucket_stable_and_uniform():
    buckets = [CanaryRouter.bucket(f"request-{i}") for i in range(10_000)]
    assert all(0 <= b < 10_000 for b in buckets)
    # same id -> same bucket
    assert CanaryRouter.bucket("abc") == CanaryRouter.bucket("abc")


def test_router_allocation_approximates_percent():
    router = CanaryRouter(sticky=True)
    state = RoutingState(production="v1", canary="v2", canary_percent=20)
    canary_hits = sum(1 for i in range(20_000) if router.route(state, f"req-{i}")[1] == "canary")
    ratio = canary_hits / 20_000
    assert 0.17 < ratio < 0.23  # hash routing is uniform within tolerance


def test_router_zero_percent_never_canaries():
    router = CanaryRouter(sticky=True)
    state = RoutingState(production="v1", canary="v2", canary_percent=0)
    assert all(router.route(state, f"r{i}")[1] == "production" for i in range(100))


def test_router_hundred_percent_always_canary():
    router = CanaryRouter(sticky=True)
    state = RoutingState(production="v1", canary="v2", canary_percent=100)
    assert all(router.route(state, f"r{i}")[1] == "canary" for i in range(100))


# --------------------------------------------------------------------------- #
# Routing state persistence
# --------------------------------------------------------------------------- #
def test_routing_state_roundtrip(settings):
    state = RoutingState(production="v1", canary="v2", canary_percent=15, updated_by="test")
    write_routing_state(settings, state)
    loaded = read_routing_state(settings)
    assert loaded.production == "v1"
    assert loaded.canary == "v2"
    assert loaded.canary_percent == 15


def test_routing_state_missing_file_is_empty(settings):
    state = read_routing_state(settings)
    assert state.production is None


# --------------------------------------------------------------------------- #
# CanaryManager
# --------------------------------------------------------------------------- #
@pytest.fixture()
def manager(settings):
    return CanaryManager(settings)


@pytest.fixture()
def two_candidates(settings, manager, trained_run):
    registry = manager.registry
    v1 = registry.register_run(Path(trained_run.run_dir))
    v2 = registry.register_run(Path(trained_run.run_dir))
    registry.mark_validated(v1.version)
    registry.mark_validated(v2.version)
    return v1.version, v2.version


def test_start_canary_requires_production(manager, two_candidates):
    v1, v2 = two_candidates
    with pytest.raises(CanaryError, match="no production model"):
        manager.start_canary(v2)


def test_deploy_production_then_canary(manager, two_candidates):
    v1, v2 = two_candidates
    state = manager.deploy_production(v1)
    assert state.production == v1 and state.canary is None
    state = manager.start_canary(v2, percent=10)
    assert (state.production, state.canary, state.canary_percent) == (v1, v2, 10)
    assert manager.registry.get(v2).status == ModelStatus.CANARY


def test_start_canary_rejects_non_candidate(manager, settings, trained_run):
    registry = manager.registry
    v1 = registry.register_run(Path(trained_run.run_dir))
    v2 = registry.register_run(Path(trained_run.run_dir))
    registry.mark_validated(v1.version)
    manager.deploy_production(v1.version)
    with pytest.raises(CanaryError):
        manager.start_canary(v2.version)  # v2 is still TRAINED


def test_set_traffic(manager, two_candidates):
    v1, v2 = two_candidates
    manager.deploy_production(v1)
    manager.start_canary(v2, percent=5)
    state = manager.set_traffic(50)
    assert state.canary_percent == 50
    with pytest.raises(CanaryError):
        manager.set_traffic(150)


def test_next_step_percent(manager):
    manager.settings.canary.promotion_steps = [25, 50, 100]
    assert manager.next_step_percent(5) == 25
    assert manager.next_step_percent(25) == 50
    assert manager.next_step_percent(50) == 100
    assert manager.next_step_percent(100) is None


def test_promote_canary(manager, two_candidates):
    v1, v2 = two_candidates
    manager.deploy_production(v1)
    manager.start_canary(v2, percent=50)
    state = manager.promote_canary()
    assert state.production == v2 and state.canary is None
    assert manager.registry.get(v1).status == ModelStatus.ARCHIVED
    assert manager.registry.previous_production().version == v1


def test_rollback_canary(manager, two_candidates):
    v1, v2 = two_candidates
    manager.deploy_production(v1)
    manager.start_canary(v2, percent=50)
    state = manager.rollback_canary(reason="test rollback")
    assert state.production == v1 and state.canary is None and state.canary_percent == 0
    assert manager.registry.get(v2).status == ModelStatus.ROLLED_BACK


def test_double_canary_rejected(manager, settings, trained_run):
    registry = manager.registry
    versions = [registry.register_run(Path(trained_run.run_dir)).version for _ in range(3)]
    for v in versions:
        registry.mark_validated(v)
    manager.deploy_production(versions[0])
    manager.start_canary(versions[1], percent=10)
    with pytest.raises(CanaryError):
        manager.start_canary(versions[2], percent=10)


def test_cooldown_active(settings):
    import time

    settings.rollback.cooldown_seconds = 60
    assert cooldown_active(settings, time.time() - 10) is True
    assert cooldown_active(settings, time.time() - 120) is False
    assert cooldown_active(settings, None) is False
