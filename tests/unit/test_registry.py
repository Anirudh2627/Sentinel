"""Unit tests: registry state machine, pointers, persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentinel.registry.manager import (
    InvalidTransitionError,
    ModelNotFoundError,
    ModelRegistry,
    RegistryError,
)
from sentinel.registry.metadata import ModelStatus


@pytest.fixture()
def registry(settings) -> ModelRegistry:
    return ModelRegistry(settings)


@pytest.fixture()
def version(registry, trained_run) -> str:
    mv = registry.register_run(Path(trained_run.run_dir))
    return mv.version


def test_register_creates_v1_trained(registry, version):
    mv = registry.get(version)
    assert version == "v1"
    assert mv.status == ModelStatus.TRAINED
    assert mv.artifact_path.endswith("v1")
    # artifacts were copied into the registry
    artifact = registry.artifact_dir(version)
    assert (artifact / "model.joblib").is_file()
    assert (artifact / "metadata.json").is_file()
    assert (artifact / "reference_stats.json").is_file()


def test_register_increments_versions(registry, trained_run):
    v1 = registry.register_run(Path(trained_run.run_dir))
    v2 = registry.register_run(Path(trained_run.run_dir))
    assert (v1.version, v2.version) == ("v1", "v2")


def test_metadata_traceability(registry, version, trained_run):
    mv = registry.get(version)
    assert mv.dataset_version == trained_run.dataset_version
    assert mv.run_id == trained_run.run_id
    assert mv.code_version  # package version recorded
    assert "val" in mv.metrics and "f1" in mv.metrics["val"]
    assert mv.history[0].to_status == ModelStatus.TRAINED


def test_valid_transition_chain(registry, version):
    registry.set_status(version, ModelStatus.VALIDATED, reason="test")
    registry.set_status(version, ModelStatus.CANDIDATE, reason="test")
    registry.set_status(version, ModelStatus.CANARY, reason="test")
    registry.set_status(version, ModelStatus.PRODUCTION, reason="test")
    assert registry.get(version).status == ModelStatus.PRODUCTION
    assert registry.production().version == version


def test_invalid_transition_rejected(registry, version):
    with pytest.raises(InvalidTransitionError):
        registry.set_status(version, ModelStatus.PRODUCTION)  # TRAINED -> PRODUCTION illegal


def test_history_is_append_only_audit(registry, version):
    registry.set_status(version, ModelStatus.VALIDATED, reason="r1", actor="tester")
    registry.set_status(version, ModelStatus.CANDIDATE, reason="r2", actor="tester")
    history = registry.get(version).history
    assert [h.to_status for h in history] == [
        ModelStatus.TRAINED,
        ModelStatus.VALIDATED,
        ModelStatus.CANDIDATE,
    ]
    assert history[-1].reason == "r2"
    assert history[-1].actor == "tester"


def test_promotion_archives_previous_production(registry, trained_run):
    v1 = registry.register_run(Path(trained_run.run_dir))
    v2 = registry.register_run(Path(trained_run.run_dir))
    registry.mark_validated(v1.version)
    registry.mark_validated(v2.version)
    registry.promote_to_production(v1.version, reason="first")
    registry.promote_to_production(v2.version, reason="second")

    assert registry.production().version == "v2"
    assert registry.previous_production().version == "v1"
    assert registry.get("v1").status == ModelStatus.ARCHIVED


def test_canary_pointers(registry, trained_run):
    v1 = registry.register_run(Path(trained_run.run_dir))
    v2 = registry.register_run(Path(trained_run.run_dir))
    registry.mark_validated(v1.version)
    registry.mark_validated(v2.version)
    registry.promote_to_production(v1.version)
    registry.start_canary(v2.version)
    assert registry.canary().version == "v2"
    registry.promote_to_production(v2.version, reason="promoted from canary")
    assert registry.canary() is None
    assert registry.production().version == "v2"


def test_only_one_canary_at_a_time(registry, trained_run):
    v1 = registry.register_run(Path(trained_run.run_dir))
    v2 = registry.register_run(Path(trained_run.run_dir))
    v3 = registry.register_run(Path(trained_run.run_dir))
    registry.mark_validated(v1.version)
    registry.promote_to_production(v1.version)
    registry.mark_validated(v2.version)
    registry.mark_validated(v3.version)
    registry.start_canary(v2.version)
    with pytest.raises(RegistryError):
        registry.start_canary(v3.version)


def test_rollback_canary(registry, trained_run):
    v1 = registry.register_run(Path(trained_run.run_dir))
    v2 = registry.register_run(Path(trained_run.run_dir))
    registry.mark_validated(v1.version)
    registry.promote_to_production(v1.version)
    registry.mark_validated(v2.version)
    registry.start_canary(v2.version)
    registry.rollback_canary(v2.version, reason="bad metrics")
    assert registry.get("v2").status == ModelStatus.ROLLED_BACK
    assert registry.canary() is None
    assert registry.production().version == "v1"


def test_restore_previous_production(registry, trained_run):
    v1 = registry.register_run(Path(trained_run.run_dir))
    v2 = registry.register_run(Path(trained_run.run_dir))
    registry.mark_validated(v1.version)
    registry.mark_validated(v2.version)
    registry.promote_to_production(v1.version)
    registry.promote_to_production(v2.version)
    restored = registry.restore_previous_production(reason="v2 is bad in prod")
    assert restored.version == "v1"
    assert registry.production().version == "v1"
    assert registry.get("v2").status == ModelStatus.ROLLED_BACK


def test_restore_without_previous_returns_none(registry, trained_run):
    v1 = registry.register_run(Path(trained_run.run_dir))
    registry.mark_validated(v1.version)
    registry.promote_to_production(v1.version)
    assert registry.restore_previous_production(reason="nothing before") is None


def test_index_persists_to_disk(settings, registry, version):
    path = settings.resolve(settings.registry.index_path)
    assert path.is_file()
    data = json.loads(path.read_text())
    assert data["versions"][version]["status"] == "trained"


def test_unknown_version_raises(registry):
    with pytest.raises(ModelNotFoundError):
        registry.get("v999")
