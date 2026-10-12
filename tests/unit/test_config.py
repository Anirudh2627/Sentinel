"""Unit tests: configuration loading, overrides and validation."""

from __future__ import annotations

import pytest

from sentinel.config.settings import ConfigError, SentinelSettings


def test_load_development_defaults():
    s = SentinelSettings.load("development")
    assert s.env == "development"
    assert s.model.name == "sentinel-churn"
    assert 0 < s.canary.initial_traffic_percent <= 100
    assert s.canary.promotion_steps == sorted(s.canary.promotion_steps)


def test_load_production_profile_overrides():
    s = SentinelSettings.load("production")
    assert s.env == "production"
    # production profile defines stricter SLOs and a longer ramp
    assert s.monitoring.max_error_rate == 0.02
    assert 10 in s.canary.promotion_steps


def test_unknown_env_raises():
    with pytest.raises(ConfigError):
        SentinelSettings.load("does-not-exist")


def test_env_var_override_nested(monkeypatch):
    monkeypatch.setenv("SENTINEL_MONITORING__MAX_ERROR_RATE", "0.42")
    monkeypatch.setenv("SENTINEL_CANARY__INITIAL_TRAFFIC_PERCENT", "7")
    monkeypatch.setenv("SENTINEL_FAULTS__ENABLED", "true")
    s = SentinelSettings.load("development")
    assert s.monitoring.max_error_rate == 0.42
    assert s.canary.initial_traffic_percent == 7
    assert s.faults.enabled is True


def test_env_var_override_deep_nested(monkeypatch):
    monkeypatch.setenv("SENTINEL_MONITORING__DRIFT__PSI_THRESHOLD", "0.11")
    s = SentinelSettings.load("development")
    assert s.monitoring.drift.psi_threshold == 0.11


def test_resolve_paths_relative_to_root(tmp_path):
    s = SentinelSettings.load("development")
    s.paths.root_dir = tmp_path
    resolved = s.resolve(s.serving.prediction_log)
    assert resolved.is_absolute()
    assert str(resolved).startswith(str(tmp_path))


def test_invalid_promotion_steps_rejected():
    from sentinel.config.settings import CanaryConfig

    with pytest.raises(ValueError):
        CanaryConfig(promotion_steps=[50, 25])
    with pytest.raises(ValueError):
        CanaryConfig(promotion_steps=[])


def test_summary_is_serialisable_and_redacted():
    import json

    s = SentinelSettings.load("development")
    dumped = json.dumps(s.summary())
    assert "config_dir" not in json.loads(dumped)


def test_port_validation():
    from sentinel.config.settings import ServingConfig

    with pytest.raises(ValueError):
        ServingConfig(port=99999)
