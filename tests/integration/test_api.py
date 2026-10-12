"""Integration tests: the FastAPI serving layer (in-process TestClient)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from sentinel.deployment.canary import CanaryManager
from sentinel.serving.app import create_app
from tests.fixtures.helpers import sample_features


@pytest.fixture()
def client(settings, registered_version):
    mv, _gate = registered_version
    settings.serving.routing_reload_interval_seconds = 0.0  # immediate routing refresh in tests
    CanaryManager(settings).deploy_production(mv.version)
    app = create_app(settings)
    with TestClient(app) as c:
        c.sentinel_version = mv.version  # type: ignore[attr-defined]
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_after_deploy(client):
    r = client.get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["ready"] is True
    assert body["production_model"] == client.sentinel_version


def test_ready_without_model(settings):
    settings.serving.routing_reload_interval_seconds = 0.0
    app = create_app(settings)
    with TestClient(app) as c:
        r = c.get("/ready")
        assert r.json()["ready"] is False


def test_predict_ok(client):
    r = client.post("/predict", json={"request_id": "r-1", "features": sample_features()})
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == client.sentinel_version
    assert body["served_by"] == "production"
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["latency_ms"] > 0.0  # measured, not fabricated
    assert body["request_id"] == "r-1"


def test_predict_rejects_invalid_features(client):
    bad = sample_features(tenure_months=500)  # domain bound is 72
    r = client.post("/predict", json={"request_id": "r-bad", "features": bad})
    assert r.status_code == 422
    assert r.json()["error"] == "validation_error"


def test_predict_rejects_unknown_category(client):
    bad = sample_features(contract_type="Lifetime")
    r = client.post("/predict", json={"request_id": "r-bad2", "features": bad})
    assert r.status_code == 422


def test_prediction_log_written(client, settings):
    client.post("/predict", json={"request_id": "r-log", "features": sample_features()})
    path = settings.resolve(settings.serving.prediction_log)
    assert path.is_file()
    records = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    rec = next(r for r in records if r["request_id"] == "r-log")
    assert rec["status"] == "ok"
    assert rec["model_version"] == client.sentinel_version
    assert rec["features"]["contract_type"] == "Month-to-month"
    assert rec["latency_ms"] >= 0.0


def test_validation_error_is_logged(client, settings):
    client.post(
        "/predict", json={"request_id": "r-val", "features": sample_features(num_services=99)}
    )
    records = [
        json.loads(ln)
        for ln in settings.resolve(settings.serving.prediction_log).read_text().splitlines()
        if ln.strip()
    ]
    rec = next(r for r in records if r.get("request_id") == "r-val")
    assert rec["status"] == "validation_error"


def test_batch_predict(client):
    payload = {
        "requests": [
            {"request_id": f"b-{i}", "features": sample_features(tenure_months=float(i))}
            for i in range(5)
        ]
    }
    r = client.post("/batch-predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 5
    assert len(body["results"]) == 5
    assert body["total_latency_ms"] > 0


def test_feedback_accepted(client, settings):
    client.post("/predict", json={"request_id": "r-fb", "features": sample_features()})
    r = client.post("/feedback", json={"request_id": "r-fb", "ground_truth": 1})
    assert r.status_code == 202
    lines = settings.resolve(settings.serving.feedback_log).read_text().splitlines()
    assert any("r-fb" in ln for ln in lines)


def test_feedback_rejects_invalid_truth(client):
    r = client.post("/feedback", json={"request_id": "x", "ground_truth": 3})
    assert r.status_code == 422


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["models"][0]["role"] == "production"
    assert body["models"][0]["version"] == client.sentinel_version
    assert body["models"][0]["metrics"]  # offline metrics exposed


def test_metrics_endpoint_prometheus_format(client):
    for i in range(3):
        client.post("/predict", json={"request_id": f"m-{i}", "features": sample_features()})
    r = client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    assert "sentinel_requests_total" in text
    assert f'model_version="{client.sentinel_version}"' in text
    assert "sentinel_request_latency_seconds_bucket" in text


def test_fault_injection_error_rate(client):
    r = client.post("/admin/faults", json={"enabled": True, "error_rate": 1.0, "latency_ms": 0})
    assert r.status_code == 200
    r = client.post("/predict", json={"request_id": "f-1", "features": sample_features()})
    assert r.status_code == 500
    assert r.json()["detail"]["error"] == "prediction_failed"
    # disable again
    client.post("/admin/faults", json={"enabled": False, "error_rate": 0.0})
    r = client.post("/predict", json={"request_id": "f-2", "features": sample_features()})
    assert r.status_code == 200


def test_fault_injection_latency_is_measured(client):
    client.post("/admin/faults", json={"enabled": True, "latency_ms": 120, "error_rate": 0.0})
    r = client.post("/predict", json={"request_id": "f-3", "features": sample_features()})
    assert r.status_code == 200
    assert r.json()["latency_ms"] >= 80  # 120ms ± jitter — real measurement
    client.post("/admin/faults", json={"enabled": False})


def test_admin_token_enforced(settings, registered_version, monkeypatch):
    mv, _ = registered_version
    monkeypatch.setenv("SENTINEL_SERVING__ADMIN_TOKEN", "s3cret-token")
    settings.serving.routing_reload_interval_seconds = 0.0
    CanaryManager(settings).deploy_production(mv.version)
    app = create_app(settings)
    with TestClient(app) as c:
        assert c.post("/admin/faults", json={"enabled": True, "error_rate": 0.1}).status_code == 403
        r = c.post(
            "/admin/faults",
            json={"enabled": True, "error_rate": 0.1},
            headers={"X-Sentinel-Admin-Token": "s3cret-token"},
        )
        assert r.status_code == 200
        # public endpoints stay open
        assert c.get("/health").status_code == 200


def test_admin_reload(client):
    r = client.post("/admin/reload")
    assert r.status_code == 200
    assert r.json()["status"] == "reloaded"


def test_canary_routing_via_api(settings, trained_run):
    """With canary at 100%, every request must be served by the canary."""
    from pathlib import Path

    from sentinel.registry.manager import ModelRegistry

    settings.serving.routing_reload_interval_seconds = 0.0
    registry = ModelRegistry(settings)
    v1 = registry.register_run(Path(trained_run.run_dir))
    v2 = registry.register_run(Path(trained_run.run_dir))
    registry.mark_validated(v1.version)
    registry.mark_validated(v2.version)
    manager = CanaryManager(settings, registry)
    manager.deploy_production(v1.version)
    manager.start_canary(v2.version, percent=100)

    app = create_app(settings)
    with TestClient(app) as c:
        r = c.post("/predict", json={"request_id": "c-1", "features": sample_features()})
        body = r.json()
        assert body["served_by"] == "canary"
        assert body["model_version"] == v2.version
