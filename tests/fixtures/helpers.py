"""Test scaffolding helpers (synthetic log records for offline tests)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np

from sentinel.config.settings import SentinelSettings


def write_prediction_log(
    settings: SentinelSettings,
    model_version: str,
    n: int,
    *,
    error_rate: float = 0.0,
    latency_ms: float = 10.0,
    positive_rate: float = 0.4,
    seed: int = 0,
    features: dict | None = None,
    start: datetime | None = None,
) -> None:
    """Append synthetic prediction-log records (test scaffolding, not fake production data)."""
    rng = np.random.default_rng(seed)
    path = settings.resolve(settings.serving.prediction_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = start or datetime.now(UTC) - timedelta(seconds=5)
    with path.open("a") as fh:
        for i in range(n):
            is_error = rng.random() < error_rate
            prediction = int(rng.random() < positive_rate)
            rec = {
                "ts": (start + timedelta(milliseconds=i)).isoformat(),
                "request_id": f"t-{model_version}-{i}",
                "model_version": model_version,
                "served_by": "production",
                "status": "error" if is_error else "ok",
                "latency_ms": float(latency_ms + rng.normal(0, 1.0)),
                "prediction": None if is_error else prediction,
                "probability": None
                if is_error
                else float(np.clip(rng.normal(0.5, 0.2), 0.01, 0.99)),
                "error_type": "injected" if is_error else None,
                "features": features or {},
            }
            fh.write(json.dumps(rec) + "\n")


def write_feedback_log(settings: SentinelSettings, entries: list[tuple[str, int]]) -> None:
    path = settings.resolve(settings.serving.feedback_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for request_id, truth in entries:
            fh.write(
                json.dumps(
                    {
                        "ts": datetime.now(UTC).isoformat(),
                        "request_id": request_id,
                        "ground_truth": truth,
                    }
                )
                + "\n"
            )


def sample_features(**over) -> dict:
    """A valid ChurnFeatures payload (overridable per field)."""
    base = {
        "tenure_months": 8.0,
        "monthly_charges": 62.5,
        "total_charges": 510.0,
        "num_services": 3,
        "support_tickets_30d": 1,
        "avg_session_minutes": 12.5,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method": "Electronic check",
        "promo_active": 0,
        "senior_citizen": 0,
    }
    base.update(over)
    return base
