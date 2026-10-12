"""Request metrics: Prometheus exposition + rolling window statistics.

The serving layer records every request into both:
    * a ``prometheus_client`` registry (scraped at ``/metrics``), and
    * the JSONL prediction log (consumed by the collector/controller).

A dedicated ``CollectorRegistry`` per instance keeps tests isolated and
avoids global mutable state.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

LATENCY_BUCKETS_S = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


class LatencyStats(BaseModel):
    p50: float
    p95: float
    p99: float
    mean: float
    max: float


class WindowStats(BaseModel):
    """Aggregated stats for one model version inside one monitoring window."""

    model_version: str
    n_requests: int
    n_errors: int
    error_rate: float
    throughput_rps: float
    latency_ms: LatencyStats | None = None
    positive_rate: float | None = None  # share of predictions == 1
    mean_probability: float | None = None
    n_predictions: int = 0
    window_start: str = ""
    window_end: str = ""


class PerformanceStats(BaseModel):
    """Delayed-ground-truth performance for one model version."""

    model_version: str
    n_labels: int
    roc_auc: float | None = None
    accuracy: float | None = None
    log_loss: float | None = None
    true_positive_rate: float | None = None


def percentile(values: list[float] | np.ndarray, q: float) -> float:
    """q in [0, 100]."""
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def compute_window_stats(
    model_version: str,
    records: list[dict[str, Any]],
    window_seconds: float,
) -> WindowStats:
    """Aggregate raw prediction-log records into window statistics."""
    n = len(records)
    errors = [r for r in records if r.get("status") != "ok"]
    latencies = [float(r["latency_ms"]) for r in records if r.get("latency_ms") is not None]
    preds = [r for r in records if r.get("prediction") is not None]
    probs = [float(r["probability"]) for r in preds if r.get("probability") is not None]
    positive = [p for p in preds if int(p["prediction"]) == 1]

    latency = None
    if latencies:
        latency = LatencyStats(
            p50=round(percentile(latencies, 50), 3),
            p95=round(percentile(latencies, 95), 3),
            p99=round(percentile(latencies, 99), 3),
            mean=round(float(np.mean(latencies)), 3),
            max=round(float(np.max(latencies)), 3),
        )

    timestamps = [r.get("ts") for r in records if r.get("ts")]
    window_start = min(timestamps) if timestamps else datetime.now(UTC).isoformat()
    window_end = max(timestamps) if timestamps else datetime.now(UTC).isoformat()

    return WindowStats(
        model_version=model_version,
        n_requests=n,
        n_errors=len(errors),
        error_rate=round(len(errors) / n, 6) if n else 0.0,
        throughput_rps=round(n / window_seconds, 4) if window_seconds > 0 else 0.0,
        latency_ms=latency,
        positive_rate=round(len(positive) / len(preds), 6) if preds else None,
        mean_probability=round(float(np.mean(probs)), 6) if probs else None,
        n_predictions=len(preds),
        window_start=window_start,
        window_end=window_end,
    )


class RequestMetrics:
    """Prometheus collectors for the serving layer."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.requests_total = Counter(
            "sentinel_requests_total",
            "Total inference requests",
            labelnames=("model_version", "endpoint", "status"),
            registry=self.registry,
        )
        self.errors_total = Counter(
            "sentinel_errors_total",
            "Total errors by type",
            labelnames=("model_version", "error_type"),
            registry=self.registry,
        )
        self.latency_seconds = Histogram(
            "sentinel_request_latency_seconds",
            "End-to-end request latency (seconds)",
            labelnames=("model_version", "endpoint"),
            buckets=LATENCY_BUCKETS_S,
            registry=self.registry,
        )
        self.predictions_total = Counter(
            "sentinel_predictions_total",
            "Predictions by class",
            labelnames=("model_version", "prediction"),
            registry=self.registry,
        )
        self.feedback_total = Counter(
            "sentinel_feedback_total",
            "Delayed ground-truth feedback received",
            labelnames=("model_version",),
            registry=self.registry,
        )
        self.model_info = Gauge(
            "sentinel_model_info",
            "Model version metadata (1 = current)",
            labelnames=("version", "role", "status"),
            registry=self.registry,
        )
        self.canary_percent = Gauge(
            "sentinel_canary_traffic_percent",
            "Current canary traffic percentage",
            registry=self.registry,
        )

    def observe_request(
        self,
        model_version: str,
        endpoint: str,
        status: str,
        latency_s: float,
        prediction: int | None = None,
        error_type: str | None = None,
    ) -> None:
        self.requests_total.labels(model_version, endpoint, status).inc()
        self.latency_seconds.labels(model_version, endpoint).observe(latency_s)
        if prediction is not None:
            self.predictions_total.labels(model_version, str(prediction)).inc()
        if error_type:
            self.errors_total.labels(model_version, error_type).inc()

    def observe_feedback(self, model_version: str) -> None:
        self.feedback_total.labels(model_version).inc()

    def set_model_info(self, version: str, role: str, status: str) -> None:
        self.model_info.labels(version, role, status).set(1)

    def set_canary_percent(self, percent: float) -> None:
        self.canary_percent.set(percent)

    def render(self) -> bytes:
        return generate_latest(self.registry)
