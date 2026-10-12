"""Production traffic simulator.

Generates labelled synthetic requests for a chosen scenario and sends them to
a running Sentinel server, optionally posting delayed ground-truth feedback
(which powers live performance monitoring). All reported numbers come from
actual server responses — the simulator never invents metrics.

Transport is pluggable so tests can drive a FastAPI ``TestClient`` in-process.
"""

from __future__ import annotations

import logging
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from sentinel.config import SentinelSettings, get_logger, log_event
from sentinel.data.ingestion import generate_synthetic_churn
from sentinel.simulation.scenarios import ScenarioSpec, get_scenario

logger = get_logger(__name__)

FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "contract_type",
    "internet_service",
    "payment_method",
    "num_services",
    "support_tickets_30d",
    "avg_session_minutes",
    "promo_active",
    "senior_citizen",
]


class HttpResponseLike(Protocol):
    status_code: int

    def json(self) -> dict[str, Any]: ...


class Transport(Protocol):
    def post(self, path: str, json: dict[str, Any]) -> HttpResponseLike: ...


class HttpxTransport:
    def __init__(self, base_url: str, timeout: float = 10.0):
        self.client = httpx.Client(base_url=base_url, timeout=timeout)

    def post(self, path: str, json: dict[str, Any]) -> HttpResponseLike:
        return self.client.post(path, json=json)

    def close(self) -> None:
        self.client.close()


class TestClientTransport:
    """Adapter for fastapi.testclient.TestClient (used in tests)."""

    def __init__(self, client: Any):
        self.client = client

    def post(self, path: str, json: dict[str, Any]) -> HttpResponseLike:
        return self.client.post(path, json=json)

    def close(self) -> None:
        pass


class SimulationReport(BaseModel):
    scenario: str
    n_sent: int = 0
    n_ok: int = 0
    n_errors: int = 0
    n_validation_errors: int = 0
    n_feedback_sent: int = 0
    duration_seconds: float = 0.0
    requests_per_second: float = 0.0
    server_latency_ms: dict[str, float] = Field(default_factory=dict)
    served_by_version: dict[str, int] = Field(default_factory=dict)
    prediction_positive_rate: float | None = None


def generate_scenario_samples(
    scenario: ScenarioSpec,
    n: int,
    seed: int,
) -> pd.DataFrame:
    """Generate labelled feature samples for a scenario."""
    df = generate_synthetic_churn(
        n_samples=n,
        seed=seed,
        drift_spec=scenario.drift_spec or None,
        concept_drift=scenario.concept_drift,
    )
    for col, value in scenario.feature_overrides.items():
        if col in df.columns:
            df[col] = value
    return df


@dataclass
class _Counters:
    lock: Any = field(default_factory=lambda: __import__("threading").Lock())
    sent: int = 0
    ok: int = 0
    errors: int = 0
    validation_errors: int = 0
    feedback: int = 0
    latencies: list[float] = field(default_factory=list)
    versions: dict[str, int] = field(default_factory=dict)
    positives: int = 0
    predictions: int = 0


class TrafficSimulator:
    def __init__(
        self,
        settings: SentinelSettings,
        scenario: str = "normal",
        transport: Transport | None = None,
        requests_per_second: float | None = None,
        duration_seconds: float | None = None,
        concurrency: int | None = None,
        feedback_rate: float | None = None,
        seed: int | None = None,
        max_requests: int | None = None,
        progress_every: int = 100,
    ):
        self.settings = settings
        self.scenario = get_scenario(scenario)
        sim = settings.simulation
        self.transport = transport or HttpxTransport(sim.base_url)
        self.rps = (requests_per_second or sim.requests_per_second) * self.scenario.rate_multiplier
        self.duration = duration_seconds if duration_seconds is not None else sim.duration_seconds
        self.concurrency = concurrency or sim.concurrency
        self.feedback_rate = sim.feedback_rate if feedback_rate is None else feedback_rate
        self.seed = sim.seed if seed is None else seed
        self.max_requests = max_requests
        self.progress_every = progress_every

    # ------------------------------------------------------------------ #
    def run(self) -> SimulationReport:
        n_total = int(self.rps * self.duration)
        if self.max_requests is not None:
            n_total = min(n_total, self.max_requests)
        n_total = max(n_total, 1)

        samples = generate_scenario_samples(self.scenario, n_total, self.seed)
        # random.Random (not numpy Generator): shared across worker threads.
        rng = random.Random(self.seed + 1)
        counters = _Counters()

        log_event(
            logger,
            logging.INFO,
            "simulation starting",
            scenario=self.scenario.name,
            requests=n_total,
            rps=round(self.rps, 2),
            concurrency=self.concurrency,
            feedback_rate=self.feedback_rate,
        )

        start = time.monotonic()
        interval = 1.0 / self.rps if self.rps > 0 else 0.0

        def send(i: int) -> None:
            row = samples.iloc[i]
            request_id = uuid.uuid4().hex
            payload = {
                "request_id": request_id,
                "features": _row_to_features(row),
            }
            try:
                resp = self.transport.post("/predict", payload)
            except Exception as exc:  # connection errors etc.
                with counters.lock:
                    counters.sent += 1
                    counters.errors += 1
                log_event(logger, logging.DEBUG, "request failed", error=str(exc))
                return
            with counters.lock:
                counters.sent += 1
                if resp.status_code == 200:
                    counters.ok += 1
                    body = resp.json()
                    counters.latencies.append(float(body.get("latency_ms", 0.0)))
                    version = str(body.get("model_version", "unknown"))
                    counters.versions[version] = counters.versions.get(version, 0) + 1
                    counters.predictions += 1
                    counters.positives += int(body.get("prediction", 0) == 1)
                elif resp.status_code == 422:
                    counters.validation_errors += 1
                else:
                    counters.errors += 1
                if counters.sent % self.progress_every == 0:
                    elapsed = time.monotonic() - start
                    log_event(
                        logger,
                        logging.INFO,
                        "simulation progress",
                        sent=counters.sent,
                        total=n_total,
                        ok=counters.ok,
                        errors=counters.errors,
                        elapsed_s=round(elapsed, 1),
                    )

            # delayed ground-truth feedback (partial, like real label pipelines)
            if rng.random() < self.feedback_rate:
                truth = int(row["churned"])
                try:
                    fb = self.transport.post(
                        "/feedback", {"request_id": request_id, "ground_truth": truth}
                    )
                    if fb.status_code in (200, 202):
                        with counters.lock:
                            counters.feedback += 1
                except Exception:  # noqa: BLE001 - feedback is best-effort
                    pass

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            for i in range(n_total):
                target = start + i * interval
                delay = target - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                pool.submit(send, i)

        elapsed = time.monotonic() - start
        latencies = np.asarray(counters.latencies, dtype=float)
        report = SimulationReport(
            scenario=self.scenario.name,
            n_sent=counters.sent,
            n_ok=counters.ok,
            n_errors=counters.errors,
            n_validation_errors=counters.validation_errors,
            n_feedback_sent=counters.feedback,
            duration_seconds=round(elapsed, 3),
            requests_per_second=round(counters.sent / elapsed, 3) if elapsed > 0 else 0.0,
            server_latency_ms=(
                {
                    "mean": round(float(latencies.mean()), 3),
                    "p50": round(float(np.percentile(latencies, 50)), 3),
                    "p95": round(float(np.percentile(latencies, 95)), 3),
                    "p99": round(float(np.percentile(latencies, 99)), 3),
                }
                if len(latencies)
                else {}
            ),
            served_by_version=dict(counters.versions),
            prediction_positive_rate=(
                round(counters.positives / counters.predictions, 4)
                if counters.predictions
                else None
            ),
        )
        log_event(
            logger,
            logging.INFO,
            "simulation finished",
            scenario=report.scenario,
            sent=report.n_sent,
            ok=report.n_ok,
            errors=report.n_errors,
            rps=report.requests_per_second,
            versions=report.served_by_version,
        )
        return report


def _row_to_features(row: pd.Series) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for col in FEATURE_COLUMNS:
        value = row[col]
        if col in ("num_services", "support_tickets_30d", "promo_active", "senior_citizen"):
            features[col] = int(value)
        elif isinstance(value, (float, np.floating)):
            features[col] = round(float(value), 2)
        elif isinstance(value, (int, np.integer)):
            features[col] = float(value) if col in ("tenure_months",) else int(value)
        else:
            features[col] = str(value)
    return features
