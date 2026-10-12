"""FastAPI inference service.

Endpoints:
    GET  /health        liveness
    GET  /ready         readiness (production model loadable)
    GET  /model-info    deployed versions + routing
    POST /predict       single inference (canary-aware, latency-measured)
    POST /batch-predict batched inference
    POST /feedback      delayed ground truth (drives live performance monitoring)
    GET  /metrics       Prometheus exposition
    POST /admin/faults  runtime fault injection (token-protected)
    GET  /admin/faults  current fault state
    POST /admin/reload  force routing/model cache refresh

Every request is measured with ``time.perf_counter`` (no fabricated numbers),
counted in Prometheus, logged as structured JSON and appended to the
prediction log consumed by the monitoring collector.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from sentinel import __version__
from sentinel.config import SentinelSettings, configure_logging, get_logger, log_event
from sentinel.monitoring.metrics import RequestMetrics
from sentinel.serving.faults import FaultInjector, InjectedFaultError
from sentinel.serving.model_loader import ModelRepository, ModelUnavailableError
from sentinel.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
    FaultStateResponse,
    FeedbackRequest,
    HealthResponse,
    ModelInfoEntry,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
    SetFaultsRequest,
)

logger = get_logger("sentinel.serving")


class JsonlAppender:
    """Thread-safe append-only JSONL writer (prediction/feedback logs)."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str)
        with self._lock, self.path.open("a") as fh:
            fh.write(line + "\n")


class ServiceContainer:
    """Composition root for the serving process (explicit dependencies)."""

    def __init__(self, settings: SentinelSettings):
        self.settings = settings
        self.repository = ModelRepository(settings)
        self.metrics = RequestMetrics()
        self.faults = FaultInjector(settings.faults)
        self.prediction_log = JsonlAppender(settings.resolve(settings.serving.prediction_log))
        self.feedback_log = JsonlAppender(settings.resolve(settings.serving.feedback_log))
        self.started_at = time.time()
        self.admin_token: str | None = os.environ.get(settings.serving.admin_token_env) or None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def create_app(settings: SentinelSettings | None = None) -> FastAPI:
    settings = settings or SentinelSettings.load()
    configure_logging(settings.serving.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = app.state.container
        routing = container.repository.routing(force=True)
        log_event(
            logger,
            logging.INFO,
            "sentinel serving started",
            version=__version__,
            env=settings.env,
            production=routing.production,
            canary=routing.canary,
            canary_percent=routing.canary_percent,
            faults=container.faults.state(),
        )
        if routing.production:
            try:
                container.repository.pipeline(routing.production)
            except ModelUnavailableError as exc:
                log_event(logger, logging.ERROR, "production model failed to load", error=str(exc))
        else:
            log_event(logger, logging.WARNING, "no production model deployed yet")
        if container.admin_token is None:
            log_event(
                logger,
                logging.WARNING,
                "admin endpoints unprotected (no token configured) — "
                "acceptable for local demos only",
            )
        yield

    app = FastAPI(
        title="Sentinel Model Server",
        version=__version__,
        description="Canary-aware ML inference service with monitoring hooks.",
        lifespan=lifespan,
    )
    app.state.container = ServiceContainer(settings)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def container(request: Request) -> ServiceContainer:
        return request.app.state.container

    def require_admin(
        request: Request, x_sentinel_admin_token: str | None = Header(default=None)
    ) -> None:
        c = container(request)
        if c.admin_token is None:
            return  # local/dev mode: admin endpoints open (warning logged at boot)
        if x_sentinel_admin_token != c.admin_token:
            raise HTTPException(status_code=403, detail="invalid admin token")

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        c = container(request)
        request_id = None
        try:
            body = await request.json()
            request_id = body.get("request_id")
        except Exception:  # noqa: BLE001 - unparseable body
            pass
        c.metrics.observe_request(
            "none", request.url.path, "validation_error", 0.0, error_type="validation"
        )
        c.prediction_log.append(
            {
                "ts": _utc_now(),
                "request_id": request_id,
                "model_version": "none",
                "served_by": None,
                "status": "validation_error",
                "latency_ms": 0.0,
                "error_type": "validation",
                "detail": str(exc.errors()[:3]),
            }
        )
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="validation_error",
                detail=f"{len(exc.errors())} field validation error(s)",
                request_id=request_id,
            ).model_dump(),
        )

    # ------------------------------------------------------------------ #
    # Health & info
    # ------------------------------------------------------------------ #
    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.get("/ready", response_model=ReadyResponse, tags=["ops"])
    def ready(request: Request) -> ReadyResponse:
        c = container(request)
        routing = c.repository.routing()
        if not routing.production and not routing.canary:
            return ReadyResponse(ready=False, detail="no model deployed")
        try:
            if routing.production:
                c.repository.pipeline(routing.production)
            return ReadyResponse(
                ready=True,
                production_model=routing.production,
                canary_model=routing.canary,
            )
        except ModelUnavailableError as exc:
            return ReadyResponse(
                ready=False,
                production_model=routing.production,
                canary_model=routing.canary,
                detail=str(exc),
            )

    @app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
    def model_info(request: Request) -> ModelInfoResponse:
        c = container(request)
        routing = c.repository.routing(force=True)
        entries: list[ModelInfoEntry] = []
        for version, role in ((routing.production, "production"), (routing.canary, "canary")):
            if not version:
                continue
            try:
                meta = c.repository.metadata(version)
                status = c.repository.registry.get(version).status.value
            except Exception:  # noqa: BLE001 - info endpoint must not fail hard
                continue
            entries.append(
                ModelInfoEntry(
                    version=version,
                    role=role,  # type: ignore[arg-type]
                    status=status,
                    dataset_version=meta.dataset_version,
                    git_commit=meta.git_commit,
                    created_at=meta.created_at,
                    metrics=meta.metrics.get("val", {}),
                )
            )
        c.metrics.set_canary_percent(routing.canary_percent)
        for entry in entries:
            c.metrics.set_model_info(entry.version, entry.role, entry.status)
        return ModelInfoResponse(
            model_name=settings.model.name,
            canary_percent=routing.canary_percent,
            models=entries,
        )

    @app.get("/metrics", tags=["ops"])
    def metrics(request: Request) -> Response:
        c = container(request)
        return Response(
            content=c.metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8"
        )

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _predict_one(c: ServiceContainer, body: PredictRequest) -> PredictResponse:
        start = time.perf_counter()
        try:
            version, served_by = c.repository.select(body.request_id)
        except ModelUnavailableError as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            c.metrics.observe_request(
                "none", "/predict", "error", latency_ms / 1000, error_type="no_model"
            )
            c.prediction_log.append(
                {
                    "ts": _utc_now(),
                    "request_id": body.request_id,
                    "model_version": "none",
                    "served_by": None,
                    "status": "error",
                    "latency_ms": round(latency_ms, 3),
                    "error_type": "no_model",
                }
            )
            raise HTTPException(
                status_code=503,
                detail=ErrorResponse(
                    error="model_unavailable", detail=str(exc), request_id=body.request_id
                ).model_dump(),
            ) from exc

        error_type: str | None = None
        prediction: int | None = None
        probability: float | None = None
        try:
            c.faults.apply()  # fault injection (latency / errors) — measured like real work
            pipeline = c.repository.pipeline(version)
            row = pd.DataFrame([body.features.to_row()])
            prediction = int(pipeline.predict(row)[0])
            probability = float(pipeline.predict_proba(row)[0][1])
        except InjectedFaultError:
            error_type = "injected"
        except Exception as exc:  # noqa: BLE001 - inference failure -> 500 + structured log
            error_type = type(exc).__name__
            log_event(
                logger,
                logging.ERROR,
                "inference failed",
                model_version=version,
                request_id=body.request_id,
                error=str(exc),
            )

        latency_ms = (time.perf_counter() - start) * 1000.0
        status = "ok" if error_type is None else "error"
        c.metrics.observe_request(
            version,
            "/predict",
            status,
            latency_ms / 1000.0,
            prediction=prediction,
            error_type=error_type,
        )
        c.prediction_log.append(
            {
                "ts": _utc_now(),
                "request_id": body.request_id,
                "model_version": version,
                "served_by": served_by,
                "status": status,
                "latency_ms": round(latency_ms, 3),
                "prediction": prediction,
                "probability": round(probability, 6) if probability is not None else None,
                "error_type": error_type,
                "features": body.features.to_row(),
            }
        )

        if error_type is not None:
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    error="prediction_failed",
                    detail=f"{error_type}",
                    request_id=body.request_id,
                ).model_dump(),
            )

        return PredictResponse(
            request_id=body.request_id,
            prediction=prediction,  # type: ignore[arg-type]
            probability=round(probability, 6),  # type: ignore[arg-type]
            model_version=version,
            model_name=settings.model.name,
            served_by=served_by,  # type: ignore[arg-type]
            latency_ms=round(latency_ms, 3),
        )

    @app.post("/predict", response_model=PredictResponse, tags=["inference"])
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        return _predict_one(container(request), body)

    @app.post("/batch-predict", response_model=BatchPredictResponse, tags=["inference"])
    def batch_predict(body: BatchPredictRequest, request: Request) -> BatchPredictResponse:
        c = container(request)
        start = time.perf_counter()
        results: list[PredictResponse] = []
        for item in body.requests:
            results.append(_predict_one(c, item))
        return BatchPredictResponse(
            results=results,
            count=len(results),
            total_latency_ms=round((time.perf_counter() - start) * 1000, 3),
        )

    @app.post("/feedback", status_code=202, tags=["inference"])
    def feedback(body: FeedbackRequest, request: Request) -> dict[str, str]:
        c = container(request)
        c.feedback_log.append(
            {
                "ts": _utc_now(),
                "request_id": body.request_id,
                "ground_truth": body.ground_truth,
            }
        )
        c.metrics.observe_feedback("unknown")
        return {"status": "accepted", "request_id": body.request_id}

    # ------------------------------------------------------------------ #
    # Admin (fault injection / reload)
    # ------------------------------------------------------------------ #
    @app.get("/admin/faults", response_model=FaultStateResponse, tags=["admin"])
    def get_faults(request: Request, _: None = Depends(require_admin)) -> FaultStateResponse:
        return FaultStateResponse(**container(request).faults.state())

    @app.post("/admin/faults", response_model=FaultStateResponse, tags=["admin"])
    def set_faults(
        body: SetFaultsRequest, request: Request, _: None = Depends(require_admin)
    ) -> FaultStateResponse:
        c = container(request)
        state = c.faults.update(
            enabled=body.enabled, latency_ms=body.latency_ms, error_rate=body.error_rate
        )
        log_event(logger, logging.WARNING, "fault injection config updated", **state)
        return FaultStateResponse(**state)

    @app.post("/admin/reload", tags=["admin"])
    def reload_models(request: Request, _: None = Depends(require_admin)) -> dict[str, str]:
        c = container(request)
        c.repository.invalidate()
        routing = c.repository.routing()
        return {
            "status": "reloaded",
            "production": routing.production or "",
            "canary": routing.canary or "",
            "canary_percent": str(routing.canary_percent),
        }

    return app


def run_server(settings: SentinelSettings | None = None) -> None:
    """Entrypoint used by ``sentinel serve`` / Docker."""
    import uvicorn

    settings = settings or SentinelSettings.load()
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.serving.host,
        port=settings.serving.port,
        log_level=settings.serving.log_level,
        # Single process keeps the in-memory Prometheus registry authoritative.
        # Scale horizontally behind a load balancer when multi-worker is needed.
        workers=1,
    )
