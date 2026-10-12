"""API request/response schemas (Pydantic = strict input validation)."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field

from sentinel.data.ingestion import CONTRACT_TYPES, INTERNET_SERVICES, PAYMENT_METHODS


class ChurnFeatures(BaseModel):
    """One inference request's raw feature vector.

    Field constraints mirror the training-time schema bounds; malformed or
    out-of-range input is rejected with HTTP 422 *before* touching the model.
    """

    tenure_months: float = Field(ge=0, le=72)
    monthly_charges: float = Field(ge=0, le=1000)
    total_charges: float = Field(ge=0, le=100_000)
    num_services: int = Field(ge=0, le=10)
    support_tickets_30d: int = Field(ge=0, le=50)
    avg_session_minutes: float = Field(ge=0, le=1440)
    contract_type: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["DSL", "Fiber optic", "No"]
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    promo_active: Literal[0, 1]
    senior_citizen: Literal[0, 1]

    def to_row(self) -> dict:
        return self.model_dump()


class PredictRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    features: ChurnFeatures


class PredictResponse(BaseModel):
    request_id: str
    prediction: int
    probability: float
    model_version: str
    model_name: str
    served_by: Literal["production", "canary"]
    latency_ms: float


class BatchPredictRequest(BaseModel):
    requests: list[PredictRequest] = Field(min_length=1, max_length=128)


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    count: int
    total_latency_ms: float


class FeedbackRequest(BaseModel):
    """Delayed ground truth for a previous prediction (request_id join)."""

    request_id: str
    ground_truth: Literal[0, 1]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str = "sentinel-serving"
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    production_model: str | None = None
    canary_model: str | None = None
    detail: str = ""


class ModelInfoEntry(BaseModel):
    version: str
    role: Literal["production", "canary"]
    status: str
    dataset_version: str
    git_commit: str | None
    created_at: str
    metrics: dict[str, float] = Field(default_factory=dict)


class ModelInfoResponse(BaseModel):
    model_name: str
    canary_percent: int
    models: list[ModelInfoEntry] = Field(default_factory=list)


class FaultStateResponse(BaseModel):
    enabled: bool
    latency_ms: float
    error_rate: float


class SetFaultsRequest(BaseModel):
    enabled: bool | None = None
    latency_ms: float | None = Field(default=None, ge=0, le=60_000)
    error_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
    request_id: str | None = None


# Convenience constants re-exported for the simulator
CONTRACT_TYPE_LEVELS = CONTRACT_TYPES
INTERNET_SERVICE_LEVELS = INTERNET_SERVICES
PAYMENT_METHOD_LEVELS = PAYMENT_METHODS
