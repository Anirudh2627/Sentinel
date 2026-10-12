"""Registry domain model: model versions, lifecycle states, audit history."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ModelStatus(StrEnum):
    """Model lifecycle states.

    TRAINED → VALIDATED → CANDIDATE → CANARY → PRODUCTION → ARCHIVED
    (plus ROLLED_BACK as an auditable terminal-ish state).
    """

    TRAINED = "trained"
    VALIDATED = "validated"
    CANDIDATE = "candidate"
    CANARY = "canary"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    ROLLED_BACK = "rolled_back"


ALLOWED_TRANSITIONS: dict[ModelStatus, set[ModelStatus]] = {
    ModelStatus.TRAINED: {ModelStatus.VALIDATED, ModelStatus.ARCHIVED},
    ModelStatus.VALIDATED: {ModelStatus.CANDIDATE, ModelStatus.ARCHIVED},
    ModelStatus.CANDIDATE: {ModelStatus.CANARY, ModelStatus.PRODUCTION, ModelStatus.ARCHIVED},
    ModelStatus.CANARY: {
        ModelStatus.PRODUCTION,
        ModelStatus.ROLLED_BACK,
        ModelStatus.CANDIDATE,  # withdraw canary without fault
        ModelStatus.ARCHIVED,
    },
    ModelStatus.PRODUCTION: {ModelStatus.ARCHIVED, ModelStatus.ROLLED_BACK},
    ModelStatus.ROLLED_BACK: {ModelStatus.ARCHIVED},
    ModelStatus.ARCHIVED: {ModelStatus.CANDIDATE, ModelStatus.PRODUCTION},  # restore paths
}


class StatusChange(BaseModel):
    """One entry in a version's immutable status history (audit trail)."""

    from_status: ModelStatus | None
    to_status: ModelStatus
    at: str
    reason: str = ""
    actor: str = "system"


class ModelVersion(BaseModel):
    """A first-class, fully traceable model version."""

    version: str  # v1, v2, ...
    run_id: str
    model_name: str
    status: ModelStatus = ModelStatus.TRAINED
    artifact_path: str  # repo-relative directory containing model.joblib etc.
    dataset_version: str
    git_commit: str | None = None
    git_dirty: bool | None = None
    code_version: str = ""
    scenario: str = "default"
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, dict[str, float]] = Field(default_factory=dict)  # {"val":..., "test":...}
    mlflow_ref: str | None = None
    history: list[StatusChange] = Field(default_factory=list)

    @property
    def primary_metrics(self) -> dict[str, float]:
        return dict(self.metrics.get("val", {}))


class RegistryIndex(BaseModel):
    """Persisted registry state (single JSON document, atomic writes)."""

    schema_version: int = 1
    next_version_number: int = 1
    versions: dict[str, ModelVersion] = Field(default_factory=dict)
    current_production: str | None = None
    previous_production: str | None = None
    current_canary: str | None = None
    current_candidate: str | None = None

    def ordered_versions(self) -> list[ModelVersion]:
        return sorted(self.versions.values(), key=lambda v: int(v.version.lstrip("v")))


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()
