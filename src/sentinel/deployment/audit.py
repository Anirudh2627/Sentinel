"""Deployment audit trail.

Every deployment-affecting action (canary start, traffic change, promotion,
rollback, gate decision, manual override) appends an immutable JSONL record.
This is the system's answer to "what happened, when, to which version, why,
based on which metric, against which threshold".
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from sentinel.config import SentinelSettings, get_logger, log_event

logger = get_logger(__name__)

AuditAction = Literal[
    "gate_decision",
    "canary_started",
    "traffic_changed",
    "canary_promoted",
    "canary_rolled_back",
    "production_deployed",
    "production_restored",
    "alert",
    "manual_override",
    "fault_injection",
]


class AuditEntry(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    action: AuditAction
    actor: str = "system"
    model_version: str | None = None
    previous_version: str | None = None
    restored_version: str | None = None
    reason: str = ""
    triggered_metric: str | None = None
    observed_value: float | None = None
    threshold: float | None = None
    canary_percent: int | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class AuditLog:
    """Append-only audit log with a thread-safe writer."""

    def __init__(self, settings: SentinelSettings):
        self.path = settings.resolve(settings.rollback.audit_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, entry: AuditEntry) -> AuditEntry:
        with self._lock, self.path.open("a") as fh:
            fh.write(entry.model_dump_json() + "\n")
        log_event(
            logger,
            logging.INFO if entry.action != "canary_rolled_back" else logging.WARNING,
            "audit event",
            action=entry.action,
            model_version=entry.model_version,
            reason=entry.reason,
            actor=entry.actor,
        )
        return entry

    def read(self, last_n: int | None = None) -> list[AuditEntry]:
        if not self.path.is_file():
            return []
        lines = [ln for ln in self.path.read_text().splitlines() if ln.strip()]
        if last_n is not None:
            lines = lines[-last_n:]
        entries = []
        for ln in lines:
            try:
                entries.append(AuditEntry.model_validate(json.loads(ln)))
            except (json.JSONDecodeError, ValueError):
                continue
        return entries

    def last_rollback_ts(self) -> float | None:
        """Epoch seconds of the most recent rollback (for cooldown checks)."""
        for entry in reversed(self.read()):
            if entry.action in ("canary_rolled_back", "production_restored"):
                try:
                    return datetime.fromisoformat(entry.timestamp).timestamp()
                except ValueError:
                    continue
        return None
