"""Rollback execution: canary rollback + previous-production restore.

Rollback is always auditable: who/what/when/why, the triggering metric, its
observed value and the configured threshold, and which version traffic was
restored to.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from sentinel.config import SentinelSettings, get_logger, log_event
from sentinel.deployment.audit import AuditEntry, AuditLog
from sentinel.deployment.canary import CanaryError, CanaryManager
from sentinel.monitoring.health import Violation
from sentinel.registry.manager import ModelRegistry

logger = get_logger(__name__)


class RollbackResult(BaseModel):
    success: bool
    action: str  # canary_rolled_back | production_restored | none
    rolled_back_version: str | None = None
    restored_version: str | None = None
    reason: str = ""
    triggered: list[Violation] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    detail: str = ""


class RollbackManager:
    def __init__(
        self,
        settings: SentinelSettings,
        registry: ModelRegistry | None = None,
        canary: CanaryManager | None = None,
        audit: AuditLog | None = None,
    ):
        self.settings = settings
        self.registry = registry or ModelRegistry(settings)
        self.canary = canary or CanaryManager(settings, self.registry)
        self.audit = audit or AuditLog(settings)

    # ------------------------------------------------------------------ #
    def should_rollback(self, violations: list[Violation]) -> list[Violation]:
        """Violations that are eligible to trigger an automatic rollback."""
        return [v for v in violations if v.severity == "critical"]

    def rollback_canary(
        self,
        reason: str,
        violations: list[Violation] | None = None,
        actor: str = "controller",
    ) -> RollbackResult:
        """Pull the canary out of traffic; production resumes 100%."""
        if not self.settings.rollback.enabled:
            return RollbackResult(
                success=False, action="none", reason="rollback disabled in config"
            )
        violations = violations or []
        state = self.canary.current_state()
        if not state.canary:
            return RollbackResult(
                success=False, action="none", reason="no active canary to roll back"
            )

        try:
            self.canary.rollback_canary(reason=reason, actor=actor)
        except CanaryError as exc:
            return RollbackResult(success=False, action="none", reason=str(exc))

        trigger = violations[0] if violations else None
        self.audit.record(
            AuditEntry(
                action="canary_rolled_back",
                actor=actor,
                model_version=state.canary,
                previous_version=state.production,
                restored_version=state.production,
                reason=reason,
                triggered_metric=trigger.kind if trigger else None,
                observed_value=trigger.value if trigger else None,
                threshold=trigger.threshold if trigger else None,
                canary_percent=state.canary_percent,
                context={"violations": [v.model_dump() for v in violations]},
            )
        )
        log_event(
            logger,
            logging.WARNING,
            "canary rolled back",
            canary=state.canary,
            restored=state.production,
            reason=reason,
            violations=len(violations),
        )
        return RollbackResult(
            success=True,
            action="canary_rolled_back",
            rolled_back_version=state.canary,
            restored_version=state.production,
            reason=reason,
            triggered=violations,
        )

    def restore_previous_production(
        self,
        reason: str,
        violations: list[Violation] | None = None,
        actor: str = "controller",
    ) -> RollbackResult:
        """Restore the archived previous production model (production itself is bad)."""
        if not self.settings.rollback.enabled:
            return RollbackResult(
                success=False, action="none", reason="rollback disabled in config"
            )
        violations = violations or []
        state = self.canary.current_state()
        previous = self.registry.previous_production()
        if previous is None:
            return RollbackResult(
                success=False, action="none", reason="no previous production version available"
            )
        current = state.production
        mv = self.registry.restore_previous_production(reason=reason)
        if mv is None:  # pragma: no cover - guarded above
            return RollbackResult(success=False, action="none", reason="restore failed")

        from sentinel.deployment.canary import RoutingState, write_routing_state

        write_routing_state(
            self.settings,
            RoutingState(production=mv.version, canary=None, canary_percent=0, updated_by=actor),
        )
        trigger = violations[0] if violations else None
        self.audit.record(
            AuditEntry(
                action="production_restored",
                actor=actor,
                model_version=current,
                previous_version=current,
                restored_version=mv.version,
                reason=reason,
                triggered_metric=trigger.kind if trigger else None,
                observed_value=trigger.value if trigger else None,
                threshold=trigger.threshold if trigger else None,
                context={"violations": [v.model_dump() for v in violations]},
            )
        )
        return RollbackResult(
            success=True,
            action="production_restored",
            rolled_back_version=current,
            restored_version=mv.version,
            reason=reason,
            triggered=violations,
        )
