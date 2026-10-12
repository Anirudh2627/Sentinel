"""Promotion logic: the quality gate and candidate deployment decisions.

The gate is the *only* path by which a trained run becomes a deployable
candidate (unless an operator explicitly forces it, which is audited).
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

from sentinel.config import SentinelSettings, get_logger, log_event
from sentinel.deployment.audit import AuditEntry, AuditLog
from sentinel.deployment.canary import CanaryManager, RoutingState, cooldown_active
from sentinel.registry.manager import ModelRegistry
from sentinel.training.evaluate import GateResult, run_quality_gate

logger = get_logger(__name__)


class DeployDecision(BaseModel):
    deployed: bool
    mode: str = ""  # production | canary | none
    routing: RoutingState | None = None
    reason: str = ""


class PromotionManager:
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
    def run_gate(self, version: str, force: bool = False, actor: str = "gate") -> GateResult:
        """Evaluate a registered version against configured thresholds."""
        mv = self.registry.get(version)
        candidate_metrics = mv.primary_metrics
        production = self.registry.production()
        production_metrics = (
            production.primary_metrics if production and production.version != version else None
        )

        gate = run_quality_gate(candidate_metrics, self.settings.evaluation, production_metrics)

        if gate.passed:
            self.registry.mark_validated(version, reason=f"gate passed ({gate.primary_metric})")
        elif force:
            log_event(
                logger,
                logging.WARNING,
                "quality gate FORCED — deploying a model that failed evaluation",
                version=version,
                reason=gate.reason,
                actor=actor,
            )
            if mv.status.value == "trained":
                self.registry.mark_validated(version, reason=f"FORCED by {actor}: {gate.reason}")
            self.audit.record(
                AuditEntry(
                    action="manual_override",
                    actor=actor,
                    model_version=version,
                    reason=f"gate forced despite failure: {gate.reason}",
                    context={"gate": gate.model_dump()},
                )
            )
        self.audit.record(
            AuditEntry(
                action="gate_decision",
                actor=actor,
                model_version=version,
                reason=("passed" if gate.passed else gate.reason)
                + (" [FORCED]" if force and not gate.passed else ""),
                context={
                    "passed": gate.passed,
                    "checks": [c.model_dump() for c in gate.checks],
                    "candidate_metrics": candidate_metrics,
                    "production_metrics": production_metrics,
                },
            )
        )
        return gate

    def write_gate_report(self, version: str, gate: GateResult) -> Path:
        out = self.settings.resolve(self.settings.paths.reports_dir) / f"gate_{version}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(gate.to_markdown())
        return out

    # ------------------------------------------------------------------ #
    def deploy_candidate(
        self,
        version: str,
        via_canary: bool = True,
        percent: int | None = None,
        actor: str = "operator",
    ) -> DeployDecision:
        """Deploy a validated candidate — as canary (default) or directly."""
        # Post-rollback cooldown guard
        last_rollback = self.audit.last_rollback_ts()
        if via_canary and cooldown_active(self.settings, last_rollback):
            return DeployDecision(
                deployed=False,
                reason=(
                    f"rollback cooldown active ({self.settings.rollback.cooldown_seconds:.0f}s); "
                    "wait or adjust rollback.cooldown_seconds"
                ),
            )

        mv = self.registry.get(version)
        if mv.status.value not in ("candidate", "archived"):
            return DeployDecision(
                deployed=False,
                reason=(
                    f"{version} is '{mv.status.value}'; "
                    "run the gate first (status must be candidate)"
                ),
            )

        state = self.canary.current_state()
        if state.production is None or not via_canary:
            routing = self.canary.deploy_production(version, actor=actor)
            self.audit.record(
                AuditEntry(
                    action="production_deployed",
                    actor=actor,
                    model_version=version,
                    previous_version=state.production,
                    restored_version=version,
                    reason="direct production deployment (no baseline or canary disabled)",
                )
            )
            return DeployDecision(deployed=True, mode="production", routing=routing)

        if not self.settings.canary.enabled:
            return DeployDecision(deployed=False, reason="canary disabled in configuration")

        routing = self.canary.start_canary(version, percent=percent, actor=actor)
        self.audit.record(
            AuditEntry(
                action="canary_started",
                actor=actor,
                model_version=version,
                previous_version=routing.production,
                canary_percent=routing.canary_percent,
                reason=f"canary started at {routing.canary_percent}% traffic",
            )
        )
        return DeployDecision(deployed=True, mode="canary", routing=routing)
