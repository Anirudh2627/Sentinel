"""Deployment controller: the automated promote / rollback / alert loop.

    collect monitoring snapshot
              ↓
    evaluate policy violations (thresholds from config)
              ↓
    critical violation on the canary?  → ROLLBACK (audited)
    healthy window?                    → count toward next promotion step
    enough healthy windows at step?    → raise traffic (5→25→50→100)
    healthy at 100%?                   → PROMOTE to production
    production violating SLOs?         → alert (optionally restore previous)

The controller is a separate process from the server: it communicates only
through the filesystem artifacts (prediction log, routing state, registry,
audit log), which makes it testable and crash-safe.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from sentinel.config import SentinelSettings, get_logger, log_event
from sentinel.deployment.audit import AuditEntry, AuditLog
from sentinel.deployment.canary import CanaryError, CanaryManager
from sentinel.deployment.rollback import RollbackManager, RollbackResult
from sentinel.monitoring.collector import MetricsCollector, MonitoringSnapshot
from sentinel.monitoring.health import Violation
from sentinel.registry.manager import ModelRegistry

logger = get_logger(__name__)

DecisionAction = Literal["none", "alert", "step_up", "promote", "rollback", "restore_previous"]


class ControllerState(BaseModel):
    phase: str = "idle"  # idle | canary | production_monitor
    production_version: str | None = None
    canary_version: str | None = None
    current_percent: int = 0
    healthy_windows: int = 0
    windows_evaluated: int = 0
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_decision: str = ""


class Decision(BaseModel):
    action: DecisionAction
    reason: str = ""
    violations: list[Violation] = Field(default_factory=list)
    snapshot_ts: str = ""
    rollback: RollbackResult | None = None
    new_percent: int | None = None


class DeploymentController:
    def __init__(
        self,
        settings: SentinelSettings,
        registry: ModelRegistry | None = None,
        collector: MetricsCollector | None = None,
        canary: CanaryManager | None = None,
        rollback: RollbackManager | None = None,
        audit: AuditLog | None = None,
    ):
        self.settings = settings
        self.registry = registry or ModelRegistry(settings)
        self.collector = collector or MetricsCollector(settings, self.registry)
        self.canary = canary or CanaryManager(settings, self.registry)
        self.rollback = rollback or RollbackManager(settings, self.registry, self.canary, audit)
        self.audit = audit or self.rollback.audit
        self.state_path = settings.resolve(settings.paths.artifacts_dir) / "controller_state.json"

    # ------------------------------------------------------------------ #
    # State persistence
    # ------------------------------------------------------------------ #
    def load_state(self) -> ControllerState:
        if self.state_path.is_file():
            try:
                return ControllerState.model_validate(json.loads(self.state_path.read_text()))
            except (json.JSONDecodeError, ValueError):
                pass
        return ControllerState()

    def save_state(self, state: ControllerState) -> None:
        state.updated_at = datetime.now(UTC).isoformat()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(state.model_dump_json(indent=2))

    # ------------------------------------------------------------------ #
    # One decision cycle
    # ------------------------------------------------------------------ #
    def step(self, snapshot: MonitoringSnapshot | None = None) -> Decision:
        snapshot = snapshot or self.collector.collect()
        state = self.load_state()
        routing = snapshot.routing

        state.windows_evaluated += 1
        state.production_version = routing.production
        state.canary_version = routing.canary
        state.current_percent = routing.canary_percent

        canary_violations = [
            v
            for v in snapshot.violations
            if v.severity == "critical" and v.model_version == routing.canary
        ]
        production_violations = [
            v
            for v in snapshot.violations
            if v.severity == "critical" and v.model_version == routing.production
        ]
        warnings = [v for v in snapshot.violations if v.severity == "warning"]

        decision: Decision

        if routing.canary:
            state.phase = "canary"
            if canary_violations:
                result = self.rollback.rollback_canary(
                    reason=self._reason(canary_violations),
                    violations=canary_violations,
                    actor="controller",
                )
                decision = Decision(
                    action="rollback",
                    reason=result.reason,
                    violations=canary_violations,
                    snapshot_ts=snapshot.ts,
                    rollback=result,
                )
                state.healthy_windows = 0
                state.canary_version = None
                state.current_percent = 0
                state.phase = "production_monitor"
            else:
                evidence = self._has_evidence(snapshot, routing.canary)
                if evidence:
                    state.healthy_windows += 1
                needed = self.settings.canary.min_healthy_windows_per_step
                if evidence and state.healthy_windows >= needed:
                    next_percent = self.canary.next_step_percent(routing.canary_percent)
                    if next_percent is not None and next_percent < 100:
                        try:
                            self.canary.set_traffic(next_percent, actor="controller")
                        except CanaryError as exc:
                            decision = Decision(
                                action="none", reason=str(exc), snapshot_ts=snapshot.ts
                            )
                        else:
                            self.audit.record(
                                AuditEntry(
                                    action="traffic_changed",
                                    actor="controller",
                                    model_version=routing.canary,
                                    previous_version=routing.production,
                                    canary_percent=next_percent,
                                    reason=(
                                        f"{state.healthy_windows} healthy windows at "
                                        f"{routing.canary_percent}% -> raising to {next_percent}%"
                                    ),
                                )
                            )
                            decision = Decision(
                                action="step_up",
                                reason=f"traffic {routing.canary_percent}% -> {next_percent}%",
                                snapshot_ts=snapshot.ts,
                                new_percent=next_percent,
                            )
                            state.healthy_windows = 0
                            state.current_percent = next_percent
                    else:
                        # At (or beyond) the top of the ramp: full promotion.
                        promote_reason = (
                            f"promoted after healthy canary at {routing.canary_percent}%"
                        )
                        try:
                            self.canary.promote_canary(
                                actor="controller",
                                reason=(
                                    f"canary healthy for {state.healthy_windows} windows "
                                    f"at {routing.canary_percent}%"
                                ),
                            )
                        except CanaryError as exc:
                            decision = Decision(
                                action="none", reason=str(exc), snapshot_ts=snapshot.ts
                            )
                        else:
                            self.audit.record(
                                AuditEntry(
                                    action="canary_promoted",
                                    actor="controller",
                                    model_version=routing.canary,
                                    previous_version=routing.production,
                                    restored_version=routing.canary,
                                    reason=promote_reason,
                                )
                            )
                            decision = Decision(
                                action="promote", reason=promote_reason, snapshot_ts=snapshot.ts
                            )
                            state.production_version = routing.canary
                            state.canary_version = None
                            state.current_percent = 0
                            state.healthy_windows = 0
                            state.phase = "production_monitor"
                else:
                    decision = Decision(
                        action="alert" if warnings else "none",
                        reason=(
                            "; ".join(v.message for v in warnings[:3])
                            if warnings
                            else (
                                f"canary healthy ({state.healthy_windows}/"
                                f"{self.settings.canary.min_healthy_windows_per_step} windows"
                                + ("" if evidence else ", awaiting minimum sample evidence")
                                + ")"
                            )
                        ),
                        violations=warnings,
                        snapshot_ts=snapshot.ts,
                    )
                    if warnings:
                        self._record_alerts(warnings)
        elif routing.production:
            state.phase = "production_monitor"
            if production_violations and self.settings.rollback.auto_restore_previous_production:
                if self.registry.previous_production() is not None:
                    result = self.rollback.restore_previous_production(
                        reason=self._reason(production_violations),
                        violations=production_violations,
                        actor="controller",
                    )
                    decision = Decision(
                        action="restore_previous",
                        reason=result.reason,
                        violations=production_violations,
                        snapshot_ts=snapshot.ts,
                        rollback=result,
                    )
                    state.production_version = result.restored_version
                else:
                    decision = Decision(
                        action="alert",
                        reason="production violating SLOs but no previous version to restore",
                        violations=production_violations,
                        snapshot_ts=snapshot.ts,
                    )
                    self._record_alerts(production_violations)
            elif production_violations or warnings:
                all_v = production_violations + warnings
                decision = Decision(
                    action="alert",
                    reason="; ".join(v.message for v in all_v[:3]),
                    violations=all_v,
                    snapshot_ts=snapshot.ts,
                )
                self._record_alerts(all_v)
            else:
                decision = Decision(
                    action="none", reason="production healthy", snapshot_ts=snapshot.ts
                )
            state.healthy_windows = 0
        else:
            state.phase = "idle"
            decision = Decision(action="none", reason="nothing deployed", snapshot_ts=snapshot.ts)

        state.last_decision = f"{decision.action}: {decision.reason}"[:500]
        self.save_state(state)
        log_event(
            logger,
            logging.WARNING
            if decision.action in ("rollback", "restore_previous", "alert")
            else logging.INFO,
            "controller decision",
            action=decision.action,
            reason=decision.reason,
            phase=state.phase,
            canary=state.canary_version,
            production=state.production_version,
            percent=state.current_percent,
            healthy_windows=state.healthy_windows,
        )
        return decision

    # ------------------------------------------------------------------ #
    def run(
        self,
        duration_seconds: float | None = None,
        interval_seconds: float | None = None,
        max_steps: int | None = None,
        stop_on_promote_or_rollback: bool = False,
    ) -> Iterator[Decision]:
        """Run decision cycles until duration/steps exhausted (generator)."""
        interval = interval_seconds or self.settings.monitoring.collection_interval_seconds
        deadline = time.monotonic() + duration_seconds if duration_seconds else None
        steps = 0
        while True:
            decision = self.step()
            steps += 1
            yield decision
            if stop_on_promote_or_rollback and decision.action in (
                "promote",
                "rollback",
                "restore_previous",
            ):
                return
            if max_steps is not None and steps >= max_steps:
                return
            if deadline is not None and time.monotonic() >= deadline:
                return
            time.sleep(interval)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _reason(violations: list[Violation]) -> str:
        first = violations[0]
        return (
            f"{first.kind} violation on {first.model_version}: "
            f"value={first.value} threshold={first.threshold} ({first.message})"
        )

    def _has_evidence(self, snapshot: MonitoringSnapshot, version: str) -> bool:
        """A window only counts as 'healthy' when it actually contains traffic."""
        stats = snapshot.stats.get(version)
        return (
            stats is not None
            and stats.n_requests >= self.settings.monitoring.min_samples_for_decision
        )

    def _record_alerts(self, violations: list[Violation]) -> None:
        for v in violations[:5]:
            self.audit.record(
                AuditEntry(
                    action="alert",
                    actor="controller",
                    model_version=v.model_version,
                    reason=v.message,
                    triggered_metric=v.kind,
                    observed_value=v.value,
                    threshold=v.threshold,
                    context={"feature": v.feature, "severity": v.severity},
                )
            )
