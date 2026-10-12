"""Canary deployment primitives: routing state + traffic allocation + manager.

Routing is a small JSON document (``artifacts/routing.json``) written by the
deployment side and read by the serving side with mtime-based refresh. In a
Kubernetes environment the same semantics map to weighted ingress / service
mesh routing (see deployment/kubernetes); locally it keeps the whole canary
mechanism observable and testable without infrastructure.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from sentinel.config import SentinelSettings, get_logger, log_event
from sentinel.registry.manager import ModelRegistry
from sentinel.registry.metadata import ModelStatus

logger = get_logger(__name__)


class RoutingState(BaseModel):
    """Which model versions receive traffic, and how it is split."""

    production: str | None = None
    canary: str | None = None
    canary_percent: int = Field(default=0, ge=0, le=100)
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_by: str = "system"


def routing_path(settings: SentinelSettings) -> Path:
    return settings.resolve(settings.serving.routing_state_path)


def read_routing_state(settings: SentinelSettings) -> RoutingState:
    path = routing_path(settings)
    if not path.is_file():
        return RoutingState()
    try:
        return RoutingState.model_validate(json.loads(path.read_text()))
    except (json.JSONDecodeError, ValueError):
        log_event(
            logger, logging.ERROR, "routing state corrupt; falling back to empty", path=str(path)
        )
        return RoutingState()


def write_routing_state(settings: SentinelSettings, state: RoutingState) -> RoutingState:
    path = routing_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at = datetime.now(UTC).isoformat()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(state.model_dump_json(indent=2))
    os.replace(tmp, path)
    log_event(
        logger,
        logging.INFO,
        "routing state written",
        production=state.production,
        canary=state.canary,
        canary_percent=state.canary_percent,
        updated_by=state.updated_by,
    )
    return state


class CanaryRouter:
    """Allocate a request to production or canary.

    Sticky mode hashes the request id, so a given client/request consistently
    lands on the same version (deterministic and testable). Non-sticky mode
    samples uniformly.
    """

    def __init__(self, sticky: bool = True, seed: int | None = None):
        self.sticky = sticky
        self._rng = random.Random(seed)

    @staticmethod
    def bucket(request_id: str) -> int:
        """Stable bucket in [0, 10000) for a request id."""
        digest = hashlib.md5(request_id.encode("utf-8")).hexdigest()
        return int(digest, 16) % 10_000

    def route(self, state: RoutingState, request_id: str) -> tuple[str | None, str]:
        """Return (model_version, served_by)."""
        if state.canary and state.canary_percent > 0 and state.production:
            hit = (
                self.bucket(request_id) < state.canary_percent * 100
                if self.sticky
                else self._rng.random() * 100 < state.canary_percent
            )
            if hit:
                return state.canary, "canary"
        if state.production:
            return state.production, "production"
        if state.canary:  # production not yet set (initial deploy edge case)
            return state.canary, "canary"
        return None, "production"


class CanaryError(RuntimeError):
    pass


class CanaryManager:
    """Coordinates registry state + routing file for canary operations."""

    def __init__(self, settings: SentinelSettings, registry: ModelRegistry | None = None):
        self.settings = settings
        self.registry = registry or ModelRegistry(settings)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def current_state(self) -> RoutingState:
        return read_routing_state(self.settings)

    def deploy_production(self, version: str, actor: str = "operator") -> RoutingState:
        """Direct (non-canary) production deployment — used for the first model."""
        mv = self.registry.get(version)
        if mv.status not in (ModelStatus.CANDIDATE, ModelStatus.VALIDATED, ModelStatus.ARCHIVED):
            raise CanaryError(f"{version} in status {mv.status.value} cannot be deployed directly")
        self.registry.promote_to_production(version, reason=f"direct deployment by {actor}")
        return write_routing_state(
            self.settings,
            RoutingState(production=version, canary=None, canary_percent=0, updated_by=actor),
        )

    def start_canary(
        self, version: str, percent: int | None = None, actor: str = "operator"
    ) -> RoutingState:
        """Put ``version`` into canary with an initial traffic percentage."""
        with self._lock:
            state = self.current_state()
            if state.production is None:
                raise CanaryError(
                    "no production model — deploy production first (canary needs a baseline)"
                )
            if state.canary and state.canary != version:
                raise CanaryError(f"canary already active: {state.canary}")
            percent = self.settings.canary.initial_traffic_percent if percent is None else percent
            if not 0 < percent <= 100:
                raise CanaryError(f"invalid canary percent: {percent}")

            mv = self.registry.get(version)
            if mv.status not in (ModelStatus.CANDIDATE, ModelStatus.ARCHIVED):
                raise CanaryError(f"{version} must be a candidate to canary (is {mv.status.value})")
            self.registry.start_canary(version, reason=f"canary started at {percent}% by {actor}")
            return write_routing_state(
                self.settings,
                RoutingState(
                    production=state.production,
                    canary=version,
                    canary_percent=percent,
                    updated_by=actor,
                ),
            )

    def set_traffic(self, percent: int, actor: str = "controller") -> RoutingState:
        with self._lock:
            state = self.current_state()
            if not state.canary:
                raise CanaryError("no active canary")
            if not 0 <= percent <= 100:
                raise CanaryError(f"invalid percent: {percent}")
            return write_routing_state(
                self.settings,
                RoutingState(
                    production=state.production,
                    canary=state.canary,
                    canary_percent=percent,
                    updated_by=actor,
                ),
            )

    def next_step_percent(self, current_percent: int) -> int | None:
        """Next configured promotion step above ``current_percent`` (None = done)."""
        steps = [s for s in self.settings.canary.promotion_steps if s > current_percent]
        return steps[0] if steps else None

    def promote_canary(
        self, actor: str = "controller", reason: str = "canary promoted"
    ) -> RoutingState:
        """Full promotion: canary becomes production; routing goes 100/0."""
        with self._lock:
            state = self.current_state()
            if not state.canary:
                raise CanaryError("no active canary to promote")
            version = state.canary
            self.registry.promote_to_production(version, reason=reason)
            return write_routing_state(
                self.settings,
                RoutingState(production=version, canary=None, canary_percent=0, updated_by=actor),
            )

    def rollback_canary(self, reason: str, actor: str = "controller") -> RoutingState:
        """Remove the canary from traffic; production keeps serving 100%."""
        with self._lock:
            state = self.current_state()
            if not state.canary:
                raise CanaryError("no active canary to roll back")
            version = state.canary
            self.registry.rollback_canary(version, reason=reason)
            return write_routing_state(
                self.settings,
                RoutingState(
                    production=state.production,
                    canary=None,
                    canary_percent=0,
                    updated_by=actor,
                ),
            )

    def withdraw_canary(self, reason: str, actor: str = "operator") -> RoutingState:
        """Withdraw canary without marking it failed (back to CANDIDATE)."""
        with self._lock:
            state = self.current_state()
            if not state.canary:
                raise CanaryError("no active canary")
            version = state.canary
            self.registry.set_status(version, ModelStatus.CANDIDATE, reason=reason, actor=actor)
            return write_routing_state(
                self.settings,
                RoutingState(
                    production=state.production, canary=None, canary_percent=0, updated_by=actor
                ),
            )


def cooldown_active(settings: SentinelSettings, last_rollback_ts: float | None) -> bool:
    """True when a new canary would violate the post-rollback cooldown."""
    if last_rollback_ts is None:
        return False
    return (time.time() - last_rollback_ts) < settings.rollback.cooldown_seconds
