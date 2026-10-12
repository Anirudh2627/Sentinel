"""Model loading for serving: cached artifact access + routing refresh.

The repository reads the registry index and routing state from disk with
mtime/interval-based refresh, so deployments performed by the controller (a
separate process) take effect in the running server without restarts.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import joblib

from sentinel.config import SentinelSettings, get_logger, log_event
from sentinel.deployment.canary import CanaryRouter, RoutingState, read_routing_state
from sentinel.registry.manager import ModelRegistry
from sentinel.training.artifacts import TrainingMetadata, load_metadata

logger = get_logger(__name__)


class ModelUnavailableError(RuntimeError):
    pass


class ModelRepository:
    """Thread-safe, lazily-cached access to model artifacts + routing state."""

    def __init__(self, settings: SentinelSettings, registry: ModelRegistry | None = None):
        self.settings = settings
        self.registry = registry or ModelRegistry(settings)
        self.router = CanaryRouter(sticky=settings.canary.sticky_routing)
        self._cache: dict[str, Any] = {}
        self._meta_cache: dict[str, TrainingMetadata] = {}
        self._lock = threading.Lock()
        self._routing_lock = threading.Lock()
        self._routing: RoutingState = RoutingState()
        self._routing_checked_at: float = 0.0
        self._routing_mtime: float | None = None

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #
    def routing(self, force: bool = False) -> RoutingState:
        """Return the routing state, refreshing from disk at most every interval.

        Double-checked locking: the mtime marker is only updated *after* the
        new state is installed, so concurrent requests can never observe a
        fresh mtime together with a stale routing state.
        """
        interval = self.settings.serving.routing_reload_interval_seconds
        if not force and time.monotonic() - self._routing_checked_at < interval:
            return self._routing
        with self._routing_lock:
            now = time.monotonic()
            if not force and now - self._routing_checked_at < interval:
                return self._routing
            self._routing_checked_at = now
            path = Path(self.settings.resolve(self.settings.serving.routing_state_path))
            try:
                mtime = path.stat().st_mtime if path.exists() else None
            except OSError:
                mtime = None
            if force or mtime != self._routing_mtime or self._routing.production is None:
                new_state = read_routing_state(self.settings)
                if new_state != self._routing:
                    log_event(
                        logger,
                        logging.INFO,
                        "routing state refreshed",
                        production=new_state.production,
                        canary=new_state.canary,
                        canary_percent=new_state.canary_percent,
                    )
                self._routing = new_state
                self._routing_mtime = mtime
            return self._routing

    def select(self, request_id: str) -> tuple[str, str]:
        """(model_version, served_by) for a request; raises when nothing is deployed."""
        state = self.routing()
        version, served_by = self.router.route(state, request_id)
        if version is None:
            raise ModelUnavailableError("no model deployed (routing state is empty)")
        return version, served_by

    # ------------------------------------------------------------------ #
    # Artifacts
    # ------------------------------------------------------------------ #
    def pipeline(self, version: str) -> Any:
        with self._lock:
            if version in self._cache:
                return self._cache[version]
        artifact_dir = self._artifact_dir(version)
        model_path = artifact_dir / "model.joblib"
        if not model_path.is_file():
            raise ModelUnavailableError(f"model artifact missing for {version}: {model_path}")
        pipeline = joblib.load(model_path)
        with self._lock:
            self._cache[version] = pipeline
        log_event(logger, logging.INFO, "model loaded", model_version=version, path=str(model_path))
        return pipeline

    def metadata(self, version: str) -> TrainingMetadata:
        with self._lock:
            if version in self._meta_cache:
                return self._meta_cache[version]
        meta = load_metadata(self._artifact_dir(version))
        with self._lock:
            self._meta_cache[version] = meta
        return meta

    def _artifact_dir(self, version: str) -> Path:
        try:
            return self.registry.artifact_dir(version)
        except Exception as exc:
            raise ModelUnavailableError(f"unknown model version {version}: {exc}") from exc

    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()
            self._meta_cache.clear()
        self.routing(force=True)

    # ------------------------------------------------------------------ #
    def registry_summary(self) -> dict[str, Any]:
        index = self.registry.index()
        return json.loads(index.model_dump_json())
