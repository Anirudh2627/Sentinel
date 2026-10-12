"""Fault injection for demos and chaos testing (server side).

Two knobs, both configurable at boot (``faults.*`` in monitoring.yaml /
environment) and at runtime via the token-protected ``POST /admin/faults``:

    * ``latency_ms``  — artificial inference latency (with jitter)
    * ``error_rate``  — fraction of requests that fail with an injected error

Faults are logged explicitly (``error_type="injected"``) so monitoring treats
them exactly like real incidents.
"""

from __future__ import annotations

import random
import threading
import time

from sentinel.config.settings import FaultsConfig


class InjectedFaultError(RuntimeError):
    """Raised when the error-injection dice hits a request."""


class FaultInjector:
    def __init__(self, config: FaultsConfig | None = None):
        self._lock = threading.Lock()
        self._rng = random.Random()
        cfg = config or FaultsConfig()
        self.enabled: bool = cfg.enabled
        self.latency_ms: float = cfg.latency_ms
        self.error_rate: float = cfg.error_rate
        self.latency_jitter: float = cfg.latency_jitter

    # ------------------------------------------------------------------ #
    def update(
        self,
        enabled: bool | None = None,
        latency_ms: float | None = None,
        error_rate: float | None = None,
    ) -> dict[str, float | bool]:
        with self._lock:
            if enabled is not None:
                self.enabled = enabled
            if latency_ms is not None:
                self.latency_ms = max(0.0, float(latency_ms))
            if error_rate is not None:
                self.error_rate = min(max(0.0, float(error_rate)), 1.0)
            return self.state()

    def state(self) -> dict[str, float | bool]:
        return {
            "enabled": self.enabled,
            "latency_ms": self.latency_ms,
            "error_rate": self.error_rate,
        }

    # ------------------------------------------------------------------ #
    def apply(self) -> None:
        """Apply configured latency / error injection. Call inside request flow."""
        if not self.enabled:
            return
        with self._lock:
            latency_ms, error_rate, jitter = self.latency_ms, self.error_rate, self.latency_jitter
        if latency_ms > 0:
            factor = 1.0 + self._rng.uniform(-jitter, jitter)
            time.sleep(max(0.0, latency_ms * factor) / 1000.0)
        if error_rate > 0 and self._rng.random() < error_rate:
            raise InjectedFaultError("injected fault (error_rate scenario)")
