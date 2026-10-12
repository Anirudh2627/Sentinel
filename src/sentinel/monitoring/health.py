"""Health evaluation: turn window statistics into policy violations.

Pure decision logic — every threshold comes from configuration
(``monitoring.yaml``). The deployment controller consumes the violations and
decides promote / rollback / alert.

Violation severity semantics:
    * ``critical`` — may trigger automatic rollback (only when the matching
      ``rollback.on_<kind>`` switch is enabled).
    * ``warning``  — alert-only (signal recorded, no automatic action).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from sentinel.config.settings import MonitoringConfig, RollbackConfig
from sentinel.monitoring.drift import DriftReport
from sentinel.monitoring.metrics import PerformanceStats, WindowStats

ViolationKind = Literal[
    "error_rate",
    "latency_p95",
    "latency_p99",
    "prediction_shift",
    "drift",
    "performance",
]


class Violation(BaseModel):
    kind: ViolationKind
    model_version: str
    value: float
    threshold: float
    severity: Literal["warning", "critical"]
    message: str
    feature: str | None = None  # for drift violations


class HealthEvaluator:
    """Evaluate SLOs / shift / drift / delayed-label performance."""

    def __init__(self, monitoring: MonitoringConfig, rollback: RollbackConfig):
        self.mon = monitoring
        self.rb = rollback

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        stats_by_version: dict[str, WindowStats],
        drift_reports: dict[str, DriftReport] | None = None,
        performance: dict[str, PerformanceStats] | None = None,
        production_version: str | None = None,
        canary_version: str | None = None,
    ) -> list[Violation]:
        violations: list[Violation] = []
        drift_reports = drift_reports or {}
        performance = performance or {}

        for version, stats in stats_by_version.items():
            if stats.n_requests < self.mon.min_samples_for_decision:
                continue  # not enough evidence to judge
            violations += self._operational_slos(version, stats)

        violations += self._prediction_shift(stats_by_version, production_version, canary_version)

        for version, report in drift_reports.items():
            violations += self._drift(version, report)

        for version, perf in performance.items():
            violations += self._performance(version, perf)

        return violations

    # ------------------------------------------------------------------ #
    def _severity(self, kind: ViolationKind, would_rollback: bool) -> str:
        enabled = {
            "error_rate": self.rb.on_error_rate,
            "latency_p95": self.rb.on_latency,
            "latency_p99": self.rb.on_latency,
            "prediction_shift": self.rb.on_prediction_shift,
            "drift": self.rb.on_drift,
            "performance": self.rb.on_performance,
        }[kind]
        return "critical" if (would_rollback and enabled and self.rb.enabled) else "warning"

    def _operational_slos(self, version: str, stats: WindowStats) -> list[Violation]:
        out: list[Violation] = []
        if stats.error_rate > self.mon.max_error_rate:
            out.append(
                Violation(
                    kind="error_rate",
                    model_version=version,
                    value=stats.error_rate,
                    threshold=self.mon.max_error_rate,
                    severity=self._severity("error_rate", True),
                    message=(
                        f"error rate {stats.error_rate:.4f} > {self.mon.max_error_rate:.4f} "
                        f"({stats.n_errors}/{stats.n_requests} requests)"
                    ),
                )
            )
        if stats.latency_ms is not None:
            if stats.latency_ms.p95 > self.mon.max_p95_latency_ms:
                out.append(
                    Violation(
                        kind="latency_p95",
                        model_version=version,
                        value=stats.latency_ms.p95,
                        threshold=self.mon.max_p95_latency_ms,
                        severity=self._severity("latency_p95", True),
                        message=(
                            f"p95 latency {stats.latency_ms.p95:.1f}ms > "
                            f"{self.mon.max_p95_latency_ms:.1f}ms"
                        ),
                    )
                )
            if stats.latency_ms.p99 > self.mon.max_p99_latency_ms:
                out.append(
                    Violation(
                        kind="latency_p99",
                        model_version=version,
                        value=stats.latency_ms.p99,
                        threshold=self.mon.max_p99_latency_ms,
                        severity=self._severity("latency_p99", True),
                        message=(
                            f"p99 latency {stats.latency_ms.p99:.1f}ms > "
                            f"{self.mon.max_p99_latency_ms:.1f}ms"
                        ),
                    )
                )
        return out

    def _prediction_shift(
        self,
        stats_by_version: dict[str, WindowStats],
        production_version: str | None,
        canary_version: str | None,
    ) -> list[Violation]:
        cfg = self.mon.prediction_shift
        if not cfg.enabled or not production_version or not canary_version:
            return []
        prod = stats_by_version.get(production_version)
        canary = stats_by_version.get(canary_version)
        if (
            prod is None
            or canary is None
            or prod.positive_rate is None
            or canary.positive_rate is None
        ):
            return []
        # Both sides need enough predictions for the comparison to mean anything
        # (a 5% canary slice of low traffic is pure noise).
        if prod.n_predictions < cfg.min_samples or canary.n_predictions < cfg.min_samples:
            return []
        delta = abs(canary.positive_rate - prod.positive_rate)
        if delta > cfg.max_delta_positive_rate:
            return [
                Violation(
                    kind="prediction_shift",
                    model_version=canary_version,
                    value=round(delta, 6),
                    threshold=cfg.max_delta_positive_rate,
                    severity=self._severity("prediction_shift", True),
                    message=(
                        f"canary positive-rate {canary.positive_rate:.4f} vs production "
                        f"{prod.positive_rate:.4f} "
                        f"(delta {delta:.4f} > {cfg.max_delta_positive_rate:.4f})"
                    ),
                )
            ]
        return []

    def _drift(self, version: str, report: DriftReport) -> list[Violation]:
        cfg = self.mon.drift
        if not cfg.enabled or not report.is_drifted:
            return []
        top = report.top_features(cfg.top_k_features)
        # Drift severity is configured explicitly; it only becomes rollback-
        # eligible when it is "critical" AND rollback.on_drift is enabled.
        would_rollback = cfg.severity == "critical"
        return [
            Violation(
                kind="drift",
                model_version=version,
                value=f.score,
                threshold=f.threshold,
                severity=self._severity("drift", would_rollback),
                feature=f.feature,
                message=(
                    f"{report.method.upper()} drift on '{f.feature}': "
                    f"{f.score:.4f} > {f.threshold:.4f} "
                    f"({report.n_drifted_features} feature(s) drifting; "
                    f"top={[t.feature for t in top]})"
                ),
            )
            for f in top
            if f.drifted
        ]

    def _performance(self, version: str, perf: PerformanceStats) -> list[Violation]:
        cfg = self.mon.performance
        if not cfg.enabled or perf.roc_auc is None or perf.n_labels < cfg.min_samples:
            return []
        if perf.roc_auc < cfg.min_auc:
            return [
                Violation(
                    kind="performance",
                    model_version=version,
                    value=perf.roc_auc,
                    threshold=cfg.min_auc,
                    severity=self._severity("performance", True),
                    message=(
                        f"live AUC on delayed labels {perf.roc_auc:.4f} < {cfg.min_auc:.4f} "
                        f"({perf.n_labels} labelled requests)"
                    ),
                )
            ]
        return []
