"""Metrics collector: read serving logs -> window stats -> drift -> snapshot.

The serving layer writes append-only JSONL logs (predictions + delayed
feedback). The collector tails them, aggregates per model version, computes
drift against each model's training reference, evaluates health policy and
persists a :class:`MonitoringSnapshot` for the deployment controller and CLI.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from sentinel.config import SentinelSettings, get_logger, log_event
from sentinel.features.engineering import FeatureEngineer
from sentinel.monitoring.drift import DriftDetector, DriftReport
from sentinel.monitoring.health import HealthEvaluator, Violation
from sentinel.monitoring.metrics import (
    PerformanceStats,
    WindowStats,
    compute_window_stats,
)
from sentinel.registry.manager import ModelRegistry
from sentinel.training.artifacts import load_reference_stats

logger = get_logger(__name__)

_TAIL_BYTES = 32 * 1024 * 1024  # only re-read the last 32MB of a log file


class RoutingSnapshot(BaseModel):
    production: str | None = None
    canary: str | None = None
    canary_percent: int = 0


class MonitoringSnapshot(BaseModel):
    """One monitoring decision window, fully serialised (audit-friendly)."""

    ts: str
    window_seconds: float
    routing: RoutingSnapshot
    stats: dict[str, WindowStats] = Field(default_factory=dict)
    drift: dict[str, DriftReport] = Field(default_factory=dict)
    performance: dict[str, PerformanceStats] = Field(default_factory=dict)
    violations: list[Violation] = Field(default_factory=list)

    @property
    def critical_violations(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "critical"]


def tail_jsonl(
    path: Path, since: datetime | None = None, max_bytes: int = _TAIL_BYTES
) -> list[dict]:
    """Read trailing JSONL lines, optionally filtering by ISO ``ts`` field."""
    if not path.is_file():
        return []
    size = path.stat().st_size
    with path.open("rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
            fh.readline()  # discard possibly partial first line
        raw = fh.read().decode("utf-8", errors="replace")
    records: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if since is not None:
            ts = rec.get("ts")
            if not ts:
                continue
            try:
                when = datetime.fromisoformat(ts)
            except ValueError:
                continue
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
            if when < since:
                continue
        records.append(rec)
    return records


class MetricsCollector:
    def __init__(self, settings: SentinelSettings, registry: ModelRegistry | None = None):
        self.settings = settings
        self.registry = registry or ModelRegistry(settings)
        self.evaluator = HealthEvaluator(settings.monitoring, settings.rollback)
        self.prediction_log = settings.resolve(settings.serving.prediction_log)
        self.feedback_log = settings.resolve(settings.serving.feedback_log)
        self.snapshot_path = settings.resolve(settings.monitoring.snapshot_path)

    # ------------------------------------------------------------------ #
    def collect(self, now: datetime | None = None, persist: bool = True) -> MonitoringSnapshot:
        """Build (and optionally persist) one monitoring snapshot."""
        now = now or datetime.now(UTC)
        window = self.settings.monitoring.window_seconds
        since = datetime.fromtimestamp(now.timestamp() - window, tz=UTC)

        predictions = tail_jsonl(self.prediction_log, since=since)
        feedback = tail_jsonl(self.feedback_log, since=None)

        routing = self._routing_snapshot()
        by_version = self._group_by_version(predictions)
        stats = {
            version: compute_window_stats(version, records, window)
            for version, records in by_version.items()
        }
        drift = self._compute_drift(by_version, routing)
        performance = self._compute_performance(by_version, feedback)

        violations = self.evaluator.evaluate(
            stats_by_version=stats,
            drift_reports=drift,
            performance=performance,
            production_version=routing.production,
            canary_version=routing.canary,
        )

        snapshot = MonitoringSnapshot(
            ts=now.isoformat(),
            window_seconds=window,
            routing=routing,
            stats=stats,
            drift=drift,
            performance=performance,
            violations=violations,
        )
        if persist:
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            with self.snapshot_path.open("a") as fh:
                fh.write(snapshot.model_dump_json() + "\n")
        log_event(
            logger,
            logging.INFO,
            "monitoring snapshot collected",
            requests=sum(s.n_requests for s in stats.values()),
            versions=list(stats.keys()),
            violations=len(violations),
            critical=len(snapshot.critical_violations),
        )
        return snapshot

    def latest_snapshot(self) -> MonitoringSnapshot | None:
        if not self.snapshot_path.is_file():
            return None
        lines = [ln for ln in self.snapshot_path.read_text().splitlines() if ln.strip()]
        if not lines:
            return None
        return MonitoringSnapshot.model_validate(json.loads(lines[-1]))

    # ------------------------------------------------------------------ #
    def _routing_snapshot(self) -> RoutingSnapshot:
        from sentinel.deployment.canary import (
            read_routing_state,  # local import (no cycle at runtime)
        )

        state = read_routing_state(self.settings)
        return RoutingSnapshot(
            production=state.production, canary=state.canary, canary_percent=state.canary_percent
        )

    @staticmethod
    def _group_by_version(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for rec in records:
            version = rec.get("model_version") or "unknown"
            out.setdefault(version, []).append(rec)
        return out

    def _compute_drift(
        self, by_version: dict[str, list[dict[str, Any]]], routing: RoutingSnapshot
    ) -> dict[str, DriftReport]:
        cfg = self.settings.monitoring.drift
        if not cfg.enabled:
            return {}
        reports: dict[str, DriftReport] = {}
        for version, records in by_version.items():
            if len(records) < cfg.min_samples:
                continue
            try:
                mv = self.registry.get(version)
                reference = load_reference_stats(self.registry.artifact_dir(version))
            except Exception as exc:  # unknown version / missing stats
                log_event(
                    logger,
                    logging.WARNING,
                    "drift skipped: reference unavailable",
                    model_version=version,
                    error=str(exc),
                )
                continue
            frames = [r.get("features") for r in records if isinstance(r.get("features"), dict)]
            if len(frames) < cfg.min_samples:
                continue
            df = pd.DataFrame(frames)
            try:
                # Compare in the same (post-engineering) space as the reference.
                df = FeatureEngineer().fit_transform(df)
            except ValueError as exc:
                log_event(
                    logger,
                    logging.WARNING,
                    "drift skipped: feature engineering failed",
                    model_version=version,
                    error=str(exc),
                )
                continue
            detector = DriftDetector(reference, cfg)
            reports[version] = detector.detect(df)
            _ = mv  # version validated above
        return reports

    def _compute_performance(
        self, by_version: dict[str, list[dict[str, Any]]], feedback: list[dict[str, Any]]
    ) -> dict[str, PerformanceStats]:
        cfg = self.settings.monitoring.performance
        if not cfg.enabled or not feedback:
            return {}
        truth_by_request = {
            str(f.get("request_id")): int(f["ground_truth"])
            for f in feedback
            if f.get("request_id") is not None and f.get("ground_truth") is not None
        }
        out: dict[str, PerformanceStats] = {}
        for version, records in by_version.items():
            y_true: list[int] = []
            y_prob: list[float] = []
            y_pred: list[int] = []
            for rec in records:
                rid = str(rec.get("request_id"))
                if rid in truth_by_request and rec.get("probability") is not None:
                    y_true.append(truth_by_request[rid])
                    y_prob.append(float(rec["probability"]))
                    y_pred.append(int(rec.get("prediction", 0)))
            if len(y_true) < cfg.min_samples:
                out[version] = PerformanceStats(model_version=version, n_labels=len(y_true))
                continue
            auc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None
            acc = float(accuracy_score(y_true, y_pred))
            eps = 1e-15
            ll = (
                float(log_loss(y_true, [min(max(p, eps), 1 - eps) for p in y_prob], labels=[0, 1]))
                if auc is not None
                else None
            )
            out[version] = PerformanceStats(
                model_version=version,
                n_labels=len(y_true),
                roc_auc=round(auc, 6) if auc is not None else None,
                accuracy=round(acc, 6),
                log_loss=round(ll, 6) if ll is not None else None,
                true_positive_rate=round(float(sum(y_true) / len(y_true)), 6),
            )
        return out
