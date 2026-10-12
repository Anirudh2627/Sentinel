"""Model evaluation: metrics + the configurable promotion quality gate.

The gate is pure logic over metrics dictionaries — the same function is used
by the CLI, the pipeline, the deployment controller and CI, so the promotion
policy has exactly one implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sentinel.config.settings import EvaluationConfig


class MetricsResult(BaseModel):
    """Classification metrics for one dataset split."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    log_loss: float
    positive_rate: float  # model prediction rate (share of predicted positives)
    true_positive_rate: float  # actual class balance
    confusion_matrix: list[list[int]]  # [[TN, FP], [FN, TP]]
    n_samples: int

    def as_flat_dict(self, prefix: str = "") -> dict[str, float]:
        out = {
            f"{prefix}accuracy": self.accuracy,
            f"{prefix}precision": self.precision,
            f"{prefix}recall": self.recall,
            f"{prefix}f1": self.f1,
            f"{prefix}roc_auc": self.roc_auc,
            f"{prefix}log_loss": self.log_loss,
            f"{prefix}positive_rate": self.positive_rate,
            f"{prefix}n_samples": float(self.n_samples),
        }
        return out


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> MetricsResult:
    """Compute the full metric bundle from raw predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) > 1 else float("nan")
    eps = 1e-15
    ll = float(log_loss(y_true, np.clip(y_prob, eps, 1 - eps), labels=[0, 1]))

    return MetricsResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=auc,
        log_loss=ll,
        positive_rate=float(y_pred.mean()),
        true_positive_rate=float(y_true.mean()),
        confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
        n_samples=int(len(y_true)),
    )


def evaluate_pipeline(pipeline: Any, X: Any, y: np.ndarray) -> MetricsResult:
    """Evaluate a fitted sklearn pipeline on a dataframe split."""
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]
    return evaluate_predictions(y, y_pred, y_prob)


# --------------------------------------------------------------------------- #
# Quality gate
# --------------------------------------------------------------------------- #
class GateCheck(BaseModel):
    name: str
    passed: bool
    value: float | None = None
    threshold: float | None = None
    detail: str = ""


class GateResult(BaseModel):
    passed: bool
    checks: list[GateCheck] = Field(default_factory=list)
    primary_metric: str = "f1"
    reason: str = ""

    def to_markdown(self) -> str:
        head = f"# Evaluation gate — {'PASSED' if self.passed else 'BLOCKED'}"
        lines = [head, "", "| check | value | threshold | result |", "|---|---|---|---|"]
        for c in self.checks:
            v = f"{c.value:.4f}" if c.value is not None else "-"
            t = f"{c.threshold:.4f}" if c.threshold is not None else "-"
            lines.append(f"| {c.name} | {v} | {t} | {'PASS' if c.passed else 'FAIL'} |")
        if self.reason:
            lines += ["", f"Reason: {self.reason}"]
        return "\n".join(lines)


def run_quality_gate(
    candidate_metrics: dict[str, float],
    config: EvaluationConfig,
    production_metrics: dict[str, float] | None = None,
) -> GateResult:
    """Decide whether a candidate may be promoted.

    Rules (all configurable, none hardcoded here):
        * candidate_<metric> >= configured minimum for f1/precision/recall/auc
        * if ``require_beat_production`` and a production model exists:
          candidate primary metric >= production primary metric + beat_margin
    """
    checks: list[GateCheck] = []

    def add(name: str, value: float | None, threshold: float) -> None:
        ok = value is not None and value >= threshold
        checks.append(
            GateCheck(
                name=name,
                passed=ok,
                value=value,
                threshold=threshold,
                detail="" if ok else f"{name} {value} < {threshold}",
            )
        )

    add("min_f1", candidate_metrics.get("f1"), config.min_f1)
    add("min_precision", candidate_metrics.get("precision"), config.min_precision)
    add("min_recall", candidate_metrics.get("recall"), config.min_recall)
    add("min_auc", candidate_metrics.get("roc_auc"), config.min_auc)

    primary = config.primary_metric
    if config.require_beat_production and production_metrics is not None:
        cand_primary = candidate_metrics.get(primary)
        prod_primary = production_metrics.get(primary)
        if cand_primary is None or prod_primary is None:
            checks.append(
                GateCheck(
                    name=f"beat_production[{primary}]",
                    passed=False,
                    value=cand_primary,
                    threshold=prod_primary,
                    detail="primary metric missing from candidate or production metrics",
                )
            )
        else:
            threshold = prod_primary + config.beat_margin
            checks.append(
                GateCheck(
                    name=f"beat_production[{primary}]",
                    passed=cand_primary >= threshold,
                    value=cand_primary,
                    threshold=threshold,
                    detail=(
                        ""
                        if cand_primary >= threshold
                        else (
                            f"candidate {primary}={cand_primary:.4f} below production "
                            f"{prod_primary:.4f} + margin {config.beat_margin}"
                        )
                    ),
                )
            )

    failed = [c for c in checks if not c.passed]
    return GateResult(
        passed=not failed,
        checks=checks,
        primary_metric=primary,
        reason="; ".join(c.detail for c in failed if c.detail),
    )
