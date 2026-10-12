"""Data validation with a machine-readable report.

Checks implemented:
    * schema consistency (missing / unexpected columns, dtype families)
    * missing-value ratios per column
    * invalid value ranges (numeric bounds, allowed categorical levels)
    * duplicate rows and duplicate ids
    * target sanity (presence, binary, class-balance bounds)
    * naive target-leakage screening (|corr(feature, target)| threshold)

The pipeline fails loudly (``DataValidationError``) in strict mode.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from sentinel.config import SentinelSettings, get_logger, log_event

logger = get_logger(__name__)

CheckStatus = Literal["passed", "failed", "warning"]


class CheckResult(BaseModel):
    name: str
    status: CheckStatus
    details: dict = Field(default_factory=dict)
    message: str = ""


class ValidationReport(BaseModel):
    passed: bool
    n_rows: int
    n_columns: int
    checks: list[CheckResult] = Field(default_factory=list)

    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "failed"]

    def to_markdown(self) -> str:
        lines = [
            f"# Data validation report — {'PASSED' if self.passed else 'FAILED'}",
            f"rows={self.n_rows}, columns={self.n_columns}",
            "",
            "| check | status | message |",
            "|---|---|---|",
        ]
        for c in self.checks:
            lines.append(f"| {c.name} | {c.status} | {c.message or '-'} |")
        return "\n".join(lines)


class DataValidationError(RuntimeError):
    def __init__(self, report: ValidationReport):
        self.report = report
        failed = "; ".join(f"{c.name}: {c.message}" for c in report.failed_checks())
        super().__init__(f"data validation failed -> {failed}")


# Expected schema for the churn dataset. Numeric bounds are *domain* bounds,
# not observed min/max — values outside them indicate upstream corruption.
NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "tenure_months": (0, 72),
    "monthly_charges": (0, 1000),
    "total_charges": (0, 100_000),
    "num_services": (0, 10),
    "support_tickets_30d": (0, 50),
    "avg_session_minutes": (0, 24 * 60),
}
CATEGORICAL_LEVELS: dict[str, set[str]] = {
    "contract_type": {"Month-to-month", "One year", "Two year"},
    "internet_service": {"DSL", "Fiber optic", "No"},
    "payment_method": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}
BINARY_COLUMNS = {"promo_active", "senior_citizen"}


class DataValidator:
    """Validate a raw dataframe against schema + quality rules."""

    def __init__(self, settings: SentinelSettings):
        self.settings = settings
        self.cfg = settings.validation
        self.feat = settings.features
        self.expected_columns = (
            [self.feat.id_column, self.feat.target]
            + list(self.feat.numeric)
            + list(self.feat.categorical)
        )

    # ------------------------------------------------------------------ #
    def validate(self, df: pd.DataFrame) -> ValidationReport:
        checks: list[CheckResult] = []
        checks.append(self._check_schema(df))
        checks.append(self._check_min_rows(df))
        checks.append(self._check_missing(df))
        checks.append(self._check_ranges(df))
        checks.append(self._check_categorical_levels(df))
        checks.append(self._check_binary_columns(df))
        checks.append(self._check_duplicates(df))
        checks.append(self._check_target(df))
        checks.append(self._check_leakage(df))

        failed = any(c.status == "failed" for c in checks)
        report = ValidationReport(
            passed=not failed, n_rows=int(len(df)), n_columns=int(df.shape[1]), checks=checks
        )
        log_event(
            logger,
            logging.INFO if report.passed else logging.ERROR,
            "data validation finished",
            passed=report.passed,
            failed_checks=[c.name for c in report.failed_checks()],
        )
        if failed and self.cfg.strict:
            raise DataValidationError(report)
        return report

    # ------------------------------------------------------------------ #
    def _check_schema(self, df: pd.DataFrame) -> CheckResult:
        missing = [c for c in self.expected_columns if c not in df.columns]
        unexpected = [c for c in df.columns if c not in self.expected_columns]
        status: CheckStatus = "passed" if not missing else "failed"
        if not missing and unexpected:
            status = "warning"
        return CheckResult(
            name="schema",
            status=status,
            details={"missing_columns": missing, "unexpected_columns": unexpected},
            message=f"missing={missing} unexpected={unexpected}" if (missing or unexpected) else "",
        )

    def _check_min_rows(self, df: pd.DataFrame) -> CheckResult:
        ok = len(df) >= self.cfg.min_rows
        return CheckResult(
            name="min_rows",
            status="passed" if ok else "failed",
            details={"n_rows": len(df), "min_rows": self.cfg.min_rows},
            message="" if ok else f"only {len(df)} rows (min {self.cfg.min_rows})",
        )

    def _check_missing(self, df: pd.DataFrame) -> CheckResult:
        ratios = df.isna().mean().to_dict()
        offenders = {
            c: float(r)
            for c, r in ratios.items()
            if r > self.cfg.max_missing_ratio and c in self.expected_columns
        }
        ok = not offenders
        return CheckResult(
            name="missing_values",
            status="passed" if ok else "failed",
            details={"offenders": offenders, "max_missing_ratio": self.cfg.max_missing_ratio},
            message=f"columns over missing budget: {offenders}" if offenders else "",
        )

    def _check_ranges(self, df: pd.DataFrame) -> CheckResult:
        offenders: dict[str, dict] = {}
        for col, (lo, hi) in NUMERIC_BOUNDS.items():
            if col not in df.columns or df[col].isna().all():
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            bad = int(((values < lo) | (values > hi)).sum())
            if bad:
                offenders[col] = {"out_of_range": bad, "bounds": [lo, hi]}
        ok = not offenders
        return CheckResult(
            name="numeric_ranges",
            status="passed" if ok else "failed",
            details=offenders,
            message=f"out-of-range values: {offenders}" if offenders else "",
        )

    def _check_categorical_levels(self, df: pd.DataFrame) -> CheckResult:
        offenders: dict[str, list] = {}
        for col, allowed in CATEGORICAL_LEVELS.items():
            if col not in df.columns:
                continue
            unknown = sorted({str(v) for v in df[col].dropna().unique()} - allowed)
            if unknown:
                offenders[col] = unknown
        ok = not offenders
        return CheckResult(
            name="categorical_levels",
            status="passed" if ok else "failed",
            details=offenders,
            message=f"unknown levels: {offenders}" if offenders else "",
        )

    def _check_binary_columns(self, df: pd.DataFrame) -> CheckResult:
        offenders: dict[str, list] = {}
        for col in BINARY_COLUMNS:
            if col not in df.columns:
                continue
            values = set(pd.to_numeric(df[col], errors="coerce").dropna().unique().tolist())
            if not values <= {0.0, 1.0}:
                offenders[col] = sorted(values)
        ok = not offenders
        return CheckResult(
            name="binary_columns",
            status="passed" if ok else "failed",
            details=offenders,
            message=f"non-binary values: {offenders}" if offenders else "",
        )

    def _check_duplicates(self, df: pd.DataFrame) -> CheckResult:
        feature_cols = [
            c for c in self.expected_columns if c in df.columns and c != self.feat.id_column
        ]
        row_dup_ratio = float(df.duplicated(subset=feature_cols).mean()) if feature_cols else 0.0
        id_dup_ratio = (
            float(df[self.feat.id_column].duplicated().mean())
            if self.feat.id_column in df.columns
            else 0.0
        )
        details = {
            "duplicate_row_ratio": row_dup_ratio,
            "duplicate_id_ratio": id_dup_ratio,
        }
        failed = (
            row_dup_ratio > self.cfg.max_duplicate_row_ratio
            or id_dup_ratio > self.cfg.max_duplicate_id_ratio
        )
        return CheckResult(
            name="duplicates",
            status="failed" if failed else "passed",
            details=details,
            message=f"dup_rows={row_dup_ratio:.5f} dup_ids={id_dup_ratio:.5f}" if failed else "",
        )

    def _check_target(self, df: pd.DataFrame) -> CheckResult:
        target = self.feat.target
        if target not in df.columns:
            return CheckResult(
                name="target", status="failed", message=f"target column '{target}' missing"
            )
        values = pd.to_numeric(df[target], errors="coerce")
        if values.isna().any():
            return CheckResult(
                name="target", status="failed", message="target contains NaN/non-numeric"
            )
        levels = set(values.unique().tolist())
        if not levels <= {0.0, 1.0}:
            return CheckResult(
                name="target",
                status="failed",
                message=f"target must be binary, got levels {sorted(levels)}",
            )
        pos_rate = float(values.mean())
        ok = self.cfg.min_positive_rate <= pos_rate <= self.cfg.max_positive_rate
        return CheckResult(
            name="target",
            status="passed" if ok else "failed",
            details={"positive_rate": pos_rate},
            message="" if ok else f"class imbalance: positive_rate={pos_rate:.4f}",
        )

    def _check_leakage(self, df: pd.DataFrame) -> CheckResult:
        """Screen for suspiciously predictive features (target leakage)."""
        target = self.feat.target
        suspects: dict[str, float] = {}
        if target not in df.columns:
            return CheckResult(
                name="leakage_screen", status="warning", message="target missing; skipped"
            )
        y = pd.to_numeric(df[target], errors="coerce")
        for col in self.feat.numeric:
            if col not in df.columns:
                continue
            x = pd.to_numeric(df[col], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < 10 or x[mask].std() == 0:
                continue
            corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
            if abs(corr) >= self.cfg.leakage_corr_threshold:
                suspects[col] = round(corr, 4)
        for col in BINARY_COLUMNS | set(CATEGORICAL_LEVELS):
            if col not in df.columns:
                continue
            # exact-match leakage: feature equals target row-for-row
            try:
                if (df[col].astype(str) == y.astype(int).astype(str)).mean() > 0.99:
                    suspects[col] = 1.0
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
        ok = not suspects
        return CheckResult(
            name="leakage_screen",
            status="passed" if ok else "failed",
            details={"suspects": suspects, "threshold": self.cfg.leakage_corr_threshold},
            message=f"leakage suspects: {suspects}" if suspects else "",
        )
