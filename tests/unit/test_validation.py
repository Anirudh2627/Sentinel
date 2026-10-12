"""Unit tests: data validation checks."""

from __future__ import annotations

import numpy as np
import pytest

from sentinel.config.settings import SentinelSettings
from sentinel.data.validation import DataValidationError, DataValidator


def _validator(settings: SentinelSettings) -> DataValidator:
    settings.validation.min_rows = 50
    return DataValidator(settings)


def test_healthy_data_passes(settings, raw_df):
    report = _validator(settings).validate(raw_df)
    assert report.passed
    assert all(c.status != "failed" for c in report.checks)


def test_missing_column_fails(settings, raw_df):
    df = raw_df.drop(columns=["monthly_charges"])
    with pytest.raises(DataValidationError) as exc:
        _validator(settings).validate(df)
    names = [c.name for c in exc.value.report.failed_checks()]
    assert "schema" in names


def test_unexpected_column_is_warning(settings, raw_df):
    df = raw_df.assign(mystery=1)
    report = _validator(settings).validate(df)
    schema = next(c for c in report.checks if c.name == "schema")
    assert schema.status == "warning"
    assert report.passed


def test_excess_missing_values_fails(settings, raw_df):
    df = raw_df.copy()
    idx = np.random.default_rng(0).choice(len(df), size=int(len(df) * 0.1), replace=False)
    df.loc[idx, "tenure_months"] = np.nan
    with pytest.raises(DataValidationError) as exc:
        _validator(settings).validate(df)
    assert "missing_values" in [c.name for c in exc.value.report.failed_checks()]


def test_out_of_range_fails(settings, raw_df):
    df = raw_df.copy()
    df.loc[df.index[:5], "tenure_months"] = 500  # domain bound is 72
    with pytest.raises(DataValidationError) as exc:
        _validator(settings).validate(df)
    assert "numeric_ranges" in [c.name for c in exc.value.report.failed_checks()]


def test_unknown_categorical_level_fails(settings, raw_df):
    df = raw_df.copy()
    df.loc[df.index[:3], "contract_type"] = "Lifetime deal"
    with pytest.raises(DataValidationError) as exc:
        _validator(settings).validate(df)
    assert "categorical_levels" in [c.name for c in exc.value.report.failed_checks()]


def test_duplicate_ids_fail(settings, raw_df):
    df = raw_df.copy()
    df.loc[df.index[10], "customer_id"] = df.loc[df.index[0], "customer_id"]
    with pytest.raises(DataValidationError) as exc:
        _validator(settings).validate(df)
    assert "duplicates" in [c.name for c in exc.value.report.failed_checks()]


def test_target_leakage_detected(settings, raw_df):
    df = raw_df.copy()
    # a feature that is an exact copy of the target must be flagged
    df["monthly_charges"] = df["churned"].astype(float) * 50.0
    with pytest.raises(DataValidationError) as exc:
        _validator(settings).validate(df)
    failed = {c.name for c in exc.value.report.failed_checks()}
    assert "leakage_screen" in failed


def test_non_binary_target_fails(settings, raw_df):
    df = raw_df.copy()
    df.loc[df.index[:10], "churned"] = 2
    with pytest.raises(DataValidationError):
        _validator(settings).validate(df)


def test_non_strict_mode_returns_failed_report(settings, raw_df):
    settings.validation.strict = False
    df = raw_df.drop(columns=["monthly_charges"])
    report = DataValidator(settings).validate(df)
    assert not report.passed  # reported, not raised


def test_report_markdown(settings, raw_df):
    report = _validator(settings).validate(raw_df)
    md = report.to_markdown()
    assert "PASSED" in md
    assert "| check | status | message |" in md
