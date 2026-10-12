"""Unit tests: drift detection (PSI, KS, detector behaviour)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sentinel.config.settings import DriftConfig
from sentinel.monitoring.drift import (
    DriftDetector,
    compute_psi,
    fit_reference_stats,
    ks_two_sample,
)

NUMERIC = ["x1", "x2"]
CATEGORICAL = ["c1"]


def _df(n=2000, seed=0, shift=0.0, cat_p=(0.5, 0.5)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x1": rng.normal(0.0 + shift, 1.0, n),
            "x2": rng.normal(5.0, 1.0, n),
            "c1": rng.choice(["a", "b"], n, p=list(cat_p)),
        }
    )


def test_psi_identical_distributions_near_zero():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert compute_psi(p, p) < 1e-9


def test_psi_shifted_distribution_large():
    p = np.array([0.7, 0.2, 0.05, 0.05])
    q = np.array([0.05, 0.05, 0.2, 0.7])
    assert compute_psi(p, q) > 0.25


def test_psi_handles_zero_bins():
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([0.4, 0.4, 0.2])
    value = compute_psi(p, q)
    assert np.isfinite(value) and value > 0


def test_ks_same_sample_high_pvalue():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 500)
    b = rng.normal(0, 1, 500)
    d, p = ks_two_sample(a, b)
    assert 0 <= d <= 1
    assert p > 0.05


def test_ks_shifted_sample_significant():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 500)
    b = rng.normal(3, 1, 500)
    d, p = ks_two_sample(a, b)
    assert d > 0.5
    assert p < 0.001


def test_fit_reference_stats_structure():
    ref = fit_reference_stats(_df(), "ds-test", NUMERIC, CATEGORICAL)
    assert ref["dataset_version"] == "ds-test"
    assert set(ref["features"]) == {"x1", "x2", "c1"}
    assert ref["features"]["x1"]["kind"] == "numeric"
    assert len(ref["features"]["x1"]["bin_edges"]) > 2
    assert ref["features"]["c1"]["kind"] == "categorical"
    assert sorted(ref["features"]["c1"]["categories"]) == ["a", "b"]


def test_detector_no_drift_on_same_distribution():
    ref = fit_reference_stats(_df(seed=0), "ds-test", NUMERIC, CATEGORICAL)
    detector = DriftDetector(ref, DriftConfig(psi_threshold=0.25))
    report = detector.detect(_df(seed=99))
    assert not report.is_drifted
    assert report.max_score < 0.25


def test_detector_flags_numeric_shift():
    ref = fit_reference_stats(_df(seed=0), "ds-test", NUMERIC, CATEGORICAL)
    detector = DriftDetector(ref, DriftConfig(psi_threshold=0.25))
    report = detector.detect(_df(seed=99, shift=2.0))
    assert report.is_drifted
    drifted = {f.feature for f in report.features if f.drifted}
    assert "x1" in drifted


def test_detector_flags_categorical_shift():
    ref = fit_reference_stats(_df(seed=0), "ds-test", NUMERIC, CATEGORICAL)
    detector = DriftDetector(ref, DriftConfig(psi_threshold=0.25))
    report = detector.detect(_df(seed=99, cat_p=(0.95, 0.05)))
    drifted = {f.feature for f in report.features if f.drifted}
    assert "c1" in drifted


def test_detector_unknown_category_bucketed():
    ref = fit_reference_stats(_df(seed=0), "ds-test", NUMERIC, CATEGORICAL)
    detector = DriftDetector(ref, DriftConfig(psi_threshold=0.25))
    current = _df(seed=5)
    current["c1"] = "zzz"  # never seen in training
    report = detector.detect(current)
    c1 = next(f for f in report.features if f.feature == "c1")
    assert c1.drifted
    assert "__unknown__" in c1.detail["current_freq"]


def test_detector_ks_method():
    ref = fit_reference_stats(_df(seed=0), "ds-test", NUMERIC, CATEGORICAL)
    cfg = DriftConfig(method="ks", ks_threshold=0.2)
    report = DriftDetector(ref, cfg).detect(_df(seed=99, shift=3.0))
    assert report.method == "ks"
    x1 = next(f for f in report.features if f.feature == "x1")
    assert x1.drifted
    assert x1.detail["ks_p_value"] < 0.01
