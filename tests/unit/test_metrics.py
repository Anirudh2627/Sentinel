"""Unit tests: metric math and window aggregation."""

from __future__ import annotations

import numpy as np

from sentinel.monitoring.metrics import compute_window_stats, percentile
from sentinel.training.evaluate import evaluate_predictions


def test_percentile_basic():
    values = list(range(1, 101))
    assert percentile(values, 50) == 50.5
    assert percentile(values, 95) == 95.05
    assert np.isnan(percentile([], 50))


def test_evaluate_predictions_perfect():
    y = np.array([0, 1, 0, 1, 1, 0])
    m = evaluate_predictions(y, y, np.array([0.01, 0.99, 0.02, 0.98, 0.97, 0.03]))
    assert m.accuracy == 1.0
    assert m.f1 == 1.0
    assert m.roc_auc == 1.0
    assert m.confusion_matrix == [[3, 0], [0, 3]]


def test_evaluate_predictions_inverted_is_bad():
    y = np.array([0, 1, 0, 1, 1, 0])
    inv = 1 - y
    m = evaluate_predictions(y, inv, np.array([0.99, 0.01, 0.98, 0.02, 0.03, 0.97]))
    assert m.accuracy == 0.0
    assert m.roc_auc == 0.0


def test_evaluate_predictions_handles_zero_division():
    y = np.array([0, 0, 0, 0])
    pred = np.array([0, 0, 0, 0])
    m = evaluate_predictions(y, pred, np.array([0.1, 0.2, 0.3, 0.4]))
    assert m.precision == 0.0  # zero_division=0
    assert np.isnan(m.roc_auc)  # single-class


def test_window_stats_aggregation():
    records = []
    for i in range(10):
        records.append(
            {
                "ts": f"2026-01-01T00:00:{i:02d}+00:00",
                "model_version": "v1",
                "status": "error" if i < 2 else "ok",
                "latency_ms": 10.0 + i,
                "prediction": 1 if i % 2 == 0 else 0,
                "probability": 0.7 if i % 2 == 0 else 0.2,
            }
        )
    stats = compute_window_stats("v1", records, window_seconds=10)
    assert stats.n_requests == 10
    assert stats.n_errors == 2
    assert stats.error_rate == 0.2
    assert stats.throughput_rps == 1.0
    assert stats.positive_rate == 0.5
    assert stats.latency_ms is not None
    assert stats.latency_ms.p50 == 14.5
    assert stats.latency_ms.max == 19.0


def test_window_stats_empty():
    stats = compute_window_stats("v1", [], window_seconds=60)
    assert stats.n_requests == 0
    assert stats.error_rate == 0.0
    assert stats.positive_rate is None
