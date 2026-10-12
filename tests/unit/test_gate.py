"""Unit tests: the promotion quality gate (threshold logic)."""

from __future__ import annotations

from sentinel.config.settings import EvaluationConfig
from sentinel.training.evaluate import run_quality_gate

GOOD = {"f1": 0.70, "precision": 0.68, "recall": 0.66, "roc_auc": 0.80}


def _cfg(**over) -> EvaluationConfig:
    base = {
        "min_f1": 0.55,
        "min_precision": 0.50,
        "min_recall": 0.45,
        "min_auc": 0.65,
        "require_beat_production": False,
        "beat_margin": 0.0,
    }
    base.update(over)
    return EvaluationConfig(**base)


def test_gate_passes_good_candidate():
    result = run_quality_gate(GOOD, _cfg())
    assert result.passed
    assert all(c.passed for c in result.checks)


def test_gate_blocks_low_f1():
    result = run_quality_gate({**GOOD, "f1": 0.40}, _cfg())
    assert not result.passed
    failed = {c.name for c in result.checks if not c.passed}
    assert failed == {"min_f1"}


def test_gate_blocks_low_precision():
    result = run_quality_gate({**GOOD, "precision": 0.30}, _cfg(min_precision=0.5))
    assert not result.passed
    assert "min_precision" in {c.name for c in result.checks if not c.passed}


def test_gate_blocks_low_recall():
    result = run_quality_gate({**GOOD, "recall": 0.10}, _cfg())
    assert not result.passed


def test_gate_blocks_low_auc():
    result = run_quality_gate({**GOOD, "roc_auc": 0.55}, _cfg())
    assert not result.passed


def test_gate_beat_production_required():
    prod = {"f1": 0.75, "precision": 0.7, "recall": 0.7, "roc_auc": 0.85}
    cfg = _cfg(require_beat_production=True, primary_metric="f1")
    result = run_quality_gate(GOOD, cfg, production_metrics=prod)
    assert not result.passed
    assert any(c.name == "beat_production[f1]" and not c.passed for c in result.checks)


def test_gate_beat_production_satisfied():
    prod = {"f1": 0.60, "precision": 0.6, "recall": 0.6, "roc_auc": 0.70}
    cfg = _cfg(require_beat_production=True, primary_metric="f1")
    result = run_quality_gate(GOOD, cfg, production_metrics=prod)
    assert result.passed


def test_gate_beat_production_margin():
    prod = {"f1": 0.69, "precision": 0.6, "recall": 0.6, "roc_auc": 0.70}
    cfg = _cfg(require_beat_production=True, primary_metric="f1", beat_margin=0.05)
    result = run_quality_gate(GOOD, cfg, production_metrics=prod)  # 0.70 < 0.69+0.05
    assert not result.passed


def test_gate_no_production_skips_comparison():
    cfg = _cfg(require_beat_production=True)
    result = run_quality_gate(GOOD, cfg, production_metrics=None)
    assert result.passed
    assert all(c.name != "beat_production[f1]" for c in result.checks)


def test_gate_missing_metric_fails_closed():
    result = run_quality_gate({"precision": 0.9}, _cfg())  # no f1/auc/recall
    assert not result.passed


def test_gate_markdown_report():
    result = run_quality_gate(GOOD, _cfg())
    md = result.to_markdown()
    assert "PASSED" in md
