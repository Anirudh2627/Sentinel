"""Data drift detection: reference (training) vs current (production).

Methods, chosen per feature type:
    * PSI (Population Stability Index) for numeric (quantile-binned) and
      categorical (frequency) features — the default, industry-standard
      measure for tabular monitoring.
    * Two-sample Kolmogorov–Smirnov statistic (+ asymptotic p-value) for
      numeric features when ``method: ks`` is configured.

Drift is an *input-distribution* signal. It does NOT by itself mean the model
is broken — severity/rollback policy for drift is configured separately from
performance-degradation signals (see monitoring.yaml).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from sentinel.config.settings import DriftConfig

EPS = 1e-6


# --------------------------------------------------------------------------- #
# Reference statistics
# --------------------------------------------------------------------------- #
def fit_reference_stats(
    df: pd.DataFrame,
    dataset_version: str,
    numeric_features: list[str],
    categorical_features: list[str],
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute the training-time distribution baseline (JSON-serialisable)."""
    features: dict[str, Any] = {}
    for col in numeric_features:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 2:
            edges = np.array([values.min() - 0.5, values.max() + 0.5])
        counts, _ = np.histogram(values, bins=edges)
        probs = counts / max(counts.sum(), 1)
        features[col] = {
            "kind": "numeric",
            "bin_edges": edges.tolist(),
            "bin_probs": probs.tolist(),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "quantiles": {
                q: float(np.quantile(values, float(q.strip("p")) / 100))
                for q in ("p5", "p25", "p50", "p75", "p95")
            },
        }
    for col in categorical_features:
        if col not in df.columns:
            continue
        freq = df[col].astype(str).value_counts(normalize=True)
        features[col] = {
            "kind": "categorical",
            "categories": freq.index.tolist(),
            "probs": freq.to_numpy(dtype=float).tolist(),
        }
    return {
        "dataset_version": dataset_version,
        "created_at": datetime.now(UTC).isoformat(),
        "n_samples": int(len(df)),
        "n_bins": n_bins,
        "features": features,
    }


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def compute_psi(reference_probs: np.ndarray, current_probs: np.ndarray) -> float:
    """Population Stability Index. <0.1 stable, 0.1–0.25 moderate, >0.25 significant."""
    p = np.clip(np.asarray(reference_probs, dtype=float), EPS, None)
    q = np.clip(np.asarray(current_probs, dtype=float), EPS, None)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def ks_two_sample(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample KS statistic D and asymptotic p-value (no scipy needed)."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if len(a) == 0 or len(b) == 0:
        return 0.0, 1.0
    all_values = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_values, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_values, side="right") / len(b)
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    n, m = len(a), len(b)
    en = math.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    # Kolmogorov distribution survival function (series approximation).
    p_value = 2.0 * sum((-1) ** (i - 1) * math.exp(-2.0 * (i**2) * (lam**2)) for i in range(1, 101))
    return d, float(min(max(p_value, 0.0), 1.0))


def _categorical_psi(ref: dict[str, Any], current: pd.Series) -> tuple[float, dict[str, Any]]:
    categories: list[str] = list(ref["categories"])
    ref_probs = np.asarray(ref["probs"], dtype=float)
    counts = current.astype(str).value_counts()
    cur_counts = np.array([float(counts.get(c, 0)) for c in categories], dtype=float)
    unknown = float(len(current) - cur_counts.sum())
    if unknown > 0:
        categories = [*categories, "__unknown__"]
        ref_probs = np.append(ref_probs, EPS)
        cur_counts = np.append(cur_counts, unknown)
    total = max(cur_counts.sum(), 1.0)
    psi = compute_psi(ref_probs, cur_counts / total)
    detail = {
        "reference_freq": dict(
            zip(
                ref["categories"],
                np.round(ref_probs[: len(ref["categories"])], 4).tolist(),
                strict=False,
            )
        ),
        "current_freq": {
            c: round(float(n) / total, 4) for c, n in zip(categories, cur_counts, strict=False)
        },
    }
    return psi, detail


# --------------------------------------------------------------------------- #
# Report models
# --------------------------------------------------------------------------- #
class FeatureDrift(BaseModel):
    feature: str
    kind: Literal["numeric", "categorical"]
    score: float
    threshold: float
    drifted: bool
    detail: dict[str, Any] = Field(default_factory=dict)


class DriftReport(BaseModel):
    method: Literal["psi", "ks"]
    reference_dataset_version: str
    n_samples: int
    is_drifted: bool
    max_score: float
    n_drifted_features: int
    features: list[FeatureDrift] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def top_features(self, k: int = 5) -> list[FeatureDrift]:
        return sorted(self.features, key=lambda f: f.score, reverse=True)[:k]


class DriftDetector:
    """Compare a current sample dataframe against persisted reference stats."""

    def __init__(self, reference: dict[str, Any], config: DriftConfig):
        self.reference = reference
        self.config = config
        self.method = config.method

    def detect(self, current_df: pd.DataFrame) -> DriftReport:
        results: list[FeatureDrift] = []

        for name, ref in self.reference.get("features", {}).items():
            if name not in current_df.columns:
                continue
            series = current_df[name]
            if ref["kind"] == "categorical":
                score, detail = _categorical_psi(ref, series)
                results.append(
                    FeatureDrift(
                        feature=name,
                        kind="categorical",
                        score=round(score, 6),
                        threshold=self.config.psi_threshold,
                        drifted=score > self.config.psi_threshold,
                        detail=detail,
                    )
                )
            else:
                values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
                if len(values) == 0:
                    continue
                if self.method == "ks":
                    # Reconstruct an approximate reference sample is not stored;
                    # KS needs raw values — use binned centers as a proxy sample
                    # weighted by reference probabilities.
                    edges = np.asarray(ref["bin_edges"], dtype=float)
                    centers = (edges[:-1] + edges[1:]) / 2.0
                    probs = np.asarray(ref["bin_probs"], dtype=float)
                    ref_sample = np.repeat(centers, np.ceil(probs * 1000).astype(int))
                    score_raw, p_value = ks_two_sample(ref_sample, values)
                    results.append(
                        FeatureDrift(
                            feature=name,
                            kind="numeric",
                            score=round(score_raw, 6),
                            threshold=self.config.ks_threshold,
                            drifted=p_value < 0.01 and score_raw > self.config.ks_threshold,
                            detail={"ks_p_value": round(p_value, 6), "mean": float(values.mean())},
                        )
                    )
                else:
                    edges = np.asarray(ref["bin_edges"], dtype=float)
                    counts, _ = np.histogram(values, bins=edges)
                    outside = len(values) - counts.sum()
                    cur_probs = counts.astype(float)
                    if outside > 0:  # fold out-of-range mass into the edge bins
                        cur_probs[0] += outside / 2.0
                        cur_probs[-1] += outside / 2.0
                    cur_probs = cur_probs / max(cur_probs.sum(), 1.0)
                    score = compute_psi(np.asarray(ref["bin_probs"], dtype=float), cur_probs)
                    results.append(
                        FeatureDrift(
                            feature=name,
                            kind="numeric",
                            score=round(score, 6),
                            threshold=self.config.psi_threshold,
                            drifted=score > self.config.psi_threshold,
                            detail={
                                "current_mean": round(float(values.mean()), 4),
                                "reference_mean": round(float(ref.get("mean", float("nan"))), 4),
                            },
                        )
                    )

        n_drifted = sum(1 for r in results if r.drifted)
        max_score = max((r.score for r in results), default=0.0)
        return DriftReport(
            method=self.method,
            reference_dataset_version=self.reference.get("dataset_version", "unknown"),
            n_samples=int(len(current_df)),
            is_drifted=n_drifted > 0,
            max_score=round(max_score, 6),
            n_drifted_features=n_drifted,
            features=sorted(results, key=lambda r: r.score, reverse=True),
        )
