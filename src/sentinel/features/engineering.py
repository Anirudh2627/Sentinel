"""Feature engineering.

``FeatureEngineer`` is a scikit-learn-compatible transformer that lives
*inside* the persisted model pipeline, guaranteeing train/serve skew-free
feature computation: whatever we compute at training time is exactly what
runs at inference time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TENURE_BUCKETS = [0, 6, 12, 24, 48, 73]  # right-open edges -> bucket index 0..4
TENURE_BUCKET_LABELS = ["0-6m", "6-12m", "12-24m", "24-48m", "48m+"]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add derived features.

    Derived columns:
        * ``charges_per_month``  — total_charges / (tenure + 1)
        * ``tickets_per_month``  — support tickets normalised by tenure
        * ``tenure_bucket``      — categorical tenure band
    """

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D102
        self._validate_columns(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: D102
        self._validate_columns(X)
        out = X.copy()
        tenure = pd.to_numeric(out["tenure_months"], errors="coerce").fillna(0.0) + 1.0
        total = pd.to_numeric(out["total_charges"], errors="coerce").fillna(0.0)
        tickets = pd.to_numeric(out["support_tickets_30d"], errors="coerce").fillna(0.0)

        out["charges_per_month"] = (total / tenure).astype(float)
        out["tickets_per_month"] = (tickets / tenure).astype(float)
        buckets = np.digitize(out["tenure_months"].to_numpy(dtype=float), TENURE_BUCKETS[1:-1])
        out["tenure_bucket"] = [TENURE_BUCKET_LABELS[int(b)] for b in buckets]
        return out

    @staticmethod
    def _validate_columns(X: pd.DataFrame) -> None:
        required = {"tenure_months", "total_charges", "support_tickets_30d"}
        missing = required - set(X.columns)
        if missing:
            raise ValueError(f"FeatureEngineer missing required columns: {sorted(missing)}")


DERIVED_NUMERIC = ["charges_per_month", "tickets_per_month"]
DERIVED_CATEGORICAL = ["tenure_bucket"]


def feature_names(numeric: list[str], categorical: list[str]) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature names *after* engineering."""
    return [*numeric, *DERIVED_NUMERIC], [*categorical, *DERIVED_CATEGORICAL]
