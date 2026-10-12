"""Unit tests: feature engineering, preprocessing pipeline, splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.data.preprocessing import build_pipeline, model_feature_names
from sentinel.data.splitting import split_dataset
from sentinel.features.engineering import DERIVED_CATEGORICAL, DERIVED_NUMERIC, FeatureEngineer


def _mini_df(n=50, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "tenure_months": rng.integers(0, 72, n),
            "monthly_charges": rng.uniform(20, 100, n).round(2),
            "total_charges": rng.uniform(100, 5000, n).round(2),
            "support_tickets_30d": rng.integers(0, 5, n),
        }
    )


def test_feature_engineer_adds_derived_columns():
    out = FeatureEngineer().fit_transform(_mini_df())
    for col in [*DERIVED_NUMERIC, *DERIVED_CATEGORICAL]:
        assert col in out.columns
    assert out["charges_per_month"].notna().all()
    assert set(out["tenure_bucket"].unique()) <= {"0-6m", "6-12m", "12-24m", "24-48m", "48m+"}


def test_feature_engineer_validates_required_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        FeatureEngineer().fit_transform(pd.DataFrame({"tenure_months": [1]}))


def test_feature_engineer_deterministic():
    df = _mini_df()
    a = FeatureEngineer().fit_transform(df)
    b = FeatureEngineer().fit_transform(df)
    pd.testing.assert_frame_equal(a, b)


def test_model_feature_names_include_derived(settings):
    numeric, categorical = model_feature_names(settings)
    assert set(DERIVED_NUMERIC) <= set(numeric)
    assert set(DERIVED_CATEGORICAL) <= set(categorical)
    assert set(settings.features.numeric) <= set(numeric)


def test_pipeline_fits_and_predicts(settings, raw_df):
    feature_cols = [*settings.features.numeric, *settings.features.categorical]
    pipe = build_pipeline(settings)
    small = raw_df.head(400)
    pipe.fit(small[feature_cols], small[settings.features.target])
    proba = pipe.predict_proba(small[feature_cols][:5])[:, 1]
    assert proba.shape == (5,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_handles_unseen_category(settings, raw_df):
    feature_cols = [*settings.features.numeric, *settings.features.categorical]
    pipe = build_pipeline(settings)
    small = raw_df.head(400).copy()
    pipe.fit(small[feature_cols], small[settings.features.target])
    row = small[feature_cols].head(1).copy()
    row["payment_method"] = "Crypto"  # unseen at training time
    assert pipe.predict(row).shape == (1,)  # handle_unknown="ignore"


def test_split_sizes_and_stratification(settings, raw_df):
    train_df, val_df, test_df, report = split_dataset(settings, raw_df, "ds-test")
    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(raw_df)
    assert report.dataset_version == "ds-test"
    # stratified: positive rates within a few points of each other
    rates = [report.train.positive_rate, report.val.positive_rate, report.test.positive_rate]
    assert max(rates) - min(rates) < 0.08
    # no id overlap between splits
    ids = settings.features.id_column
    assert set(train_df[ids]) & set(val_df[ids]) == set()
    assert set(train_df[ids]) & set(test_df[ids]) == set()


def test_split_is_reproducible(settings, raw_df):
    a_train, _, _, a_report = split_dataset(settings, raw_df, "ds-test")
    b_train, _, _, b_report = split_dataset(settings, raw_df, "ds-test")
    assert a_report.train.content_sha256 == b_report.train.content_sha256
    pd.testing.assert_frame_equal(a_train, b_train)
