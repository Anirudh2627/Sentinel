"""Preprocessing: build the persisted sklearn pipeline.

The full transform chain (feature engineering -> imputation -> encoding ->
scaling -> estimator) is a single ``sklearn.pipeline.Pipeline`` artifact, so
serving can never diverge from training.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sentinel.config import SentinelSettings
from sentinel.features.engineering import FeatureEngineer, feature_names


def build_estimator(settings: SentinelSettings):
    """Instantiate the configured estimator with its hyperparameters."""
    algo = settings.model.algorithm
    params = settings.model.params_for(algo)
    random_state = settings.model.random_state
    if algo == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state, **params)
    if algo == "random_forest":
        return RandomForestClassifier(random_state=random_state, **params)
    if algo == "logistic_regression":
        return LogisticRegression(random_state=random_state, **params)
    raise ValueError(f"unknown algorithm: {algo}")


def build_pipeline(settings: SentinelSettings) -> Pipeline:
    """Assemble the train/serve-safe preprocessing + model pipeline."""
    numeric, categorical = feature_names(settings.features.numeric, settings.features.categorical)

    numeric_steps: list = [("imputer", SimpleImputer(strategy="median"))]
    if settings.model.algorithm == "logistic_regression":
        # Tree ensembles are scale-invariant; the linear model is not.
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("feature_engineer", FeatureEngineer()),
            ("preprocess", preprocessor),
            ("estimator", build_estimator(settings)),
        ]
    )


def model_feature_names(settings: SentinelSettings) -> tuple[list[str], list[str]]:
    """Feature names (post-engineering) the model expects, split by type."""
    return feature_names(settings.features.numeric, settings.features.categorical)
