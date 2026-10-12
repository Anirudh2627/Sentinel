"""Dataset ingestion: reproducible synthetic generator + CSV loading.

The default source is a seeded synthetic customer-churn dataset so the whole
platform (including CI) is reproducible without external downloads. A CSV
source is supported for real data via ``data.source: csv``.

Every ingested dataset gets a content-addressed version (``ds-<sha1[:12]>``)
and a persisted *dataset card* (JSON) with provenance metadata.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from sentinel.config import SentinelSettings, get_logger, log_event

logger = get_logger(__name__)

CONTRACT_TYPES = ["Month-to-month", "One year", "Two year"]
INTERNET_SERVICES = ["DSL", "Fiber optic", "No"]
PAYMENT_METHODS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


class DatasetCard(BaseModel):
    """Provenance metadata for a dataset version."""

    dataset_version: str
    source: str  # synthetic | csv
    created_at: str
    n_rows: int
    n_columns: int
    positive_rate: float
    content_sha256: str
    seed: int | None = None
    generator_params: dict = Field(default_factory=dict)
    source_path: str | None = None
    column_dtypes: dict[str, str] = Field(default_factory=dict)


def _content_hash(df: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(df, index=False).values.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _dataset_version(content_sha: str) -> str:
    return f"ds-{content_sha[:12]}"


def generate_synthetic_churn(
    n_samples: int = 20_000,
    seed: int = 42,
    drift_spec: dict | None = None,
    concept_drift: bool = False,
) -> pd.DataFrame:
    """Generate a reproducible customer-churn dataset with known signal.

    The generator embeds realistic relationships (contract type, tenure,
    support tickets, charges) plus interaction effects and label noise, so
    models reach meaningful-but-imperfect performance — good enough to make
    evaluation gates and drift detection exercise real code paths.

    Simulation hooks (used to produce honest production degradation):
        * ``drift_spec`` — transforms applied to the *features before* labels
          are computed, i.e. pure covariate shift (the label mechanism is
          unchanged, so a good model can still perform acceptably).
        * ``concept_drift`` — regenerates labels under an *inverted*
          relationship for charges/tickets, i.e. genuine concept drift that
          measurably degrades a model trained on the original concept.
    """
    rng = np.random.default_rng(seed)

    tenure_months = np.clip(rng.gamma(shape=2.2, scale=14.0, size=n_samples), 0, 72).round()
    contract_type = rng.choice(CONTRACT_TYPES, size=n_samples, p=[0.55, 0.25, 0.20])
    internet_service = rng.choice(INTERNET_SERVICES, size=n_samples, p=[0.34, 0.44, 0.22])
    payment_method = rng.choice(PAYMENT_METHODS, size=n_samples, p=[0.30, 0.22, 0.24, 0.24])

    base_charge = 25.0
    monthly_charges = np.clip(
        base_charge
        + rng.normal(0.0, 18.0, size=n_samples)
        + np.where(internet_service == "Fiber optic", 22.0, 0.0)
        + np.where(internet_service == "DSL", 8.0, 0.0)
        + 4.5 * rng.integers(0, 4, size=n_samples),
        18.0,
        120.0,
    ).round(2)

    num_services = rng.integers(1, 7, size=n_samples)
    support_tickets_30d = rng.poisson(
        lam=0.6 + 0.10 * np.where(internet_service == "Fiber optic", 1, 0), size=n_samples
    )
    support_tickets_30d = np.minimum(support_tickets_30d, 9)
    avg_session_minutes = np.clip(rng.normal(14.0, 6.0, size=n_samples), 0.5, 45.0).round(2)
    promo_active = rng.integers(0, 2, size=n_samples)
    senior_citizen = (rng.random(size=n_samples) < 0.16).astype(int)

    # --- optional covariate drift (applied BEFORE labels => consistent data)
    drift_spec = drift_spec or {}
    numeric_ops = drift_spec.get("numeric") or {}
    if "monthly_charges" in numeric_ops:
        monthly_charges = _apply_numeric_ops(
            monthly_charges, numeric_ops["monthly_charges"], rng, clip=(18.0, 999.0)
        )
    if "tenure_months" in numeric_ops:
        tenure_months = _apply_numeric_ops(
            tenure_months.astype(float), numeric_ops["tenure_months"], rng, clip=(0, 72)
        ).round()
    if "support_tickets_30d" in numeric_ops:
        support_tickets_30d = _apply_numeric_ops(
            support_tickets_30d.astype(float), numeric_ops["support_tickets_30d"], rng, clip=(0, 49)
        ).round()
    if "avg_session_minutes" in numeric_ops:
        avg_session_minutes = _apply_numeric_ops(
            avg_session_minutes, numeric_ops["avg_session_minutes"], rng, clip=(0.5, 1440.0)
        ).round(2)
    categorical_ops = drift_spec.get("categorical") or {}
    if "contract_type" in categorical_ops:
        contract_type = _resample_categorical(contract_type, categorical_ops["contract_type"], rng)
    if "internet_service" in categorical_ops:
        internet_service = _resample_categorical(
            internet_service, categorical_ops["internet_service"], rng
        )
    if "payment_method" in categorical_ops:
        payment_method = _resample_categorical(
            payment_method, categorical_ops["payment_method"], rng
        )

    total_charges = np.clip(
        monthly_charges * np.maximum(tenure_months, 1) * rng.normal(1.0, 0.08, size=n_samples),
        18.0,
        None,
    ).round(2)

    # Latent churn propensity (logistic model with interactions).
    if concept_drift:
        # Inverted concept: high charges/low tickets now indicate churn.
        z = (
            -1.20
            - 0.030 * monthly_charges
            - 0.45 * support_tickets_30d
            + 0.021 * tenure_months
            + 0.90 * (contract_type == "Two year")
            + 0.012 * avg_session_minutes
            - 0.30 * senior_citizen
            + rng.normal(0.0, 0.9, size=n_samples)
        )
    else:
        z = (
            -1.55
            + 1.35 * (contract_type == "Month-to-month")
            - 0.55 * (contract_type == "Two year")
            + 0.028 * monthly_charges
            + 0.34 * support_tickets_30d
            - 0.023 * tenure_months
            - 0.017 * avg_session_minutes
            + 0.30 * senior_citizen
            - 0.25 * promo_active
            + 0.30 * (internet_service == "Fiber optic")
            + 0.45 * ((contract_type == "Month-to-month") & (support_tickets_30d >= 2))
            + rng.normal(0.0, 0.9, size=n_samples)  # irreducible noise
        )
    prob = 1.0 / (1.0 + np.exp(-z))
    churned = (rng.random(size=n_samples) < prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": [f"C-{seed}-{i:07d}" for i in range(n_samples)],
            "tenure_months": tenure_months.astype(int),
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "num_services": num_services.astype(int),
            "support_tickets_30d": support_tickets_30d.astype(int),
            "avg_session_minutes": avg_session_minutes,
            "promo_active": promo_active.astype(int),
            "senior_citizen": senior_citizen.astype(int),
            "churned": churned,
        }
    )
    return df


def _apply_numeric_ops(
    values: np.ndarray, ops: dict, rng: np.random.Generator, clip: tuple[float, float] | None = None
) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if "multiply" in ops:
        out = out * float(ops["multiply"])
    if "add" in ops:
        out = out + float(ops["add"])
    if "set" in ops:
        out = np.full_like(out, float(ops["set"]))
    if "noise_std" in ops:
        out = out + rng.normal(0.0, float(ops["noise_std"]), size=len(out))
    if clip is not None:
        out = np.clip(out, clip[0], clip[1])
    return out


def _resample_categorical(values: np.ndarray, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    probs = cfg.get("probabilities")
    if not probs:
        return values
    categories = list(probs.keys())
    weights = np.array([float(probs[c]) for c in categories])
    weights = weights / weights.sum()
    return rng.choice(categories, size=len(values), p=weights)


def apply_drift(df: pd.DataFrame, spec: dict, seed: int = 0) -> pd.DataFrame:
    """Apply a drift specification to a dataframe (post-hoc transform).

    ``spec`` shape::

        {
          "numeric": {"monthly_charges": {"multiply": 1.5, "add": 3.0, "set": 99.0}, ...},
          "categorical": {"contract_type": {"probabilities": {"Month-to-month": 0.8, ...}}},
        }

    Note: for *label-consistent* covariate-shift streams, prefer passing
    ``drift_spec`` to :func:`generate_synthetic_churn` (drift applied before
    labels are drawn). This function is for transforming existing frames.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    for col, ops in (spec.get("numeric") or {}).items():
        if col not in out.columns:
            continue
        out[col] = _apply_numeric_ops(
            pd.to_numeric(out[col], errors="coerce").fillna(0.0).to_numpy(), ops, rng
        ).round(2)
    for col, cfg in (spec.get("categorical") or {}).items():
        if col not in out.columns:
            continue
        out[col] = _resample_categorical(out[col].to_numpy(), cfg, rng)
    return out


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV source not found: {path}")
    return pd.read_csv(path)


def build_dataset(settings: SentinelSettings, n_samples: int | None = None) -> pd.DataFrame:
    """Load/generate the raw dataset according to configuration."""
    if settings.data.source == "csv":
        df = load_csv(settings.resolve(settings.data.csv_path))
    else:
        df = generate_synthetic_churn(
            n_samples=n_samples or settings.data.n_samples,
            seed=settings.data.seed,
        )
    return df


def persist_dataset(
    settings: SentinelSettings,
    df: pd.DataFrame,
    source: str,
    seed: int | None = None,
    generator_params: dict | None = None,
) -> tuple[Path, DatasetCard]:
    """Write the dataset + card to ``data/processed`` and return (csv_path, card)."""
    content_sha = _content_hash(df)
    version = _dataset_version(content_sha)
    out_dir = settings.resolve(settings.paths.data_processed) / version
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "data.csv"
    df.to_csv(csv_path, index=False)

    target = settings.features.target
    positive_rate = float(df[target].mean()) if target in df.columns else float("nan")
    card = DatasetCard(
        dataset_version=version,
        source=source,
        created_at=datetime.now(UTC).isoformat(),
        n_rows=int(len(df)),
        n_columns=int(df.shape[1]),
        positive_rate=positive_rate,
        content_sha256=content_sha,
        seed=seed,
        generator_params=generator_params or {},
        source_path=str(csv_path),
        column_dtypes={c: str(t) for c, t in df.dtypes.items()},
    )
    (out_dir / "dataset_card.json").write_text(card.model_dump_json(indent=2))
    log_event(
        logger,
        logging.INFO,
        "dataset persisted",
        dataset_version=version,
        rows=card.n_rows,
        positive_rate=round(positive_rate, 4),
        path=str(csv_path),
    )
    return csv_path, card


def load_dataset_card(settings: SentinelSettings, dataset_version: str) -> DatasetCard:
    path = settings.resolve(settings.paths.data_processed) / dataset_version / "dataset_card.json"
    if not path.is_file():
        raise FileNotFoundError(f"dataset card not found: {path}")
    return DatasetCard.model_validate(json.loads(path.read_text()))


def load_dataset_frame(settings: SentinelSettings, dataset_version: str) -> pd.DataFrame:
    path = settings.resolve(settings.paths.data_processed) / dataset_version / "data.csv"
    if not path.is_file():
        raise FileNotFoundError(f"dataset not found: {path}")
    return pd.read_csv(path)
