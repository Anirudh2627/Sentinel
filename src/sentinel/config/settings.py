"""Typed, layered configuration for Sentinel.

Precedence (highest wins):
    1. Environment variables  ``SENTINEL_<SECTION>__<KEY>[__<SUBKEY>]=value``
    2. Environment profile    ``configs/{SENTINEL_ENV}.yaml``
    3. Shared configs         ``configs/monitoring.yaml``, ``configs/model.yaml``
    4. Pydantic defaults declared below

Settings objects are immutable-ish value objects passed explicitly into
components (no global mutable state). Use :meth:`SentinelSettings.load`.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

# Repository root = .../sentinel (this file lives in src/sentinel/config/)
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]

ENV_VAR_PREFIX = "SENTINEL_"
ENV_VAR_NESTING = "__"


class ConfigError(RuntimeError):
    """Raised when configuration is invalid or cannot be located."""


# --------------------------------------------------------------------------- #
# Section models
# --------------------------------------------------------------------------- #
class PathsConfig(BaseModel):
    root_dir: Path = Path(".")
    data_raw: Path = Path("data/raw")
    data_processed: Path = Path("data/processed")
    data_synthetic: Path = Path("data/synthetic")
    models_dir: Path = Path("models")
    artifacts_dir: Path = Path("artifacts")
    monitoring_dir: Path = Path("monitoring")
    reports_dir: Path = Path("reports")


class DataConfig(BaseModel):
    source: Literal["synthetic", "csv"] = "synthetic"
    csv_path: Path = Path("data/raw/churn.csv")
    n_samples: int = Field(default=20_000, gt=0)
    seed: int = 42
    val_size: float = Field(default=0.15, gt=0, lt=0.5)
    test_size: float = Field(default=0.15, gt=0, lt=0.5)


class FeaturesConfig(BaseModel):
    target: str = "churned"
    id_column: str = "customer_id"
    numeric: list[str] = Field(default_factory=list)
    categorical: list[str] = Field(default_factory=list)
    derived: list[str] = Field(default_factory=list)


class ValidationConfig(BaseModel):
    strict: bool = True
    max_missing_ratio: float = 0.02
    max_duplicate_row_ratio: float = 0.001
    max_duplicate_id_ratio: float = 0.0
    leakage_corr_threshold: float = 0.98
    min_rows: int = 500
    min_positive_rate: float = 0.02
    max_positive_rate: float = 0.98


class DegradedScenarioConfig(BaseModel):
    label_flip_prob: float = Field(default=0.9, ge=0.0, le=1.0)
    subsample_frac: float = Field(default=0.05, gt=0.0, le=1.0)


class ModelConfig(BaseModel):
    name: str = "sentinel-churn"
    algorithm: Literal["logistic_regression", "gradient_boosting", "random_forest"] = (
        "gradient_boosting"
    )
    random_state: int = 42
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    degraded: DegradedScenarioConfig = Field(default_factory=DegradedScenarioConfig)

    def params_for(self, algorithm: str | None = None) -> dict[str, Any]:
        algo = algorithm or self.algorithm
        return dict(self.params.get(algo, {}))


class EvaluationConfig(BaseModel):
    primary_metric: Literal["f1", "roc_auc", "precision", "recall", "accuracy"] = "f1"
    min_f1: float = 0.55
    min_precision: float = 0.50
    min_recall: float = 0.45
    min_auc: float = 0.65
    require_beat_production: bool = True
    beat_margin: float = 0.0


class MLflowConfig(BaseModel):
    enabled: bool = True
    tracking_uri: str = ""  # empty -> local ./mlruns file store
    experiment_name: str = "sentinel-churn"
    registered_model_name: str = "sentinel-churn"


class ExperimentTrackingConfig(BaseModel):
    backend: Literal["auto", "mlflow", "local", "none"] = "auto"
    local_dir: Path = Path("artifacts/experiments")
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)


class RegistryConfig(BaseModel):
    index_path: Path = Path("models/registry.json")
    sync_to_mlflow: bool = True


class ServingConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"
    prediction_log: Path = Path("monitoring/prediction_log.jsonl")
    feedback_log: Path = Path("monitoring/feedback_log.jsonl")
    routing_state_path: Path = Path("artifacts/routing.json")
    admin_token_env: str = "SENTINEL_SERVING__ADMIN_TOKEN"
    routing_reload_interval_seconds: float = 2.0

    @field_validator("port")
    @classmethod
    def _valid_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("port must be between 1 and 65535")
        return v


class CanaryConfig(BaseModel):
    enabled: bool = True
    initial_traffic_percent: int = Field(default=5, ge=0, le=100)
    promotion_steps: list[int] = Field(default_factory=lambda: [25, 50, 100])
    min_healthy_windows_per_step: int = Field(default=3, ge=1)
    sticky_routing: bool = True

    @field_validator("promotion_steps")
    @classmethod
    def _valid_steps(cls, v: list[int]) -> list[int]:
        if not v or any(not 0 < s <= 100 for s in v):
            raise ValueError("promotion_steps must be non-empty ints in (0, 100]")
        if v != sorted(v):
            raise ValueError("promotion_steps must be ascending")
        return v


class PredictionShiftConfig(BaseModel):
    enabled: bool = True
    max_delta_positive_rate: float = Field(default=0.15, gt=0, le=1)
    # Minimum predictions per version before the comparison is statistically
    # meaningful — at small canary percentages a handful of requests says nothing.
    min_samples: int = Field(default=50, ge=1)


class PerformanceConfig(BaseModel):
    enabled: bool = True
    min_samples: int = 50
    min_auc: float = 0.60


class DriftConfig(BaseModel):
    enabled: bool = True
    method: Literal["psi", "ks"] = "psi"
    psi_threshold: float = 0.25
    ks_threshold: float = 0.20
    min_samples: int = 50
    severity: Literal["warning", "critical"] = "warning"
    top_k_features: int = 5


class MonitoringConfig(BaseModel):
    window_seconds: float = 60.0
    collection_interval_seconds: float = 10.0
    snapshot_path: Path = Path("monitoring/snapshots.jsonl")
    min_samples_for_decision: int = 30
    max_error_rate: float = 0.05
    max_p95_latency_ms: float = 500.0
    max_p99_latency_ms: float = 1500.0
    prediction_shift: PredictionShiftConfig = Field(default_factory=PredictionShiftConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)


class RollbackConfig(BaseModel):
    enabled: bool = True
    cooldown_seconds: float = 300.0
    audit_path: Path = Path("artifacts/audit/deployment_audit.jsonl")
    on_error_rate: bool = True
    on_latency: bool = True
    on_prediction_shift: bool = True
    on_performance: bool = True
    on_drift: bool = False
    auto_restore_previous_production: bool = False


class FaultsConfig(BaseModel):
    enabled: bool = False
    latency_ms: float = 0.0
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_jitter: float = Field(default=0.3, ge=0.0, le=1.0)


class SimulationConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8000"
    requests_per_second: float = 10.0
    concurrency: int = 4
    duration_seconds: float = 30.0
    feedback_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    seed: int = 7


# --------------------------------------------------------------------------- #
# Root settings
# --------------------------------------------------------------------------- #
class SentinelSettings(BaseModel):
    """Root configuration object for the whole platform."""

    env: str = "development"
    config_dir: Path = _PACKAGE_ROOT / "configs"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    experiment_tracking: ExperimentTrackingConfig = Field(default_factory=ExperimentTrackingConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    canary: CanaryConfig = Field(default_factory=CanaryConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    rollback: RollbackConfig = Field(default_factory=RollbackConfig)
    faults: FaultsConfig = Field(default_factory=FaultsConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, env: str | None = None, config_dir: Path | None = None) -> SentinelSettings:
        env = env or os.environ.get(f"{ENV_VAR_PREFIX}ENV", "development")
        cdir = Path(
            config_dir or os.environ.get(f"{ENV_VAR_PREFIX}CONFIG_DIR") or _PACKAGE_ROOT / "configs"
        )
        if not cdir.is_dir():
            raise ConfigError(f"config directory not found: {cdir}")

        merged: dict[str, Any] = {}
        for name in ("model.yaml", "monitoring.yaml", f"{env}.yaml"):
            path = cdir / name
            if path.is_file():
                loaded = yaml.safe_load(path.read_text()) or {}
                if not isinstance(loaded, dict):
                    raise ConfigError(f"{path} must contain a YAML mapping")
                merged = _deep_merge(merged, loaded)
            elif name == f"{env}.yaml":
                raise ConfigError(f"environment profile not found: {path}")

        merged["env"] = env
        merged["config_dir"] = str(cdir)
        _apply_env_overrides(merged)

        try:
            settings = cls.model_validate(merged)
        except ValidationError as exc:  # pragma: no cover - defensive
            raise ConfigError(f"invalid configuration:\n{exc}") from exc
        settings._ensure_dirs()
        return settings

    # ------------------------------------------------------------------ #
    # Path helpers
    # ------------------------------------------------------------------ #
    @property
    def root(self) -> Path:
        root = Path(self.paths.root_dir)
        if not root.is_absolute():
            root = _PACKAGE_ROOT / root
        return root.resolve()

    def resolve(self, path: str | Path) -> Path:
        """Resolve a configured (possibly relative) path against the repo root."""
        p = Path(path)
        return p if p.is_absolute() else (self.root / p).resolve()

    def _ensure_dirs(self) -> None:
        for p in (
            self.paths.data_raw,
            self.paths.data_processed,
            self.paths.data_synthetic,
            self.paths.models_dir,
            self.paths.artifacts_dir,
            self.paths.monitoring_dir,
            self.paths.reports_dir,
        ):
            self.resolve(p).mkdir(parents=True, exist_ok=True)
        self.resolve(self.rollback.audit_path).parent.mkdir(parents=True, exist_ok=True)
        self.resolve(self.experiment_tracking.local_dir).parent.mkdir(parents=True, exist_ok=True)

    def summary(self) -> dict[str, Any]:
        """Redacted, JSON-serialisable view (safe for logs/CI summaries)."""
        data = copy.deepcopy(self.model_dump(mode="json"))
        data.pop("config_dir", None)
        return data


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _apply_env_overrides(data: dict[str, Any]) -> None:
    """Apply ``SENTINEL_<SECTION>__<KEY>=value`` overrides in-place."""
    for key, raw in os.environ.items():
        if not key.startswith(ENV_VAR_PREFIX) or key in (f"{ENV_VAR_PREFIX}ENV",):
            continue
        if key == f"{ENV_VAR_PREFIX}CONFIG_DIR":
            continue
        parts = key[len(ENV_VAR_PREFIX) :].lower().split(ENV_VAR_NESTING.lower())
        node = data
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = _coerce(raw)
