"""End-to-end pipeline orchestration.

``run_pipeline`` chains the explicit stages:

    ingest → validate → split → train → evaluate → track → persist →
    register → quality gate

Deployment is intentionally NOT part of this pipeline: promotion to canary /
production is a separate, gated, audited decision (see deployment package).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel

from sentinel.config import SentinelSettings, get_logger, log_event
from sentinel.registry.manager import ModelRegistry
from sentinel.registry.metadata import ModelVersion
from sentinel.training.evaluate import GateResult
from sentinel.training.tracking import ExperimentTracker, get_tracker
from sentinel.training.train import TrainingResult, train_model

logger = get_logger(__name__)


class PipelineResult(BaseModel):
    ok: bool
    run_id: str
    dataset_version: str
    model_version: str | None = None
    gate_passed: bool | None = None
    gate_reason: str = ""
    reports: dict[str, str] = {}
    val_metrics: dict[str, float] = {}
    test_metrics: dict[str, float] = {}


def run_pipeline(
    settings: SentinelSettings,
    scenario: str = "default",
    n_samples: int | None = None,
    register: bool = True,
    gate: bool = True,
    tracker: ExperimentTracker | None = None,
) -> PipelineResult:
    """Execute the train→evaluate→register→gate pipeline."""
    reports: dict[str, str] = {}

    training: TrainingResult = train_model(
        settings, scenario=scenario, n_samples=n_samples, tracker=tracker or get_tracker(settings)
    )

    reports_dir = settings.resolve(settings.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / f"metrics_{training.run_id}.json"
    metrics_path.write_text(
        json.dumps(
            {
                "run_id": training.run_id,
                "dataset_version": training.dataset_version,
                "scenario": scenario,
                "val": training.val_metrics.model_dump(),
                "test": training.test_metrics.model_dump(),
            },
            indent=2,
        )
    )
    reports["metrics"] = str(metrics_path)

    model_version: ModelVersion | None = None
    gate_result: GateResult | None = None

    if register:
        registry = ModelRegistry(settings)
        model_version = registry.register_run(Path(training.run_dir))
        log_event(
            logger,
            logging.INFO,
            "registered model version",
            version=model_version.version,
            run_id=training.run_id,
        )

        if gate:
            from sentinel.deployment.promotion import PromotionManager

            promotion = PromotionManager(settings, registry=registry)
            gate_result = promotion.run_gate(model_version.version)
            gate_path = promotion.write_gate_report(model_version.version, gate_result)
            reports["gate"] = str(gate_path)

    result = PipelineResult(
        ok=True,
        run_id=training.run_id,
        dataset_version=training.dataset_version,
        model_version=model_version.version if model_version else None,
        gate_passed=gate_result.passed if gate_result else None,
        gate_reason=gate_result.reason if gate_result else "",
        reports=reports,
        val_metrics=training.val_metrics.as_flat_dict(),
        test_metrics=training.test_metrics.as_flat_dict(),
    )
    log_event(
        logger,
        logging.INFO,
        "pipeline finished",
        run_id=result.run_id,
        model_version=result.model_version,
        gate_passed=result.gate_passed,
        val_f1=round(training.val_metrics.f1, 4),
    )
    return result
