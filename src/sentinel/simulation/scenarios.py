"""Traffic scenarios for the simulator (failure-injection demos).

Every scenario produces *real* measurable effects — nothing is faked:

    normal             — same distribution as training data
    high_traffic       — rate multiplier (load test)
    distribution_shift — covariate shift: features move, label mechanism
                         unchanged (drift detectors should fire; model
                         performance may stay acceptable)
    concept_drift      — the feature→label relationship itself changes, so a
                         model trained on the old concept genuinely degrades
                         (live AUC on delayed labels collapses)
    bad_features       — extreme-but-valid feature values (edge-of-domain)
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ScenarioSpec(BaseModel):
    name: str
    description: str
    drift_spec: dict = Field(default_factory=dict)  # applied pre-label (covariate shift)
    concept_drift: bool = False
    feature_overrides: dict = Field(default_factory=dict)  # post-hoc fixed columns
    rate_multiplier: float = 1.0


SCENARIOS: dict[str, ScenarioSpec] = {
    "normal": ScenarioSpec(
        name="normal",
        description="in-distribution traffic matching the training data",
    ),
    "high_traffic": ScenarioSpec(
        name="high_traffic",
        description="5x request rate (load/soak test)",
        rate_multiplier=5.0,
    ),
    "distribution_shift": ScenarioSpec(
        name="distribution_shift",
        description=(
            "covariate shift: charges up ~45%, tenure down, 85% month-to-month, "
            "70% fiber — labels still follow the original mechanism"
        ),
        drift_spec={
            "numeric": {
                "monthly_charges": {"multiply": 1.45, "add": 4.0},
                "tenure_months": {"multiply": 0.4},
                "support_tickets_30d": {"add": 1.2},
            },
            "categorical": {
                "contract_type": {
                    "probabilities": {"Month-to-month": 0.85, "One year": 0.10, "Two year": 0.05}
                },
                "internet_service": {
                    "probabilities": {"DSL": 0.15, "Fiber optic": 0.70, "No": 0.15}
                },
            },
        },
    ),
    "concept_drift": ScenarioSpec(
        name="concept_drift",
        description=(
            "genuine concept drift: churn is now driven by the OPPOSITE "
            "charge/ticket relationship — old models measurably degrade"
        ),
        concept_drift=True,
    ),
    "bad_features": ScenarioSpec(
        name="bad_features",
        description="edge-of-domain feature values (valid per schema, far from training mass)",
        feature_overrides={
            "monthly_charges": 950.0,
            "total_charges": 95_000.0,
            "tenure_months": 0.0,
            "avg_session_minutes": 0.5,
            "support_tickets_30d": 45,
        },
    ),
}


def get_scenario(name: str) -> ScenarioSpec:
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario '{name}'; available: {sorted(SCENARIOS)}")
    return SCENARIOS[name]


def scenario_names() -> list[str]:
    return sorted(SCENARIOS)
