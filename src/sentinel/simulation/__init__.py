"""Traffic simulation & failure-injection scenarios."""

from sentinel.simulation.scenarios import SCENARIOS, ScenarioSpec, get_scenario, scenario_names
from sentinel.simulation.traffic import (
    SimulationReport,
    TrafficSimulator,
    generate_scenario_samples,
)

__all__ = [
    "SCENARIOS",
    "ScenarioSpec",
    "get_scenario",
    "scenario_names",
    "SimulationReport",
    "TrafficSimulator",
    "generate_scenario_samples",
]
