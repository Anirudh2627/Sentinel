#!/usr/bin/env python3
"""Inject data drift by driving drifted traffic at the running server.

The drift is REAL: feature distributions (covariate shift) or the label
mechanism itself (concept drift) change in the simulated traffic, and the
monitoring stack measures the effect.

Usage:
    python scripts/inject_drift.py --mode covariate|concept --duration 60 --rps 10
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.config import SentinelSettings, configure_logging  # noqa: E402
from sentinel.simulation.traffic import TrafficSimulator  # noqa: E402

MODE_TO_SCENARIO = {"covariate": "distribution_shift", "concept": "concept_drift"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="covariate", choices=sorted(MODE_TO_SCENARIO))
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--rps", type=float, default=None)
    parser.add_argument("--feedback-rate", type=float, default=None)
    parser.add_argument("--url", default=None)
    parser.add_argument("--env", default=None)
    args = parser.parse_args()

    settings = SentinelSettings.load(args.env)
    configure_logging(settings.serving.log_level, json_format=False)
    if args.url:
        settings.simulation.base_url = args.url

    scenario = MODE_TO_SCENARIO[args.mode]
    print(f"injecting drift via scenario '{scenario}' ...")
    report = TrafficSimulator(
        settings,
        scenario=scenario,
        requests_per_second=args.rps,
        duration_seconds=args.duration,
        feedback_rate=args.feedback_rate,
    ).run()
    print(json.dumps(report.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
