#!/usr/bin/env python3
"""Send simulated production traffic to a running Sentinel server.

Usage:
    python scripts/simulate_traffic.py --scenario normal --duration 30 --rps 10
Scenarios: normal | high_traffic | distribution_shift | concept_drift | bad_features
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.config import SentinelSettings, configure_logging  # noqa: E402
from sentinel.simulation.scenarios import scenario_names  # noqa: E402
from sentinel.simulation.traffic import TrafficSimulator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="normal", choices=scenario_names())
    parser.add_argument("--rps", type=float, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument("--feedback-rate", type=float, default=None)
    parser.add_argument("--url", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--env", default=None)
    args = parser.parse_args()

    settings = SentinelSettings.load(args.env)
    configure_logging(settings.serving.log_level, json_format=False)
    if args.url:
        settings.simulation.base_url = args.url

    simulator = TrafficSimulator(
        settings,
        scenario=args.scenario,
        requests_per_second=args.rps,
        duration_seconds=args.duration,
        max_requests=args.max_requests,
        feedback_rate=args.feedback_rate,
        seed=args.seed,
    )
    report = simulator.run()
    reports = settings.resolve(settings.paths.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    out = reports / f"simulation_{args.scenario}_{int(time.time())}.json"
    out.write_text(report.model_dump_json(indent=2))
    print(json.dumps(report.model_dump(), indent=2))
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
