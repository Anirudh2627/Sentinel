#!/usr/bin/env python3
"""Run the deployment controller loop (promote / rollback / alert).

Usage:
    python scripts/run_controller.py --duration 300 --interval 10
    python scripts/run_controller.py --once
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.config import SentinelSettings, configure_logging  # noqa: E402
from sentinel.deployment.controller import DeploymentController  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--interval", type=float, default=None)
    parser.add_argument("--stop-on-terminal", action="store_true")
    parser.add_argument("--env", default=None)
    args = parser.parse_args()

    settings = SentinelSettings.load(args.env)
    configure_logging(settings.serving.log_level, json_format=False)
    ctrl = DeploymentController(settings)

    if args.once:
        print(json.dumps(ctrl.step().model_dump(), indent=2))
        return 0
    for decision in ctrl.run(
        duration_seconds=args.duration,
        interval_seconds=args.interval,
        stop_on_promote_or_rollback=args.stop_on_terminal,
    ):
        print(json.dumps(decision.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
