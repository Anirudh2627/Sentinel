#!/usr/bin/env python3
"""Train a model (full stage: ingest -> validate -> split -> train -> evaluate -> persist).

Usage:
    python scripts/train.py [--scenario default|degraded] [--n-samples N] \
        [--seed S] [--env development]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.config import SentinelSettings, configure_logging  # noqa: E402
from sentinel.training.train import train_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="default", choices=["default", "degraded"])
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--env", default=None)
    args = parser.parse_args()

    settings = SentinelSettings.load(args.env)
    configure_logging(settings.serving.log_level, json_format=False)
    if args.seed is not None:
        settings.data.seed = args.seed

    result = train_model(settings, scenario=args.scenario, n_samples=args.n_samples)
    print(json.dumps(result.model_dump(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
