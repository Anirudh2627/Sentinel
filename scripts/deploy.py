#!/usr/bin/env python3
"""Deploy a candidate model version (canary by default).

Usage:
    python scripts/deploy.py v2 [--direct] [--percent 5] [--env development]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.config import SentinelSettings, configure_logging  # noqa: E402
from sentinel.deployment.promotion import PromotionManager  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument(
        "--direct", action="store_true", help="skip canary, deploy straight to production"
    )
    parser.add_argument("--percent", type=int, default=None, help="initial canary traffic percent")
    parser.add_argument("--env", default=None)
    args = parser.parse_args()

    settings = SentinelSettings.load(args.env)
    configure_logging(settings.serving.log_level, json_format=False)

    decision = PromotionManager(settings).deploy_candidate(
        args.version, via_canary=not args.direct, percent=args.percent, actor="scripts/deploy.py"
    )
    print(json.dumps(decision.model_dump(), indent=2))
    return 0 if decision.deployed else 2


if __name__ == "__main__":
    raise SystemExit(main())
