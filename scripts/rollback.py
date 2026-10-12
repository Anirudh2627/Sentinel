#!/usr/bin/env python3
"""Manually roll back the active canary (or restore previous production).

Usage:
    python scripts/rollback.py [--reason "..."] [--production]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinel.config import SentinelSettings, configure_logging  # noqa: E402
from sentinel.deployment.rollback import RollbackManager  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reason", default="manual rollback via scripts/rollback.py")
    parser.add_argument(
        "--production", action="store_true", help="restore previous production instead"
    )
    parser.add_argument("--env", default=None)
    args = parser.parse_args()

    settings = SentinelSettings.load(args.env)
    configure_logging(settings.serving.log_level, json_format=False)
    manager = RollbackManager(settings)
    result = (
        manager.restore_previous_production(reason=args.reason, actor="scripts/rollback.py")
        if args.production
        else manager.rollback_canary(reason=args.reason, actor="scripts/rollback.py")
    )
    print(json.dumps(result.model_dump(), indent=2))
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
