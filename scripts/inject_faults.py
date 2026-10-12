#!/usr/bin/env python3
"""Inject server-side faults (latency / errors) via the admin API.

Usage:
    python scripts/inject_faults.py --latency-ms 800            # slow inference
    python scripts/inject_faults.py --error-rate 0.2            # failing requests
    python scripts/inject_faults.py --disable                   # back to normal
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402

from sentinel.config import SentinelSettings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency-ms", type=float, default=0.0)
    parser.add_argument("--error-rate", type=float, default=0.0)
    parser.add_argument("--disable", action="store_true")
    parser.add_argument("--url", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--env", default=None)
    args = parser.parse_args()

    settings = SentinelSettings.load(args.env)
    base = (args.url or settings.simulation.base_url).rstrip("/")
    headers = {"X-Sentinel-Admin-Token": args.token} if args.token else {}
    payload = (
        {"enabled": False, "latency_ms": 0, "error_rate": 0}
        if args.disable
        else {"enabled": True, "latency_ms": args.latency_ms, "error_rate": args.error_rate}
    )
    resp = httpx.post(f"{base}/admin/faults", json=payload, headers=headers, timeout=10)
    print(resp.status_code, json.dumps(resp.json(), indent=2))
    return 0 if resp.status_code == 200 else 2


if __name__ == "__main__":
    raise SystemExit(main())
