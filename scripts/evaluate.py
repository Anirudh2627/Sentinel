#!/usr/bin/env python3
"""Evaluate a registered model version on its persisted holdout split.

Usage:
    python scripts/evaluate.py v1 [--split test|val] [--env development]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from sentinel.config import SentinelSettings, configure_logging  # noqa: E402
from sentinel.registry.manager import ModelRegistry  # noqa: E402
from sentinel.training.artifacts import load_pipeline  # noqa: E402
from sentinel.training.evaluate import evaluate_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--env", default=None)
    args = parser.parse_args()

    settings = SentinelSettings.load(args.env)
    configure_logging(settings.serving.log_level, json_format=False)

    registry = ModelRegistry(settings)
    mv = registry.get(args.version)
    pipeline = load_pipeline(registry.artifact_dir(args.version))
    split_path = (
        settings.resolve(settings.paths.data_processed) / mv.dataset_version / f"{args.split}.csv"
    )
    if not split_path.is_file():
        print(f"split not found: {split_path}", file=sys.stderr)
        return 2

    df = pd.read_csv(split_path)
    feature_cols = [*settings.features.numeric, *settings.features.categorical]
    metrics = evaluate_pipeline(pipeline, df[feature_cols], df[settings.features.target].to_numpy())

    reports = settings.resolve(settings.paths.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    out = reports / f"evaluation_{args.version}_{args.split}.json"
    out.write_text(json.dumps(metrics.model_dump(), indent=2))
    print(json.dumps(metrics.model_dump(), indent=2))
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
