"""Reproducible stratified dataset splitting with persisted split metadata."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from sentinel.config import SentinelSettings


class SplitInfo(BaseModel):
    n_rows: int
    positive_rate: float
    content_sha256: str


class SplitReport(BaseModel):
    dataset_version: str
    seed: int
    val_size: float
    test_size: float
    created_at: str
    train: SplitInfo
    val: SplitInfo
    test: SplitInfo


def _frame_info(df: pd.DataFrame, target: str) -> SplitInfo:
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=False).values.tobytes()
    ).hexdigest()
    return SplitInfo(
        n_rows=int(len(df)),
        positive_rate=float(df[target].mean()),
        content_sha256=digest,
    )


def split_dataset(
    settings: SentinelSettings,
    df: pd.DataFrame,
    dataset_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitReport]:
    """Stratified train/val/test split; returns frames + report."""
    target = settings.features.target
    seed = settings.data.seed

    train_df, holdout = train_test_split(
        df,
        test_size=settings.data.val_size + settings.data.test_size,
        random_state=seed,
        stratify=df[target],
    )
    relative_test = settings.data.test_size / (settings.data.val_size + settings.data.test_size)
    val_df, test_df = train_test_split(
        holdout,
        test_size=relative_test,
        random_state=seed,
        stratify=holdout[target],
    )

    report = SplitReport(
        dataset_version=dataset_version,
        seed=seed,
        val_size=settings.data.val_size,
        test_size=settings.data.test_size,
        created_at=datetime.now(UTC).isoformat(),
        train=_frame_info(train_df, target),
        val=_frame_info(val_df, target),
        test=_frame_info(test_df, target),
    )
    return train_df, val_df, test_df, report


def persist_splits(
    settings: SentinelSettings,
    dataset_version: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    report: SplitReport,
) -> Path:
    """Persist split CSVs + report under ``data/processed/<version>/``."""
    out_dir = settings.resolve(settings.paths.data_processed) / dataset_version
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    (out_dir / "split_report.json").write_text(report.model_dump_json(indent=2))
    return out_dir


def load_split_report(settings: SentinelSettings, dataset_version: str) -> SplitReport:
    path = settings.resolve(settings.paths.data_processed) / dataset_version / "split_report.json"
    return SplitReport.model_validate(json.loads(path.read_text()))
