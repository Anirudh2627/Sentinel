"""File-backed model registry with atomic writes and an auditable state machine.

The local registry (``models/registry.json`` + ``models/v<n>/``) is the
operational source of truth for serving and the deployment controller; MLflow
mirrors experiment data and (best effort) model versions for UI exploration.
This keeps the platform runnable with zero external services while remaining
MLflow-integrated.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sentinel.config import SentinelSettings, get_logger, log_event
from sentinel.registry.metadata import (
    ALLOWED_TRANSITIONS,
    ModelStatus,
    ModelVersion,
    RegistryIndex,
    StatusChange,
    utc_now_iso,
)
from sentinel.training.artifacts import TrainingMetadata, load_metadata

logger = get_logger(__name__)

INDEX_FILE = "registry.json"


class RegistryError(RuntimeError):
    pass


class InvalidTransitionError(RegistryError):
    pass


class ModelNotFoundError(RegistryError):
    pass


class ModelRegistry:
    """Registry facade. Every mutation is lock-guarded, atomic and audited."""

    def __init__(self, settings: SentinelSettings):
        self.settings = settings
        self.models_dir = settings.resolve(settings.paths.models_dir)
        self.index_path = settings.resolve(settings.registry.index_path)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #
    def _load(self) -> RegistryIndex:
        if not self.index_path.exists():
            return RegistryIndex()
        return RegistryIndex.model_validate(json.loads(self.index_path.read_text()))

    def _save(self, index: RegistryIndex) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(index.model_dump_json(indent=2))
        os.replace(tmp, self.index_path)

    @contextmanager
    def _lock(self, timeout: float = 10.0) -> Iterator[None]:
        lock_path = self.index_path.with_suffix(".lock")
        deadline = time.monotonic() + timeout
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                break
            except FileExistsError:
                if time.monotonic() > deadline:
                    raise RegistryError(f"could not acquire registry lock: {lock_path}") from None
                time.sleep(0.05)
        try:
            yield
        finally:
            lock_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def index(self) -> RegistryIndex:
        return self._load()

    def get(self, version: str) -> ModelVersion:
        index = self._load()
        mv = index.versions.get(version)
        if mv is None:
            raise ModelNotFoundError(f"model version not found: {version}")
        return mv

    def list_versions(self, status: ModelStatus | None = None) -> list[ModelVersion]:
        versions = self._load().ordered_versions()
        if status is not None:
            versions = [v for v in versions if v.status == status]
        return versions

    def production(self) -> ModelVersion | None:
        index = self._load()
        return index.versions.get(index.current_production) if index.current_production else None

    def previous_production(self) -> ModelVersion | None:
        index = self._load()
        return index.versions.get(index.previous_production) if index.previous_production else None

    def canary(self) -> ModelVersion | None:
        index = self._load()
        return index.versions.get(index.current_canary) if index.current_canary else None

    def candidate(self) -> ModelVersion | None:
        index = self._load()
        return index.versions.get(index.current_candidate) if index.current_candidate else None

    def artifact_dir(self, version: str) -> Path:
        return self.settings.resolve(self.get(version).artifact_path)

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    def register_run(
        self, run_path: str | Path, metadata: TrainingMetadata | None = None
    ) -> ModelVersion:
        """Copy a training run into ``models/v<n>/`` and record it as TRAINED."""
        run_path = Path(run_path)
        metadata = metadata or load_metadata(run_path)
        with self._lock():
            index = self._load()
            version = f"v{index.next_version_number}"
            dest_rel = Path(self.settings.paths.models_dir) / version
            dest = self.settings.resolve(dest_rel)
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(run_path, dest)

            mv = ModelVersion(
                version=version,
                run_id=metadata.run_id,
                model_name=metadata.model_name,
                status=ModelStatus.TRAINED,
                artifact_path=str(dest_rel),
                dataset_version=metadata.dataset_version,
                git_commit=metadata.git_commit,
                git_dirty=metadata.git_dirty,
                code_version=metadata.code_version,
                scenario=metadata.scenario,
                params=metadata.params,
                metrics={
                    k: v
                    for k, v in metadata.metrics.items()
                    if isinstance(v, dict) and _is_float_dict(v)
                },
                history=[
                    StatusChange(
                        from_status=None,
                        to_status=ModelStatus.TRAINED,
                        at=utc_now_iso(),
                        reason="registered from training run",
                        actor="registry",
                    )
                ],
            )
            index.versions[version] = mv
            index.next_version_number += 1
            self._save(index)
        log_event(
            logger,
            logging.INFO,
            "model registered",
            version=version,
            run_id=metadata.run_id,
            dataset_version=metadata.dataset_version,
            git_commit=metadata.git_commit,
        )
        return mv

    # ------------------------------------------------------------------ #
    # State machine
    # ------------------------------------------------------------------ #
    def set_status(
        self, version: str, status: ModelStatus, reason: str = "", actor: str = "system"
    ) -> ModelVersion:
        with self._lock():
            index = self._load()
            mv = index.versions.get(version)
            if mv is None:
                raise ModelNotFoundError(version)
            if status not in ALLOWED_TRANSITIONS[mv.status]:
                raise InvalidTransitionError(
                    f"illegal transition {mv.status.value} -> {status.value} for {version}"
                )
            change = StatusChange(
                from_status=mv.status,
                to_status=status,
                at=utc_now_iso(),
                reason=reason,
                actor=actor,
            )
            mv.status = status
            mv.updated_at = change.at
            mv.history.append(change)
            self._sync_pointers(index, mv)
            self._save(index)
        log_event(
            logger,
            logging.INFO,
            "model status changed",
            version=version,
            to=status.value,
            reason=reason,
            actor=actor,
        )
        return mv

    def _sync_pointers(self, index: RegistryIndex, mv: ModelVersion) -> None:
        """Keep convenience pointers consistent after a status change."""
        if mv.status == ModelStatus.PRODUCTION:
            if index.current_production and index.current_production != mv.version:
                index.previous_production = index.current_production
            index.current_production = mv.version
            if index.current_canary == mv.version:
                index.current_canary = None
            if index.current_candidate == mv.version:
                index.current_candidate = None
        elif mv.status == ModelStatus.CANARY:
            index.current_canary = mv.version
        elif mv.status == ModelStatus.CANDIDATE:
            index.current_candidate = mv.version
        elif mv.status in (ModelStatus.ROLLED_BACK, ModelStatus.ARCHIVED):
            if index.current_canary == mv.version:
                index.current_canary = None
            if index.current_candidate == mv.version:
                index.current_candidate = None
            if index.current_production == mv.version:
                index.current_production = None

    # ------------------------------------------------------------------ #
    # High-level lifecycle operations
    # ------------------------------------------------------------------ #
    def mark_validated(self, version: str, reason: str = "quality gate passed") -> ModelVersion:
        self.set_status(version, ModelStatus.VALIDATED, reason=reason, actor="gate")
        return self.set_status(version, ModelStatus.CANDIDATE, reason=reason, actor="gate")

    def start_canary(self, version: str, reason: str = "canary deployment started") -> ModelVersion:
        """Transition a candidate to CANARY. Only one canary at a time."""
        index = self._load()
        if index.current_canary and index.current_canary != version:
            raise RegistryError(
                f"another canary is active: {index.current_canary}; "
                f"resolve it before deploying {version}"
            )
        return self.set_status(version, ModelStatus.CANARY, reason=reason, actor="deployment")

    def promote_to_production(self, version: str, reason: str = "promoted") -> ModelVersion:
        """CANARY|CANDIDATE|ARCHIVED -> PRODUCTION; old production is archived."""
        with self._lock():
            index = self._load()
            mv = index.versions.get(version)
            if mv is None:
                raise ModelNotFoundError(version)
            old_prod = index.current_production
            if old_prod and old_prod != version:
                old = index.versions[old_prod]
                if old.status == ModelStatus.PRODUCTION:
                    old.status = ModelStatus.ARCHIVED
                    old.updated_at = utc_now_iso()
                    old.history.append(
                        StatusChange(
                            from_status=ModelStatus.PRODUCTION,
                            to_status=ModelStatus.ARCHIVED,
                            at=old.updated_at,
                            reason=f"replaced by {version}",
                            actor="registry",
                        )
                    )
                index.previous_production = old_prod
            mv.status = ModelStatus.PRODUCTION
            mv.updated_at = utc_now_iso()
            mv.history.append(
                StatusChange(
                    from_status=mv.history[-1].to_status if mv.history else None,
                    to_status=ModelStatus.PRODUCTION,
                    at=mv.updated_at,
                    reason=reason,
                    actor="registry",
                )
            )
            index.current_production = version
            if index.current_canary == version:
                index.current_canary = None
            if index.current_candidate == version:
                index.current_candidate = None
            self._save(index)
        log_event(
            logger, logging.INFO, "model promoted to production", version=version, reason=reason
        )
        return self.get(version)

    def rollback_canary(self, version: str, reason: str) -> ModelVersion:
        """Mark the active canary as ROLLED_BACK and clear the pointer."""
        mv = self.set_status(version, ModelStatus.ROLLED_BACK, reason=reason, actor="controller")
        log_event(logger, logging.WARNING, "canary rolled back", version=version, reason=reason)
        return mv

    def restore_previous_production(self, reason: str) -> ModelVersion | None:
        """Roll production back to the previously archived production model."""
        index = self._load()
        prev = index.previous_production
        current = index.current_production
        if not prev:
            log_event(logger, logging.ERROR, "no previous production model to restore")
            return None
        if current:
            self.set_status(current, ModelStatus.ROLLED_BACK, reason=reason, actor="controller")
        mv = self.set_status(prev, ModelStatus.PRODUCTION, reason=reason, actor="controller")
        with self._lock():
            index = self._load()
            index.previous_production = current
            self._save(index)
        log_event(
            logger, logging.WARNING, "previous production restored", version=prev, reason=reason
        )
        return mv

    def archive(self, version: str, reason: str = "manual archive") -> ModelVersion:
        return self.set_status(version, ModelStatus.ARCHIVED, reason=reason, actor="operator")


def _is_float_dict(d: dict) -> bool:
    return all(isinstance(v, (int, float)) for v in d.values())
