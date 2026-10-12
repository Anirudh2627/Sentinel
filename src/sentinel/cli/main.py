"""Sentinel command-line interface.

Every command maps to real package functionality (no stubs). Use
``sentinel <command> --help`` for details. The CLI is also what the Makefile,
scripts/ and CI workflows invoke.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import typer

from sentinel import __version__
from sentinel.config import SentinelSettings, configure_logging, get_logger

app = typer.Typer(
    help="Sentinel — production MLOps pipeline: train, gate, deploy, monitor, rollback.",
    no_args_is_help=True,
    add_completion=False,
)
logger = get_logger("sentinel.cli")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _settings(env: str | None = None) -> SentinelSettings:
    settings = SentinelSettings.load(env)
    configure_logging(settings.serving.log_level, json_format=False)
    return settings


def _echo_json(obj) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


def _latest_run_dir(settings: SentinelSettings) -> Path:
    runs = settings.resolve(settings.paths.artifacts_dir) / "runs"
    candidates = (
        sorted([p for p in runs.iterdir() if p.is_dir()], key=lambda p: p.name)
        if runs.is_dir()
        else []
    )
    if not candidates:
        typer.echo("no training runs found — run `sentinel train` first", err=True)
        raise typer.Exit(2)
    return candidates[-1]


# --------------------------------------------------------------------------- #
# Config / status
# --------------------------------------------------------------------------- #
@app.command()
def config(
    env: str | None = typer.Option(None, help="Config profile (development|production)."),
) -> None:
    """Print the effective (redacted) configuration as JSON."""
    _echo_json(_settings(env).summary())


@app.command()
def status(env: str | None = typer.Option(None)) -> None:
    """Show registry pointers, routing, controller state and latest snapshot."""
    settings = _settings(env)
    from sentinel.deployment.canary import read_routing_state
    from sentinel.deployment.controller import DeploymentController
    from sentinel.monitoring.collector import MetricsCollector
    from sentinel.registry.manager import ModelRegistry

    registry = ModelRegistry(settings)
    index = registry.index()
    routing = read_routing_state(settings)
    controller = DeploymentController(settings)
    state = controller.load_state()
    snapshot = MetricsCollector(settings, registry).latest_snapshot()

    out = {
        "env": settings.env,
        "registry": {
            "versions": {
                v: {
                    "status": mv.status.value,
                    "dataset": mv.dataset_version,
                    "created_at": mv.created_at,
                }
                for v, mv in index.versions.items()
            },
            "current_production": index.current_production,
            "previous_production": index.previous_production,
            "current_canary": index.current_canary,
            "current_candidate": index.current_candidate,
        },
        "routing": routing.model_dump(),
        "controller": state.model_dump(),
        "latest_snapshot": (
            {
                "ts": snapshot.ts,
                "requests": {v: s.n_requests for v, s in snapshot.stats.items()},
                "error_rate": {v: s.error_rate for v, s in snapshot.stats.items()},
                "violations": [v.model_dump() for v in snapshot.violations],
            }
            if snapshot
            else None
        ),
    }
    _echo_json(out)


@app.command()
def versions(
    env: str | None = typer.Option(None),
    status_filter: str | None = typer.Option(None, "--status"),
) -> None:
    """List registered model versions."""
    settings = _settings(env)
    from sentinel.registry.manager import ModelRegistry
    from sentinel.registry.metadata import ModelStatus

    registry = ModelRegistry(settings)
    st = ModelStatus(status_filter) if status_filter else None
    rows = []
    for mv in registry.list_versions(st):
        rows.append(
            {
                "version": mv.version,
                "status": mv.status.value,
                "scenario": mv.scenario,
                "dataset_version": mv.dataset_version,
                "git_commit": (mv.git_commit or "")[:12] or None,
                "val_f1": round(mv.primary_metrics.get("f1", float("nan")), 4),
                "val_auc": round(mv.primary_metrics.get("roc_auc", float("nan")), 4),
                "created_at": mv.created_at,
            }
        )
    _echo_json(rows)


@app.command()
def audit(
    last: int = typer.Option(20, help="Show the last N audit entries."),
    env: str | None = typer.Option(None),
) -> None:
    """Print the deployment audit trail."""
    settings = _settings(env)
    from sentinel.deployment.audit import AuditLog

    entries = AuditLog(settings).read(last_n=last)
    _echo_json([e.model_dump() for e in entries])


# --------------------------------------------------------------------------- #
# Data / training / evaluation
# --------------------------------------------------------------------------- #
@app.command()
def generate(
    n_samples: int | None = typer.Option(None, "--n-samples"),
    env: str | None = typer.Option(None),
) -> None:
    """Generate (or load) + validate + persist the dataset."""
    settings = _settings(env)
    from sentinel.data.ingestion import build_dataset, persist_dataset
    from sentinel.data.validation import DataValidator

    df = build_dataset(settings, n_samples=n_samples)
    report = DataValidator(settings).validate(df)
    _, card = persist_dataset(settings, df, source=settings.data.source, seed=settings.data.seed)
    typer.echo(report.to_markdown())
    _echo_json(card.model_dump())


@app.command()
def train(
    scenario: str = typer.Option("default", help="default | degraded (failure injection)."),
    n_samples: int | None = typer.Option(None),
    seed: int | None = typer.Option(None, help="Override the data seed (new dataset version)."),
    env: str | None = typer.Option(None),
    no_track: bool = typer.Option(False, help="Skip experiment tracking."),
) -> None:
    """Run the training stage (ingest → validate → split → train → evaluate → persist)."""
    settings = _settings(env)
    if seed is not None:
        settings.data.seed = seed
    from sentinel.training.tracking import NullTracker, get_tracker
    from sentinel.training.train import train_model

    result = train_model(
        settings,
        scenario=scenario,
        n_samples=n_samples,
        tracker=NullTracker() if no_track else get_tracker(settings),
    )
    _echo_json(result.model_dump())


@app.command()
def register(
    run: str | None = typer.Option(None, "--run", help="Run directory (default: latest)."),
    env: str | None = typer.Option(None),
) -> None:
    """Register a training run as a new model version (status: TRAINED)."""
    settings = _settings(env)
    from sentinel.registry.manager import ModelRegistry

    run_path = Path(run) if run else _latest_run_dir(settings)
    mv = ModelRegistry(settings).register_run(run_path)
    _echo_json(mv.model_dump())


@app.command()
def evaluate(
    version: str = typer.Argument(..., help="Registered model version, e.g. v2."),
    split: str = typer.Option("test", help="test | val"),
    env: str | None = typer.Option(None),
) -> None:
    """Evaluate a registered model on its persisted holdout split (real metrics)."""
    settings = _settings(env)
    import pandas as pd

    from sentinel.data.splitting import load_split_report  # noqa: F401  (ensures split exists)
    from sentinel.registry.manager import ModelRegistry
    from sentinel.training.artifacts import load_pipeline
    from sentinel.training.evaluate import evaluate_pipeline

    registry = ModelRegistry(settings)
    mv = registry.get(version)
    pipeline = load_pipeline(registry.artifact_dir(version))
    split_path = (
        settings.resolve(settings.paths.data_processed) / mv.dataset_version / f"{split}.csv"
    )
    if not split_path.is_file():
        typer.echo(f"split not found: {split_path}", err=True)
        raise typer.Exit(2)
    df = pd.read_csv(split_path)
    feature_cols = [*settings.features.numeric, *settings.features.categorical]
    metrics = evaluate_pipeline(pipeline, df[feature_cols], df[settings.features.target].to_numpy())

    reports_dir = settings.resolve(settings.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_json = reports_dir / f"evaluation_{version}_{split}.json"
    out_json.write_text(json.dumps(metrics.model_dump(), indent=2))
    typer.echo(f"# Evaluation of {version} on '{split}' split ({metrics.n_samples} rows)", err=True)
    _echo_json(metrics.model_dump())


@app.command()
def gate(
    version: str = typer.Argument(...),
    force: bool = typer.Option(False, help="FORCE past a failed gate (audited override)."),
    env: str | None = typer.Option(None),
) -> None:
    """Run the promotion quality gate. Exit code 2 when the gate blocks."""
    settings = _settings(env)
    from sentinel.deployment.promotion import PromotionManager

    promotion = PromotionManager(settings)
    result = promotion.run_gate(version, force=force, actor="cli")
    report_path = promotion.write_gate_report(version, result)
    typer.echo(result.to_markdown())
    typer.echo(f"report: {report_path}", err=True)
    if not result.passed and not force:
        raise typer.Exit(2)


@app.command()
def pipeline(
    scenario: str = typer.Option("default"),
    n_samples: int | None = typer.Option(None),
    seed: int | None = typer.Option(None),
    no_register: bool = typer.Option(False),
    no_gate: bool = typer.Option(False),
    force: bool = typer.Option(False, help="Force past a failed gate (audited)."),
    env: str | None = typer.Option(None),
) -> None:
    """Full pipeline: train → evaluate → register → gate. Exit code 2 if gate blocks."""
    settings = _settings(env)
    if seed is not None:
        settings.data.seed = seed
    from sentinel.pipeline.orchestration import run_pipeline

    result = run_pipeline(
        settings, scenario=scenario, n_samples=n_samples, register=not no_register, gate=not no_gate
    )

    if force and result.model_version and result.gate_passed is False:
        from sentinel.deployment.promotion import PromotionManager

        gate_result = PromotionManager(settings).run_gate(
            result.model_version, force=True, actor="cli-force"
        )
        typer.echo("gate FORCED despite failure:", err=True)
        typer.echo(gate_result.to_markdown(), err=True)
        result.gate_passed = gate_result.passed
        result.gate_reason = f"FORCED override: {gate_result.reason}"

    _echo_json(result.model_dump())
    if result.gate_passed is False and not force:
        raise typer.Exit(2)


# --------------------------------------------------------------------------- #
# Deployment
# --------------------------------------------------------------------------- #
@app.command()
def deploy(
    version: str = typer.Argument(...),
    direct: bool = typer.Option(
        False, "--direct", help="Skip canary (first deployment / emergencies)."
    ),
    percent: int | None = typer.Option(None, help="Initial canary traffic percent."),
    env: str | None = typer.Option(None),
) -> None:
    """Deploy a candidate: canary (default) or direct-to-production."""
    settings = _settings(env)
    from sentinel.deployment.promotion import PromotionManager

    decision = PromotionManager(settings).deploy_candidate(
        version, via_canary=not direct, percent=percent, actor="cli"
    )
    _echo_json(decision.model_dump())
    if not decision.deployed:
        raise typer.Exit(2)


@app.command()
def traffic(
    percent: int = typer.Argument(..., help="New canary traffic percent (0-100)."),
    env: str | None = typer.Option(None),
) -> None:
    """Manually set canary traffic percentage."""
    settings = _settings(env)
    from sentinel.deployment.canary import CanaryManager

    state = CanaryManager(settings).set_traffic(percent, actor="cli")
    _echo_json(state.model_dump())


@app.command()
def promote(
    env: str | None = typer.Option(None), reason: str = typer.Option("manual promotion via CLI")
) -> None:
    """Manually promote the active canary to production."""
    settings = _settings(env)
    from sentinel.deployment.audit import AuditEntry, AuditLog
    from sentinel.deployment.canary import CanaryManager

    canary = CanaryManager(settings)
    state_before = canary.current_state()
    state = canary.promote_canary(actor="cli", reason=reason)
    AuditLog(settings).record(
        AuditEntry(
            action="canary_promoted",
            actor="cli",
            model_version=state_before.canary,
            previous_version=state_before.production,
            restored_version=state.production,
            reason=reason,
        )
    )
    _echo_json(state.model_dump())


@app.command()
def rollback(
    reason: str = typer.Option("manual rollback via CLI"),
    production: bool = typer.Option(
        False, help="Restore the PREVIOUS production model instead of rolling back the canary."
    ),
    env: str | None = typer.Option(None),
) -> None:
    """Roll back the active canary (default) or restore previous production."""
    settings = _settings(env)
    from sentinel.deployment.rollback import RollbackManager

    manager = RollbackManager(settings)
    result = (
        manager.restore_previous_production(reason=reason, actor="cli")
        if production
        else manager.rollback_canary(reason=reason, actor="cli")
    )
    _echo_json(result.model_dump())
    if not result.success:
        raise typer.Exit(2)


# --------------------------------------------------------------------------- #
# Serving / monitoring / control loop
# --------------------------------------------------------------------------- #
@app.command()
def serve(env: str | None = typer.Option(None)) -> None:
    """Run the FastAPI inference server."""
    from sentinel.serving.app import run_server

    run_server(_settings(env))


@app.command()
def monitor(
    watch: bool = typer.Option(False, help="Loop instead of a single collection."),
    interval: float | None = typer.Option(None),
    duration: float | None = typer.Option(None, help="Stop watching after N seconds."),
    env: str | None = typer.Option(None),
) -> None:
    """Collect monitoring snapshots (window stats + drift + health)."""
    settings = _settings(env)
    from sentinel.monitoring.collector import MetricsCollector

    collector = MetricsCollector(settings)
    interval = interval or settings.monitoring.collection_interval_seconds
    deadline = time.monotonic() + duration if duration else None
    while True:
        snapshot = collector.collect()
        _print_snapshot(snapshot.model_dump())
        if not watch:
            break
        if deadline and time.monotonic() >= deadline:
            break
        time.sleep(interval)


def _print_snapshot(data: dict) -> None:
    compact = {
        "ts": data["ts"],
        "routing": data["routing"],
        "stats": {
            v: {
                "requests": s["n_requests"],
                "errors": s["n_errors"],
                "error_rate": s["error_rate"],
                "p95_ms": s["latency_ms"]["p95"] if s.get("latency_ms") else None,
                "positive_rate": s.get("positive_rate"),
            }
            for v, s in data["stats"].items()
        },
        "drift": {
            v: {
                "is_drifted": d["is_drifted"],
                "max_score": d["max_score"],
                "n_drifted": d["n_drifted_features"],
            }
            for v, d in data["drift"].items()
        },
        "performance": {
            v: p.model_dump() if hasattr(p, "model_dump") else p
            for v, p in data["performance"].items()
        },
        "violations": data["violations"],
    }
    _echo_json(compact)


@app.command()
def controller(
    once: bool = typer.Option(False, help="Run a single decision cycle."),
    duration: float | None = typer.Option(None, help="Run for N seconds."),
    interval: float | None = typer.Option(None),
    stop_on_terminal: bool = typer.Option(False, help="Stop after a promote/rollback decision."),
    env: str | None = typer.Option(None),
) -> None:
    """Run the deployment controller (promote / rollback / alert decisions)."""
    settings = _settings(env)
    from sentinel.deployment.controller import DeploymentController

    ctrl = DeploymentController(settings)
    if once:
        decision = ctrl.step()
        _echo_json(decision.model_dump())
        return
    for decision in ctrl.run(
        duration_seconds=duration,
        interval_seconds=interval,
        stop_on_promote_or_rollback=stop_on_terminal,
    ):
        _echo_json(decision.model_dump())


# --------------------------------------------------------------------------- #
# Simulation / failure injection
# --------------------------------------------------------------------------- #
@app.command()
def simulate(
    scenario: str = typer.Option(
        "normal", help="normal | high_traffic | distribution_shift | concept_drift | bad_features"
    ),
    rps: float | None = typer.Option(None),
    duration: float | None = typer.Option(None),
    max_requests: int | None = typer.Option(None),
    feedback_rate: float | None = typer.Option(None),
    url: str | None = typer.Option(None, help="Server base URL."),
    seed: int | None = typer.Option(None),
    save: bool = typer.Option(True, help="Persist the report to reports/."),
    env: str | None = typer.Option(None),
) -> None:
    """Send simulated production traffic to a running server."""
    settings = _settings(env)
    if url:
        settings.simulation.base_url = url
    from sentinel.simulation.traffic import TrafficSimulator

    simulator = TrafficSimulator(
        settings,
        scenario=scenario,
        requests_per_second=rps,
        duration_seconds=duration,
        feedback_rate=feedback_rate,
        seed=seed,
        max_requests=max_requests,
    )
    report = simulator.run()
    if save:
        reports = settings.resolve(settings.paths.reports_dir)
        reports.mkdir(parents=True, exist_ok=True)
        out = reports / f"simulation_{scenario}_{int(time.time())}.json"
        out.write_text(report.model_dump_json(indent=2))
        typer.echo(f"report: {out}", err=True)
    _echo_json(report.model_dump())


@app.command()
def faults(
    latency_ms: float = typer.Option(0.0, help="Injected latency per request."),
    error_rate: float = typer.Option(0.0, help="Fraction of requests that fail (0..1)."),
    disable: bool = typer.Option(False, help="Disable fault injection."),
    url: str | None = typer.Option(None, help="Server base URL."),
    token: str | None = typer.Option(None, help="Admin token (if the server requires one)."),
    env: str | None = typer.Option(None),
) -> None:
    """Configure server-side fault injection via the admin API."""
    import httpx

    settings = _settings(env)
    base = url or settings.simulation.base_url
    headers = {"X-Sentinel-Admin-Token": token} if token else {}
    payload: dict = {"enabled": not disable, "latency_ms": latency_ms, "error_rate": error_rate}
    if disable:
        payload = {"enabled": False, "latency_ms": 0, "error_rate": 0}
    resp = httpx.post(f"{base.rstrip('/')}/admin/faults", json=payload, headers=headers, timeout=10)
    if resp.status_code != 200:
        typer.echo(f"failed to set faults: {resp.status_code} {resp.text}", err=True)
        raise typer.Exit(2)
    _echo_json(resp.json())


@app.command()
def clean(
    everything: bool = typer.Option(
        False, "--all", help="Also remove models/registry (full reset)."
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation."),
    env: str | None = typer.Option(None),
) -> None:
    """Remove runtime state (logs, routing, snapshots) for a clean demo run."""
    settings = _settings(env)
    if not yes:
        typer.confirm(f"This deletes runtime state under {settings.root}. Continue?", abort=True)
    import shutil

    targets = [
        settings.resolve(settings.serving.prediction_log),
        settings.resolve(settings.serving.feedback_log),
        settings.resolve(settings.monitoring.snapshot_path),
        settings.resolve(settings.serving.routing_state_path),
        settings.resolve(settings.paths.artifacts_dir) / "controller_state.json",
        settings.resolve(settings.rollback.audit_path),
    ]
    if everything:
        targets += [
            settings.resolve(settings.paths.models_dir),
            settings.resolve(settings.paths.artifacts_dir) / "runs",
            settings.resolve(settings.paths.data_processed),
            settings.resolve(settings.paths.reports_dir),
        ]
    for target in targets:
        if target.is_file():
            target.unlink()
            typer.echo(f"removed {target}")
        elif target.is_dir():
            shutil.rmtree(target)
            typer.echo(f"removed {target}/")
    settings._ensure_dirs()
    typer.echo("clean complete")


@app.command()
def version() -> None:
    """Print the Sentinel version."""
    typer.echo(f"sentinel {__version__} (python {sys.version.split()[0]})")


if __name__ == "__main__":
    app()
