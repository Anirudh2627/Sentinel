# Sentinel

**A production-grade MLOps platform demonstrating the complete model lifecycle:
training → evaluation gates → model registry → canary deployment → monitoring →
drift detection → automatic rollback — all runnable on a laptop.**

```
Training Data → Data Validation → Training → Evaluation → Registry
      → CI Quality Gate → Deployment → Canary Release → Monitoring
      → Degradation Detection → Automatic Rollback
```

Sentinel is an **ML platform engineering project**, not a notebook project.
The ML model (a scikit-learn churn classifier) is deliberately boring — the
point is everything around it: reproducible pipelines, first-class model
versions, configurable promotion gates, real canary traffic splitting,
measured monitoring signals, and audited automatic rollback.

> **No fabricated numbers.** Every metric this system reports — F1 scores,
> latency percentiles, drift scores, live AUC, rollback reasons — is measured
> from actual executions. Examples in this README were captured from real
> runs in this repository and are marked as such.

---

## Table of contents

1. [Why MLOps matters](#why-mlops-matters)
2. [Problem statement](#problem-statement)
3. [Quickstart (5 minutes)](#quickstart-5-minutes)
4. [The portfolio story (one-command demo)](#the-portfolio-story-one-command-demo)
5. [Architecture](#architecture)
6. [MLOps lifecycle](#mlops-lifecycle)
7. [Repository structure](#repository-structure)
8. [Data pipeline](#data-pipeline)
9. [Training pipeline](#training-pipeline)
10. [MLflow usage](#mlflow-usage)
11. [Model registry & versioning](#model-registry--versioning)
12. [Evaluation & quality gate](#evaluation--quality-gate)
13. [Serving API](#serving-api)
14. [Monitoring & observability](#monitoring--observability)
15. [Drift detection](#drift-detection)
16. [Degradation detection (drift ≠ failure)](#degradation-detection-drift--failure)
17. [Canary deployment](#canary-deployment)
18. [Automatic rollback](#automatic-rollback)
19. [Failure injection](#failure-injection)
20. [Traffic simulation](#traffic-simulation)
21. [CI/CD](#cicd)
22. [Docker](#docker)
23. [Kubernetes (optional)](#kubernetes-optional)
24. [Testing](#testing)
25. [Configuration](#configuration)
26. [Security](#security)
27. [CLI & Makefile reference](#cli--makefile-reference)
28. [Limitations](#limitations)
29. [Future improvements](#future-improvements)

---

## Why MLOps matters

A model in a notebook delivers zero business value and zero risk visibility.
In production, models fail in ways unit tests don't catch: input
distributions shift, labels arrive late, a "better" offline model behaves
worse on today's traffic, and a bad deploy can silently burn revenue for
days. MLOps is the engineering discipline that makes models **deployable,
observable, and reversible**:

* **Reproducible**: same code + same data ⇒ same model, provably (content-hashed datasets, seeded training, git-stamped artifacts).
* **Gated**: nothing reaches production without passing explicit, configured quality thresholds.
* **Observable**: latency, errors, prediction mix, input drift and live accuracy per model version.
* **Reversible**: bad versions are rolled back automatically in seconds, with a forensic audit trail.

## Problem statement

> Deploying a new churn model shouldn't be an act of faith.
>
> Sentinel answers: *how does a model travel from training data to 100% of
> production traffic — and how does the system protect users when that model
> turns out to be bad?*

The system must therefore:

1. validate data before training and fail loudly on quality problems,
2. version every model with full provenance (data, code, params, metrics),
3. enforce a configurable quality gate before promotion,
4. expose the model behind a production-style API with canary traffic splitting,
5. monitor operational + statistical + business signals per version,
6. distinguish *input drift* from *model degradation*,
7. roll back automatically on degradation — and record exactly why.

## Quickstart (5 minutes)

Requirements: Python 3.11+, ~1 GB disk. No Docker, no cloud, no GPU needed.

```bash
git clone <your-fork-url> sentinel && cd sentinel
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: make install

# 1. Train → evaluate → register → gate (creates model v1)
make pipeline

# 2. Start the inference server (http://127.0.0.1:8000/docs)
make serve &

# 3. Deploy v1 to production, send traffic, watch monitoring
./bin/sentinel deploy v1 --direct
./bin/sentinel simulate --scenario normal --rps 20 --duration 20
./bin/sentinel monitor
```

Or skip straight to the full lifecycle: **[`make demo`](#the-portfolio-story-one-command-demo)**.

Run the test suite (no external services required):

```bash
make test        # 157 unit + integration tests
make lint        # ruff check + format
```

## The portfolio story (one-command demo)

```bash
make demo        # = bash scripts/run_demo.sh  (~2–3 minutes)
```

The script starts the server, then walks the entire lifecycle with **real
traffic and real measurements**:

| Phase | What happens | What proves it |
|---|---|---|
| 1 | Train v1 → validation → gate → production deploy | gate report, registry status |
| 2 | Train v2 (new data seed) → canary at 10% → healthy traffic → controller ramps 10→25→50→**promote** | measured traffic split, controller decisions |
| 3 | Inject **covariate drift** → PSI explodes → controller **alerts only** | drift ≠ failure, by policy |
| 4 | Inject **800ms latency fault** → p95 SLO violated → production alert | server-measured latency |
| 5 | Train a **deliberately degraded** model → gate **blocks** it → operator **forces** (audited) → canary → live AUC collapses on delayed labels → **automatic rollback** | audit trail entry with metric/threshold |
| 6 | Print the full audit trail | JSONL forensic log |

Output captured from an actual demo run in this workspace (your numbers will
differ slightly — they are measured, not scripted):

```text
Phase 2: sent=480 ok=480 errors=0 split={'v1': 429, 'v2': 51}     ← 10.6% canary share
         DECISION: step_up — traffic 10% -> 25%
         DECISION: step_up — traffic 25% -> 50%
         DECISION: promote — promoted after healthy canary at 50%
Phase 3: DECISION: alert — PSI drift on 'tickets_per_month': 2.3933 > 0.2500 …
Phase 4: traffic under fault: p95=1040.5ms (measured by server)
Phase 5: gate blocked v3: min_f1 0.4107 < 0.55; min_auc 0.3207 < 0.65
         DECISION: rollback — live AUC on delayed labels 0.2860 < 0.6000 (177 labelled requests)
Phase 6: 2026-…T21:11:33  canary_rolled_back  ver=v3  metric=performance  value=0.286  threshold=0.6
```

## Architecture

```
                         ┌─────────────────────┐
                         │     Training Data   │  seeded synthetic generator
                         └──────────┬──────────┘  (or your CSV)
                                    ↓
                         ┌─────────────────────┐
                         │   Data Validation   │  schema/ranges/dupes/leakage
                         └──────────┬──────────┘  → fails the pipeline loudly
                                    ↓
                         ┌─────────────────────┐
                         │  Training Pipeline  │  one sklearn Pipeline artifact
                         └──────────┬──────────┘  (features + model, no skew)
                                    ↓
                         ┌─────────────────────┐
                         │  Evaluation Engine  │  f1/precision/recall/AUC/CM
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │    Model Registry   │  v1, v2, v3… + state machine
                         └──────────┬──────────┘  + MLflow mirror
                                    ↓
                         ┌─────────────────────┐
                         │    CI Quality Gate  │  configured thresholds only
                         └──────────┬──────────┘
                                    ↓
                     ┌──────────────┴──────────────┐
                     ↓                             ↓
             ┌───────────────┐             ┌───────────────┐
             │   Model V1    │             │   Model V2    │
             │  Production   │             │    Canary     │
             └───────┬───────┘             └───────┬───────┘
                     │   sticky-hash weighted routing
                     └──────────────┬──────────────┘
                                    ↓
                         ┌─────────────────────┐
                         │   Monitoring Layer  │  latency p50/p95/p99 · errors
                         ├─────────────────────┤  throughput · prediction mix
                         │                     │  PSI/KS drift · live AUC
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │ Deployment          │  promote · step up · alert
                         │ Controller          │  rollback — all audited
                         └─────────────────────┘
```

Processes integrate through **filesystem artifacts** (registry index,
routing state, JSONL logs, audit trail) — zero infrastructure, crash-safe,
fully inspectable. Details: [`docs/architecture.md`](docs/architecture.md).

## MLOps lifecycle

```
Code → Data → Validate → Train → Evaluate → Register → Gate → Deploy
     → Canary → Monitor → (Detect degradation) → Promote | Rollback → Audit
```

Every stage is an explicit command with typed inputs/outputs and persisted
artifacts. Stage ownership, artifact contracts and failure modes:
[`docs/mlops-lifecycle.md`](docs/mlops-lifecycle.md).

## Repository structure

```
sentinel/
├── src/sentinel/
│   ├── config/          # typed layered settings (YAML + env), JSON logging
│   ├── data/            # ingestion, validation, preprocessing, splitting
│   ├── features/        # FeatureEngineer (inside the persisted pipeline)
│   ├── training/        # trainer, metrics, quality gate, artifacts, MLflow tracking
│   ├── registry/        # model versions + lifecycle state machine (atomic JSON index)
│   ├── serving/         # FastAPI app, model repository, canary routing, fault injection
│   ├── monitoring/      # window stats, PSI/KS drift, health policy, collector
│   ├── deployment/      # canary manager, promotion, rollback, audit, controller
│   ├── simulation/      # traffic simulator + failure scenarios
│   ├── pipeline/        # stage orchestration
│   └── cli/             # `sentinel` command surface (typer)
├── configs/             # model.yaml · monitoring.yaml · development.yaml · production.yaml
├── scripts/             # train/evaluate/deploy/simulate/inject/rollback + run_demo.sh + ci_deploy.sh
├── tests/               # unit/ + integration/ + fixtures/ (all local, no services)
├── deployment/
│   ├── docker/          # MLflow Dockerfile, Prometheus config, Grafana provisioning+dashboard
│   └── kubernetes/      # optional minimal manifests (Deployment/Service/ConfigMap/PVC)
├── .github/workflows/   # ci.yml · train.yml · deploy.yml
├── docs/                # architecture · mlops-lifecycle · deployment · monitoring · rollback
├── data/ models/ artifacts/ monitoring/ reports/   # runtime state (gitignored, .gitkeep'd)
├── Dockerfile · docker-compose.yml · Makefile · pyproject.toml · .env.example
└── bin/sentinel         # zero-install launcher (puts src/ on PYTHONPATH)
```

## Data pipeline

**Problem**: customer churn prediction (binary classification, tabular).
**Default source**: a seeded synthetic generator (`data/ingestion.py`) with
realistic structure — contract type, tenure, charges, support tickets,
session behaviour, interactions and label noise — so the platform is fully
reproducible with zero downloads. Bring your own data via
`data.source: csv` + `data.csv_path`. A 200-row sample lives in
`data/synthetic/sample_churn.csv`.

Stages (all persisted, all versioned):

1. **Acquire** — generate/load → content-addressed `dataset_version` (`ds-<sha256[:12]>`) + `dataset_card.json` (rows, positive rate, seed, hash).
2. **Validate** — schema consistency, per-column missing budgets, domain range checks, categorical level checks, binary checks, duplicate rows/ids, target sanity (binary, class-balance bounds), and a **target-leakage screen** (|corr| ≥ threshold or feature ≡ target ⇒ fail). Strict mode raises `DataValidationError`; a Markdown report is produced either way.
3. **Split** — stratified train/val/test (`data.val_size`, `data.test_size`) with per-split hashes recorded in `split_report.json`.
4. **Persist metadata** — everything above lives under `data/processed/<dataset_version>/`.

## Training pipeline

`sentinel train` / `make train` runs: load validated data → fit a single
sklearn `Pipeline` (`FeatureEngineer` → imputation → one-hot/scaling →
estimator) → evaluate on val **and** test → compute training reference
distributions (the drift baseline) → persist run artifacts → log to the
experiment tracker.

* Algorithms: `gradient_boosting` (default), `random_forest`, `logistic_regression` — all hyperparameters in `configs/model.yaml`.
* Feature engineering is **inside** the persisted pipeline: serving computes exactly what training computed (no train/serve skew by construction).
* Runs live in `artifacts/runs/<run_id>/` (`model.joblib`, `metadata.json`, `reference_stats.json`, `split_report.json`, `metrics.json`) until the registry promotes them to `models/v<N>/`.
* Determinism: fixed seeds end-to-end; same seed ⇒ same dataset hash ⇒ same metrics (asserted in `tests/integration/test_pipeline.py`).

## MLflow usage

MLflow is the experiment-tracking backend **when available** — one component
of Sentinel, not the project itself:

* `experiment_tracking.backend: auto|mlflow|local|none` — `auto` uses MLflow if importable/reachable and silently falls back to a local JSONL tracker otherwise (this is why CI and tests need no MLflow server).
* Logged per run: parameters (algorithm, hyperparams, seed, dataset version), metrics (`val_*`, `test_*`), and the full artifact directory.
* Local default backend: `sqlite:///mlflow.db` (MLflow 3 put the plain file store into maintenance mode). Browse with `make mlflow-ui`.
* Registry mirroring: on registration, the model is also logged to the MLflow Model Registry (`sentinel-churn`) with `sentinel_version`/`sentinel_status` tags — best-effort, never blocks the pipeline.
* In Docker Compose, the API and host pipelines point at the MLflow server via `SENTINEL_EXPERIMENT_TRACKING__MLFLOW__TRACKING_URI=http://mlflow:5000`.

## Model registry & versioning

`models/registry.json` (atomic writes, lock-guarded) makes every version a
first-class entity:

```
TRAINED → VALIDATED → CANDIDATE → CANARY → PRODUCTION → ARCHIVED
                         │           │          │
                         └───────────┴──────────┴──→ ROLLED_BACK → ARCHIVED
```

Each `ModelVersion` records: artifact path, `run_id`, `dataset_version`,
`git_commit`/`git_dirty` (populated dynamically; `null` outside a git
checkout — never faked), `code_version`, scenario, params, offline metrics,
MLflow ref, and an append-only **status history** (from/to/when/why/who).

The registry always answers: *current production*, *previous production*,
*current canary*, *current candidate*. Illegal transitions raise
`InvalidTransitionError`. Inspect with `sentinel status` / `sentinel versions`.

## Evaluation & quality gate

Metrics per split: accuracy, precision, recall, F1, ROC-AUC, log-loss,
confusion matrix, predicted positive rate. The gate is one pure function
(`run_quality_gate`) used identically by CLI, pipeline, controller and CI:

```yaml
evaluation:
  primary_metric: f1
  min_f1: 0.55
  min_precision: 0.50
  min_recall: 0.45
  min_auc: 0.65
  require_beat_production: true   # candidate ≥ production + beat_margin
  beat_margin: 0.0
```

* **pass** → `VALIDATED → CANDIDATE`, report in `reports/gate_<v>.md`, audit entry.
* **fail** → stays `TRAINED`, CI exits non-zero, deployment refused.
* **forced override** (`--force`) → allowed but written to the audit log as `manual_override` with the full failing context.

Real behaviour observed in this workspace: a retrained candidate with
val F1 0.6125 was **blocked** because production had 0.6339 — the gate did
its job (`sentinel versions` shows it stuck at `trained`).

## Serving API

FastAPI + uvicorn (`sentinel serve`), canary-aware:

```
GET  /health          liveness
GET  /ready           readiness (production model loadable?)
GET  /model-info      deployed versions, roles, offline metrics, canary %
POST /predict         {request_id?, features{…}} → prediction + provenance
POST /batch-predict   up to 128 requests
POST /feedback        delayed ground truth {request_id, ground_truth}
GET  /metrics         Prometheus exposition
POST /admin/faults    runtime fault injection (token-protected)
POST /admin/reload    force routing/model cache refresh
```

Example real response (captured in this workspace):

```json
{
  "request_id": "manual-1",
  "prediction": 1,
  "probability": 0.951698,
  "model_version": "v1",
  "model_name": "sentinel-churn",
  "served_by": "production",
  "latency_ms": 46.251
}
```

* **Input validation**: Pydantic schemas with domain bounds; invalid input → structured 422 (`{"error": "validation_error", …}`), counted and logged, never reaching the model.
* **Latency**: measured with `time.perf_counter()` around routing + fault injection + inference — never fabricated.
* **Model loading**: artifacts cached per version; routing state re-read on change (mtime polling), so deploys apply without restarts.
* **Structured logs**: one JSON object per event (stdout/stderr split: logs to stderr, results to stdout — pipe-friendly CLI).

## Monitoring & observability

Per request the server writes a JSONL record (timestamp, version, latency,
status, prediction, probability, features). The collector tails these logs
per window and computes, **per model version**:

* request count, error rate, throughput,
* latency p50/p95/p99 (exact percentiles over the window),
* prediction distribution (positive rate, mean probability),
* PSI/KS drift vs the model's training reference,
* live AUC/accuracy/log-loss from delayed `/feedback` labels.

Results land in `monitoring/snapshots.jsonl` + policy violations. Prometheus
metrics (`sentinel_requests_total`, `sentinel_request_latency_seconds`,
`sentinel_predictions_total`, `sentinel_errors_total`,
`sentinel_canary_traffic_percent`, `sentinel_model_info`, …) are scraped by
the compose Prometheus and rendered by the provisioned Grafana dashboard
(queries only — no baked-in values). Details:
[`docs/monitoring.md`](docs/monitoring.md).

## Drift detection

Reference vs current, in identical (post-feature-engineering) coordinates:

* **PSI** (default) over quantile bins for numerics and frequency tables for categoricals; unseen categories fold into an `__unknown__` bucket (drift, not crash). Interpretation: `<0.1` stable · `0.1–0.25` moderate · `>0.25` significant.
* **KS** two-sample statistic + asymptotic p-value (implemented without scipy) as an alternative method.
* Thresholds, method and severity are config (`monitoring.drift.*`).

Real capture from the demo (Phase 3, covariate-shift traffic):
`PSI drift on 'tickets_per_month': 2.3933 > 0.2500 (8 features drifting)` —
while an in-distribution window on the same model scored `max_psi 0.038`
(no drift). Same detector, honest contrast.

## Degradation detection (drift ≠ failure)

Sentinel separates three questions:

| Question | Signal | Default action |
|---|---|---|
| Did the **inputs** move? | PSI/KS drift | `warning` → alert only |
| Is the model **behaving** differently than production? | prediction-rate delta (with per-version minimum sample sizes) | critical → rollback-eligible |
| Is the model **actually worse** on today's traffic? | live AUC on delayed ground truth | critical → rollback-eligible |

Drift alone never rolls anything back unless you explicitly configure
`drift.severity: critical` **and** `rollback.on_drift: true`. The demo
shows drifted traffic producing alerts while the canary keeps serving.

## Canary deployment

```
100% ──→ production v1                      (baseline)
deploy candidate v2 at canary.initial_traffic_percent (default 5%)
   healthy windows ≥ canary.min_healthy_windows_per_step?
      → step up along canary.promotion_steps (default 25 → 50 → 100)
      → at the top of the ramp: PROMOTE (v1 archived as previous_production)
   critical violation attributed to v2?
      → ROLLBACK (v2 → ROLLED_BACK, v1 resumes 100%)
```

* Percentages are configuration, never hardcoded (`5%` appears nowhere in code).
* Allocation: sticky hash of `request_id` (deterministic, uniform — measured 10.6% actual vs 10% configured in the demo) with optional non-sticky sampling.
* One canary at a time; post-rollback cooldown prevents flapping.
* Routing state (`artifacts/routing.json`) is the single source of truth shared by server/controller/CLI; the K8s manifests keep identical semantics.

## Automatic rollback

The deployment controller (`sentinel controller`) is a standalone decision
loop — collect snapshot → evaluate policy → act → audit:

* **rollback triggers** (each independently switchable in `rollback.*`): error rate, p95/p99 latency, prediction shift, live-performance AUC, optionally drift.
* **execution**: routing flips to production-only, registry transitions `CANARY → ROLLED_BACK`, cooldown starts.
* **auditability**: every rollback writes timestamp, actor, rolled-back version, previous/restored version, reason string, triggering metric, observed value, threshold, canary percent at failure, and the full violation payloads.
* **production itself failing**: alerts; optionally auto-restores `previous_production` (`rollback.auto_restore_previous_production`).

A real audit entry from the demo (Phase 5):

```json
{
  "action": "canary_rolled_back", "actor": "controller",
  "model_version": "v3", "previous_version": "v2", "restored_version": "v2",
  "triggered_metric": "performance", "observed_value": 0.285959, "threshold": 0.6,
  "canary_percent": 40,
  "reason": "performance violation on v3: … live AUC on delayed labels 0.2860 < 0.6000 (177 labelled requests)"
}
```

Deep dive: [`docs/rollback.md`](docs/rollback.md).

## Failure injection

| Failure | Command | Detector exercised |
|---|---|---|
| **Bad model** | `sentinel pipeline --scenario degraded` (corrupts training labels/subsample) → gate blocks → `gate --force` (audited) → deploy | live AUC, prediction shift |
| **Covariate drift** | `python scripts/inject_drift.py --mode covariate` | PSI warnings (alert-only) |
| **Concept drift** | `python scripts/inject_drift.py --mode concept` | live AUC collapse on honest labels |
| **High latency** | `python scripts/inject_faults.py --latency-ms 800` | p95/p99 SLOs |
| **Error injection** | `python scripts/inject_faults.py --error-rate 0.2` | error-rate SLO |
| **Bad features** | `sentinel simulate --scenario bad_features` | drift + edge-of-domain behaviour |
| Reset | `python scripts/inject_faults.py --disable` | — |

Server-side faults pass through the same measurement path as real failures
(injected latency is inside the `perf_counter` span; injected errors are
logged with `error_type: "injected"`), so monitoring treats them identically.

## Traffic simulation

`sentinel simulate --scenario <name>` drives the real HTTP API with labelled
synthetic traffic: `normal`, `high_traffic` (5×), `distribution_shift`,
`concept_drift`, `bad_features`. Configurable rate, duration, concurrency,
seed and `feedback_rate` (partial delayed ground truth — exactly like real
label pipelines). Reports persist to `reports/simulation_*.json`; all
latencies in reports are the server's own measurements. The transport is
pluggable, which is how integration tests drive the full stack in-process.

## CI/CD

* **`ci.yml`** (push/PR): ruff lint + format check, configuration-schema validation for both profiles, unit tests, integration tests, coverage summary. Matrix: Python 3.11/3.12. No secrets required.
* **`train.yml`** (manual dispatch): runs the real pipeline (data validation → training → evaluation → gate). Gate failure fails the job (exit 2). Writes **measured** metrics into the job summary and uploads a model bundle (registry + artifacts + reports + processed data).
* **`deploy.yml`** (manual dispatch): downloads a bundle by Train-run id, **re-verifies the gate from the bundle's audit trail**, emits a deployment plan (always, dry-run). Execution only happens with `execute=true` + a `production` GitHub Environment (manual approval) + a self-hosted runner labelled `sentinel-deploy`, running `scripts/ci_deploy.sh` (sync bundle → canary deploy → controller with automatic promote/rollback).

No workflow assumes secrets, cloud infra, or a Kubernetes cluster exists;
the deployment path fails loudly rather than pretending.

## Docker

```bash
docker compose up --build -d
```

* **api** — the model server (production profile, non-root user, curl healthcheck).
* **mlflow** — tracking server (SQLite backend + artifact volume).
* **prometheus** — scrapes the API every 5s (`deployment/docker/prometheus.yml`).
* **grafana** — provisioned datasource + *Sentinel — Model Serving Overview* dashboard: traffic split by version, latency percentiles, errors, prediction distribution, canary %, feedback rate, deployed-models table.
* **controller** (optional profile) — `docker compose --profile controller up controller`.

Host and containers share `./models`, `./artifacts`, `./monitoring`,
`./data`, `./reports` — train on the host, serve in Docker, control from
either. See [`docs/deployment.md`](docs/deployment.md).

## Kubernetes (optional)

`deployment/kubernetes/` contains minimal manifests: Namespace, ConfigMap +
Secret (placeholder token), PVC, Deployment (readiness `/ready`, liveness
`/health`, resource requests/limits, `maxUnavailable: 0` rolling update,
Prometheus scrape annotations) and Service; apply with
`kubectl apply -k deployment/kubernetes/`.

Deliberately minimal: the project is fully functional without Kubernetes,
and file-based state assumes `replicas: 1` / RWO storage (documented swap
points for scaling). K8s is an extension path, not the core.

## Testing

```bash
make test        # full suite (unit + integration), no external services
make test-cov    # with coverage
```

* **Unit (120 tests)**: config loading/env overrides, every validation check, feature engineering, preprocessing, splitting, PSI/KS math (known values, edge cases, unseen categories), window stats, gate threshold matrix, registry state machine (transitions, pointers, history, one-canary rule, restore-previous), canary routing (determinism, uniformity, 0%/100% edges, cooldown), health policy matrix (each violation kind × severity × switches), rollback manager (audit content, disabled mode, no-canary no-op), audit log, controller decision matrix (hand-built snapshots).
* **Integration (37 tests)**: full pipeline (incl. degraded-scenario gate block + reproducibility), serving API (predict/validate/batch/feedback/faults/admin-token/metrics/canary routing), monitoring collector over realistic logs (stats, drift on shifted traffic, live-performance join, snapshot persistence), and the **complete deployment lifecycle in-process**: deploy → canary → real simulated traffic → step-up → promote → degraded canary → measured AUC collapse → automatic rollback → audit assertions.

Tests use tiny data, fast models, tmp-path state and in-process HTTP — the
whole suite runs in well under a minute on a laptop.

## Configuration

Layered, typed, validated (Pydantic):

```
configs/model.yaml        # data, features, validation, model, evaluation, tracking, registry
configs/monitoring.yaml   # monitoring SLOs, drift, rollback policy, faults
configs/development.yaml  # paths, serving, canary, simulation (dev profile)
configs/production.yaml   # stricter SLOs, longer ramp/soak (prod profile)
```

Precedence: **env vars** (`SENTINEL_<SECTION>__<KEY>[__<SUBKEY>]`, e.g.
`SENTINEL_MONITORING__DRIFT__PSI_THRESHOLD=0.1`) → profile YAML → shared
YAMLs → schema defaults. `SENTINEL_ENV` selects the profile;
`sentinel config` prints the effective, redacted configuration. Every
threshold mentioned in this README lives in these files — business logic
contains no magic numbers.

## Security

* `.env.example` documents every variable; **no secrets anywhere in code or configs**; `.env` is gitignored.
* `/admin/*` (fault injection, reload) is token-protected when `SENTINEL_SERVING__ADMIN_TOKEN` is set; unset = open with a boot-time warning (local demos only).
* Strict input validation with domain bounds on every feature; structured error responses that never leak internals.
* Admin-token comparison happens server-side; tokens are never logged (logging is field-explicit).
* Server runs as non-root in Docker; registry/index writes are atomic.

## CLI & Makefile reference

```
make install / install-dev / lint / format / test / test-cov
make data / train / register / evaluate V=v1 / gate V=v1 / pipeline
make serve / monitor [WATCH=1] / controller DURATION=600 / status / versions / audit
make deploy V=v1 / canary V=v2 P=10 / promote / rollback
make simulate S=normal R=20 D=30 / inject-drift M=covariate|concept
make inject-latency L=800 / inject-errors E=0.2 / faults-off
make up / down / demo / mlflow-ui / k8s-apply / k8s-delete / clean
```

CLI equivalents (`sentinel --help`): `config, status, versions, audit,
generate, train, register, evaluate, gate, pipeline, deploy, traffic,
promote, rollback, serve, monitor, controller, simulate, faults, clean`.
Every target maps to real code — no stubs.

## Limitations

Honest constraints of a local-first design:

* **File-based state** (registry, routing, logs) is single-writer-oriented: fine for one host/compose; multi-replica K8s needs RWX storage or a database-backed registry (interfaces are isolated for that swap).
* **Single serving process** keeps the in-memory Prometheus registry authoritative; horizontal scaling needs a pushgateway/shared metrics layer.
* **In-window percentiles** are exact but memory-bounded by the log tail (32MB) — high-volume production would stream into a TSDB instead.
* **Feedback labels** assume a `request_id` join and arrive via the simulator; real label pipelines have their own latency/backfill semantics.
* The **MLflow registry mirror** is best-effort; the local registry is the operational truth (by design, so the platform runs with zero services).
* **Drift KS mode** reconstructs the reference sample from bin centers (references store distributions, not raw data) — PSI mode is exact by construction.
* The churn model is intentionally modest; a 0.74 AUC on synthetic data is not a claim about real churn prediction.

## Future improvements

* Database-backed registry (Postgres) + S3/GCS artifact store; MLflow registry as primary with webhooks.
* Shadow deployment mode (mirror traffic, compare outputs, serve nothing) before canary.
* Multi-armed-bandit routing as an alternative promotion policy behind the same `CanaryRouter` interface.
* Streaming monitoring (Kafka/Pub-Sub → Flink/Bytewasm) instead of log-tailing; OpenTelemetry traces across train→serve→control.
* Online-learning detectors (ADWIN/DDM) for concept drift alongside window statistics.
* Weighted-ingress integration for Kubernetes (Argo Rollouts / Istio) reusing the audit + controller decisions.
* Signed model artifacts and policy-as-code (OPA) for gate rules.

---

## License

MIT — see [LICENSE](LICENSE).
