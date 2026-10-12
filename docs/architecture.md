# Architecture

## System context

Sentinel is a **local-first ML platform**: one Python package (`src/sentinel`)
that owns the full model lifecycle, plus thin operational surfaces (CLI,
FastAPI server, scripts, CI workflows, Docker/K8s manifests).

```
                          ┌─────────────────────┐
                          │     Training Data   │   synthetic generator (seeded)
                          └──────────┬──────────┘   or CSV (data.source: csv)
                                     ↓
                          ┌─────────────────────┐
                          │   Data Validation   │   schema, ranges, duplicates,
                          └──────────┬──────────┘   leakage screen → fail fast
                                     ↓
                          ┌─────────────────────┐
                          │  Training Pipeline  │   sklearn Pipeline persisted
                          └──────────┬──────────┘   with feature engineering
                                     ↓
                          ┌─────────────────────┐
                          │  Evaluation Engine  │   f1/precision/recall/AUC/
                          └──────────┬──────────┘   confusion matrix/log-loss
                                     ↓
                          ┌─────────────────────┐
                          │    Model Registry   │   v1, v2, v3 … first-class
                          └──────────┬──────────┘   versions + state machine
                                     ↓
                          ┌─────────────────────┐
                          │    CI Quality Gate  │   configurable thresholds,
                          └──────────┬──────────┘   beat-production rule
                                     ↓
                      ┌──────────────┴──────────────┐
                      ↓                             ↓
              ┌───────────────┐             ┌───────────────┐
              │   Model V1    │             │   Model V2    │
              │  Production   │             │    Canary     │
              └───────┬───────┘             └───────┬───────┘
                      │      weighted routing       │
                      └──────────────┬──────────────┘
                                     ↓
                          ┌─────────────────────┐
                          │   Monitoring Layer  │  latency p50/p95/p99, errors,
                          ├─────────────────────┤  throughput, prediction mix,
                          │                     │  data drift (PSI/KS),
                          │                     │  delayed-label performance
                          └──────────┬──────────┘
                                     ↓
                          ┌─────────────────────┐
                          │ Deployment          │  promote / step-up /
                          │ Controller          │  rollback / alert — every
                          └─────────────────────┘  action audited (JSONL)
```

## Process architecture (who talks to whom)

Sentinel deliberately uses **filesystem artifacts as the integration bus**
between processes. There are three long-lived/short-lived actors:

```
┌────────────────────┐        ┌───────────────────────┐
│  Pipeline process  │        │   Serving process     │
│  (train/evaluate/  │        │   (FastAPI, uvicorn)  │
│   register/gate)   │        │                       │
└─────────┬──────────┘        └──────────┬────────────┘
          │ writes                       │ reads (mtime polling)
          ↓                              ↓
   models/registry.json  ←──────  artifacts/routing.json
   models/v<N>/*                         │ writes
   artifacts/runs/*                      ↓
          │                    monitoring/prediction_log.jsonl
          │                    monitoring/feedback_log.jsonl
          │                              │ reads (windowed tail)
          │                              ↓
          │                  ┌───────────────────────┐
          │                  │  Controller process   │
          └─────────────────→│  (monitor → evaluate  │
             reads registry  │   → promote/rollback) │
                             └──────────┬────────────┘
                                        │ writes routing.json,
                                        │ registry.json,
                                        ↓ audit JSONL
```

Why files and not a message bus?

* Zero infrastructure — the whole platform runs on a laptop.
* Crash-safe and inspectable: every state transition is a JSON document you
  can `cat`, diff, and put in an artifact archive.
* The same semantics map cleanly onto Kubernetes (ConfigMap/volume or a
  weighted ingress) when you outgrow the single host.

## Component map

| Package | Responsibility | Key files |
|---|---|---|
| `config` | Layered YAML + env-var typed settings, JSON logging | `settings.py`, `logging.py` |
| `data` | Ingestion (synthetic/CSV), validation, splits, preprocessing pipeline | `ingestion.py`, `validation.py`, `preprocessing.py`, `splitting.py` |
| `features` | Train/serve-consistent feature engineering (inside the sklearn Pipeline) | `engineering.py` |
| `training` | Trainer, metrics, quality gate, run artifacts, MLflow/local tracking | `train.py`, `evaluate.py`, `artifacts.py`, `tracking.py` |
| `registry` | Model versions as first-class entities; lifecycle state machine; atomic locked JSON index | `metadata.py`, `manager.py` |
| `serving` | FastAPI inference, canary routing, fault injection, prediction/feedback logs, Prometheus | `app.py`, `model_loader.py`, `faults.py`, `schemas.py` |
| `monitoring` | Window stats, PSI/KS drift, delayed-label performance, health policy | `collector.py`, `drift.py`, `health.py`, `metrics.py` |
| `deployment` | Canary manager, promotion gate, rollback, audit log, controller loop | `canary.py`, `promotion.py`, `rollback.py`, `audit.py`, `controller.py` |
| `simulation` | Traffic generator + failure-injection scenarios | `traffic.py`, `scenarios.py` |
| `pipeline` | Stage orchestration (train→evaluate→register→gate) | `orchestration.py` |
| `cli` | Operator surface (`sentinel …`), used by Makefile/scripts/CI | `main.py` |

## Key design decisions

1. **Model versions are first-class.** `models/registry.json` stores a
   `ModelVersion` per release with artifact path, dataset version, git
   commit, code version, params, offline metrics, status and a complete
   status-change history. Serving and deployment only ever talk to the
   registry — never to loose files.

2. **One artifact = one Pipeline object.** Feature engineering, imputation,
   encoding, scaling and the estimator are a single persisted sklearn
   `Pipeline`, eliminating train/serve skew by construction.

3. **Thresholds live in configuration, never in code.** The gate
   (`evaluation.*`), SLOs (`monitoring.*`), canary ramp (`canary.*`) and
   rollback policy (`rollback.*`) are YAML/env-driven. Business logic only
   *evaluates* configuration.

4. **Drift ≠ degradation.** Drift (input distribution) produces `warning`
   violations by default — alerts without automatic action. Model
   degradation is measured separately via delayed ground-truth labels
   (live AUC) and prediction-distribution shift. Each signal has its own
   threshold and its own rollback switch.

5. **Everything auditable.** Every gate decision, canary start, traffic
   change, promotion, rollback, alert and forced override appends an
   immutable JSONL audit record with timestamp, actor, versions, triggering
   metric, observed value and threshold.

6. **MLflow is a component, not the project.** MLflow receives parameters,
   metrics and artifacts, and mirrors registry versions when available.
   The operational source of truth for serving/deployment is the local
   registry, so the platform runs with zero external services
   (`experiment_tracking.backend: auto|mlflow|local|none`).

## Data & artifact layout (runtime)

```
data/processed/<dataset_version>/     data.csv, dataset_card.json, train/val/test.csv, split_report.json
artifacts/runs/<run_id>/              model.joblib, metadata.json, reference_stats.json, metrics.json
models/v<N>/                          promoted copy of a run (registry-managed)
models/registry.json                  registry index (atomic writes, file lock)
artifacts/routing.json                live traffic split (production/canary/percent)
artifacts/controller_state.json       controller phase + healthy-window counters
artifacts/audit/deployment_audit.jsonl  immutable deployment audit trail
monitoring/prediction_log.jsonl       per-request serving log (features, latency, outcome)
monitoring/feedback_log.jsonl         delayed ground truth (request_id join)
monitoring/snapshots.jsonl            monitoring decision windows
reports/                              gate/metrics/evaluation/simulation reports
```
