# MLOps Lifecycle

This document walks through each lifecycle stage, the component that owns it,
the artifacts it produces, and the failure modes it guards against.

```
Code → Data → Validate → Train → Evaluate → Register → Gate → Deploy
     → Canary → Monitor → (Detect degradation) → Promote | Rollback → Audit
```

## 1. Code & configuration

* All behaviour is driven by `configs/*.yaml` with env-var overrides
  (`SENTINEL_<SECTION>__<KEY>`). Profiles: `development`, `production`.
* Model versions record `git_commit` + `git_dirty` (populated dynamically;
  `null` when the workspace is not a git checkout — never faked) and
  `code_version` (package version), so any artifact can be traced to code.

## 2. Data ingestion

* Default source: **seeded synthetic churn generator** — content-addressed
  dataset versions (`ds-<sha256[:12]>`) make runs reproducible and dataset
  identity verifiable. A CSV source is supported (`data.source: csv`).
* Output: `data/processed/<dataset_version>/data.csv` + `dataset_card.json`
  (rows, positive rate, content hash, seed, provenance).

## 3. Data validation (fail fast)

`sentinel.data.validation.DataValidator` runs schema and quality checks:

| Check | Fails when |
|---|---|
| schema | expected column missing (unexpected columns → warning) |
| min_rows | dataset smaller than `validation.min_rows` |
| missing_values | any column over `max_missing_ratio` |
| numeric_ranges | domain bounds violated (e.g. tenure > 72) |
| categorical_levels | unseen category levels |
| binary_columns | 0/1 columns contain other values |
| duplicates | duplicated rows/ids above tolerance |
| target | missing, non-binary, or class ratio out of bounds |
| leakage_screen | \|corr(feature, target)\| ≥ `leakage_corr_threshold` or feature equals target |

In strict mode (`validation.strict: true`) a failed check raises
`DataValidationError` and the pipeline stops before training.

## 4. Training

`sentinel train` / `run_pipeline()`:

1. stratified train/val/test split (persisted with hashes in `split_report.json`),
2. fit a single sklearn `Pipeline` (feature engineering → imputation →
   one-hot/scaling → estimator),
3. evaluate on val **and** test,
4. compute **reference distributions** (quantile bins + categorical
   frequencies) from training features — the drift baseline,
5. persist run artifacts under `artifacts/runs/<run_id>/`,
6. log parameters/metrics/artifacts to the configured tracker
   (MLflow when available; local JSONL fallback).

Failure injection at this stage: `--scenario degraded` corrupts the
*training signal* (label flips + subsampling per `model.degraded.*`),
producing a genuinely bad model for gate/rollback demos.

## 5. Evaluation & the quality gate

Metrics: accuracy, precision, recall, F1, ROC-AUC, log-loss, confusion
matrix, predicted positive rate (validation + test splits).

The gate (`sentinel.training.evaluate.run_quality_gate`) is pure policy
evaluation over configuration:

```
candidate passes iff
    f1        ≥ evaluation.min_f1
AND precision ≥ evaluation.min_precision
AND recall    ≥ evaluation.min_recall
AND roc_auc   ≥ evaluation.min_auc
AND (if require_beat_production and a production model exists)
    candidate[primary_metric] ≥ production[primary_metric] + beat_margin
```

Gate outcomes are written to `reports/gate_<version>.md`, recorded in the
audit log, and drive registry transitions:

* **pass** → `TRAINED → VALIDATED → CANDIDATE`
* **fail** → stays `TRAINED` (not deployable)
* **forced override** → candidate, but a `manual_override` audit entry with
  the full failing gate context is written. CI exits non-zero on failure.

## 6. Registry & lifecycle states

```
TRAINED → VALIDATED → CANDIDATE → CANARY → PRODUCTION → ARCHIVED
                          │          │          │
                          └──────────┴──────────┴──→ ROLLED_BACK → ARCHIVED
```

The registry (`models/registry.json`, atomic writes + file lock) always
answers: current production, previous production, current canary, current
candidate, and the full status history (who/when/why) per version.

## 7. Deployment & canary

`sentinel deploy <version>` (canary by default):

```
production v1: 95% ─┐
                    ├─ sticky hash routing (md5(request_id) % 10000)
canary v2:      5% ─┘
```

* Initial percent: `canary.initial_traffic_percent`; ramp:
  `canary.promotion_steps` (e.g. 25 → 50 → 100).
* A post-rollback **cooldown** (`rollback.cooldown_seconds`) blocks new
  canaries immediately after a rollback.
* First deployment (no production baseline) goes direct-to-production.

## 8. Monitoring

The server logs every request (features, latency, prediction, outcome) to
`monitoring/prediction_log.jsonl`; delayed ground truth arrives via
`POST /feedback` into `monitoring/feedback_log.jsonl`. The collector joins
them per window and computes:

* operational SLOs: error rate, p95/p99 latency, throughput (per version),
* prediction-distribution shift: canary vs production positive rate,
* data drift: PSI (default) or KS against the model's training reference,
* live performance: AUC/accuracy/log-loss on delayed labels (per version).

## 9. Degradation handling (promote / rollback)

The controller (`sentinel controller`) loops: collect → evaluate → decide.

| Situation | Decision |
|---|---|
| critical violation attributed to the **canary** | **ROLLBACK** canary (audited; production resumes 100%) |
| enough healthy windows at current step | **STEP UP** traffic to next ramp value |
| healthy at top of ramp | **PROMOTE** canary → production (old production archived as previous) |
| warnings (e.g. drift with default severity) | **ALERT** only (audit entries; no traffic action) |
| critical violation on **production** | alert; optionally restore previous production (`rollback.auto_restore_previous_production`) |

Every decision updates `artifacts/controller_state.json` and appends audit
records. See `docs/rollback.md` for the rollback anatomy.

## 10. Reproducibility contract

* Same seed + same config ⇒ same dataset version (content hash) ⇒ same
  model metrics.
* Every model version directory contains everything needed to re-derive its
  story: `metadata.json` (params, dataset, git, metrics), `model.joblib`,
  `reference_stats.json`, `split_report.json`.
* Experiment tracking mirrors parameters/metrics/artifacts to MLflow when
  enabled (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).
