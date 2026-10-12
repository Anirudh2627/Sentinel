# Monitoring & Drift Detection

## Signal inventory

| Signal | Source | Granularity | Default policy |
|---|---|---|---|
| Request count / throughput | prediction log + Prometheus counter | per model version, per window | informational |
| Error rate | prediction log (`status != ok`) | per version | SLO: `monitoring.max_error_rate` → rollback-eligible |
| Latency p50/p95/p99 | server-measured `perf_counter` | per version | SLO: `max_p95_latency_ms`, `max_p99_latency_ms` → rollback-eligible |
| Prediction distribution | prediction log (`prediction`) | per version | canary-vs-production delta: `prediction_shift.max_delta_positive_rate` → rollback-eligible |
| Data drift (PSI/KS) | prediction-log features vs training reference | per version | `drift.severity: warning` → **alert only** by default |
| Live performance (AUC/acc/log-loss) | `/feedback` delayed labels joined to predictions | per version | `performance.min_auc` → rollback-eligible |

All thresholds are configuration (`configs/monitoring.yaml`); nothing is
hardcoded. Violations carry `severity: warning|critical`; only critical
violations **attributed to the active canary** trigger automatic rollback
(see `docs/rollback.md`).

## Data flow

```
FastAPI /predict ──→ monitoring/prediction_log.jsonl   (features, latency, prediction, status)
FastAPI /feedback ─→ monitoring/feedback_log.jsonl     (request_id → ground truth)
prometheus_client → GET /metrics ──→ Prometheus ──→ Grafana dashboards

MetricsCollector (sentinel monitor / controller)
  ├─ tails both logs for the evaluation window
  ├─ groups by model_version → WindowStats (error rate, percentiles, positive rate…)
  ├─ rebuilds engineered features (FeatureEngineer) and compares against
  │  each model's reference_stats.json → DriftReport (PSI or KS per feature)
  ├─ joins delayed labels → PerformanceStats (live AUC etc.)
  ├─ HealthEvaluator → Violation[] (thresholds from config)
  └─ appends MonitoringSnapshot → monitoring/snapshots.jsonl
```

Because the collector consumes the *same logs the server writes*, the
numbers shown by `sentinel monitor`, stored in snapshots, exposed to
Prometheus and used by the controller are all derived from real traffic.

## Drift detection details

Reference baseline (persisted at training time per model):

* numeric features: quantile bin edges (default 10 bins) + bin
  probabilities, plus mean/std/percentiles,
* categorical features: observed level frequencies,
* computed in **post-feature-engineering space**, and production samples are
  passed through the same stateless `FeatureEngineer` before comparison —
  reference and current live in identical coordinates.

Methods (`monitoring.drift.method`):

* **PSI** (default): `Σ (q−p)·ln(q/p)` over bins. Rule of thumb:
  `<0.1` stable, `0.1–0.25` moderate, `>0.25` significant. Unknown
  categories are folded into an `__unknown__` bucket with an epsilon
  reference mass (so unseen levels register as drift instead of crashing).
* **KS**: two-sample Kolmogorov–Smirnov statistic with the asymptotic
  p-value (implemented directly — no scipy dependency); a feature counts as
  drifted when `p < 0.01` **and** `D > ks_threshold`.

### Drift is not failure

A drift violation means *inputs moved*, which can happen for benign reasons
(seasonality, marketing campaign, upstream schema-adjacent behaviour). By
default drift produces `warning` violations → audit `alert` entries, no
traffic action. Two switches make it actionable when you want that:

```yaml
monitoring.drift.severity: critical   # drift violations become critical
rollback.on_drift: true               # critical drift may trigger rollback
```

The demo (`scripts/run_demo.sh`, Phase 3) shows drifted traffic producing
PSI ≫ threshold while the controller deliberately only alerts.

## Delayed-label performance (true degradation signal)

The simulator (and, in real life, your label pipeline) posts ground truth to
`POST /feedback {request_id, ground_truth}`. The collector joins labels to
predictions per model version and computes live AUC/accuracy/log-loss once
`performance.min_samples` labels exist in the window. This is what separates
"inputs moved but the model still works" from "the model is genuinely
failing on today's traffic" — the Phase 5 demo rollback is triggered by a
measured live AUC collapse, not by drift.

## Prometheus & Grafana

Metrics exposed at `GET /metrics` (per-model-version labels throughout):

```
sentinel_requests_total{model_version,endpoint,status}
sentinel_errors_total{model_version,error_type}
sentinel_request_latency_seconds_bucket{model_version,endpoint,le}
sentinel_predictions_total{model_version,prediction}
sentinel_feedback_total{model_version}
sentinel_model_info{version,role,status}
sentinel_canary_traffic_percent
```

`docker compose up -d` provisions Prometheus (5s scrape) and Grafana with
the **Sentinel — Model Serving Overview** dashboard: request rate split by
version/status, latency percentiles, error rate, prediction distribution,
canary traffic %, feedback rate and deployed-model table. Every panel is a
live query — no baked-in values.

## Statistical hygiene

* Windows with fewer than `min_samples_for_decision` requests never produce
  SLO violations (no judging a model on 3 requests).
* Prediction-shift comparison requires `prediction_shift.min_samples`
  predictions **per version** — at a 5% canary slice, a handful of requests
  is noise, and the evaluator treats it as such.
* Percentiles are computed from the actual latency samples in the window
  (exact, not t-digest-approximated — window sizes here are small).
