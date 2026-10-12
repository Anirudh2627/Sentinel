# Canary Deployment & Automatic Rollback

## Canary mechanics

Traffic allocation is **sticky-hash routing** inside the serving layer:

```
bucket = int(md5(request_id).hexdigest(), 16) % 10000
route  = canary if bucket < canary_percent * 100 else production
```

Properties: deterministic per request id (replayable, testable), uniform
(verified in `tests/unit/test_canary.py` within ±3pp at 20k samples), and
independent of client identity spoofing — the split is a server-side policy.
Non-sticky sampling is available via `canary.sticky_routing: false`.

The split itself lives in `artifacts/routing.json`:

```json
{
  "production": "v1",
  "canary": "v2",
  "canary_percent": 25,
  "updated_at": "…",
  "updated_by": "controller"
}
```

The API server polls this file (mtime-checked, `routing_reload_interval_seconds`)
— deployments take effect without restarts, and the file is the single
source of truth shared by server, controller and CLI.

## The ramp

```
deploy candidate (canary.initial_traffic_percent, default 5%)
        │
        ▼
┌────────────────────────┐   healthy window? (traffic evidence + no critical violations)
│  count healthy windows │──────────────────────────────┐
└───────────┬────────────┘                              │ ≥ canary.min_healthy_windows_per_step
            │ critical violation (canary-attributed)    ▼
            ▼                                   next step in canary.promotion_steps
      ROLLBACK (audited)                        (e.g. 5 → 25 → 50)
                                                       │ top of ramp reached
                                                       ▼
                                               PROMOTE to production
                                        (old production → ARCHIVED,
                                         kept as previous_production)
```

Windows without enough traffic (`monitoring.min_samples_for_decision`)
neither promote nor condemn — they simply don't count as evidence.

## Rollback triggers

A rollback fires when the window contains a **critical** violation
attributed to the active canary and the matching policy switch is on:

| Trigger | Config switch | Default threshold |
|---|---|---|
| Error rate | `rollback.on_error_rate` | `monitoring.max_error_rate: 0.05` |
| Latency p95/p99 | `rollback.on_latency` | `max_p95_latency_ms: 500`, `max_p99_latency_ms: 1500` |
| Prediction-distribution shift vs production | `rollback.on_prediction_shift` | `max_delta_positive_rate: 0.15` |
| Live performance on delayed labels | `rollback.on_performance` | `performance.min_auc: 0.60` |
| Data drift | `rollback.on_drift` (**off** by default) | `drift.psi_threshold: 0.25` + `drift.severity` |
| Master switch | `rollback.enabled` | true |

Production-attributed critical violations never roll back the *canary*.
They raise alerts, and optionally restore the previous production model
(`rollback.auto_restore_previous_production`, off by default) using the
registry's `previous_production` pointer.

## Rollback execution

`RollbackManager.rollback_canary()`:

1. routing state rewritten → `canary: null, canary_percent: 0`
   (production instantly serves 100%; the API picks it up on next poll),
2. registry transition `CANARY → ROLLED_BACK` (illegal-transition guarded),
3. audit entry appended (below),
4. cooldown starts: `deploy_candidate` refuses new canaries for
   `rollback.cooldown_seconds` (prevents rollback→redeploy flapping).

Manual equivalents: `sentinel rollback` (canary) and
`sentinel rollback --production` (restore previous production). Both go
through the same audited code paths.

## Audit anatomy

Every rollback leaves a forensic record in
`artifacts/audit/deployment_audit.jsonl`:

```json
{
  "timestamp": "2026-09-26T21:11:33.153649+00:00",
  "action": "canary_rolled_back",
  "actor": "controller",
  "model_version": "v3",
  "previous_version": "v2",
  "restored_version": "v2",
  "reason": "performance violation on v3: value=0.285959 threshold=0.6 (live AUC on delayed labels 0.2860 < 0.6000 (177 labelled requests))",
  "triggered_metric": "performance",
  "observed_value": 0.285959,
  "threshold": 0.6,
  "canary_percent": 40,
  "context": { "violations": [ … full violation payloads … ] }
}
```

(Real entry captured from a demo run — the degraded canary's live AUC on
177 delayed labels was 0.286 against a 0.60 threshold.)

Audited actions: `gate_decision`, `manual_override` (forced gates),
`canary_started`, `traffic_changed`, `canary_promoted`,
`canary_rolled_back`, `production_deployed`, `production_restored`,
`alert`, `fault_injection`.

## Failure-injection toolbox

To exercise rollback without waiting for a real incident:

| Failure | How | What trips |
|---|---|---|
| Bad model | `sentinel pipeline --scenario degraded` then `sentinel gate vN --force` + `sentinel deploy vN` | live AUC collapse + prediction shift |
| Covariate drift | `scripts/inject_drift.py --mode covariate` | PSI warnings (alert-only by default) |
| Concept drift | `scripts/inject_drift.py --mode concept` | live AUC degradation (labels genuinely change) |
| High latency | `scripts/inject_faults.py --latency-ms 800` | p95/p99 SLO violations |
| Errors | `scripts/inject_faults.py --error-rate 0.2` | error-rate SLO violations |
| Bad features | `sentinel simulate --scenario bad_features` | edge-of-domain behaviour (drift + validation counts) |

Fault injection is server-side (`FaultInjector`), measured by the same
latency/error instrumentation as real failures, and only enabled via config
or the token-protected `/admin/faults` endpoint.
