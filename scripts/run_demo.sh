#!/usr/bin/env bash
# ===========================================================================
# Sentinel — complete MLOps lifecycle demo (single command).
#
#   Phase 0  start model server
#   Phase 1  train v1 → data validation → evaluation → gate → register
#            → deploy to production (100%)
#   Phase 2  train v2 (new data seed) → gate → canary at 10%
#            → simulated healthy traffic → controller ramps 10→25→50→promote
#   Phase 3  covariate drift injection → drift detected → WARNING alert only
#            (drift ≠ model failure; policy comes from configs/monitoring.yaml)
#   Phase 4  latency fault injection → p95 SLO violation → production alert
#   Phase 5  deliberately degraded model v3 → gate BLOCKS it → operator
#            FORCE (audited) → canary → real traffic + delayed labels →
#            controller measures live AUC collapse → AUTOMATIC ROLLBACK
#   Phase 6  show the full audit trail
#
# Every number shown is measured from the actual running system.
# Tuning env vars below make decisions fast enough for a demo; production
# profiles use longer soak windows (configs/production.yaml).
# ===========================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# --- demo tuning (fast decisions, small windows) -----------------------------
export SENTINEL_ENV="${SENTINEL_ENV:-development}"
export SENTINEL_CANARY__MIN_HEALTHY_WINDOWS_PER_STEP=1
export SENTINEL_MONITORING__MIN_SAMPLES_FOR_DECISION=10
export SENTINEL_MONITORING__PERFORMANCE__MIN_SAMPLES=20
export SENTINEL_MONITORING__DRIFT__MIN_SAMPLES=20
export SENTINEL_MONITORING__PREDICTION_SHIFT__MIN_SAMPLES=20
export SENTINEL_ROLLBACK__COOLDOWN_SECONDS=10
# v2 promotion story: absolute thresholds still apply; the beat-production
# rule is disabled only to keep the scripted demo deterministic.
export SENTINEL_EVALUATION__REQUIRE_BEAT_PRODUCTION=false

N_SAMPLES="${DEMO_N_SAMPLES:-12000}"
BASE_URL="http://127.0.0.1:8000"
SERVER_LOG="artifacts/demo_server.log"
S="./bin/sentinel"

banner() { printf '\n\033[1;36m========== %s ==========\033[0m\n' "$1"; }
note()   { printf '\033[0;33m» %s\033[0m\n' "$1"; }

jq_py() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

controller_step() {  # run one controller cycle, print action + reason
  $S controller --once 2>/dev/null | jq_py "'DECISION: ' + d['action'] + ' — ' + d['reason'][:150]"
}

# --------------------------------------------------------------------------- #
banner "Phase 0 — clean runtime state & start model server"
$S clean --all --yes >/dev/null 2>&1 || true
mkdir -p artifacts
PYTHONPATH=src python3 -m sentinel.cli serve > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
note "server pid=${SERVER_PID}, log=${SERVER_LOG}"
python3 - "$BASE_URL" <<'PY'
import sys, time
import httpx
deadline = time.time() + 60
while time.time() < deadline:
    try:
        r = httpx.get(sys.argv[1] + "/health", timeout=2)
        if r.status_code == 200:
            print("server is up:", r.json())
            sys.exit(0)
    except Exception:
        pass
    time.sleep(0.5)
print("server failed to start — see artifacts/demo_server.log", file=sys.stderr)
sys.exit(1)
PY

# --------------------------------------------------------------------------- #
banner "Phase 1 — train v1, pass the quality gate, deploy to production"
note "pipeline: ingest → validate schema/quality → split → train → evaluate → register → gate"
V1=$($S pipeline --seed 42 --n-samples "${N_SAMPLES}" 2>/dev/null | jq_py "d['model_version']")
note "trained and gated: ${V1}"
$S evaluate "${V1}" --split test 2>/dev/null | jq_py "'test metrics: f1=%.4f auc=%.4f precision=%.4f recall=%.4f' % (d['f1'], d['roc_auc'], d['precision'], d['recall'])"
$S deploy "${V1}" --direct 2>/dev/null | jq_py "'deployed: ' + str(d['deployed']) + ' mode=' + d['mode'] + ' routing=' + str(d['routing']['production'])"
sleep 2
note "server sees production model: $($S status 2>/dev/null | jq_py "d['routing']['production']")"

# --------------------------------------------------------------------------- #
banner "Phase 2 — train v2 (fresh data seed), canary at 10%, ramp & promote"
V2=$($S pipeline --seed 43 --n-samples "${N_SAMPLES}" 2>/dev/null | jq_py "d['model_version']")
note "candidate: ${V2}"
$S deploy "${V2}" --percent 10 2>/dev/null | jq_py "'canary started: prod=' + d['routing']['production'] + ' canary=' + str(d['routing']['canary']) + ' at ' + str(d['routing']['canary_percent']) + '%'"
sleep 2
note "sending healthy traffic (with delayed ground-truth feedback)…"
$S simulate --scenario normal --rps 60 --duration 8 --feedback-rate 0.8 2>/dev/null | jq_py "'sent=%d ok=%d errors=%d split=%s' % (d['n_sent'], d['n_ok'], d['n_errors'], d['served_by_version'])"
for i in 1 2 3; do
  controller_step
done
note "routing now: $($S status 2>/dev/null | jq_py "d['routing']")"

# --------------------------------------------------------------------------- #
banner "Phase 3 — inject COVARIATE DRIFT (features shift, labels keep mechanism)"
note "expect: drift detectors fire (PSI), severity=warning → alert only, NO rollback"
$S simulate --scenario distribution_shift --rps 60 --duration 6 2>/dev/null | jq_py "'drifted traffic sent: %d requests' % d['n_sent']"
controller_step
$S monitor 2>/dev/null | jq_py "'drift report: ' + str({v: {'is_drifted': r['is_drifted'], 'max_psi': r['max_score']} for v, r in d['drift'].items()})"

# --------------------------------------------------------------------------- #
banner "Phase 4 — inject LATENCY fault (800ms) → production SLO alert"
python3 scripts/inject_faults.py --latency-ms 800 >/dev/null
$S simulate --scenario normal --rps 40 --duration 5 2>/dev/null | jq_py "'traffic under fault: p95=%.1fms (measured by server)' % d['server_latency_ms']['p95']"
controller_step
python3 scripts/inject_faults.py --disable >/dev/null
note "faults disabled again"

# --------------------------------------------------------------------------- #
banner "Phase 5 — deploy a DELIBERATELY DEGRADED model → automatic rollback"
note "training v3 with corrupted labels (scenario=degraded). The gate WILL block it…"
set +e
$S pipeline --scenario degraded --seed 43 --n-samples "${N_SAMPLES}" 2>/dev/null > /tmp/degraded_out.json
GATE_EXIT=$?
set -e
V3=$(jq_py "d['model_version']" < /tmp/degraded_out.json)
jq_py "'gate blocked %s: %s' % (d['model_version'], d['gate_reason'][:120])" < /tmp/degraded_out.json
note "operator FORCES the deployment anyway (audited override) to demo rollback…"
$S gate "${V3}" --force >/dev/null 2>&1
$S deploy "${V3}" --percent 40 2>/dev/null | jq_py "'canary started: ' + str(d['routing']['canary']) + ' at ' + str(d['routing']['canary_percent']) + '%'"
sleep 2
$S simulate --scenario normal --rps 60 --duration 8 --feedback-rate 1.0 2>/dev/null | jq_py "'traffic: sent=%d split=%s' % (d['n_sent'], d['served_by_version'])"
note "controller evaluates live metrics (incl. delayed-label AUC)…"
controller_step
note "routing after rollback: $($S status 2>/dev/null | jq_py "d['routing']")"

# --------------------------------------------------------------------------- #
banner "Phase 6 — the audit trail tells the whole story"
$S versions 2>/dev/null | jq_py "'\n'.join('%-4s %-12s scenario=%-9s val_f1=%.4f' % (v['version'], v['status'], v['scenario'], v['val_f1']) for v in d)"
echo
note "last audit events:"
$S audit --last 8 2>/dev/null | jq_py "'\n'.join('%s  %-20s ver=%-4s metric=%-16s value=%s threshold=%s' % (e['timestamp'][:19], e['action'], str(e.get('model_version')), str(e.get('triggered_metric')), e.get('observed_value'), e.get('threshold')) for e in d)"

banner "Demo complete — everything above was measured live"
note "Grafana/Prometheus views available via: docker compose up -d (see docs/monitoring.md)"
