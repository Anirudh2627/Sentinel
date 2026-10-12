#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# ci_deploy.sh — execute a canary deployment from a verified model bundle.
#
# Intended to run on a host that already runs the Sentinel stack
# (e.g. `docker compose up -d` from this repo, or any host with the
# dependencies installed). It is deliberately explicit: no magic, no cloud
# assumptions. If the prerequisites are missing it FAILS LOUDLY instead of
# pretending to deploy.
#
# Usage:  bash scripts/ci_deploy.sh <bundle-dir>
# Env:    MODEL_VERSION   version to deploy (required)
#         CANARY_PERCENT  initial canary traffic (default 5)
#         CONTROLLER_DURATION  seconds to run the controller (default 1800)
# ---------------------------------------------------------------------------
set -euo pipefail

BUNDLE_DIR="${1:?usage: ci_deploy.sh <bundle-dir>}"
MODEL_VERSION="${MODEL_VERSION:?MODEL_VERSION must be set}"
CANARY_PERCENT="${CANARY_PERCENT:-5}"
CONTROLLER_DURATION="${CONTROLLER_DURATION:-1800}"

echo "== ci_deploy.sh =="
echo "bundle:     ${BUNDLE_DIR}"
echo "version:    ${MODEL_VERSION}"
echo "canary:     ${CANARY_PERCENT}%"
echo "controller: ${CONTROLLER_DURATION}s"

# --- sanity: bundle must contain a registry with the requested version ------
REGISTRY="${BUNDLE_DIR}/models/registry.json"
if [[ ! -f "${REGISTRY}" ]]; then
  echo "ERROR: ${REGISTRY} not found — not a valid bundle" >&2
  exit 1
fi
if ! grep -q "\"${MODEL_VERSION}\"" "${REGISTRY}"; then
  echo "ERROR: ${MODEL_VERSION} not present in bundle registry" >&2
  exit 1
fi

# --- sync bundle state into the runtime workspace ---------------------------
# (registry + model artifacts; operational logs are NOT overwritten)
mkdir -p models artifacts reports
cp -r "${BUNDLE_DIR}/models/." models/
if [[ -d "${BUNDLE_DIR}/artifacts/runs" ]]; then
  mkdir -p artifacts/runs
  cp -r "${BUNDLE_DIR}/artifacts/runs/." artifacts/runs/
fi
if [[ -d "${BUNDLE_DIR}/data/processed" ]]; then
  mkdir -p data/processed
  cp -r "${BUNDLE_DIR}/data/processed/." data/processed/
fi

SENTINEL="${SENTINEL_CMD:-./bin/sentinel}"

# --- deploy as canary --------------------------------------------------------
${SENTINEL} deploy "${MODEL_VERSION}" --percent "${CANARY_PERCENT}"

# --- let the controller drive promotion/rollback -----------------------------
set +e
${SENTINEL} controller --duration "${CONTROLLER_DURATION}" --interval 15 --stop-on-terminal
CONTROLLER_EXIT=$?
set -e

echo "== post-deployment status =="
${SENTINEL} status
${SENTINEL} audit --last 10

if [[ ${CONTROLLER_EXIT} -ne 0 ]]; then
  echo "controller exited with ${CONTROLLER_EXIT}" >&2
  exit ${CONTROLLER_EXIT}
fi
echo "ci_deploy.sh finished"
