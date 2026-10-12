# Deployment Guide

Three supported topologies, from zero-infrastructure to cluster:

1. **Bare local process** (default; nothing to install beyond Python deps)
2. **Docker Compose stack** (API + MLflow + Prometheus + Grafana)
3. **Kubernetes** (optional, minimal manifests)

---

## 1. Local process

```bash
pip install -r requirements.txt        # or: pip install -e ".[dev]"

# train → evaluate → register → gate
./bin/sentinel pipeline                # (or: make pipeline)

# serve (reads registry + routing from disk; picks up deploys without restart)
./bin/sentinel serve                   # http://127.0.0.1:8000/docs

# in another shell: deploy / canary / control
./bin/sentinel deploy v1 --direct      # first model → production
./bin/sentinel deploy v2 --percent 10  # candidate → canary @10%
./bin/sentinel controller --duration 600 --interval 15
```

`bin/sentinel` is a launcher that puts `src/` on `PYTHONPATH`, so everything
works **without installing the package**. After `pip install -e .` you can
use the `sentinel` entrypoint directly.

State locations (development profile): `models/` (registry+artifacts),
`artifacts/routing.json` (traffic split), `monitoring/*.jsonl` (serving
logs), `artifacts/audit/deployment_audit.jsonl` (audit trail).

Reset everything: `./bin/sentinel clean --all --yes`.

## 2. Docker Compose

```bash
docker compose up --build -d
```

| Service | URL | Purpose |
|---|---|---|
| api | http://localhost:8000/docs | FastAPI model server (`/health`, `/ready`, `/predict`, `/metrics`, `/admin/*`) |
| mlflow | http://localhost:5000 | experiment tracking + model registry UI |
| prometheus | http://localhost:9090 | scrapes `api:/metrics` every 5s |
| grafana | http://localhost:3000 | provisioned datasource + "Sentinel — Model Serving Overview" dashboard (admin / `$GRAFANA_PASSWORD`, default `admin`) |

Host↔container integration: `./models`, `./artifacts`, `./monitoring`,
`./data`, `./reports` are bind-mounted, so you can train on the host and
deploy into the container (or vice versa) — routing changes are picked up
by the API within `serving.routing_reload_interval_seconds`.

Run the deployment controller as a container (on-demand profile):

```bash
docker compose --profile controller up controller
```

Train from the host against the containerized MLflow:

```bash
export SENTINEL_EXPERIMENT_TRACKING__MLFLOW__TRACKING_URI=http://localhost:5000
./bin/sentinel pipeline
```

Security notes:

* Set `SENTINEL_ADMIN_TOKEN` in `.env` to protect `/admin/*` (fault
  injection). Unset = open admin endpoints, with a warning logged at boot —
  acceptable for a throwaway local demo only.
* The Grafana admin password defaults to `admin` for local use; override via
  `GRAFANA_PASSWORD` in `.env`. Never reuse these values outside localhost.

## 3. Kubernetes (optional)

Minimal manifests live in `deployment/kubernetes/`
(Deployment + Service + ConfigMap + Secret + PVC + Namespace, with
readiness `/ready`, liveness `/health`, resource limits and a
`maxUnavailable: 0` rolling-update strategy):

```bash
kubectl apply -k deployment/kubernetes/     # or: make k8s-apply
```

Caveats (by design — K8s is not the core of this project):

* The image reference is `sentinel-api:local`; push it to your registry and
  update `deployment.yaml` first.
* Registry/routing/logs are file-based on a PVC (`ReadWriteOnce` ⇒
  `replicas: 1`). For multi-replica clusters use `ReadWriteMany` storage or
  move the registry to a database — the interfaces (`ModelRegistry`,
  `read/write_routing_state`) are the swap points.
* Canary weights are still driven by Sentinel's routing state (identical
  semantics everywhere). To delegate the split to the mesh instead, point
  your ingress at two Services and let `CanaryManager` render weights —
  the audit trail and controller decisions remain unchanged.

## CI/CD deployment (GitHub Actions)

* `.github/workflows/ci.yml` — lint/format/config validation/tests on every push & PR (no secrets needed).
* `.github/workflows/train.yml` — manual dispatch: runs the real pipeline, fails when the gate blocks, writes measured metrics to the job summary, uploads a **model bundle** artifact.
* `.github/workflows/deploy.yml` — manual dispatch: downloads a bundle from a Train run, **re-verifies the gate from the bundle's audit trail**, prints a deployment plan (dry-run), and — only with `execute=true`, a `production` environment approval and a self-hosted runner labelled `sentinel-deploy` — runs `scripts/ci_deploy.sh` (sync bundle → canary deploy → controller with automatic promote/rollback).

No workflow requires secrets for the dry-run path; nothing pretends a
deployment happened when the target infrastructure is absent.
