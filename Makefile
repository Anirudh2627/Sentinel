# ---------------------------------------------------------------------------
# Sentinel — developer commands. Every target maps to real code (no stubs).
# Works WITHOUT installing the package: bin/sentinel puts src/ on PYTHONPATH.
# ---------------------------------------------------------------------------

PYTHON   ?= python3
SENTINEL ?= ./bin/sentinel
ENV      ?= development
export SENTINEL_ENV := $(ENV)

# Demo tuning lives inside scripts/run_demo.sh so `bash scripts/run_demo.sh`
# and `make demo` behave identically.

.PHONY: help install install-dev lint format test test-cov
.PHONY: data train evaluate register gate pipeline
.PHONY: serve monitor controller status versions audit clean
.PHONY: deploy canary promote rollback simulate inject-drift inject-latency inject-errors faults-off
.PHONY: mlflow-ui up down demo k8s-apply k8s-delete

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------- #
# Setup & quality
# --------------------------------------------------------------------------- #
install: ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-dev: ## Install with dev/test tooling + editable package
	$(PYTHON) -m pip install -e ".[dev]"

lint: ## Ruff lint + format check
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m ruff format --check src tests scripts

format: ## Auto-format code
	$(PYTHON) -m ruff format src tests scripts
	$(PYTHON) -m ruff check --fix src tests scripts

test: ## Run the full test suite
	PYTHONPATH=src $(PYTHON) -m pytest tests -q

test-cov: ## Tests with coverage report
	PYTHONPATH=src $(PYTHON) -m pytest tests --cov=src/sentinel --cov-report=term-missing

# --------------------------------------------------------------------------- #
# ML lifecycle
# --------------------------------------------------------------------------- #
data: ## Generate + validate + persist the dataset
	$(SENTINEL) generate

train: ## Train a model (run stage only, prints run dir)
	$(SENTINEL) train

register: ## Register the latest training run in the model registry
	$(SENTINEL) register

evaluate: ## Evaluate a registered version on its test split (V=v2 make evaluate)
	$(SENTINEL) evaluate $(or $(V),v1)

gate: ## Run the promotion quality gate (V=v1 make gate)
	$(SENTINEL) gate $(or $(V),v1)

pipeline: ## Full pipeline: train → evaluate → register → gate
	$(SENTINEL) pipeline

# --------------------------------------------------------------------------- #
# Serving & monitoring
# --------------------------------------------------------------------------- #
serve: ## Run the inference API on :8000
	$(SENTINEL) serve

monitor: ## Collect one monitoring snapshot (WATCH=1 for loop)
	$(SENTINEL) monitor $(if $(WATCH),--watch,)

controller: ## Run the deployment controller for DURATION seconds (default 300)
	$(SENTINEL) controller --duration $(or $(DURATION),300) --interval $(or $(INTERVAL),10)

status: ## Registry + routing + controller + latest snapshot
	$(SENTINEL) status

versions: ## List model versions
	$(SENTINEL) versions

audit: ## Show the deployment audit trail
	$(SENTINEL) audit --last 20

clean: ## Remove runtime state (logs, routing, snapshots) — models kept
	$(SENTINEL) clean --yes

mlflow-ui: ## Browse experiments (requires mlflow; uses local sqlite backend)
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

# --------------------------------------------------------------------------- #
# Deployment operations
# --------------------------------------------------------------------------- #
deploy: ## Deploy V directly to production (V=v1 make deploy)
	$(SENTINEL) deploy $(or $(V),v1) --direct

canary: ## Start canary for V at P percent (V=v2 P=10 make canary)
	$(SENTINEL) deploy $(or $(V),v2) --percent $(or $(P),10)

promote: ## Manually promote the active canary
	$(SENTINEL) promote

rollback: ## Manually roll back the active canary
	$(SENTINEL) rollback --reason "manual rollback via make"

# --------------------------------------------------------------------------- #
# Traffic simulation & failure injection
# --------------------------------------------------------------------------- #
simulate: ## Send normal traffic (S=scenario R=rps D=seconds)
	$(SENTINEL) simulate --scenario $(or $(S),normal) --rps $(or $(R),20) --duration $(or $(D),30)

inject-drift: ## Inject covariate drift traffic (M=covariate|concept)
	$(PYTHON) scripts/inject_drift.py --mode $(or $(M),covariate) --duration $(or $(D),60) --rps $(or $(R),20)

inject-latency: ## Inject L ms artificial latency (L=800 make inject-latency)
	$(PYTHON) scripts/inject_faults.py --latency-ms $(or $(L),800)

inject-errors: ## Inject failing requests at rate E (E=0.2 make inject-errors)
	$(PYTHON) scripts/inject_faults.py --error-rate $(or $(E),0.2)

faults-off: ## Disable all fault injection
	$(PYTHON) scripts/inject_faults.py --disable

# --------------------------------------------------------------------------- #
# Docker & Kubernetes
# --------------------------------------------------------------------------- #
up: ## Start the local stack (api + mlflow + prometheus + grafana)
	docker compose up --build -d

down: ## Stop the local stack
	docker compose down

demo: ## Run the complete lifecycle demo (server started/stopped automatically)
	bash scripts/run_demo.sh

k8s-apply: ## Apply the (optional) Kubernetes manifests
	kubectl apply -f deployment/kubernetes/

k8s-delete: ## Delete the Kubernetes manifests
	kubectl delete -f deployment/kubernetes/
