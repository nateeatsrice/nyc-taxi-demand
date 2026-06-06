# nyc-taxi-demand Makefile
# Run `make help` (or just `make`) for the target list.

.DEFAULT_GOAL := help
SHELL := /bin/bash

PROJECT := nyc-taxi-demand
PY := python
TF_PERSISTENT := terraform/persistent
TF_EPHEMERAL := terraform/ephemeral

# --- guard helper: require a variable, e.g. $(call require,RUN_ID) ---
define require
	@if [ -z "$($(1))" ]; then \
		echo "ERROR: missing required arg $(1)."; \
		echo "Usage: make $(MAKECMDGOALS) $(1)=<value>"; \
		exit 1; \
	fi
endef

##@ Setup
.PHONY: setup
setup: ## Create venv + install LAPTOP-SAFE core deps + dev tools + pre-commit
	uv venv --python 3.12
	uv pip install -e . --group dev
	. .venv/bin/activate && pre-commit install
	@echo "Done. Activate with: . .venv/bin/activate"

##@ Quality
.PHONY: lint format test test-cov
lint: ## Ruff lint
	ruff check src tests flows

format: ## Ruff format (writes)
	ruff format src tests flows

test: ## Run tests
	pytest -q

test-cov: ## Run tests with coverage report
	pytest -q --cov=nyc_taxi_demand --cov-report=term-missing

##@ Training & MLflow
.PHONY: train train-batch mlflow-ui promote
train: ## Train + compare locally (MLflow file store), promote best
	$(PY) -m nyc_taxi_demand.cli train

train-batch: ## Submit training to AWS Batch (spot) via Metaflow
	$(PY) flows/training_flow.py run --with batch

mlflow-ui: ## Sync MLflow store from S3 and launch the UI locally
	@mkdir -p .mlflow
	aws s3 sync s3://nateeatsrice-master-s3/mlruns/$(PROJECT)/ .mlflow/ || true
	mlflow ui --backend-store-uri sqlite:///.mlflow/mlflow.db --port 5000

promote: ## Promote a specific run: make promote RUN_ID=<id> RMSE=<val>
	$(call require,RUN_ID)
	$(call require,RMSE)
	$(PY) -m nyc_taxi_demand.cli promote $(RUN_ID) $(RMSE)

##@ Serving (local)
.PHONY: serve-api serve-ui
serve-api: ## Run FastAPI locally against the Production model
	uvicorn nyc_taxi_demand.serving.api.main:app --reload --port 8000

serve-ui: ## Run the Streamlit UI locally (expects API on :8000)
	streamlit run src/nyc_taxi_demand/serving/ui/app.py

##@ Inference & Monitoring
.PHONY: batch-infer monitor
batch-infer: ## Batch-predict all zones: make batch-infer START=YYYY-MM-DD [DAYS=1]
	$(call require,START)
	$(PY) -m nyc_taxi_demand.cli batch-infer --start-date $(START) --days $(or $(DAYS),1)

monitor: ## Run the Evidently drift report (deferrable phase; needs monitoring extra)
	@echo "Run the drift job; see src/nyc_taxi_demand/monitoring/drift.py"
	$(PY) -c "print('Wire reference/current frames then call run_drift_report().')"

##@ Docker / ECR
.PHONY: build push
build: ## Build all three images locally
	docker build -f docker/api.Dockerfile -t $(PROJECT):api .
	docker build -f docker/ui.Dockerfile -t $(PROJECT):ui .
	docker build -f docker/training.Dockerfile -t $(PROJECT):train .

push: ## Push images to ECR: make push ECR_URL=<repo_url> TAG=<tag>
	$(call require,ECR_URL)
	$(call require,TAG)
	docker tag $(PROJECT):api   $(ECR_URL):api-$(TAG)   && docker push $(ECR_URL):api-$(TAG)
	docker tag $(PROJECT):ui    $(ECR_URL):ui-$(TAG)    && docker push $(ECR_URL):ui-$(TAG)
	docker tag $(PROJECT):train $(ECR_URL):train-$(TAG) && docker push $(ECR_URL):train-$(TAG)

##@ EKS
.PHONY: deploy
deploy: ## Apply k8s manifests to the current kube context
	kubectl apply -f k8s/

##@ Terraform -- EPHEMERAL (default scope for plan/apply/destroy)
.PHONY: tf-init tf-plan tf-apply tf-destroy
tf-init: ## terraform init (ephemeral)
	cd $(TF_EPHEMERAL) && terraform init

tf-plan: ## terraform plan (ephemeral)
	cd $(TF_EPHEMERAL) && terraform plan

tf-apply: ## terraform apply (ephemeral)
	cd $(TF_EPHEMERAL) && terraform apply

tf-destroy: ## terraform destroy (EPHEMERAL ONLY -- never touches data/ECR/Glue/bucket)
	cd $(TF_EPHEMERAL) && terraform destroy

##@ Terraform -- PERSISTENT (apply MANUALLY; no destroy target on purpose)
.PHONY: tf-persistent-init tf-persistent-apply
tf-persistent-init: ## terraform init (persistent)
	cd $(TF_PERSISTENT) && terraform init

tf-persistent-apply: ## terraform apply (persistent) -- ECR + Batch job defs; run deliberately
	cd $(TF_PERSISTENT) && terraform apply
# NOTE: there is intentionally NO tf-persistent-destroy target. Persistent
# resources carry prevent_destroy and hold images/metadata you do not want to lose.
# Tear them down by hand only if you truly mean to.

##@ Help
.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage: make \033[36m<target>\033[0m\n"} \
		/^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)
