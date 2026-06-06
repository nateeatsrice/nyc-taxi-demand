# nyc-taxi-demand

An **MLOps platform** that consumes the gold feature tables produced by a separate
**data platform** (`data-pipeline`) and turns them into a trained, served, and
monitored **hourly taxi-demand forecasting** model.

This repo is the *consumer* in a two-project platform. It does **not** ingest raw
data or own the data lake — it reads the curated gold tables, trains and compares
several models, promotes the best to a registry, serves it behind an API + UI, and
monitors it. Compute is ephemeral and scales to zero; data, artifacts, and history
are persistent in S3 and survive `terraform destroy`.

---

## Architecture

```
┌──────────────────────── data-pipeline (separate repo, already built) ─────────────────────┐
│  raw → bronze → silver → GOLD feature tables  (Glue: data_pipeline_gold_dev)               │
│      s3://nateeatsrice-master-s3/data-lake/gold/features/                                   │
│        • location_hourly_features   (~145k rows, 20 monthly partitions)                     │
│        • trip_weather_daily         (~600 rows, daily weather)                              │
└───────────────────────────────────────────────────────────────────────────────────────────┘
                                   │  (read-only, prefix-scoped IAM)
                                   ▼
┌──────────────────────────────── nyc-taxi-demand (THIS repo) ───────────────────────────────┐
│                                                                                             │
│  data/        load gold + join weather by date → temporal split → snapshot descriptor      │
│  features/    SHARED transform (leakage-free, pre-knowable only)  ◄── imported by BOTH ──┐  │
│  training/    XGBoost · LightGBM · HistGradientBoosting → MLflow (file/SQLite→S3) + SHAP │  │
│  registry/    promote best: None → Staging → Production  (MLflow legacy stage API)       │  │
│  serving/     FastAPI (loads Production model) + Streamlit UI  ──────────────────────────┘  │
│  inference/   batch predict all zones, pre-knowable features only → S3                       │
│  monitoring/  CloudWatch (infra/app) + Evidently drift reports → S3                          │
│                                                                                             │
│  Orchestration: Metaflow + @batch (AWS Batch spot EC2)                                       │
│  Serving infra: EKS + ALB    CI/CD: GitHub Actions → ECR → EKS                               │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

         PERSISTENT (survives destroy)              EPHEMERAL (full destroy blast radius)
         ───────────────────────────────           ─────────────────────────────────────
         • master S3 bucket  (owned outside TF)     • VPC / NAT / subnets / ALB
         • Glue catalog      (owned by data repo)   • EKS cluster + node groups (spot)
         • DynamoDB tflock   (owned outside TF)     • Batch COMPUTE ENVIRONMENT (spot) + queue
         • ECR repo                                 • K8s deployments/services/ingress
         • Batch JOB DEFINITIONS                    • ALL IAM roles/policies (scoped access)
         • MLflow runs, artifacts, predictions,
           Evidently reports  (all in S3)
```

## Persistent vs ephemeral (and why)

Two independent Terraform root modules with a producer/consumer relationship via
remote state:

- **`terraform/persistent/`** — durable, rarely changed, *not* in the destroy blast
  radius: the ECR repo and the Batch **job definitions** (cheap metadata). Anything
  whose loss would be costly carries `prevent_destroy`. Applied manually.
- **`terraform/ephemeral/`** — everything that costs money to keep running or is safe
  to recreate: VPC/NAT/ALB, EKS + spot node group, the Batch **compute environment**
  (spot EC2) + queue, the K8s workloads, and **all IAM** (it's assumed by ephemeral
  compute). Reads persistent outputs via `terraform_remote_state`. This is what
  `make tf-destroy` tears down.

**Backend resources are global and owned OUTSIDE Terraform** (created via CLI, shared
across all repos): the state bucket `nateeatsrice-master-s3` and the lock table
`nateeatsrice-tflock`. This repo's Terraform never creates, manages, or destroys
them, nor the Glue catalog.

**Persistence guarantee:** MLflow runs/metrics, model artifacts, batch predictions,
Evidently reports, and monitoring logs all live in S3 under the master bucket and
**survive `terraform destroy`**. Only compute is destroyed. A destroy never deletes
data, artifacts, ECR images, the Glue catalog, or the bucket.

## Leakage handling

The target is `trip_count` — how many trips occurred in a zone-hour. Several gold
columns exist *only because those trips already happened* and would not be knowable
when predicting a **future** hour. Feeding them to the model is target leakage.

**Dropped (leaky) columns:** `total_revenue`, `avg_tip`, `avg_fare`, `avg_distance`,
`avg_duration_min`, `unique_destinations`.

**Kept (pre-knowable) features:** `pickup_location_id`, `pickup_hour`, `day_of_week`,
`is_weekend`, `time_of_day`, and forecasted weather (`temp_avg_fahrenheit`,
`is_rainy`).

Validation uses a **temporal split** (train on earlier dates, validate on later
dates) — never a random shuffle, which would leak future days into training. The
**batch-inference path constructs rows from pre-knowable inputs alone**, which is the
practical proof the feature design is leakage-free. See `docs/leakage.md`.

## Train/serve consistency

A single shared module (`features/transform.py`) is imported by **both** training and
serving, so the feature logic can never diverge between them. The risk it prevents —
**train/serve skew** — is when the API derives a feature (e.g. the `time_of_day`
bucket boundaries) differently than training did, so the model sees a different
distribution at inference than it learned on. `tests/test_train_serve_consistency.py`
asserts the serving path (derive calendar from date+hour) reproduces the training
path exactly. See `docs/train-serve-consistency.md`.

## Quickstart (local, macOS 11 Intel)

```bash
make setup        # uv venv + core (laptop-safe) deps + dev tools + pre-commit
make test         # run the test suite
make train        # train + compare locally (MLflow file store), promote best
make mlflow-ui    # sync MLflow store from S3 and open the UI
make serve-api    # run FastAPI locally against the Production model
make serve-ui     # run the Streamlit UI
```

See `docs/build-plan.md` for the full phased build/debug checklist.

## Dependency policy

Local dev runs on an **Intel Mac on macOS 11 (Big Sur)**; all *local* deps are pinned
to versions with prebuilt Big Sur Intel wheels (no source builds). Cloud/Linux-only
deps (Streamlit serving, Metaflow, Evidently) live in optional-dependency groups
installed only in Docker/CI. MLflow is pinned `>=2.9,<3` because this repo
intentionally uses the legacy stage-based registry API. Details and pin ceilings:
`docs/dependency-pins.md`.

## Future work (explicitly out of scope)

- **Auth** on the API/UI (currently none — do not expose publicly as-is).
- **Canary / A-B deploys** for model rollout.
- **Feature store** (currently the shared transform module fills this role).
- **Streaming / real-time** features and inference.
- **Live weather-forecast provider** for batch inference (currently an explicit input).
