# Phased build plan (working checklist)

Bring the system up in dependency order so you debug one layer at a time. Each
phase lists what it builds, how to verify it, and what it depends on. Check items
off as you go.

> Ordering note: Phase 5 (Batch training) depends on the persistent stack (Phase 4)
> for the job definition, but it only *runs* on the compute environment created in
> Phase 7. So `make train-batch` is genuinely exercisable only after Phase 7. Build
> 4 → 6 first (everything works locally), then 7, then circle back to verify 5.

## Phase 0 — Repo skeleton & tooling
- [ ] `make setup` resolves + installs on macOS 11 Intel with **zero source builds**
- [ ] `make lint` and `make test` pass
- Depends on: nothing

## Phase 1 — Data loading + shared feature module (local)
- [ ] Glue/S3 reader for both gold tables; weather join by date
- [ ] Shared feature module: leakage drops, calendar derivation, time_of_day buckets
- [ ] Temporal split; snapshot descriptor
- [ ] Verify: `pytest tests/test_features.py tests/test_data_loading.py tests/test_train_serve_consistency.py`
- Depends on: Phase 0

## Phase 2 — Train + compare + MLflow (file store, local)
- [ ] `make train` produces 3 runs; best selected by RMSE
- [ ] `make mlflow-ui` shows runs; SHAP plots present as artifacts
- Depends on: Phase 1

## Phase 3 — Containerize training; MLflow store synced to S3
- [ ] `docker build -f docker/training.Dockerfile` succeeds
- [ ] Runs write MLflow store + artifacts to S3 under `mlruns/` + `ml-artifacts/`
- [ ] On a clean machine `make mlflow-ui` reconstructs history from S3
- [ ] Confirm serial-runs assumption is understood (no concurrent Batch runs)
- Depends on: Phase 2

## Phase 4 — Terraform persistent (ECR, Batch job defs)
- [ ] `make tf-persistent-init && make tf-persistent-apply`
- [ ] ECR repo + Batch job definition created; image pushes to ECR
- [ ] Does NOT touch bucket / Glue / lock table
- Depends on: Phase 3

## Phase 5 — Training on AWS Batch (spot) via Metaflow
- [ ] `make train-batch` runs on spot, scales to zero afterward
- [ ] Runs land in the S3 MLflow store
- Depends on: Phase 4 (+ Phase 7 compute env to actually execute)

## Phase 6 — Registry promotion + serving locally
- [ ] Promote best run (None → Staging → Production)
- [ ] `make serve-api` + `make serve-ui` serve real predictions from Production
- Depends on: Phase 2 (file store); full loop after Phase 5

## Phase 7 — Terraform ephemeral (VPC/EKS/Batch compute env/ALB/IAM)
- [ ] `make tf-init && make tf-apply`
- [ ] EKS up; Batch compute env + queue up; IAM scoped roles created
- [ ] `make tf-destroy` tears down compute only — S3/ECR/Glue/bucket survive
- Depends on: Phase 4

## Phase 8 — Deploy serving to EKS + ALB
- [ ] Annotate `k8s/serviceaccount.yaml` with the IRSA role ARN
- [ ] `make deploy`; ALB DNS serves API + UI end-to-end
- Depends on: Phases 6, 7

## Phase 9 — CI/CD (GitHub Actions)
- [ ] PR runs lint + test (ci.yml)
- [ ] Merge builds + pushes images and rolls EKS (cd.yml)
- [ ] Manual `train` workflow dispatches a Batch run (train.yml)
- Depends on: Phases 4, 8

## Phase 10 — Batch inference path
- [ ] `make batch-infer START=YYYY-MM-DD` writes predictions to S3
- [ ] Leakage-safe: pre-knowable features only
- Depends on: Phase 6

## Phase 11 — CloudWatch monitoring (baseline)
- [ ] EKS/ALB/Batch metrics + app log groups populate under load
- Depends on: Phases 7, 8

## Phase 12 — Evidently drift reporting (deferrable)
- [ ] `make monitor` writes an HTML/JSON drift report to S3 under `monitoring/`
- Depends on: Phases 2, 10
