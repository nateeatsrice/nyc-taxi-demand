# Kubernetes manifests

Applied to the EKS cluster created by `terraform/ephemeral`.

- `serviceaccount.yaml` — IRSA-annotated SA (`nyc-taxi-demand-sa`) granting the
  serving pods scoped S3 access. **TODO(you):** set the role ARN from the
  `serving_irsa_role_arn` Terraform output.
- `api-deployment.yaml` — FastAPI deployment + ClusterIP service.
- `ui-deployment.yaml` — Streamlit deployment + ClusterIP service (talks to the API
  in-cluster).
- `ingress.yaml` — ALB ingress (`/` → UI, `/api` → API). Requires the AWS Load
  Balancer Controller in the cluster.

**TODO(you):** replace `<ECR_REPO_URL>` / `<ACCOUNT_ID>` placeholders (CD does this
automatically via `kubectl set image`).
