# Terraform

Two independent root modules with a producer/consumer relationship via remote
state. Both back their state in the **externally-owned** global resources
(`nateeatsrice-master-s3` bucket + `nateeatsrice-tflock` DynamoDB table); neither
creates or manages those.

- **`persistent/`** — ECR repo + Batch job definitions. Durable, `prevent_destroy`,
  applied **manually** (`make tf-persistent-apply`). No destroy target.
- **`ephemeral/`** — VPC/NAT/ALB, EKS + spot nodes, Batch compute env + queue, K8s
  workloads, and all scoped IAM. Full destroy blast radius (`make tf-destroy`).
  Reads persistent outputs via `terraform_remote_state`.

A `terraform destroy` of `ephemeral` never deletes data, artifacts, ECR images,
the Glue catalog, or the bucket.
