# Persistent resources: durable, cheap, NOT in the destroy blast radius.
# Holds the ECR repository (Docker images) and the AWS Batch JOB DEFINITION
# (metadata only -- the compute environment that runs jobs is ephemeral and lives
# in the ephemeral stack).

data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  image_uri  = "${local.account_id}.dkr.ecr.${var.region}.amazonaws.com/${var.project}:${var.training_image_tag}"
}

# --- ECR repository for all images (training, api, ui) ---
resource "aws_ecr_repository" "this" {
  name                 = var.project
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  # Losing the repo means losing every pushed image -> protect it.
  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_ecr_lifecycle_policy" "this" {
  repository = aws_ecr_repository.this.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 20 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 20
      }
      action = { type = "expire" }
    }]
  })
}

# --- Batch job definition for training (metadata; persists across runs) ---
# The job RUNS on the ephemeral compute environment + queue created in the
# ephemeral stack. The execution/job role ARNs are also defined in ephemeral
# (IAM lives there); passed in here via remote_state would create a cycle, so the
# job def references roles by a stable naming convention instead.
# TODO(you): confirm the role names match what the ephemeral stack creates.
resource "aws_batch_job_definition" "train" {
  name = "${var.project}-train"
  type = "container"

  platform_capabilities = ["EC2"]

  container_properties = jsonencode({
    image   = local.image_uri
    command = ["python", "flows/training_flow.py", "run", "--with", "batch"]
    resourceRequirements = [
      { type = "VCPU", value = tostring(var.batch_job_vcpus) },
      { type = "MEMORY", value = tostring(var.batch_job_memory_mb) },
    ]
    # TODO(you): these role ARNs are assumed by the job; the ephemeral stack
    # creates them. Confirm names or wire via SSM parameter if you prefer.
    jobRoleArn       = "arn:aws:iam::${local.account_id}:role/${var.project}-batch-job-role"
    executionRoleArn = "arn:aws:iam::${local.account_id}:role/${var.project}-batch-execution-role"
    environment = [
      { name = "NTD_REGION", value = var.region },
      { name = "NTD_ACCOUNT_ID", value = local.account_id },
    ]
  })

  lifecycle {
    prevent_destroy = true
  }
}
