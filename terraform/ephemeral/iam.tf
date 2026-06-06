# IAM roles/policies granting PREFIX-SCOPED least-privilege access to the shared
# master bucket and the Glue catalog. IAM lives in the ephemeral stack because it
# is assumed by ephemeral compute (Batch jobs, EKS pods). None of this grants any
# ability to create/delete the bucket or catalog -- only scoped data access.

locals {
  bucket_arn = "arn:aws:s3:::${var.master_bucket}"

  # Read-only on gold; read/write on this project's own prefixes.
  read_prefixes  = ["data-lake/gold/*"]
  write_prefixes = [
    "ml-artifacts/${var.project}/*",
    "mlruns/${var.project}/*",
    "batch-predictions/${var.project}/*",
    "monitoring/${var.project}/*",
  ]
}

data "aws_iam_policy_document" "data_access" {
  # List the bucket, constrained to relevant prefixes.
  statement {
    sid       = "ListBucketScoped"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [local.bucket_arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = concat(local.read_prefixes, local.write_prefixes)
    }
  }

  statement {
    sid       = "ReadGold"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = [for p in local.read_prefixes : "${local.bucket_arn}/${p}"]
  }

  statement {
    sid     = "ReadWriteProjectPrefixes"
    effect  = "Allow"
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = [for p in local.write_prefixes : "${local.bucket_arn}/${p}"]
  }

  # Glue catalog: read-only metadata for the gold database/tables (Athena reads).
  statement {
    sid    = "GlueRead"
    effect = "Allow"
    actions = [
      "glue:GetDatabase",
      "glue:GetTable",
      "glue:GetTables",
      "glue:GetPartition",
      "glue:GetPartitions",
    ]
    resources = ["*"] # TODO(you): tighten to the specific catalog/db/table ARNs.
  }

  # Athena query execution for awswrangler reads.
  statement {
    sid    = "AthenaQuery"
    effect = "Allow"
    actions = [
      "athena:StartQueryExecution",
      "athena:GetQueryExecution",
      "athena:GetQueryResults",
      "athena:StopQueryExecution",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "data_access" {
  name   = "${var.project}-data-access"
  policy = data.aws_iam_policy_document.data_access.json
}

# --- Batch job role (assumed by training containers) ---
# Name must match what the persistent stack's job definition references.
resource "aws_iam_role" "batch_job" {
  name = "${var.project}-batch-job-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "batch_job_data" {
  role       = aws_iam_role.batch_job.name
  policy_arn = aws_iam_policy.data_access.arn
}

# --- Batch execution role (pulls image, writes logs) ---
resource "aws_iam_role" "batch_execution" {
  name = "${var.project}-batch-execution-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "batch_execution" {
  role       = aws_iam_role.batch_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# --- IRSA role for serving pods (FastAPI loads Production model from S3) ---
data "aws_iam_policy_document" "irsa_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [module.eks.oidc_provider_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${module.eks.oidc_provider}:sub"
      values   = ["system:serviceaccount:default:${var.project}-sa"]
    }
  }
}

resource "aws_iam_role" "serving_irsa" {
  name               = "${var.project}-serving-irsa"
  assume_role_policy = data.aws_iam_policy_document.irsa_assume.json
}

resource "aws_iam_role_policy_attachment" "serving_data" {
  role       = aws_iam_role.serving_irsa.name
  policy_arn = aws_iam_policy.data_access.arn
}
