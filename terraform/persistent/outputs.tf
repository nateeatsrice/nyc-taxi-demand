# Outputs consumed by the ephemeral stack via terraform_remote_state.

output "ecr_repository_url" {
  description = "ECR repository URL for pushing/pulling images."
  value       = aws_ecr_repository.this.repository_url
}

output "ecr_repository_arn" {
  value = aws_ecr_repository.this.arn
}

output "batch_job_definition_arn" {
  description = "ARN of the training Batch job definition."
  value       = aws_batch_job_definition.train.arn
}

output "batch_job_definition_name" {
  value = aws_batch_job_definition.train.name
}

output "account_id" {
  value = local.account_id
}
