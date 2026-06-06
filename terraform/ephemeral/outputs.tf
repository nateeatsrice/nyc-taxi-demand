output "cluster_name" {
  value = module.eks.cluster_name
}

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "configure_kubectl" {
  description = "Run this to point kubectl at the cluster."
  value       = "aws eks update-kubeconfig --region ${var.region} --name ${module.eks.cluster_name}"
}

output "batch_job_queue" {
  value = aws_batch_job_queue.main.name
}

output "serving_irsa_role_arn" {
  description = "Annotate the K8s service account with this for S3 access."
  value       = aws_iam_role.serving_irsa.arn
}

output "ecr_repository_url" {
  description = "Pulled from the persistent stack for convenience."
  value       = data.terraform_remote_state.persistent.outputs.ecr_repository_url
}
