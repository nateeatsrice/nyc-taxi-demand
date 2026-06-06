variable "region" {
  description = "AWS region (must match the data-pipeline platform)."
  type        = string
  default     = "us-east-2"
}

variable "project" {
  description = "Project namespace, used for resource naming."
  type        = string
  default     = "nyc-taxi-demand"
}

variable "training_image_tag" {
  description = "Tag the Batch job definition points at in ECR."
  type        = string
  default     = "latest"
}

variable "batch_job_vcpus" {
  description = "vCPUs requested by the training Batch job definition."
  type        = number
  default     = 4
}

variable "batch_job_memory_mb" {
  description = "Memory (MiB) for the training Batch job definition."
  type        = number
  default     = 8192
}
