variable "region" {
  type    = string
  default = "us-east-2"
}

variable "project" {
  type    = string
  default = "nyc-taxi-demand"
}

variable "vpc_cidr" {
  type    = string
  default = "10.20.0.0/16"
}

variable "cluster_version" {
  type    = string
  default = "1.29"
}

variable "master_bucket" {
  description = "Shared master bucket (externally owned). IAM here grants scoped access only."
  type        = string
  default     = "nateeatsrice-master-s3"
}

variable "glue_gold_database" {
  type    = string
  default = "data_pipeline_gold_dev"
}

variable "spot_instance_types" {
  type    = list(string)
  default = ["m5.large", "m5a.large", "m6i.large"]
}
