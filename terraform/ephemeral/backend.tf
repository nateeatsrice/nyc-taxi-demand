# Ephemeral stack backend. Same global, externally-owned backend resources as the
# persistent stack -- never created/managed here.
terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.40"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.27"
    }
  }

  backend "s3" {
    bucket         = "nateeatsrice-master-s3"
    key            = "terraform-state/nyc-taxi-demand/ephemeral/terraform.tfstate"
    region         = "us-east-2" # TODO(you): confirm region
    dynamodb_table = "nateeatsrice-tflock"
    encrypt        = true
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "nyc-taxi-demand"
      Stack     = "ephemeral"
      ManagedBy = "terraform"
    }
  }
}

# Consume persistent stack outputs (ECR, Batch job def).
data "terraform_remote_state" "persistent" {
  backend = "s3"
  config = {
    bucket         = "nateeatsrice-master-s3"
    key            = "terraform-state/nyc-taxi-demand/persistent/terraform.tfstate"
    region         = var.region
    dynamodb_table = "nateeatsrice-tflock"
  }
}

data "aws_caller_identity" "current" {}

# Kubernetes provider authenticates against the EKS cluster created below.
provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_name, "--region", var.region]
  }
}
