# Persistent stack backend.
#
# State lives in the GLOBAL master bucket and locks on the GLOBAL DynamoDB table.
# Both are created and owned OUTSIDE Terraform (via CLI) and shared across all
# repos -- this stack must NEVER create or manage them. They are referenced here
# only as backend configuration.
terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.40"
    }
  }

  backend "s3" {
    bucket         = "nateeatsrice-master-s3"
    key            = "terraform-state/nyc-taxi-demand/persistent/terraform.tfstate"
    region         = "us-east-2"
    dynamodb_table = "nateeatsrice-tflock"
    encrypt        = true
  }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = {
      Project   = "nyc-taxi-demand"
      Stack     = "persistent"
      ManagedBy = "terraform"
    }
  }
}
