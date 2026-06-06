# VPC + EKS. Uses the well-maintained community modules so this is real, runnable
# infra rather than hundreds of lines of hand-rolled networking. Everything here
# is in the FULL destroy blast radius -- `make tf-destroy` tears it all down.

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  azs = slice(data.aws_availability_zones.available.names, 0, 2)
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.5"

  name = "${var.project}-vpc"
  cidr = var.vpc_cidr

  azs             = local.azs
  private_subnets = [cidrsubnet(var.vpc_cidr, 4, 0), cidrsubnet(var.vpc_cidr, 4, 1)]
  public_subnets  = [cidrsubnet(var.vpc_cidr, 4, 2), cidrsubnet(var.vpc_cidr, 4, 3)]

  enable_nat_gateway = true
  single_nat_gateway = true # one NAT to keep cost down (ephemeral anyway)

  # Tags required by the AWS Load Balancer Controller for subnet discovery.
  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
  }
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
  }
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.8"

  cluster_name    = "${var.project}-eks"
  cluster_version = var.cluster_version

  cluster_endpoint_public_access = true

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  enable_cluster_creator_admin_permissions = true

  eks_managed_node_groups = {
    # Spot node group for serving workloads (FastAPI + Streamlit).
    spot = {
      ami_type       = "AL2_x86_64"
      instance_types = var.spot_instance_types
      capacity_type  = "SPOT"

      min_size     = 1
      max_size     = 3
      desired_size = 2
    }
  }
}
