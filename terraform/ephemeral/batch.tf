# AWS Batch compute environment (spot EC2) + job queue. This is the EPHEMERAL half
# of training compute: it scales to zero when idle (min_vcpus = 0) and is fully
# destroyed by `make tf-destroy`. The job DEFINITION (what to run) is persistent
# and lives in the persistent stack.

# --- IAM for the Batch compute environment (the EC2/spot fleet) ---
resource "aws_iam_role" "batch_service" {
  name = "${var.project}-batch-service-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "batch.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "batch_service" {
  role       = aws_iam_role.batch_service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

# Instance role + profile for the EC2 instances in the compute environment.
resource "aws_iam_role" "batch_instance" {
  name = "${var.project}-batch-instance-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "batch_instance_ecs" {
  role       = aws_iam_role.batch_instance.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "batch_instance" {
  name = "${var.project}-batch-instance-profile"
  role = aws_iam_role.batch_instance.name
}

# Spot fleet role.
resource "aws_iam_role" "spot_fleet" {
  name = "${var.project}-spot-fleet-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "spotfleet.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "spot_fleet" {
  role       = aws_iam_role.spot_fleet.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole"
}

resource "aws_security_group" "batch" {
  name        = "${var.project}-batch-sg"
  description = "Batch compute environment egress"
  vpc_id      = module.vpc.vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_batch_compute_environment" "spot" {
  compute_environment_name = "${var.project}-spot"
  type                     = "MANAGED"
  service_role             = aws_iam_role.batch_service.arn

  compute_resources {
    type                = "SPOT"
    bid_percentage      = 100
    allocation_strategy = "SPOT_CAPACITY_OPTIMIZED"

    min_vcpus     = 0 # scale to zero when idle
    max_vcpus     = 16
    desired_vcpus = 0

    instance_type = ["optimal"]

    subnets            = module.vpc.private_subnets
    security_group_ids = [aws_security_group.batch.id]

    instance_role    = aws_iam_instance_profile.batch_instance.arn
    spot_iam_fleet_role = aws_iam_role.spot_fleet.arn
  }

  depends_on = [aws_iam_role_policy_attachment.batch_service]
}

resource "aws_batch_job_queue" "main" {
  name     = "${var.project}-queue"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.spot.arn
  }
}
