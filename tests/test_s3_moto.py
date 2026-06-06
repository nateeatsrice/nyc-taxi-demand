"""S3 interaction test using moto to mock S3."""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from nyc_taxi_demand.common.config import get_settings


@pytest.fixture
def s3_bucket():
    with mock_aws():
        settings = get_settings()
        client = boto3.client("s3", region_name=settings.region)
        client.create_bucket(
            Bucket=settings.master_bucket,
            CreateBucketConfiguration={"LocationConstraint": settings.region},
        )
        yield settings.master_bucket


def test_put_json_and_list(s3_bucket):
    from nyc_taxi_demand.common.s3 import list_keys, put_json

    uri = put_json("batch-predictions/nyc-taxi-demand/test/out.json", {"a": 1})
    assert uri.startswith("s3://")

    keys = list_keys("batch-predictions/nyc-taxi-demand/")
    assert "batch-predictions/nyc-taxi-demand/test/out.json" in keys
