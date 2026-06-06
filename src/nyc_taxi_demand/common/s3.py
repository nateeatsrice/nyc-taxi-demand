"""Thin S3 helpers shared across the codebase.

Kept deliberately small: most heavy reads go through awswrangler (Glue/Athena).
These cover JSON/text artifact writes and the MLflow store sync.
"""

from __future__ import annotations

import json
from typing import Any

import boto3

from nyc_taxi_demand.common.config import get_settings


def s3_client(region: str | None = None):
    settings = get_settings()
    return boto3.client("s3", region_name=region or settings.region)


def put_json(key: str, obj: Any, bucket: str | None = None) -> str:
    """Write a JSON-serializable object to s3://bucket/key. Returns the URI."""
    settings = get_settings()
    bucket = bucket or settings.master_bucket
    s3_client().put_object(
        Bucket=bucket,
        Key=key.lstrip("/"),
        Body=json.dumps(obj, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    return f"s3://{bucket}/{key.lstrip('/')}"


def put_text(key: str, text: str, content_type: str, bucket: str | None = None) -> str:
    settings = get_settings()
    bucket = bucket or settings.master_bucket
    s3_client().put_object(
        Bucket=bucket,
        Key=key.lstrip("/"),
        Body=text.encode("utf-8"),
        ContentType=content_type,
    )
    return f"s3://{bucket}/{key.lstrip('/')}"


def list_keys(prefix: str, bucket: str | None = None) -> list[str]:
    settings = get_settings()
    bucket = bucket or settings.master_bucket
    paginator = s3_client().get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.lstrip("/")):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return keys
