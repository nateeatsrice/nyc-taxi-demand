"""Central configuration -- the single source of truth for every external
resource name this repo touches.

All S3 prefixes are namespaced under the project name so this repo coexists
with other repos sharing the master bucket. Nothing here creates resources;
these are just references.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT = "nyc-taxi-demand"


class Settings(BaseSettings):
    """Environment-overridable settings. Defaults match the platform layout.

    Override any value via env var (prefix ``NTD_``) or a local ``.env`` file.
    """

    model_config = SettingsConfigDict(env_prefix="NTD_", env_file=".env", extra="ignore")

    # --- AWS ---
    # TODO(you): confirm region matches the data-pipeline platform.
    region: str = "us-east-2"
    # TODO(you): set to your AWS account ID (used to build ECR/role ARNs locally).
    account_id: str = Field(default="", description="12-digit AWS account ID")

    # --- shared master bucket (created & owned OUTSIDE Terraform) ---
    master_bucket: str = "nateeatsrice-master-s3"

    # --- upstream gold tables (owned by the data-pipeline repo) ---
    glue_gold_database: str = "data_pipeline_gold_dev"
    table_location_hourly: str = "location_hourly_features"
    table_trip_weather_daily: str = "trip_weather_daily"
    gold_features_prefix: str = "data-lake/gold/features"

    # --- this repo's persistent prefixes (all under the master bucket) ---
    ml_artifacts_prefix: str = f"ml-artifacts/{PROJECT}"
    mlruns_prefix: str = f"mlruns/{PROJECT}"
    batch_predictions_prefix: str = f"batch-predictions/{PROJECT}"
    monitoring_prefix: str = f"monitoring/{PROJECT}"

    # --- MLflow ---
    # Local synced store dir; `make mlflow-ui` syncs S3 -> here, then serves.
    mlflow_local_store: str = ".mlflow"
    mlflow_experiment: str = f"{PROJECT}-demand"
    registered_model_name: str = "nyc_taxi_demand_hourly"

    # --- helpers ---
    def s3_uri(self, prefix: str, *parts: str) -> str:
        """Build an s3:// URI under the master bucket."""
        path = "/".join([prefix.strip("/"), *[p.strip("/") for p in parts]])
        return f"s3://{self.master_bucket}/{path}"

    @property
    def gold_features_uri(self) -> str:
        return self.s3_uri(self.gold_features_prefix)

    @property
    def mlflow_artifact_root(self) -> str:
        return self.s3_uri(self.ml_artifacts_prefix)

    @property
    def mlruns_s3_uri(self) -> str:
        return self.s3_uri(self.mlruns_prefix)

    @property
    def ecr_repo_uri(self) -> str:
        """ECR repo URI. Requires account_id + region to be set."""
        return f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/{PROJECT}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
