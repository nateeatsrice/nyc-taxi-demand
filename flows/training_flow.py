"""Metaflow training flow.

Stages: load/validate -> feature prep -> train-compare -> register+promote ->
(optional) batch inference -> monitoring report.

The heavy ``train`` step is decorated with ``@batch`` so it runs on AWS Batch spot
EC2; the lightweight orchestration steps run locally (or on the scheduler). The
compute environment is provisioned by ``terraform/ephemeral``; the job definition
metadata by ``terraform/persistent``.

Run locally end-to-end (no Batch):   python flows/training_flow.py run
Run training on Batch:               python flows/training_flow.py run --with batch

TODO(you): set the @batch image/queue to your ECR image + job queue, or pass them
via Metaflow config / environment. Values left generic on purpose.
"""

from __future__ import annotations

from metaflow import FlowSpec, Parameter, batch, step


class TrainingFlow(FlowSpec):
    year = Parameter("year", help="Limit to a single year partition", default=None)
    val_fraction = Parameter("val_fraction", default=0.2)
    do_promote = Parameter("promote", default=True)
    models = Parameter("models", help="Comma-separated models (default: all but svr)", default=None)
    trials = Parameter("trials", help="Optuna trials per model", default=50)

    @step
    def start(self):
        """Validate config / data reachability before spending compute."""
        from nyc_taxi_demand.common.config import get_settings

        self.settings_snapshot = get_settings().model_dump()
        self.next(self.train)

    # TODO(you): replace image/queue with your ECR URI + Batch job queue name.
    @batch(
        cpu=4,
        memory=8000,
        image="# TODO(you): <account>.dkr.ecr.<region>.amazonaws.com/nyc-taxi-demand:latest",
    )
    @step
    def train(self):
        """Tune all models with Optuna on AWS Batch (spot)."""
        from nyc_taxi_demand.training.train import train_and_compare

        model_list = [m.strip() for m in self.models.split(",")] if self.models else None
        summary = train_and_compare(
            models=model_list,
            n_trials=self.trials,
            year=self.year,
            val_fraction=self.val_fraction,
        )
        self.results = [r.__dict__ for r in summary.results]
        self.best = summary.best.__dict__
        self.next(self.register)

    @step
    def register(self):
        """Register + promote the best run via the legacy stage API."""
        if self.do_promote:
            from nyc_taxi_demand.registry.promote import promote_best_run

            self.promotion = promote_best_run(self.best["run_id"], self.best["rmse"])
        else:
            self.promotion = {"stage": "skipped"}
        self.next(self.end)

    @step
    def end(self):
        print(f"Best: {self.best['model']} rmse={self.best['rmse']:.3f}")
        print(f"Promotion: {self.promotion}")


if __name__ == "__main__":
    TrainingFlow()
