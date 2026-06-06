"""Command-line entrypoint (``ntd``).

Thin wrapper over the library so the Makefile targets and CI call stable
commands. Each command is intentionally small -- the logic lives in the modules.
"""

from __future__ import annotations

import datetime as dt

import typer
from rich import print as rprint

app = typer.Typer(help="nyc-taxi-demand MLOps CLI", no_args_is_help=True)


@app.command()
def train(
    models: str = typer.Option(
        None,
        help="Comma-separated models to tune (default: all except svr). "
        "E.g. --models xgboost,ridge,random_forest. Use `list-models` to see options.",
    ),
    trials: int = typer.Option(50, help="Optuna trials per model."),
    year: int = typer.Option(None, help="Limit to a single year partition (lighter local runs)."),
    val_fraction: float = typer.Option(0.2, help="Temporal validation fraction."),
    promote: bool = typer.Option(True, help="Register + promote the global best run."),
) -> None:
    """Tune each model with Optuna (one experiment per model); promote the best."""
    from nyc_taxi_demand.registry.promote import promote_best_run
    from nyc_taxi_demand.training.train import train_and_compare

    model_list = [m.strip() for m in models.split(",")] if models else None
    summary = train_and_compare(
        models=model_list, n_trials=trials, year=year, val_fraction=val_fraction
    )

    rprint(
        f"[bold]Results across {len(summary.results)} models "
        f"({trials} trials each), best first:[/bold]"
    )
    for r in summary.results:
        marker = "[green]*[/green]" if r is summary.best else " "
        rprint(f"  {marker} {r.model:<24} rmse={r.rmse:.3f}  mae={r.mae:.3f}  r2={r.r2:.3f}")

    if promote:
        best = summary.best
        outcome = promote_best_run(best.run_id, best.rmse)
        rprint(f"[green]Promoted[/green] {best.model}: {outcome}")


@app.command("list-models")
def list_models() -> None:
    """List available models and their families."""
    from nyc_taxi_demand.training.algorithms import DEFAULT_MODELS, MODELS

    rprint("[bold]Available models:[/bold]")
    for name, spec in MODELS.items():
        default = "" if name in DEFAULT_MODELS else " [dim](not in default set)[/dim]"
        poisson = "poisson" if spec.poisson else "squared-error"
        rprint(f"  {name:<24} family={spec.family:<7} {poisson}{default}")


@app.command("batch-infer")
def batch_infer(
    start_date: str = typer.Option(..., help="YYYY-MM-DD start of the future window."),
    days: int = typer.Option(1, help="Number of days to predict."),
    temp_f: float = typer.Option(60.0, help="Default forecasted temp (F)."),
    rainy: bool = typer.Option(False, help="Default forecasted rain."),
) -> None:
    """Predict demand for all zones over a future window; write to S3."""
    from nyc_taxi_demand.inference.batch import run_batch_inference

    uri = run_batch_inference(
        start_date=dt.date.fromisoformat(start_date),
        days=days,
        default_temp_f=temp_f,
        default_rainy=rainy,
    )
    rprint(f"[green]Wrote predictions:[/green] {uri}")


@app.command()
def promote(run_id: str = typer.Argument(...), rmse: float = typer.Argument(...)) -> None:
    """Manually register + promote a specific run by id."""
    from nyc_taxi_demand.registry.promote import promote_best_run

    rprint(promote_best_run(run_id, rmse))


if __name__ == "__main__":
    app()
