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
    year: int = typer.Option(None, help="Limit to a single year partition (lighter local runs)."),
    val_fraction: float = typer.Option(0.2, help="Temporal validation fraction."),
    promote: bool = typer.Option(True, help="Register + promote the best run after training."),
) -> None:
    """Train + compare all algorithms; optionally promote the best."""
    from nyc_taxi_demand.registry.promote import promote_best_run
    from nyc_taxi_demand.training.train import train_and_compare

    results = train_and_compare(year=year, val_fraction=val_fraction)
    rprint("[bold]Results (best first):[/bold]")
    for r in results:
        rprint(f"  {r.algorithm:<24} rmse={r.rmse:.3f}  mae={r.mae:.3f}  r2={r.r2:.3f}")

    if promote:
        best = results[0]
        outcome = promote_best_run(best.run_id, best.rmse)
        rprint(f"[green]Promoted[/green] {outcome}")


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
