# Model search & hyperparameter tuning

## Why this exists

The first cut of this project fit three models once each with hardcoded
hyperparameters. That is not model selection -- it is three arbitrary points in a
huge configuration space. This module replaces that with a proper search.

## What it does

For each requested model, an **Optuna** study runs N trials (default 50). Each
trial samples hyperparameters from that model's search space, builds the pipeline,
fits on the temporal-train split, and scores validation RMSE. Optuna's TPE sampler
uses earlier trials to focus later ones on promising regions.

**One MLflow experiment per model** (`nyc-taxi-demand-<model>`), so the UI shows ten
experiments, each with its trials as runs. Within an experiment you get a study
parent run, one nested run per trial (with that trial's unique hyperparameters and
metrics), and a `-best` run carrying the servable model artifact. The single best
run across all models is registered and promoted.

## Models

| Model | Family | Objective | In default set |
|---|---|---|---|
| xgboost | tree | Poisson (`count:poisson`) | yes |
| lightgbm | tree | Poisson | yes |
| hist_gradient_boosting | tree | Poisson | yes |
| poisson_regression | linear | Poisson GLM | yes |
| ridge | linear | squared-error (L2) | yes |
| lasso | linear | squared-error (L1) | yes |
| elastic_net | linear | squared-error (L1+L2) | yes |
| linear_regression | linear | squared-error (OLS) | yes |
| random_forest | other | squared-error | yes |
| svr | other | epsilon-insensitive | no (slow) |

**Poisson where it fits.** Trip counts are non-negative event counts, so the tree
models and the Poisson GLM use a Poisson objective and cannot predict negatives.
Plain OLS and SVR minimize squared error and *can* go negative -- they are kept as
honest baselines, and seeing them underperform (or predict negatives) is itself a
useful comparison. The serving layer clamps at 0 regardless.

**Polynomial features** are not a separate model: the linear specs expose a
`poly_degree` hyperparameter (1-3) so Optuna decides whether polynomial expansion
helps, which is the cleaner formulation than a standalone "polynomial regression".

## Usage

```bash
make train                                  # all default models, 50 trials each
make train MODELS=xgboost,ridge TRIALS=100  # subset, more trials
make train MODELS=svr TRIALS=20             # opt into the slow one
make list-models                            # see all options
```

`svr` is excluded from the default set because it is slow on ~145k rows; request it
explicitly if you want it.

## Reproducibility

Every trial logs the data snapshot descriptor (paths, partitions, row count, hash)
alongside its hyperparameters, so any run's exact training data and configuration
are recoverable.
