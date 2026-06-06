# Train/serve consistency

## The risk: train/serve skew

A model learns the distribution of the features it was trained on. If the serving
code computes a feature even slightly differently than the training code did, the
model receives inputs from a distribution it never saw, and predictions degrade —
silently, with no error. This is **train/serve skew**, and it is one of the most
common and hardest-to-debug failure modes in production ML.

The classic example here is `time_of_day`. If training bucketed hours as
`morning = 6–11` but the API bucketed them as `morning = 0–11`, then every early
prediction would land in the wrong category and the model's learned relationship
would not apply.

## The mitigation: one shared module

There is exactly one place where raw inputs become model features:
`src/nyc_taxi_demand/features/transform.py`. Both paths import it:

- **Training** calls `build_features(frame, derive_calendar=False)` — the gold rows
  already carry `day_of_week` / `is_weekend` / `time_of_day`.
- **Serving / batch** calls `build_features(rows, derive_calendar=True)` — only a
  date + hour + zone + weather are supplied, and the calendar features are derived
  from them via the same `time_of_day_bucket` and `compute_calendar_features`
  helpers training relies on.

Because both go through the same code, the bucket boundaries (and everything else)
cannot drift between them.

## The guarantee: a test that fails on skew

`tests/test_train_serve_consistency.py` takes gold rows, runs them through the
training path, then strips the calendar columns and runs the same rows through the
serving path, and asserts the two feature frames are identical. If anyone ever
changes one path without the other, this test goes red.

> During the initial build this test caught a real mismatch: a test fixture used a
> naive `morning/afternoon` rule instead of the real bucket boundaries. That is
> exactly the class of bug it exists to catch.
