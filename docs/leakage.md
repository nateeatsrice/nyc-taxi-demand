# Leakage handling

## The problem

The target is `trip_count`: how many trips occurred in a given zone during a given
hour. The gold `location_hourly_features` table also contains aggregates that are
computed *from those same trips* — they only have a value because the trips already
happened. Using them as model inputs is **target leakage**: at prediction time, for
a *future* hour, none of them would be known.

## Columns dropped (leaky)

| Column                | Why it leaks                                  |
|-----------------------|-----------------------------------------------|
| `total_revenue`       | Sum over the trips being counted              |
| `avg_tip`             | Derived from completed trips                   |
| `avg_fare`            | Derived from completed trips                   |
| `avg_distance`        | Known only after trips occur                   |
| `avg_duration_min`    | Known only after trips occur                   |
| `unique_destinations` | Counted from the trips themselves              |

These are enumerated in `features.transform.LEAKY_COLUMNS` and dropped defensively
in `build_features()`, so neither the training nor the serving path can ever use
them, regardless of what the caller passes in.

## Features kept (pre-knowable)

`pickup_location_id`, `pickup_hour`, `day_of_week`, `is_weekend`, `time_of_day`,
and forecasted weather (`temp_avg_fahrenheit`, `is_rainy`). Every one of these is
knowable *before* the target hour.

## Temporal validation

`data.split.temporal_split` trains on earlier dates and validates on strictly later
dates. A random shuffle split would place rows from the same future day in both
train and validation, leaking future signal and producing optimistic, meaningless
metrics.

## The practical proof

The batch-inference path (`inference.batch`) constructs prediction rows from
pre-knowable inputs **alone** — zone, target date/hour calendar context, and a
forecasted-weather input — with no access to any gold aggregate. If the model can
run there, the feature design is leakage-free by construction.

> Weather caveat: there is no live forecast provider wired in. Batch inference takes
> a weather assumption per date (or a flat default); the live API takes it as user
> input. In production this would come from a forecast API. Keeping it an explicit
> input keeps the leakage boundary obvious.
