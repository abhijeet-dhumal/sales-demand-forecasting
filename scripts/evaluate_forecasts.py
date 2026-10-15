"""Rolling-origin comparison for the one-week-ahead sales forecast demo."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


MODEL_NAME = "hist_gradient_boosting"
CATEGORICAL_COLUMNS = ("store_index", "department_index")


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    error = predicted - actual
    denominator = float(np.abs(actual).sum())
    if denominator == 0:
        raise ValueError("WAPE and weighted bias are undefined when every actual is zero")
    return {
        "rows": int(actual.size),
        "series": 0,
        "wape": float(np.abs(error).sum() / denominator),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "weighted_bias": float(error.sum() / denominator),
    }


def _series_metrics(
    rows: pd.DataFrame, actual_column: str, prediction_column: str
) -> list[dict[str, Any]]:
    result = []
    for (store_id, dept_id), group in rows.groupby(
        ["store_id", "dept_id"], dropna=False, sort=True
    ):
        actual = group[actual_column].to_numpy(dtype=float)
        predicted = group[prediction_column].to_numpy(dtype=float)
        error = predicted - actual
        denominator = float(np.abs(actual).sum())
        values = {
            "rows": int(actual.size),
            "wape": float(np.abs(error).sum() / denominator) if denominator else None,
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "weighted_bias": float(error.sum() / denominator) if denominator else None,
        }
        values.update({"series": 1, "store_id": str(store_id), "dept_id": str(dept_id)})
        result.append(values)
    return result


def _summary(
    actual: Sequence[float],
    predictions: Mapping[str, Sequence[float]],
    fold_metrics: Mapping[str, list[dict[str, Any]]] | None = None,
    series_count: int = 0,
) -> dict[str, Any]:
    y = np.asarray(actual, dtype=float)
    summary: dict[str, Any] = {}
    for name, values in predictions.items():
        metrics = _metrics(y, np.asarray(values, dtype=float))
        metrics["series"] = int(series_count)
        if fold_metrics is not None:
            per_fold = fold_metrics[name]
            metrics["mean_fold_wape"] = float(np.mean([fold["wape"] for fold in per_fold]))
        summary[name] = metrics
    return summary


def _fit_predict(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    feature_columns: Sequence[str],
    max_iter: int,
    random_state: int,
) -> tuple[np.ndarray, float, float]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    train_x = train.loc[:, feature_columns].copy()
    predict_x = predict.loc[:, feature_columns].copy()
    for column in CATEGORICAL_COLUMNS:
        categories = pd.Index(train_x[column].dropna().unique())
        unknown = predict_x[column].notna() & ~predict_x[column].isin(categories)
        if unknown.any():
            raise ValueError(f"Unseen {column} values in a later forecast week")
        train_x[column] = pd.Categorical(train_x[column], categories=categories)
        predict_x[column] = pd.Categorical(predict_x[column], categories=categories)

    model = HistGradientBoostingRegressor(
        max_iter=max_iter,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        categorical_features=list(CATEGORICAL_COLUMNS),
        early_stopping=False,
        random_state=random_state,
    )
    fit_started = perf_counter()
    model.fit(train_x, train["weekly_sales"].to_numpy(dtype=float))
    fit_seconds = perf_counter() - fit_started
    predict_started = perf_counter()
    # Walmart's net weekly sales include returns and can be negative.
    predictions = model.predict(predict_x)
    predict_seconds = perf_counter() - predict_started
    return predictions, fit_seconds, predict_seconds


def evaluate_forecasts(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate baselines and a pooled tree on shared chronological folds.

    Week settings are one-based positions in the complete target timeline.
    Validation fits one model per target week. Holdout fits once using only
    targets before the holdout and reuses that frozen model for every holdout
    week; lagged input values still advance using each forecast's known origin.
    """
    config = dict(config or {})
    evaluation_started = perf_counter()
    validation_start = int(config.get("validation_start_week", 65))
    validation_end = int(config.get("validation_end_week", 96))
    holdout_start = int(config.get("holdout_start_week", 97))
    holdout_end = int(config.get("holdout_end_week", 104))
    max_iter = int(config.get("max_iter", 40))
    random_state = int(config.get("random_state", 42))
    min_improvement = float(config.get("min_wape_improvement", 0.02))
    max_bias = float(config.get("max_abs_weighted_bias", 0.05))

    if not feature_columns:
        raise ValueError("At least one model feature is required")
    if set(feature_columns) & {"weekly_sales", "event_timestamp", "origin_week", "target_week"}:
        raise ValueError("Labels and timeline metadata cannot be model inputs")
    if not set(CATEGORICAL_COLUMNS).issubset(feature_columns):
        raise ValueError("The model feature list must include store and department identity")
    if validation_start > validation_end or validation_end >= holdout_start or holdout_start > holdout_end:
        raise ValueError("Expected ordered validation weeks followed by a holdout window")

    required = {
        "store_id", "dept_id", "event_timestamp", "origin_week", "weekly_sales",
        *feature_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Feast historical rows are missing columns: {', '.join(missing)}")

    rows = frame.copy().reset_index(drop=True)
    rows["event_timestamp"] = pd.to_datetime(rows["event_timestamp"], utc=True)
    rows["origin_week"] = pd.to_datetime(rows["origin_week"], utc=True)
    if rows[["event_timestamp", "origin_week"]].isna().any().any():
        raise ValueError("Forecast origin and target timestamps must be present")
    if not ((rows["origin_week"] < rows["event_timestamp"]).all()):
        raise ValueError("Every forecast origin must precede its target week")
    if not (
        rows["event_timestamp"] - rows["origin_week"] == pd.Timedelta(weeks=1)
    ).all():
        raise ValueError("Each example must forecast exactly one week ahead")
    if rows.duplicated(["store_id", "dept_id", "event_timestamp"]).any():
        raise ValueError("Feast returned duplicate entity/target-week rows")

    target_weeks = rows["event_timestamp"].drop_duplicates().sort_values().tolist()
    if len(target_weeks) < holdout_end:
        raise ValueError(
            f"Backtest needs at least {holdout_end} target weeks; got {len(target_weeks)}"
        )
    week_numbers = {week: index + 1 for index, week in enumerate(target_weeks)}
    rows["target_week_number"] = rows["event_timestamp"].map(week_numbers)
    rows["weekly_sales"] = pd.to_numeric(rows["weekly_sales"], errors="coerce")
    numeric_features = rows.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    eligible = rows["weekly_sales"].notna() & np.isfinite(rows["weekly_sales"])
    eligible &= numeric_features.notna().all(axis=1) & np.isfinite(numeric_features).all(axis=1)
    rows = rows.loc[eligible].copy()
    if rows.empty:
        raise ValueError("No Feast rows have complete labels and model features")

    lookup = {
        (row.store_id, row.dept_id, int(row.target_week_number)): float(row.weekly_sales)
        for row in rows.itertuples(index=False)
    }
    candidate_names = ("last_week_naive", "seasonal_naive", MODEL_NAME)
    validation_actual: list[float] = []
    validation_predictions: dict[str, list[float]] = {name: [] for name in candidate_names}
    validation_fold_metrics: dict[str, list[dict[str, Any]]] = {
        name: [] for name in candidate_names
    }
    validation_series: set[tuple[Any, Any]] = set()
    validation_rows: list[pd.DataFrame] = []
    folds = []

    for target_week_number in range(validation_start, validation_end + 1):
        train = rows.loc[rows["target_week_number"] < target_week_number]
        valid = rows.loc[rows["target_week_number"] == target_week_number]
        if train.empty or valid.empty:
            raise ValueError(f"Fold {target_week_number} has an empty train or validation set")
        if not (train["event_timestamp"].max() < valid["event_timestamp"].min()):
            raise ValueError(f"Fold {target_week_number} crosses target weeks")

        actual = valid["weekly_sales"].to_numpy(dtype=float)
        last_week = valid["lag_1"].to_numpy(dtype=float)
        seasonal_values = []
        for row in valid.itertuples(index=False):
            prior = lookup.get((row.store_id, row.dept_id, target_week_number - 52))
            if prior is None:
                raise ValueError(
                    f"Seasonal naive is unavailable for target week {target_week_number}"
                )
            seasonal_values.append(prior)
        tree_prediction, fit_seconds, predict_seconds = _fit_predict(
            train, valid, feature_columns, max_iter=max_iter, random_state=random_state
        )
        predictions = {
            "last_week_naive": last_week,
            "seasonal_naive": np.asarray(seasonal_values, dtype=float),
            MODEL_NAME: tree_prediction,
        }
        fold = {
            "target_week_number": target_week_number,
            "target_week": valid["event_timestamp"].min().strftime("%Y-%m-%d"),
            "origin_week": valid["origin_week"].min().strftime("%Y-%m-%d"),
            "train_through_target_week_number": int(train["target_week_number"].max()),
            "train_rows": int(len(train)),
            "validation_rows": int(len(valid)),
            "series": int(valid[["store_id", "dept_id"]].drop_duplicates().shape[0]),
            "hist_gradient_boosting_fit_seconds": fit_seconds,
            "hist_gradient_boosting_predict_seconds": predict_seconds,
            "metrics": {},
        }
        validation_series.update(
            valid[["store_id", "dept_id"]].itertuples(index=False, name=None)
        )
        fold_rows = valid[["store_id", "dept_id"]].copy()
        fold_rows["actual"] = actual
        for name, predicted in predictions.items():
            metrics = _metrics(actual, predicted)
            metrics["series"] = int(valid[["store_id", "dept_id"]].drop_duplicates().shape[0])
            fold["metrics"][name] = metrics
            validation_fold_metrics[name].append(metrics)
            validation_predictions[name].extend(predicted.tolist())
            fold_rows[name] = predicted
        validation_actual.extend(actual.tolist())
        validation_rows.append(fold_rows)
        folds.append(fold)

    series_count = len(validation_series)
    validation = _summary(
        validation_actual,
        validation_predictions,
        validation_fold_metrics,
        series_count=series_count,
    )
    validation_frame = pd.concat(validation_rows, ignore_index=True)
    validation_per_series = {
        name: _series_metrics(validation_frame, "actual", name) for name in candidate_names
    }
    seasonal_validation_wape = validation["seasonal_naive"]["mean_fold_wape"]
    candidate_validation_wape = validation[MODEL_NAME]["mean_fold_wape"]
    relative_improvement = (
        (seasonal_validation_wape - candidate_validation_wape) / seasonal_validation_wape
        if seasonal_validation_wape > 0
        else None
    )
    validation_gate = {
        "relative_wape_improvement": relative_improvement,
        "minimum_relative_wape_improvement": min_improvement,
        "absolute_weighted_bias": abs(validation[MODEL_NAME]["weighted_bias"]),
        "maximum_absolute_weighted_bias": max_bias,
        "passed": bool(
            relative_improvement is not None
            and relative_improvement >= min_improvement
            and abs(validation[MODEL_NAME]["weighted_bias"]) <= max_bias
        ),
    }

    holdout_train = rows.loc[rows["target_week_number"] < holdout_start]
    holdout = rows.loc[
        rows["target_week_number"].between(holdout_start, holdout_end)
    ].copy()
    if holdout_train.empty or holdout.empty:
        raise ValueError("The configured untouched holdout is empty")
    if int(holdout_train["target_week_number"].max()) >= holdout_start:
        raise ValueError("Holdout labels leaked into model fit")

    frozen_predictions, holdout_fit_seconds, holdout_predict_seconds = _fit_predict(
        holdout_train,
        holdout,
        feature_columns,
        max_iter=max_iter,
        random_state=random_state,
    )
    holdout["actual"] = holdout["weekly_sales"].astype(float)
    holdout["last_week_naive"] = holdout["lag_1"].astype(float)
    seasonal_holdout = []
    for row in holdout.itertuples(index=False):
        prior = lookup.get((row.store_id, row.dept_id, int(row.target_week_number) - 52))
        if prior is None:
            raise ValueError(
                f"Seasonal naive is unavailable for holdout target week {row.target_week_number}"
            )
        seasonal_holdout.append(prior)
    holdout["seasonal_naive"] = np.asarray(seasonal_holdout, dtype=float)
    holdout[MODEL_NAME] = frozen_predictions
    holdout_actual = holdout["actual"].to_numpy(dtype=float)
    holdout_predictions = {
        name: holdout[name].to_numpy(dtype=float) for name in candidate_names
    }
    holdout_series_count = int(holdout[["store_id", "dept_id"]].drop_duplicates().shape[0])
    holdout_summary = _summary(
        holdout_actual, holdout_predictions, series_count=holdout_series_count
    )
    holdout_series = {
        name: _series_metrics(holdout, "actual", name) for name in candidate_names
    }
    holdout_gate = {
        "hist_gradient_boosting_wape_no_worse_than_seasonal_naive": bool(
            holdout_summary[MODEL_NAME]["wape"] <= holdout_summary["seasonal_naive"]["wape"]
        ),
        "absolute_weighted_bias": abs(holdout_summary[MODEL_NAME]["weighted_bias"]),
        "maximum_absolute_weighted_bias": max_bias,
        "passed": bool(
            holdout_summary[MODEL_NAME]["wape"] <= holdout_summary["seasonal_naive"]["wape"]
            and abs(holdout_summary[MODEL_NAME]["weighted_bias"]) <= max_bias
        ),
    }

    return {
        "protocol": {
            "forecast_horizon_weeks": 1,
            "target_timeline_weeks": int(len(target_weeks)),
            "validation_target_weeks": [validation_start, validation_end],
            "holdout_target_weeks": [holdout_start, holdout_end],
            "holdout_fit_through_target_week_number": int(
                holdout_train["target_week_number"].max()
            ),
            "holdout_model_refit_between_weeks": False,
            "holdout_used_for_candidate_selection": False,
            "per_fold_temporal_order_asserted": True,
        },
        "runtime": {
            "total_evaluation_seconds": perf_counter() - evaluation_started,
            "validation_tree_fit_seconds": float(
                sum(fold["hist_gradient_boosting_fit_seconds"] for fold in folds)
            ),
            "validation_tree_predict_seconds": float(
                sum(fold["hist_gradient_boosting_predict_seconds"] for fold in folds)
            ),
            "holdout_tree_fit_seconds": holdout_fit_seconds,
            "holdout_tree_predict_seconds": holdout_predict_seconds,
        },
        "resources": dict(config.get("resource_profile", {})),
        "dataset": {
            "rows_with_complete_model_features": int(len(rows)),
            "series": int(rows[["store_id", "dept_id"]].drop_duplicates().shape[0]),
            "first_target_week": target_weeks[0].strftime("%Y-%m-%d"),
            "last_target_week": target_weeks[-1].strftime("%Y-%m-%d"),
        },
        "candidates": {
            "last_week_naive": {"family": "baseline"},
            "seasonal_naive": {"family": "baseline", "lag_weeks": 52},
            MODEL_NAME: {
                "family": "pooled_cpu_gradient_boosting",
                "max_iter": max_iter,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 20,
                "l2_regularization": 1.0,
                "categorical_features": list(CATEGORICAL_COLUMNS),
                "random_state": random_state,
            },
            "mlp": {"status": "not_compared_on_these_folds; integration_example_only"},
        },
        "validation": validation,
        "validation_per_series": validation_per_series,
        "validation_gate": validation_gate,
        "validation_folds": folds,
        "holdout": holdout_summary,
        "holdout_per_series": holdout_series,
        "holdout_gate": holdout_gate,
        "decision": {
            "candidate_gates_passed": bool(validation_gate["passed"] and holdout_gate["passed"]),
            "selected_or_promoted_model": None,
            "reason": (
                "This report compares baselines and a tree candidate only; the MLP, final refit, "
                "artifact contract, and serving path are not yet compared and validated."
            ),
        },
    }
