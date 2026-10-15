"""Leakage-safe one-week-ahead sales forecasting feature contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HORIZON_WEEKS = 1
BACKTEST_MIN_WEEKS = 104
MODEL_FEATURE_COLUMNS = (
    "store_index",
    "department_index",
    "lag_1",
    "lag_2",
    "lag_4",
    "lag_8",
    "rolling_mean_4w",
    "rolling_std_4w",
    "sales_vs_avg",
    "week_of_year",
    "month",
    "quarter",
    "week_of_month",
    "is_month_end",
    "is_holiday",
    "days_to_holiday",
    "store_size",
)


def load_walmart_dataset(
    data_dir: str | Path, min_weeks: int = BACKTEST_MIN_WEEKS
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load the Walmart Kaggle dataset tables and keep complete weekly series.

    The caller places train.csv, features.csv, and stores.csv (or their
    .csv.zip files) from the Walmart Kaggle dataset in ``data_dir``.
    """
    root = Path(data_dir)

    train = _read_walmart_file(root, "train.csv")
    features = _read_walmart_file(root, "features.csv")
    stores = _read_walmart_file(root, "stores.csv")

    required = {
        "train.csv": {"Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"},
        "features.csv": {"Store", "Date"},
        "stores.csv": {"Store", "Type", "Size"},
    }
    for name, frame in (("train.csv", train), ("features.csv", features), ("stores.csv", stores)):
        missing = sorted(required[name] - set(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")

    train = train.copy()
    features = features.drop(columns=["IsHoliday"], errors="ignore").copy()
    stores = stores.copy()
    for frame in (train, features):
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise").dt.normalize()

    if train[["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"]].isna().any().any():
        raise ValueError("train.csv has missing entity keys, dates, sales, or holiday flags")
    if features[["Store", "Date"]].duplicated().any():
        raise ValueError("features.csv has duplicate Store/Date rows")
    if stores["Store"].duplicated().any():
        raise ValueError("stores.csv has duplicate Store rows")

    merged = train.merge(
        features, on=["Store", "Date"], how="left", validate="many_to_one"
    ).merge(stores, on="Store", how="left", validate="many_to_one")
    if merged[["Type", "Size"]].isna().any().any():
        raise ValueError("Some sales rows do not have matching store metadata")

    dates = merged["Date"]
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert("UTC").dt.tz_localize(None)
    week_end = dates.dt.to_period("W-FRI").dt.end_time.dt.normalize()
    merged["event_timestamp"] = week_end.dt.tz_localize("UTC")
    merged = merged.rename(
        columns={
            "Store": "store_id",
            "Dept": "dept_id",
            "Weekly_Sales": "weekly_sales",
            "IsHoliday": "is_holiday",
            "Type": "store_type",
            "Size": "store_size",
            "Temperature": "temperature",
            "Fuel_Price": "fuel_price",
            "CPI": "cpi",
            "Unemployment": "unemployment",
        }
    )
    for column in ("store_id", "dept_id", "store_size", "is_holiday"):
        merged[column] = pd.to_numeric(merged[column], errors="raise").astype("int64")
    merged["weekly_sales"] = pd.to_numeric(merged["weekly_sales"], errors="raise")
    merged["store_type"] = merged["store_type"].astype(str)

    entity_keys = ["store_id", "dept_id", "event_timestamp"]
    if merged.duplicated(entity_keys).any():
        raise ValueError("Walmart data has duplicate Store/Dept/week rows")
    if merged.empty:
        raise ValueError("train.csv has no rows after joining the Walmart dataset files")
    weeks = pd.DatetimeIndex(merged["event_timestamp"].drop_duplicates().sort_values())
    expected_weeks = pd.date_range(weeks[0], periods=len(weeks), freq="7D")
    if not weeks.equals(expected_weeks):
        raise ValueError("Walmart data must contain consecutive weekly dates")
    if len(weeks) < min_weeks:
        raise ValueError(
            f"Backtest needs at least {min_weeks} weekly dates; found {len(weeks)}"
        )

    coverage = merged.groupby(["store_id", "dept_id"])["event_timestamp"].nunique()
    complete_entities = coverage.index[coverage.eq(len(weeks))]
    row_entities = pd.MultiIndex.from_frame(merged[["store_id", "dept_id"]])
    complete = merged.loc[row_entities.isin(complete_entities)].copy()
    if complete.empty:
        raise ValueError("No Store/Dept series has complete weekly coverage")

    keep = entity_keys + [
        "weekly_sales", "is_holiday", "store_type", "store_size",
        "temperature", "fuel_price", "cpi", "unemployment",
    ]
    complete = complete[[column for column in keep if column in complete.columns]]
    complete = complete.sort_values(entity_keys).reset_index(drop=True)
    return complete, {
        "weeks": len(weeks),
        "series_kept": len(complete_entities),
        "series_dropped": len(coverage) - len(complete_entities),
        "rows": len(complete),
    }


def _read_walmart_file(root: Path, name: str) -> pd.DataFrame:
    for path in (root / name, root / f"{name}.zip"):
        if path.is_file():
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"Missing {name} or {name}.zip in {root}. Download the Walmart "
        "dataset files from Kaggle and extract them there."
    )


def append_walmart_forecast_rows(
    sales: pd.DataFrame, data_dir: str | Path
) -> tuple[pd.DataFrame, dict[str, int | str], pd.DatetimeIndex]:
    """Append the first Kaggle test week for entities with complete sales history.

    The appended rows have null labels and known test/calendar fields. They are
    for online materialization and inference only; training must exclude them.
    """
    root = Path(data_dir)
    test = _read_walmart_file(root, "test.csv")
    features = _read_walmart_file(root, "features.csv")
    required_test = {"Store", "Dept", "Date", "IsHoliday"}
    required_features = {"Store", "Date", "IsHoliday"}
    missing_test = sorted(required_test - set(test.columns))
    missing_features = sorted(required_features - set(features.columns))
    if missing_test:
        raise ValueError(f"test.csv is missing required columns: {', '.join(missing_test)}")
    if missing_features:
        raise ValueError(f"features.csv is missing required columns: {', '.join(missing_features)}")

    test = test.copy()
    features = features.copy()
    for frame in (test, features):
        frame["Date"] = pd.to_datetime(frame["Date"], errors="raise").dt.normalize()
    if test[["Store", "Dept", "Date", "IsHoliday"]].isna().any().any():
        raise ValueError("test.csv has missing entity keys, dates, or holiday flags")
    if test.duplicated(["Store", "Dept", "Date"]).any():
        raise ValueError("test.csv has duplicate Store/Dept/week rows")
    if features[["Store", "Date"]].duplicated().any():
        raise ValueError("features.csv has duplicate Store/Date rows")

    latest_timestamp = pd.to_datetime(sales["event_timestamp"], utc=True).max()
    latest_date = latest_timestamp.tz_convert("UTC").tz_localize(None).normalize()
    target_date = test["Date"].min()
    if target_date != latest_date + pd.Timedelta(weeks=1):
        raise ValueError(
            "This one-week demo expects the first test.csv date to immediately "
            f"follow the latest train.csv week ({latest_date.date()}); found {target_date.date()}"
        )

    test_week = test.loc[test["Date"].eq(target_date)].rename(
        columns={"IsHoliday": "test_is_holiday"}
    )
    context_columns = [
        column
        for column in ("Temperature", "Fuel_Price", "CPI", "Unemployment")
        if column in features.columns
    ]
    calendar = features.loc[
        features["Date"].eq(target_date),
        ["Store", "Date", "IsHoliday", *context_columns],
    ].rename(columns={"IsHoliday": "feature_is_holiday"})
    future = test_week.merge(
        calendar, on=["Store", "Date"], how="left", validate="many_to_one"
    )
    if future["feature_is_holiday"].isna().any():
        raise ValueError("features.csv has no calendar row for every first-week test store")
    test_holidays = pd.to_numeric(future["test_is_holiday"], errors="raise").astype("int32")
    feature_holidays = pd.to_numeric(future["feature_is_holiday"], errors="raise").astype("int32")
    if not test_holidays.eq(feature_holidays).all():
        raise ValueError("test.csv and features.csv disagree on the first test week's holiday flag")

    history_entities = sales.loc[
        pd.to_datetime(sales["event_timestamp"], utc=True).eq(latest_timestamp),
        ["store_id", "dept_id", "store_type", "store_size"],
    ].drop_duplicates(["store_id", "dept_id"])
    future = future.rename(columns={"Store": "store_id", "Dept": "dept_id"})
    future = future.merge(
        history_entities,
        on=["store_id", "dept_id"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    no_history = int(future["_merge"].ne("both").sum())
    future = future.loc[future["_merge"].eq("both")].drop(columns="_merge").copy()
    if future.empty:
        raise ValueError("No first-week Kaggle test entities have complete training history")

    future["event_timestamp"] = target_date.tz_localize("UTC")
    future["weekly_sales"] = np.nan
    future["is_holiday"] = pd.to_numeric(future["feature_is_holiday"], errors="raise").astype("int32")
    for source, target in (
        ("Temperature", "temperature"),
        ("Fuel_Price", "fuel_price"),
        ("CPI", "cpi"),
        ("Unemployment", "unemployment"),
    ):
        if source in future:
            future[target] = future[source]

    output_columns = [
        "store_id", "dept_id", "event_timestamp", "weekly_sales", "is_holiday",
        "store_type", "store_size", "temperature", "fuel_price", "cpi", "unemployment",
    ]
    future = future[[column for column in output_columns if column in future.columns]]
    extended = pd.concat([sales, future], ignore_index=True, sort=False)

    holiday_rows = features.loc[
        pd.to_numeric(features["IsHoliday"], errors="coerce").fillna(0).eq(1), "Date"
    ]
    holiday_dates = (
        holiday_rows.dt.to_period("W-FRI").dt.end_time.dt.normalize()
        .drop_duplicates().sort_values().dt.tz_localize("UTC")
    )
    return extended, {
        "forecast_week": target_date.strftime("%Y-%m-%d"),
        "test_entities": int(len(test_week)),
        "forecast_entities": int(len(future)),
        "entities_without_complete_history": no_history,
    }, pd.DatetimeIndex(holiday_dates)


def build_forecast_frame(
    sales: pd.DataFrame,
    known_holiday_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build target-week rows using only sales strictly before that week.

    ``event_timestamp`` and ``weekly_sales`` identify the target week and label.
    Calendar fields describe that target week and are known in advance. Actual
    weather and economic observations are retained for diagnostics but are not
    model inputs because they are not known at the forecast origin.
    """
    required = {"store_id", "dept_id", "event_timestamp", "weekly_sales"}
    missing = sorted(required - set(sales.columns))
    if missing:
        raise ValueError(f"Sales data is missing required columns: {', '.join(missing)}")

    frame = sales.copy()
    frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"], utc=True)
    frame = frame.sort_values(["store_id", "dept_id", "event_timestamp"]).reset_index(drop=True)
    if frame.duplicated(["store_id", "dept_id", "event_timestamp"]).any():
        raise ValueError("Sales data has duplicate entity/week rows")
    intervals = frame.groupby(["store_id", "dept_id"], sort=False)["event_timestamp"].diff().dropna()
    if not intervals.eq(pd.Timedelta(weeks=1)).all():
        raise ValueError("Sales data must contain one observation per entity in consecutive weeks")

    if "is_holiday" in frame:
        frame["is_holiday"] = pd.to_numeric(frame["is_holiday"], errors="coerce").fillna(0).astype("int32")
    else:
        frame["is_holiday"] = pd.Series(0, index=frame.index, dtype="int32")
    frame["week_of_year"] = frame["event_timestamp"].dt.isocalendar().week.astype("int32")
    frame["month"] = frame["event_timestamp"].dt.month.astype("int32")
    frame["quarter"] = frame["event_timestamp"].dt.quarter.astype("int32")
    frame["week_of_month"] = ((frame["event_timestamp"].dt.day - 1) // 7 + 1).astype("int32")
    frame["is_month_end"] = (
        frame["event_timestamp"].add(pd.Timedelta(weeks=1)).dt.month
        != frame["event_timestamp"].dt.month
    ).astype("int32")

    if known_holiday_dates is None:
        holiday_dates = pd.DatetimeIndex(
            frame.loc[frame["is_holiday"].eq(1), "event_timestamp"].drop_duplicates().sort_values()
        )
    else:
        holiday_dates = pd.DatetimeIndex(pd.to_datetime(known_holiday_dates, utc=True))
        holiday_dates = holiday_dates.drop_duplicates().sort_values()
    holiday_values = holiday_dates.asi8
    timestamp_values = frame["event_timestamp"].array.asi8
    days_to_holiday = np.full(len(frame), 365, dtype=np.int32)
    if len(holiday_values):
        next_holiday = np.searchsorted(holiday_values, timestamp_values, side="left")
        known = next_holiday < len(holiday_values)
        days_to_holiday[known] = (
            (holiday_values[next_holiday[known]] - timestamp_values[known])
            // pd.Timedelta(days=1).value
        ).astype("int32")
    frame["days_to_holiday"] = days_to_holiday

    entity_keys = ["store_id", "dept_id", "event_timestamp"]
    history = frame[entity_keys + ["weekly_sales"]]
    for lag in (1, 2, 3, 4, 8):
        prior = history.copy()
        prior["event_timestamp"] += pd.Timedelta(weeks=lag)
        prior = prior.rename(columns={"weekly_sales": f"lag_{lag}"})
        frame = frame.merge(prior, on=entity_keys, how="left", validate="one_to_one", sort=False)

    prior_four = frame[[f"lag_{lag}" for lag in (1, 2, 3, 4)]]
    frame["rolling_mean_4w"] = prior_four.mean(axis=1, skipna=False)
    frame["rolling_std_4w"] = prior_four.std(axis=1, skipna=False)
    rolling_mean = frame["rolling_mean_4w"].mask(frame["rolling_mean_4w"].eq(0))
    momentum = frame["lag_1"] / rolling_mean - 1.0
    frame["sales_vs_avg"] = momentum.replace([float("inf"), float("-inf")], 0).fillna(0)

    frame["target_week"] = frame["event_timestamp"].dt.strftime("%Y-%m-%d")
    frame["origin_week"] = (
        frame["event_timestamp"] - pd.Timedelta(weeks=HORIZON_WEEKS)
    ).dt.strftime("%Y-%m-%d")
    frame["horizon"] = HORIZON_WEEKS
    return frame


def chronological_split(
    frame: pd.DataFrame, validation_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split whole target weeks so rows from one week cannot cross the boundary."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    timestamp_column = "event_timestamp" if "event_timestamp" in frame else "date"
    if timestamp_column not in frame:
        raise ValueError("Forecast data must contain event_timestamp or date")

    timestamps = pd.to_datetime(frame[timestamp_column], utc=True)
    weeks = sorted(timestamps.dropna().unique())
    split_at = int(len(weeks) * (1.0 - validation_fraction))
    if split_at < 1 or split_at >= len(weeks):
        raise ValueError("Not enough unique target weeks for the requested split")
    cutoff = weeks[split_at]
    train = frame.loc[timestamps < cutoff].copy()
    validation = frame.loc[timestamps >= cutoff].copy()
    if train.empty or validation.empty:
        raise ValueError("Chronological split produced an empty partition")
    return train, validation
