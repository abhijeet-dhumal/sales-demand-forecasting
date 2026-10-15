"""
Walmart Sales Forecasting Feature Definitions

This module defines Feast feature store resources for the Walmart-style sales forecasting use case.

Architecture:
- Registry: PostgreSQL (SQL-queryable)
- Offline Store: File-based (Parquet) for apply/materialize
- Online Store: PostgreSQL (low-latency serving)
- Compute Engine: Ray (distributed get_historical_features)

Entities:
- store_id: Store identifier
- dept_id: Department identifier

Feature Views:
- sales_features: Target-week calendar and strictly prior sales features
- store_features: Static store metadata (type, size, region)

Feature Services:
- training_features: Full feature set for model training (includes target)
- inference_features: Features for inference (excludes target)

Target-week rows are one-week-ahead forecasts. Actual future weather and
economic observations are deliberately excluded from both FeatureServices.
"""

from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, FeatureService
from feast.types import Float32, Int32, String


# =============================================================================
# ENTITIES
# =============================================================================

store_entity = Entity(
    name="store_id",
    description="Store identifier",
    join_keys=["store_id"],
    tags={"team": "sales", "priority": "high"},
)

dept_entity = Entity(
    name="dept_id",
    description="Department identifier",
    join_keys=["dept_id"],
    tags={"team": "sales"},
)


# =============================================================================
# DATA SOURCES (Parquet files on shared PVC - pre-sorted for fast PIT joins)
# =============================================================================
# Callers set FEAST_DATA_ROOT to the shared feature data path before loading this module.
import os

_DATA_ROOT = os.environ.get("FEAST_DATA_ROOT")
if not _DATA_ROOT:
    raise RuntimeError("Set FEAST_DATA_ROOT before loading Feast features")

sales_source = FileSource(
    path=f"{_DATA_ROOT}/sales_features.parquet",
    timestamp_field="event_timestamp",
    description="Weekly sales with optimized features (pre-sorted by entity + timestamp)",
)

store_source = FileSource(
    path=f"{_DATA_ROOT}/store_features.parquet",
    timestamp_field="event_timestamp",
    description="Static store metadata",
)


# =============================================================================
# FEATURE VIEWS
# =============================================================================

sales_features = FeatureView(
    name="sales_features",
    description="One-week-ahead target rows with strictly prior sales history",
    entities=[store_entity, dept_entity],
    # Competition timestamps are historical; this demo replays the first test week today.
    ttl=timedelta(0),
    schema=[
        # Target variable
        Field(name="weekly_sales", dtype=Float32, description="Weekly sales amount ($)"),
        
        # Lag and rolling fields are computed strictly before the target week.
        Field(name="lag_1", dtype=Float32, description="Sales from the forecast origin week"),
        Field(name="lag_2", dtype=Float32, description="Sales from 2 weeks ago"),
        Field(name="lag_4", dtype=Float32, description="Sales from 4 weeks ago"),
        Field(name="lag_8", dtype=Float32, description="Sales from 8 weeks ago"),
        
        # === ROLLING STATISTICS (28% importance) - Trend & volatility ===
        Field(name="rolling_mean_4w", dtype=Float32, description="Prior four observed weeks through the origin"),
        Field(name="rolling_std_4w", dtype=Float32, description="Prior four-week sales volatility"),
        Field(name="sales_vs_avg", dtype=Float32, description="Origin-week sales relative to prior four-week mean"),
        
        # === TEMPORAL FEATURES (18% importance) - Seasonality ===
        Field(name="week_of_year", dtype=Int32, description="Week 1-52 (yearly seasonality)"),
        Field(name="month", dtype=Int32, description="Month 1-12"),
        Field(name="quarter", dtype=Int32, description="Quarter 1-4 (Q4 retail boost)"),
        Field(name="week_of_month", dtype=Int32, description="Week 1-5 within month"),
        Field(name="is_month_end", dtype=Int32, description="Last week of month (0/1)"),
        
        # === HOLIDAY FEATURES (10% importance) ===
        Field(name="is_holiday", dtype=Int32, description="Holiday week indicator (0/1)"),
        Field(name="days_to_holiday", dtype=Int32, description="Days until next major holiday"),
        
        Field(name="origin_week", dtype=String, description="Last week available to the forecaster"),
        Field(name="target_week", dtype=String, description="Week being forecast"),
    ],
    source=sales_source,
    online=True,
    tags={"type": "time_series", "domain": "sales"},
)

store_features = FeatureView(
    name="store_features",
    description="Static store metadata",
    entities=[store_entity, dept_entity],
    ttl=None,
    schema=[
        Field(name="store_index", dtype=Int32, description="Numeric store identifier supplied as a model input"),
        Field(name="department_index", dtype=Int32, description="Numeric department identifier supplied as a model input"),
        Field(name="store_type", dtype=String, description="Store type: A, B, or C"),
        Field(name="store_size", dtype=Int32, description="Store square footage"),
        Field(name="region", dtype=String, description="Geographic region"),
    ],
    source=store_source,
    online=True,
    tags={"type": "dimension", "domain": "store"},
)


# =============================================================================
# FEATURE SERVICES
# =============================================================================

# Training includes the label and the same forecast inputs used at inference.
training_features = FeatureService(
    name="training_features",
    description="One-week-ahead training rows with a separate target label",
    features=[
        sales_features[[
            "weekly_sales",
            "lag_1", "lag_2", "lag_4", "lag_8",
            "rolling_mean_4w", "rolling_std_4w", "sales_vs_avg",
            "week_of_year", "month", "quarter", "week_of_month", "is_month_end",
            "is_holiday", "days_to_holiday",
            "origin_week", "target_week",
        ]],
        store_features[["store_index", "department_index", "store_size"]],
    ],
    tags={"stage": "training", "model": "demand-forecasting", "version": "one-week-ahead-v1"},
)

# Inference: excludes target variable
inference_features = FeatureService(
    name="inference_features",
    description="Forecast inputs and target-week metadata, excluding target sales",
    features=[
        sales_features[[
            "lag_1", "lag_2", "lag_4", "lag_8",
            "rolling_mean_4w", "rolling_std_4w", "sales_vs_avg",
            "week_of_year", "month", "quarter", "week_of_month", "is_month_end",
            "is_holiday", "days_to_holiday",
            "origin_week", "target_week",
        ]],
        store_features[["store_index", "department_index", "store_size"]],
    ],
    tags={"stage": "inference", "model": "demand-forecasting", "version": "one-week-ahead-v1"},
)
