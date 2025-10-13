"""
Sales Demand Forecasting Feature Definitions

This module defines feature store resources for the Walmart sales forecasting use case.

Dataset: Real Walmart Sales Forecasting data from Kaggle (421K records)

Key Features:
- Entities: store, dept (composite key handled by Feature Store)
- Sales features: weekly_sales, is_holiday, time-series features (lags, rolling averages)
- External features: temperature, CPI, fuel_price, unemployment, markdowns
- On-demand features: normalization, interactions, business logic

Feature Engineering Pattern:
- Time-series features pre-computed in download_data.py (ETL pipeline)
- Feature Store handles feature serving, consistency, and online/offline stores
- On-demand transformations for runtime feature generation
"""

from datetime import timedelta
import pandas as pd

from feast import Entity, FeatureView, Field, FileSource, FeatureService, ValueType
from feast.types import Float64, Int32, Int64, String
from feast.on_demand_feature_view import on_demand_feature_view


# ======================================================================================
# ENTITY DEFINITIONS
# ======================================================================================

store_entity = Entity(
    name="store",
    value_type=ValueType.INT64,
    description="Walmart store number (1-45). Each store represents a physical retail location.",
    tags={
        "owner": "retail_analytics_team",
        "team": "data_science",
        "domain": "retail_sales",
    },
)

dept_entity = Entity(
    name="dept",
    value_type=ValueType.INT64,
    description="Department number (1-99). Departments represent product categories within stores.",
    tags={
        "owner": "retail_analytics_team",
        "team": "data_science",
        "domain": "retail_sales",
    },
)


# ======================================================================================
# DATA SOURCES
# ======================================================================================

# Sales Data Source (with pre-computed time-series features)
sales_source = FileSource(
    name="sales_source",
    path="/shared/feature_repo/data/sales_features.parquet",
    timestamp_field="date",
)

store_external_source = FileSource(
    name="store_external_source",
    path="/shared/feature_repo/data/store_features.parquet",
    timestamp_field="date",
)


# ======================================================================================
# SALES HISTORY FEATURE VIEW
# ======================================================================================
# Time-series features are pre-computed in the ETL pipeline (download_data.py)

sales_history_features = FeatureView(
    name="sales_history_features",
    description="Historical sales patterns with pre-computed time-series features. "
                "Features computed in ETL pipeline for efficient serving.",
    entities=[store_entity, dept_entity],
    ttl=timedelta(days=730),
    schema=[
        # Base features
        Field(
            name="weekly_sales",
            dtype=Float64,
            description="Weekly sales amount in USD"
        ),
        Field(
            name="is_holiday",
            dtype=Int64,
            description="Binary indicator (0/1) for holiday weeks"
        ),
        # Time-series features (pre-computed)
        Field(
            name="sales_lag_1",
            dtype=Float64,
            description="Sales from 1 week ago (t-1)"
        ),
        Field(
            name="sales_lag_2",
            dtype=Float64,
            description="Sales from 2 weeks ago (t-2)"
        ),
        Field(
            name="sales_lag_4",
            dtype=Float64,
            description="Sales from 4 weeks ago (t-4)"
        ),
        Field(
            name="sales_rolling_mean_4",
            dtype=Float64,
            description="4-week rolling average of sales"
        ),
        Field(
            name="sales_rolling_mean_12",
            dtype=Float64,
            description="12-week rolling average of sales"
        ),
        Field(
            name="sales_rolling_std_4",
            dtype=Float64,
            description="4-week rolling standard deviation (volatility measure)"
        ),
    ],
    source=sales_source,
    online=True,
    tags={
        "owner": "retail_analytics_team",
        "team": "data_science",
        "priority": "critical",
        "feature_category": "time_series",
        "etl": "pre_computed",
    },
)


# Store external features
store_external_features = FeatureView(
    name="store_external_features",
    description="Store-level external factors.",
    entities=[store_entity, dept_entity],  # Multiple entities = composite key
    ttl=timedelta(days=730),
    schema=[
        Field(name="temperature", dtype=Float64),
        Field(name="fuel_price", dtype=Float64),
        Field(name="cpi", dtype=Float64),
        Field(name="unemployment", dtype=Float64),
        Field(name="markdown1", dtype=Float64),
        Field(name="markdown2", dtype=Float64),
        Field(name="markdown3", dtype=Float64),
        Field(name="markdown4", dtype=Float64),
        Field(name="markdown5", dtype=Float64),
        Field(name="total_markdown", dtype=Float64),
        Field(name="has_markdown", dtype=Int32),
        Field(name="store_type", dtype=String),
        Field(name="store_size", dtype=Int64),
    ],
    source=store_external_source,
    tags={
        "owner": "retail_analytics_team",
        "feature_category": "external_contextual",
    },
    online=True,
)


# ======================================================================================
# ON-DEMAND TRANSFORMATIONS
# ======================================================================================
# Runtime feature transformations applied during feature retrieval

@on_demand_feature_view(
    sources=[sales_history_features, store_external_features],
    schema=[
        Field(name="sales_normalized", dtype=Float64),
        Field(name="temperature_normalized", dtype=Float64),
        Field(name="sales_per_sqft", dtype=Float64),
        Field(name="markdown_efficiency", dtype=Float64),
    ],
    description="Runtime normalization and interaction features",
)
def feature_transformations(inputs: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["sales_normalized"] = inputs["weekly_sales"].clip(0, 200000) / 200000
    df["temperature_normalized"] = ((inputs["temperature"] - 5) / 95).clip(0, 1)
    df["sales_per_sqft"] = inputs["weekly_sales"] / (inputs["store_size"] + 1)
    df["markdown_efficiency"] = inputs["weekly_sales"] / (inputs["total_markdown"] + 1)
    return df


@on_demand_feature_view(
    sources=[sales_history_features],
    schema=[
        Field(name="sales_velocity", dtype=Float64),
        Field(name="sales_acceleration", dtype=Float64),
        Field(name="demand_stability_score", dtype=Float64),
    ],
    description="Temporal features using pre-computed lags and rolling features",
)
def temporal_transformations(inputs: pd.DataFrame) -> pd.DataFrame:
    """Compute velocity, acceleration, and stability from pre-computed time-series features."""
    df = pd.DataFrame()
    df["sales_velocity"] = (
        (inputs["sales_lag_1"] - inputs["sales_lag_2"]) / (inputs["sales_lag_2"] + 1)
    )
    velocity_prev = (inputs["sales_lag_2"] - inputs["sales_lag_4"]) / (inputs["sales_lag_4"] + 1)
    df["sales_acceleration"] = df["sales_velocity"] - velocity_prev
    df["demand_stability_score"] = 1 - (
        inputs["sales_rolling_std_4"] / (inputs["sales_rolling_mean_4"] + 1)
    ).clip(0, 1)
    return df


# ======================================================================================
# FEATURE SERVICES
# ======================================================================================

demand_forecasting_service = FeatureService(
    name="demand_forecasting_service",
    description="Complete feature set for sales demand forecasting. "
                "Includes pre-computed time-series features + external factors + on-demand transformations.",
    features=[
        sales_history_features,  # Sales + pre-computed time-series features
        store_external_features,  # External factors (temperature, CPI, etc.)
        feature_transformations,  # On-demand: normalization, interactions
        temporal_transformations,  # On-demand: velocity, acceleration, stability
    ],
    tags={
        "owner": "retail_analytics_team",
        "use_case": "demand_forecasting",
        "pattern": "etl_precompute",
    },
)

