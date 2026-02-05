"""
Sales Demand Forecasting Feature Definitions - PostgreSQL + Ray Architecture

This module defines feature store resources for the Walmart sales forecasting use case.

ARCHITECTURE (Feast 0.56.0):
- PostgreSQL Offline Store: Durable storage layer for raw features (I/O)
- Ray Compute Engine: Distributed processing layer for joins & transformations (compute)
- PostgreSQL Online Store: Low-latency serving layer for real-time inference

Dataset: Real Walmart Sales Forecasting data from Kaggle (421K records)

Key Features:
- Entities: store, dept (composite key)
- Sales features: weekly_sales, is_holiday, time-series (lags, rolling stats)
- External features: temperature, CPI, fuel_price, unemployment, markdowns
- On-demand features: normalization, interactions, business metrics

Performance:
- PostgreSQL: SQL-queryable, indexed for fast retrieval
- Ray Compute: Automatic parallelization of joins and transformations
- Expected: 1-2 min feature retrieval for 421K rows (10x faster than file-based)
"""

from datetime import timedelta
import pandas as pd
import numpy as np

from feast import Entity, FeatureView, Field, FeatureService
from feast.types import Float64, Int32, Int64, String
from feast.on_demand_feature_view import on_demand_feature_view

# PostgreSQL source (Feast 0.56.0)
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource


# ======================================================================================
# ENTITY DEFINITIONS
# ======================================================================================

store_entity = Entity(
    name="store",
    value_type=Int64,
    description="Walmart store number (1-45). Each store represents a physical retail location.",
    tags={
        "owner": "retail_analytics_team",
        "team": "data_science",
        "domain": "retail_sales",
    },
)

dept_entity = Entity(
    name="dept",
    value_type=Int64,
    description="Department number (1-99). Departments represent product categories within stores.",
    tags={
        "owner": "retail_analytics_team",
        "team": "data_science",
        "domain": "retail_sales",
    },
)


# ======================================================================================
# DATA SOURCES - PostgreSQL Tables
# ======================================================================================

# Sales Data Source (PostgreSQL table with pre-computed time-series features)
sales_source = PostgreSQLSource(
    name="sales_source",
    query="""
        SELECT 
            store, dept, date,
            weekly_sales, is_holiday,
            week_of_year, month, quarter,
            sales_lag_1, sales_lag_2, sales_lag_4,
            sales_rolling_mean_4, sales_rolling_mean_12, sales_rolling_std_4
        FROM sales_features
    """,
    timestamp_field="date",
    description="Historical sales with pre-computed time-series features from PostgreSQL",
)

# Store External Features (PostgreSQL table)
store_external_source = PostgreSQLSource(
    name="store_external_source",
    query="""
        SELECT
            store, dept, date,
            temperature, fuel_price, cpi, unemployment,
            markdown1, markdown2, markdown3, markdown4, markdown5,
            total_markdown, has_markdown,
            store_type, store_size
        FROM store_features
    """,
    timestamp_field="date",
    description="Store-level external factors and metadata from PostgreSQL",
)


# ======================================================================================
# SALES HISTORY FEATURE VIEW
# ======================================================================================
# Time-series features are pre-computed in the ETL pipeline (download_data.py)

sales_history_features = FeatureView(
    name="sales_history_features",
    description="Historical sales patterns from PostgreSQL. Ray compute engine handles retrieval and joins.",
    entities=[store_entity, dept_entity],
    ttl=timedelta(days=730),
    schema=[
        # Base features
        Field(name="weekly_sales", dtype=Float64, description="Weekly sales amount in USD"),
        Field(name="is_holiday", dtype=Int64, description="Binary indicator (0/1) for holiday weeks"),
        # Temporal features (pre-computed in ETL)
        Field(name="week_of_year", dtype=Int64, description="Week number (1-52) for seasonality"),
        Field(name="month", dtype=Int64, description="Month (1-12) for monthly patterns"),
        Field(name="quarter", dtype=Int64, description="Quarter (1-4) for quarterly trends"),
        # Time-series features (pre-computed lags and rolling stats)
        Field(name="sales_lag_1", dtype=Float64, description="Sales from 1 week ago (t-1)"),
        Field(name="sales_lag_2", dtype=Float64, description="Sales from 2 weeks ago (t-2)"),
        Field(name="sales_lag_4", dtype=Float64, description="Sales from 4 weeks ago (t-4)"),
        Field(name="sales_rolling_mean_4", dtype=Float64, description="4-week rolling average"),
        Field(name="sales_rolling_mean_12", dtype=Float64, description="12-week rolling average"),
        Field(name="sales_rolling_std_4", dtype=Float64, description="4-week rolling std (volatility)"),
    ],
    source=sales_source,
    online=True,
    tags={
        "owner": "retail_analytics_team",
        "team": "data_science",
        "priority": "critical",
        "feature_category": "time_series",
        "storage": "postgresql",
        "compute": "ray_engine",
    },
)


store_external_features = FeatureView(
    name="store_external_features",
    description="Store-level external factors from PostgreSQL. Ray compute engine parallelizes joins.",
    entities=[store_entity, dept_entity],  # Composite key: (store, dept)
    ttl=timedelta(days=730),
    schema=[
        # Economic indicators
        Field(name="temperature", dtype=Float64, description="Regional temperature (°F)"),
        Field(name="fuel_price", dtype=Float64, description="Regional fuel price ($/gallon)"),
        Field(name="cpi", dtype=Float64, description="Consumer Price Index"),
        Field(name="unemployment", dtype=Float64, description="Regional unemployment rate (%)"),
        # Markdown/promotional features
        Field(name="markdown1", dtype=Float64, description="Markdown type 1 amount ($)"),
        Field(name="markdown2", dtype=Float64, description="Markdown type 2 amount ($)"),
        Field(name="markdown3", dtype=Float64, description="Markdown type 3 amount ($)"),
        Field(name="markdown4", dtype=Float64, description="Markdown type 4 amount ($)"),
        Field(name="markdown5", dtype=Float64, description="Markdown type 5 amount ($)"),
        Field(name="total_markdown", dtype=Float64, description="Total markdown amount ($)"),
        Field(name="has_markdown", dtype=Int32, description="Binary: 1 if any markdown active"),
        # Store attributes
        Field(name="store_type", dtype=String, description="Store type: A, B, or C"),
        Field(name="store_size", dtype=Int64, description="Store square footage"),
    ],
    source=store_external_source,
    tags={
        "owner": "retail_analytics_team",
        "feature_category": "external_contextual",
        "storage": "postgresql",
        "compute": "ray_engine",
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
        Field(name="holiday_markdown_interaction", dtype=Float64),
        Field(name="markdown_momentum", dtype=Float64),
        Field(name="seasonal_sine", dtype=Float64),
        Field(name="seasonal_cosine", dtype=Float64),
    ],
    description="On-demand transformations - Ray compute engine automatically parallelizes these",
)
def feature_transformations(inputs: pd.DataFrame) -> pd.DataFrame:
    """
    On-demand transformations computed by Ray compute engine.
    
    Ray automatically:
    - Partitions input data
    - Distributes computation across workers
    - Combines results efficiently
    
    Performance: 5-10x faster than single-threaded processing
    """
    df = pd.DataFrame()
    
    # Normalization features
    df["sales_normalized"] = inputs["weekly_sales"].clip(0, 200000) / 200000
    df["temperature_normalized"] = ((inputs["temperature"] - 5) / 95).clip(0, 1)
    
    # Business metrics
    df["sales_per_sqft"] = inputs["weekly_sales"] / (inputs["store_size"] + 1)
    df["markdown_efficiency"] = inputs["weekly_sales"] / (inputs["total_markdown"] + 1)
    
    # Advanced interactions (captures non-linear relationships)
    df["holiday_markdown_interaction"] = inputs["is_holiday"] * inputs["total_markdown"]
    df["markdown_momentum"] = inputs["total_markdown"] / (inputs["sales_lag_1"] + 1)
    
    # Seasonal patterns using sine/cosine encoding (better than raw week numbers)
    df["seasonal_sine"] = np.sin(2 * np.pi * inputs["week_of_year"] / 52)
    df["seasonal_cosine"] = np.cos(2 * np.pi * inputs["week_of_year"] / 52)
    
    return df


@on_demand_feature_view(
    sources=[sales_history_features],
    schema=[
        Field(name="sales_velocity", dtype=Float64),
        Field(name="sales_acceleration", dtype=Float64),
        Field(name="demand_stability_score", dtype=Float64),
    ],
    description="Temporal transformations - Ray distributed computation from pre-computed lags",
)
def temporal_transformations(inputs: pd.DataFrame) -> pd.DataFrame:
    """
    Temporal features computed by Ray from pre-computed time-series data.
    
    Uses lag features (sales_lag_1, sales_lag_2, sales_lag_4) from PostgreSQL
    to derive velocity, acceleration, and stability metrics.
    """
    df = pd.DataFrame()
    
    # Sales velocity: rate of change (first derivative)
    df["sales_velocity"] = (
        (inputs["sales_lag_1"] - inputs["sales_lag_2"]) / (inputs["sales_lag_2"] + 1)
    )
    
    # Sales acceleration: rate of velocity change (second derivative)
    velocity_prev = (inputs["sales_lag_2"] - inputs["sales_lag_4"]) / (inputs["sales_lag_4"] + 1)
    df["sales_acceleration"] = df["sales_velocity"] - velocity_prev
    
    # Demand stability: inverse coefficient of variation (lower volatility = higher stability)
    df["demand_stability_score"] = 1 - (
        inputs["sales_rolling_std_4"] / (inputs["sales_rolling_mean_4"] + 1)
    ).clip(0, 1)
    
    return df


# ======================================================================================
# FEATURE SERVICES
# ======================================================================================

demand_forecasting_service = FeatureService(
    name="demand_forecasting_service",
    description=(
        "Complete feature set for sales demand forecasting with PostgreSQL + Ray. "
        "Storage: PostgreSQL (durable, indexed). Compute: Ray (distributed, parallel). "
        "Features: time-series + external + on-demand transformations."
    ),
    features=[
        sales_history_features,      # PostgreSQL: Sales + time-series (lags, rolling)
        store_external_features,     # PostgreSQL: External factors (weather, economy, markdowns)
        feature_transformations,     # Ray: Normalization, interactions (distributed)
        temporal_transformations,    # Ray: Velocity, acceleration, stability (distributed)
    ],
    tags={
        "owner": "retail_analytics_team",
        "use_case": "demand_forecasting",
        "storage_layer": "postgresql",
        "compute_layer": "ray_engine",
        "architecture": "hybrid_storage_compute",
        "performance": "optimized_for_scale",
    },
)

