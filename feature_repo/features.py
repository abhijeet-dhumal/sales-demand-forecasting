"""
Sales Demand Forecasting Feature Definitions

This module defines Feast feature store resources for the Walmart-style sales forecasting use case.

Architecture:
- Registry: PostgreSQL (SQL-queryable)
- Offline Store: File-based (Parquet) for apply/materialize
- Online Store: PostgreSQL (low-latency serving)
- Compute Engine: Ray (distributed get_historical_features)

Entities:
- store_id: Store identifier (1-45)
- dept_id: Department identifier (1-10)

Feature Views:
- sales_features: Weekly sales with lag features, temporal, and economic indicators
- store_features: Static store metadata (type, size, region)

Feature Services:
- training_features: Full feature set for model training (includes target)
- inference_features: Features for inference (excludes target)

Feature Importance (typical retail forecasting):
- Lag_1, Lag_2        : 35% - Most predictive (recent history)
- Rolling stats       : 28% - Trend and volatility
- Week_of_year, Month : 18% - Seasonality patterns
- Is_holiday          : 10% - Holiday effects
- Economic indicators :  7% - External factors
- Store features      :  2% - Store context
"""

from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, FeatureService
from feast.types import Float32, Int32, String


# =============================================================================
# ENTITIES
# =============================================================================

store_entity = Entity(
    name="store_id",
    description="Store identifier (1-45)",
    join_keys=["store_id"],
    tags={"team": "sales", "priority": "high"},
)

dept_entity = Entity(
    name="dept_id",
    description="Department identifier (1-10)",
    join_keys=["dept_id"],
    tags={"team": "sales"},
)


# =============================================================================
# DATA SOURCES (Parquet files on shared PVC - pre-sorted for fast PIT joins)
# =============================================================================
# NOTE: Path depends on mount point:
#   - RHOAI Workbench: /opt/app-root/src/shared/data/
#   - TrainJob/Pods:   /shared/data/

import os
# Auto-detect mount path: /shared (TrainJob/Ray) vs /opt/app-root/src/shared (Workbench)
def _get_data_root():
    if os.environ.get("FEAST_DATA_ROOT"):
        return os.environ["FEAST_DATA_ROOT"]
    if os.path.exists("/shared/data"):
        return "/shared/data"
    return "/opt/app-root/src/shared/data"
_DATA_ROOT = _get_data_root()

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
    description="Weekly sales metrics with optimized features for accurate forecasting",
    entities=[store_entity, dept_entity],
    ttl=timedelta(days=90),
    schema=[
        # Target variable
        Field(name="weekly_sales", dtype=Float32, description="Weekly sales amount ($)"),
        
        # === LAG FEATURES (35% importance) - Recent history ===
        Field(name="lag_1", dtype=Float32, description="Sales from 1 week ago"),
        Field(name="lag_2", dtype=Float32, description="Sales from 2 weeks ago"),
        Field(name="lag_4", dtype=Float32, description="Sales from 4 weeks ago"),
        Field(name="lag_8", dtype=Float32, description="Sales from 8 weeks ago"),
        
        # === ROLLING STATISTICS (28% importance) - Trend & volatility ===
        Field(name="rolling_mean_4w", dtype=Float32, description="4-week rolling average"),
        Field(name="rolling_std_4w", dtype=Float32, description="4-week rolling std (volatility)"),
        Field(name="sales_vs_avg", dtype=Float32, description="Current/rolling_mean ratio (momentum)"),
        
        # === TEMPORAL FEATURES (18% importance) - Seasonality ===
        Field(name="week_of_year", dtype=Int32, description="Week 1-52 (yearly seasonality)"),
        Field(name="month", dtype=Int32, description="Month 1-12"),
        Field(name="quarter", dtype=Int32, description="Quarter 1-4 (Q4 retail boost)"),
        Field(name="week_of_month", dtype=Int32, description="Week 1-5 within month"),
        Field(name="is_month_end", dtype=Int32, description="Last week of month (0/1)"),
        
        # === HOLIDAY FEATURES (10% importance) ===
        Field(name="is_holiday", dtype=Int32, description="Holiday week indicator (0/1)"),
        Field(name="days_to_holiday", dtype=Int32, description="Days until next major holiday"),
        
        # === ECONOMIC INDICATORS (7% importance) ===
        Field(name="temperature", dtype=Float32, description="Regional temperature (°F)"),
        Field(name="fuel_price", dtype=Float32, description="Fuel price ($/gallon)"),
        Field(name="cpi", dtype=Float32, description="Consumer Price Index"),
        Field(name="unemployment", dtype=Float32, description="Unemployment rate (%)"),
    ],
    source=sales_source,
    online=True,
    tags={"type": "time_series", "domain": "sales"},
)

store_features = FeatureView(
    name="store_features",
    description="Static store metadata",
    entities=[store_entity, dept_entity],
    ttl=timedelta(days=365),
    schema=[
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

# Training: includes target variable (weekly_sales)
# Optimized feature set for maximum prediction accuracy
training_features = FeatureService(
    name="training_features",
    description="Optimized feature set for model training (includes target)",
    features=[
        sales_features[[
            # Target
            "weekly_sales",
            # Lag features (35% importance)
            "lag_1", "lag_2", "lag_4", "lag_8",
            # Rolling stats (28% importance)
            "rolling_mean_4w", "rolling_std_4w", "sales_vs_avg",
            # Temporal (18% importance)
            "week_of_year", "month", "quarter", "week_of_month", "is_month_end",
            # Holiday (10% importance)
            "is_holiday", "days_to_holiday",
            # Economic (7% importance)
            "temperature", "fuel_price", "cpi", "unemployment",
        ]],
        store_features[["store_type", "store_size", "region"]],
    ],
    tags={"stage": "training", "model": "demand-forecasting", "version": "v2-optimized"},
)

# Inference: excludes target variable
inference_features = FeatureService(
    name="inference_features",
    description="Features for real-time inference (excludes target)",
    features=[
        sales_features[[
            # Lag features
            "lag_1", "lag_2", "lag_4", "lag_8",
            # Rolling stats
            "rolling_mean_4w", "rolling_std_4w", "sales_vs_avg",
            # Temporal
            "week_of_year", "month", "quarter", "week_of_month", "is_month_end",
            # Holiday
            "is_holiday", "days_to_holiday",
            # Economic
            "temperature", "fuel_price", "cpi", "unemployment",
        ]],
        store_features[["store_type", "store_size", "region"]],
    ],
    tags={"stage": "inference", "model": "demand-forecasting", "version": "v2-optimized"},
)
