"""
Feast Feature Definitions for Sales Demand Forecasting

This module defines:
- Entities: store, dept
- Feature Views: sales_features, store_features
- Feature Service: sales_forecasting_features
"""

from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, FeatureService
from feast.types import Float64, Int64, String

# =============================================================================
# ENTITIES
# =============================================================================

store = Entity(
    name="store_id",
    join_keys=["store_id"],
    description="Retail store identifier",
)

dept = Entity(
    name="dept_id", 
    join_keys=["dept_id"],
    description="Department within a store",
)

# =============================================================================
# DATA SOURCES (Parquet files on shared storage)
# =============================================================================

# Sales history with lag features
sales_source = FileSource(
    name="sales_source",
    path="/shared/data/sales_features.parquet",
    timestamp_field="date",
    created_timestamp_column="created_timestamp",
)

# Store attributes and external factors
store_source = FileSource(
    name="store_source", 
    path="/shared/data/store_features.parquet",
    timestamp_field="date",
    created_timestamp_column="created_timestamp",
)

# =============================================================================
# FEATURE VIEWS
# =============================================================================

sales_features = FeatureView(
    name="sales_features",
    entities=[store, dept],
    ttl=timedelta(days=365 * 3),  # 3 years
    schema=[
        # Lag features (historical sales - safe, no leakage)
        Field(name="lag_1", dtype=Float64, description="Sales 1 week ago"),
        Field(name="lag_2", dtype=Float64, description="Sales 2 weeks ago"),
        Field(name="lag_4", dtype=Float64, description="Sales 4 weeks ago"),
        Field(name="lag_8", dtype=Float64, description="Sales 8 weeks ago"),
        Field(name="lag_52", dtype=Float64, description="Sales 52 weeks ago (YoY)"),
        
        # Rolling statistics (computed from past data only)
        Field(name="rolling_mean_4w", dtype=Float64, description="4-week rolling mean"),
        Field(name="rolling_std_4w", dtype=Float64, description="4-week rolling std"),
        Field(name="rolling_mean_8w", dtype=Float64, description="8-week rolling mean"),
        Field(name="rolling_std_8w", dtype=Float64, description="8-week rolling std"),
        Field(name="rolling_mean_52w", dtype=Float64, description="52-week rolling mean"),
    ],
    source=sales_source,
    online=True,
    description="Historical sales features with lag and rolling statistics",
)

store_features = FeatureView(
    name="store_features",
    entities=[store],
    ttl=timedelta(days=365 * 3),
    schema=[
        # Store attributes
        Field(name="store_size", dtype=Int64, description="Store size in sq ft"),
        
        # External factors (known ahead of time)
        Field(name="temperature", dtype=Float64, description="Weekly avg temperature"),
        Field(name="fuel_price", dtype=Float64, description="Regional fuel price"),
        Field(name="cpi", dtype=Float64, description="Consumer Price Index"),
        Field(name="unemployment", dtype=Float64, description="Regional unemployment rate"),
        
        # Promotions/Markdowns
        Field(name="markdown1", dtype=Float64, description="Markdown 1 amount"),
        Field(name="markdown2", dtype=Float64, description="Markdown 2 amount"),
        Field(name="markdown3", dtype=Float64, description="Markdown 3 amount"),
        Field(name="markdown4", dtype=Float64, description="Markdown 4 amount"),
        Field(name="markdown5", dtype=Float64, description="Markdown 5 amount"),
        
        # Calendar features
        Field(name="is_holiday", dtype=Int64, description="Holiday week flag"),
        Field(name="week_of_year", dtype=Int64, description="Week number (1-52)"),
        Field(name="month", dtype=Int64, description="Month (1-12)"),
    ],
    source=store_source,
    online=True,
    description="Store attributes and external factors",
)

# =============================================================================
# FEATURE SERVICE (groups features for retrieval)
# =============================================================================

sales_forecasting_service = FeatureService(
    name="sales_forecasting_features",
    features=[
        sales_features,
        store_features,
    ],
    description="All features needed for sales demand forecasting model",
)
