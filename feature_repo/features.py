"""
Feast Feature Definitions for Sales Forecasting

Entities:
    - store_id: Retail store identifier (1-45)
    - dept_id: Department identifier (1-14)

Feature Views:
    - sales_features: Time-series sales data with lags and rolling statistics
    - store_features: Static store metadata

Feature Services:
    - training_features: All features including target (weekly_sales)
    - inference_features: Features excluding target for real-time predictions
"""
from datetime import timedelta
from feast import Entity, FeatureView, FeatureService, Field, FileSource
from feast.types import Float32, Float64, Int32, Int64, String

# =============================================================================
# Entities
# =============================================================================
store_entity = Entity(
    name="store_id",
    join_keys=["store_id"],
    description="Retail store identifier"
)

dept_entity = Entity(
    name="dept_id",
    join_keys=["dept_id"],
    description="Department identifier within a store"
)

# =============================================================================
# Data Sources
# =============================================================================
sales_source = FileSource(
    name="sales_source",
    path="/shared/data/sales_features.parquet",
    timestamp_field="event_timestamp",
)

store_source = FileSource(
    name="store_source",
    path="/shared/data/store_features.parquet",
    timestamp_field="event_timestamp",
)

# =============================================================================
# Feature Views
# =============================================================================
sales_features = FeatureView(
    name="sales_features",
    entities=[store_entity, dept_entity],
    ttl=timedelta(days=365 * 3),
    schema=[
        # Target variable
        Field(name="weekly_sales", dtype=Float64),
        
        # Lag features (most predictive)
        Field(name="lag_1", dtype=Float64),
        Field(name="lag_2", dtype=Float64),
        Field(name="lag_4", dtype=Float64),
        Field(name="lag_8", dtype=Float64),
        
        # Rolling statistics
        Field(name="rolling_mean_4w", dtype=Float64),
        Field(name="rolling_std_4w", dtype=Float64),
        Field(name="sales_vs_avg", dtype=Float64),
        
        # Temporal features
        Field(name="week_of_year", dtype=Int64),
        Field(name="month", dtype=Int64),
        Field(name="quarter", dtype=Int64),
        Field(name="week_of_month", dtype=Int64),
        Field(name="is_month_end", dtype=Int64),
        Field(name="is_holiday", dtype=Int64),
        Field(name="days_to_holiday", dtype=Int64),
        
        # Economic indicators
        Field(name="temperature", dtype=Float64),
        Field(name="fuel_price", dtype=Float64),
        Field(name="cpi", dtype=Float64),
        Field(name="unemployment", dtype=Float64),
    ],
    source=sales_source,
    online=True,
    tags={"team": "ml-platform", "component": "sales-forecasting"},
)

store_features = FeatureView(
    name="store_features",
    entities=[store_entity, dept_entity],
    ttl=timedelta(days=365 * 10),
    schema=[
        Field(name="store_type", dtype=String),
        Field(name="store_size", dtype=Int64),
        Field(name="region", dtype=String),
    ],
    source=store_source,
    online=True,
    tags={"team": "ml-platform", "component": "sales-forecasting"},
)

# =============================================================================
# Feature Services
# =============================================================================

# Training: All features including target (weekly_sales)
training_features = FeatureService(
    name="training_features",
    features=[
        sales_features,
        store_features,
    ],
    tags={"use_case": "training"},
    description="All features for model training (includes weekly_sales target)"
)

# Inference: Features excluding target for real-time predictions
inference_features = FeatureService(
    name="inference_features",
    features=[
        # Lag and rolling features
        sales_features[["lag_1", "lag_2", "lag_4", "lag_8"]],
        sales_features[["rolling_mean_4w", "rolling_std_4w", "sales_vs_avg"]],
        
        # Temporal features
        sales_features[["week_of_year", "month", "quarter", "week_of_month"]],
        sales_features[["is_month_end", "is_holiday", "days_to_holiday"]],
        
        # Economic features
        sales_features[["temperature", "fuel_price", "cpi", "unemployment"]],
        
        # Store metadata
        store_features,
    ],
    tags={"use_case": "inference"},
    description="Features for inference (excludes weekly_sales target)"
)
