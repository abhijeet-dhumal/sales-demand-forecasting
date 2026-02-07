#!/usr/bin/env python3
"""
Feature Engineering with Feast + Ray Distributed Processing

Usage: python 01_feast_features.py

This script implements the Feast feature store with Ray integration:
1. Create Feast project structure (feature_store.yaml + feature definitions)
2. Generate source data (parquet files)
3. feast apply - register features to PostgreSQL registry
4. feast materialize - distributed materialization via Ray compute engine
5. FeatureStore SDK - query features (historical + online)

Architecture:
- Registry: PostgreSQL (durable, SQL-queryable)
- Offline Store: Ray (distributed file I/O for parquet)
- Batch Engine: Ray (distributed PIT joins, materialization)
- Online Store: PostgreSQL (low-latency serving)

Ray Modes:
- Direct: ray_address=ray://feast-ray-head:10001
- KubeRay: Via CodeFlare SDK with env vars (FEAST_RAY_USE_KUBERAY=true)

Reference: https://docs.feast.dev/reference/offline-stores/ray
"""
import os
import sys
import time
import json
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

NAMESPACE = os.getenv("NAMESPACE", "feast-trainer-demo")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", f"postgres.{NAMESPACE}.svc.cluster.local")
RAY_HEAD_ADDRESS = os.getenv("RAY_HEAD_ADDRESS", f"ray://feast-ray-head.{NAMESPACE}.svc.cluster.local:10001")
USE_KUBERAY = os.getenv("FEAST_RAY_USE_KUBERAY", "false").lower() == "true"

K8S_TOKEN = os.getenv("K8S_TOKEN")
K8S_API_SERVER = os.getenv("K8S_API_SERVER")

DATA_CONFIG = {
    "start_date": "2022-01-01",
    "weeks": 104,           # 2 years of data
    "stores": 10,           # Number of stores
    "departments": 5,       # Departments per store
    "seed": 42
}

# =============================================================================
# FEAST PROJECT FILES
# =============================================================================

# feature_store.yaml - Feast configuration with Ray + PostgreSQL
FEATURE_STORE_YAML = '''# Feast Feature Store - Ray Distributed Processing
# Architecture: PostgreSQL (registry/online) + Ray (offline/compute)
project: sales_forecasting
provider: local

# PostgreSQL registry for feature definitions
registry:
  registry_type: sql
  path: postgresql+psycopg2://feast:feast123@{postgres_host}:5432/feast
  cache_ttl_seconds: 60

# Ray Offline Store - distributed file I/O
# Handles: read/write parquet, pull_latest
offline_store:
  type: ray
  storage_path: /shared/data/ray_storage
  broadcast_join_threshold_mb: 100
  max_parallelism_multiplier: 2
  target_partition_size_mb: 64
  enable_ray_logging: true

# Ray Compute Engine - distributed processing
# Handles: PIT joins, aggregations, materialization
batch_engine:
  type: ray.engine
  ray_address: "{ray_address}"
  max_workers: 4
  max_parallelism_multiplier: 2
  enable_optimization: true
  broadcast_join_threshold_mb: 100
  target_partition_size_mb: 64
  window_size_for_joins: "1H"
  enable_distributed_joins: true
  enable_ray_logging: true

# PostgreSQL online store for low-latency serving
online_store:
  type: postgres
  host: {postgres_host}
  port: 5432
  database: feast
  user: feast
  password: feast123

entity_key_serialization_version: 3
'''

# features.py - Feature definitions
FEATURES_PY = '''"""
Sales Forecasting Feature Definitions

Feast Objects:
- Entities: Business objects (stores, departments)
- FeatureViews: Collections of features with schema
- FeatureServices: Groups of features for serving

Reference: https://docs.feast.dev/getting-started/concepts
"""
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, FeatureService
from feast.types import Float32, Int64, String
from feast.value_type import ValueType

# =============================================================================
# ENTITIES - Primary keys for feature lookups
# =============================================================================

store = Entity(
    name="store_id",
    join_keys=["store_id"],
    value_type=ValueType.INT64,
    description="Retail store identifier"
)

department = Entity(
    name="dept_id", 
    join_keys=["dept_id"],
    value_type=ValueType.INT64,
    description="Department within store"
)

# =============================================================================
# FEATURE VIEWS - Feature collections with schema
# =============================================================================

# Sales features with lag columns for time-series forecasting
sales_features = FeatureView(
    name="sales_features",
    description="Weekly sales metrics with lag features",
    entities=[store, department],
    ttl=timedelta(days=365),
    schema=[
        Field(name="weekly_sales", dtype=Float32, description="Total weekly sales ($)"),
        Field(name="is_holiday", dtype=Int64, description="Holiday week indicator (0/1)"),
        Field(name="temperature", dtype=Float32, description="Average temperature (°F)"),
        Field(name="fuel_price", dtype=Float32, description="Regional fuel price ($)"),
        Field(name="cpi", dtype=Float32, description="Consumer Price Index"),
        Field(name="unemployment", dtype=Float32, description="Unemployment rate (%)"),
        # Lag features for time-series
        Field(name="lag_1", dtype=Float32, description="Sales 1 week ago"),
        Field(name="lag_2", dtype=Float32, description="Sales 2 weeks ago"),
        Field(name="lag_4", dtype=Float32, description="Sales 4 weeks ago"),
        Field(name="lag_8", dtype=Float32, description="Sales 8 weeks ago"),
        Field(name="lag_52", dtype=Float32, description="Sales 52 weeks ago (YoY)"),
        Field(name="rolling_mean_4w", dtype=Float32, description="4-week rolling mean"),
    ],
    source=FileSource(
        path="/shared/data/sales_features.parquet",
        timestamp_field="event_timestamp"
    ),
    online=True
)

# Store static attributes
store_features = FeatureView(
    name="store_features",
    description="Static store attributes",
    entities=[store],
    ttl=timedelta(days=365),
    schema=[
        Field(name="store_type", dtype=String, description="Store type (A=large, B=medium, C=small)"),
        Field(name="store_size", dtype=Int64, description="Store size (sq ft)"),
        Field(name="region", dtype=String, description="Geographic region")
    ],
    source=FileSource(
        path="/shared/data/store_features.parquet",
        timestamp_field="event_timestamp"
    ),
    online=True
)

# =============================================================================
# FEATURE SERVICES - Groups of features for serving
# =============================================================================

# Training feature service - all features for model training
training_features = FeatureService(
    name="training_features",
    description="All features for model training (historical retrieval)",
    features=[
        sales_features[["weekly_sales", "lag_1", "lag_2", "lag_4", "lag_8", "lag_52", 
                       "rolling_mean_4w", "temperature", "fuel_price", "cpi", "unemployment"]],
        store_features[["store_type", "store_size"]]
    ]
)

# Inference feature service - subset for real-time predictions
inference_features = FeatureService(
    name="inference_features",
    description="Features for real-time inference (online lookup)",
    features=[
        sales_features[["lag_1", "lag_2", "lag_4", "rolling_mean_4w", 
                       "temperature", "fuel_price", "cpi", "unemployment"]],
        store_features[["store_size"]]
    ]
)
'''

# data_generator.py - Generate sample data
DATA_GENERATOR_PY = '''"""Generate sample sales data for Feast demo"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

def generate_sales_data(config: dict, data_dir: str):
    """Generate realistic sales data with seasonal patterns and lag features"""
    np.random.seed(config.get("seed", 42))
    base_date = datetime.fromisoformat(config["start_date"]).replace(tzinfo=timezone.utc)
    
    records = []
    for week in range(config["weeks"]):
        week_date = base_date + timedelta(weeks=week)
        week_of_year = week % 52
        
        # Realistic patterns
        seasonal = 1 + 0.3 * np.sin(2 * np.pi * week_of_year / 52)  # Seasonality
        holiday = 1.5 if 47 <= week_of_year <= 52 else 1.0           # Holiday boost
        
        for store_id in range(1, config["stores"] + 1):
            store_base = 50000 + store_id * 5000  # Store-specific baseline
            
            for dept_id in range(1, config["departments"] + 1):
                dept_factor = 0.5 + dept_id * 0.2  # Department factor
                
                records.append({
                    "store_id": store_id,
                    "dept_id": dept_id,
                    "event_timestamp": week_date,
                    "weekly_sales": round(max(0, store_base * dept_factor * seasonal * holiday 
                                              + np.random.normal(0, 2000)), 2),
                    "is_holiday": int(holiday > 1),
                    "temperature": round(60 + 20 * np.sin(2 * np.pi * week_of_year / 52) 
                                        + np.random.normal(0, 5), 1),
                    "fuel_price": round(3 + 0.5 * np.random.random(), 2),
                    "cpi": round(220 + week * 0.1, 1),
                    "unemployment": round(5 + np.random.normal(0, 0.5), 1)
                })
    
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame and add lag features
    sales_df = pd.DataFrame(records)
    sales_df = sales_df.sort_values(["store_id", "dept_id", "event_timestamp"])
    
    # Add lag features (critical for time-series forecasting)
    for lag in [1, 2, 4, 8, 52]:
        sales_df[f"lag_{lag}"] = sales_df.groupby(["store_id", "dept_id"])["weekly_sales"].shift(lag)
    
    # Rolling mean
    sales_df["rolling_mean_4w"] = sales_df.groupby(["store_id", "dept_id"])["weekly_sales"].transform(
        lambda x: x.rolling(4, min_periods=1).mean()
    )
    
    # Fill NaN values from lag features with forward fill then 0
    sales_df = sales_df.fillna(0)
    
    # Save sales features (for Feast FeatureView)
    sales_df.to_parquet(f"{data_dir}/sales_features.parquet", index=False)
    
    # Also save as training data (with target column renamed)
    training_df = sales_df.copy()
    training_df = training_df.rename(columns={"event_timestamp": "date"})
    training_df.to_parquet(f"{data_dir}/features.parquet", index=False)
    
    # Store features (static)
    stores_df = pd.DataFrame([{
        "store_id": i,
        "event_timestamp": base_date,
        "store_type": ["A", "B", "C"][i % 3],
        "store_size": 100000 + i * 10000,
        "region": f"region_{(i - 1) // 3 + 1}"
    } for i in range(1, config["stores"] + 1)])
    stores_df.to_parquet(f"{data_dir}/store_features.parquet", index=False)
    
    # Create entity file for training (store_id, dept_id, date)
    entities_df = sales_df[["store_id", "dept_id", "event_timestamp"]].copy()
    entities_df = entities_df.rename(columns={"event_timestamp": "date"})
    entities_df.to_parquet(f"{data_dir}/entities.parquet", index=False)
    
    print(f"Generated {len(sales_df):,} sales records")
    print(f"Generated {len(stores_df)} store records")
    print(f"Files saved to: {data_dir}")
    print(f"  - sales_features.parquet (Feast)")
    print(f"  - store_features.parquet (Feast)")
    print(f"  - features.parquet (Training)")
    print(f"  - entities.parquet (Entity DataFrame)")
    
    return len(sales_df), len(stores_df)

if __name__ == "__main__":
    import json
    config = json.loads(os.getenv("DATA_CONFIG", "{}"))
    data_dir = os.getenv("DATA_DIR", "/shared/data")
    generate_sales_data(config, data_dir)
'''

# run.sh - Main execution script
RUN_SCRIPT = '''#!/bin/bash
set -e

echo "============================================================"
echo "🍕 FEAST + RAY DISTRIBUTED FEATURE ENGINEERING"
echo "============================================================"

DATA_DIR="${DATA_DIR:-/shared/data}"
FEATURE_REPO="${FEATURE_REPO_DIR:-/shared/feature_repo}"

# Show Ray configuration
echo ""
echo "📋 Ray Configuration:"
echo "   RAY_ADDRESS: ${RAY_ADDRESS:-not set}"
echo "   FEAST_RAY_USE_KUBERAY: ${FEAST_RAY_USE_KUBERAY:-false}"
if [ "$FEAST_RAY_USE_KUBERAY" = "true" ]; then
    echo "   FEAST_RAY_CLUSTER_NAME: ${FEAST_RAY_CLUSTER_NAME:-not set}"
    echo "   FEAST_RAY_NAMESPACE: ${FEAST_RAY_NAMESPACE:-not set}"
fi

# Step 1: Generate data
echo ""
echo "📊 Step 1: Generate Sales Data"
echo "============================================================"
python /scripts/data_generator.py
echo "   ✅ Data saved to $DATA_DIR"

# Step 2: Setup Feast project
echo ""
echo "⚙️  Step 2: Setup Feast Project"
echo "============================================================"
mkdir -p "$FEATURE_REPO"
cp /scripts/feature_store.yaml "$FEATURE_REPO/"
cp /scripts/features.py "$FEATURE_REPO/"
echo "   ✅ Project: $FEATURE_REPO"
ls -la "$FEATURE_REPO"

# Create ray_storage directory
mkdir -p "$DATA_DIR/ray_storage"

# Step 3: Apply features (register to registry)
echo ""
echo "📝 Step 3: feast apply (register features)"
echo "============================================================"
cd "$FEATURE_REPO"
feast apply
echo "   ✅ Features registered to PostgreSQL registry"

# Step 4: Materialize to online store (uses Ray compute engine!)
echo ""
echo "🚀 Step 4: feast materialize (Ray Distributed)"
echo "============================================================"
echo "   This uses Ray compute engine for distributed processing!"
echo "   Check Ray Dashboard for job visibility."
feast materialize 2022-01-01T00:00:00 2024-01-01T00:00:00
echo "   ✅ Features materialized to PostgreSQL online store"

# Step 5: Test historical features with Ray (optional)
echo ""
echo "🔍 Step 5: Test Historical Feature Retrieval (Ray)"
echo "============================================================"
python - << 'EOF'
from feast import FeatureStore
import pandas as pd
from datetime import datetime, timezone, timedelta
import time

store = FeatureStore(repo_path=".")

print("   Entities:", [e.name for e in store.list_entities()])
print("   FeatureViews:", [fv.name for fv in store.list_feature_views()])
print("   FeatureServices:", [fs.name for fs in store.list_feature_services()])

# Test online lookup
print("\\n   Testing online lookup...")
online = store.get_online_features(
    features=[
        "sales_features:weekly_sales",
        "sales_features:lag_1",
        "store_features:store_size"
    ],
    entity_rows=[{"store_id": 1, "dept_id": 1}]
).to_dict()
print(f"   Online: store=1, dept=1 -> weekly_sales=${online['weekly_sales'][0]:,.0f}")

# Test historical features (uses Ray compute engine for PIT joins!)
print("\\n   Testing historical retrieval (Ray distributed)...")
entity_df = pd.DataFrame({
    "store_id": [1, 2, 3] * 10,
    "dept_id": [1, 2, 3, 4, 5] * 6,
    "event_timestamp": [datetime.now(timezone.utc) - timedelta(days=i) for i in range(30)]
})

start = time.time()
historical = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "sales_features:weekly_sales",
        "sales_features:lag_1",
        "sales_features:rolling_mean_4w",
        "store_features:store_size"
    ]
).to_df()
elapsed = time.time() - start

print(f"   Historical: Retrieved {len(historical)} rows in {elapsed:.2f}s")
print(f"   Columns: {list(historical.columns)}")
print("   ✅ Ray compute engine working!")
EOF

echo ""
echo "============================================================"
echo "✅ FEAST + RAY FEATURE ENGINEERING COMPLETE!"
echo "============================================================"
echo "   📁 Data: $DATA_DIR"
echo "   📝 Features: $FEATURE_REPO"
echo "   🔗 Registry: PostgreSQL (SQL)"
echo "   ⚡ Offline Store: Ray (distributed)"
echo "   🚀 Compute Engine: Ray (distributed)"
echo "   💾 Online Store: PostgreSQL (low-latency)"
'''

# =============================================================================
# MAIN: Submit Job to Cluster
# =============================================================================

if __name__ == "__main__":
    from kubernetes import client as k8s
    
    print(f"{'='*70}")
    print("🍕 FEAST + RAY DISTRIBUTED FEATURE ENGINEERING")
    print(f"{'='*70}")
    print("\n📋 Architecture:")
    print("   Registry:     PostgreSQL (SQL-based, durable)")
    print("   Offline Store: Ray (distributed file I/O)")
    print("   Compute:      Ray (distributed PIT joins, materialization)")
    print("   Online Store: PostgreSQL (low-latency serving)")
    print("\n📋 Feast CLI Commands:")
    print("   feast apply       - Register features to PostgreSQL registry")
    print("   feast materialize - Distributed materialization via Ray")
    
    # Auth
    if K8S_TOKEN and K8S_API_SERVER:
        cfg = k8s.Configuration()
        cfg.host, cfg.verify_ssl = K8S_API_SERVER, False
        cfg.api_key = {"authorization": f"Bearer {K8S_TOKEN}"}
        k8s.Configuration.set_default(cfg)
    else:
        from kubernetes import config
        config.load_kube_config()
    
    batch, core = k8s.BatchV1Api(), k8s.CoreV1Api()
    job_id = datetime.now().strftime('%m%d-%H%M')
    job_name = f"feast-ray-{job_id}"
    
    # Create ConfigMap with all files
    print(f"\n📦 Creating ConfigMap...")
    try:
        core.delete_namespaced_config_map("feast-ray-scripts", NAMESPACE)
    except:
        pass
    
    core.create_namespaced_config_map(NAMESPACE, k8s.V1ConfigMap(
        metadata=k8s.V1ObjectMeta(
            name="feast-ray-scripts", 
            labels={"app": "sales-forecasting", "job-type": "feast-ray"}
        ),
        data={
            "feature_store.yaml": FEATURE_STORE_YAML.format(
                postgres_host=POSTGRES_HOST,
                ray_address=RAY_HEAD_ADDRESS
            ),
            "features.py": FEATURES_PY,
            "data_generator.py": DATA_GENERATOR_PY,
            "run.sh": RUN_SCRIPT
        }
    ))
    
    # Environment variables for the job
    env_vars = [
        {"name": "POSTGRES_HOST", "value": POSTGRES_HOST},
        {"name": "DATA_DIR", "value": "/shared/data"},
        {"name": "FEATURE_REPO_DIR", "value": "/shared/feature_repo"},
        {"name": "DATA_CONFIG", "value": json.dumps(DATA_CONFIG)},
        {"name": "RAY_ADDRESS", "value": RAY_HEAD_ADDRESS},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
    ]
    
    # Add KubeRay env vars if enabled
    if USE_KUBERAY:
        env_vars.extend([
            {"name": "FEAST_RAY_USE_KUBERAY", "value": "true"},
            {"name": "FEAST_RAY_CLUSTER_NAME", "value": "feast-ray"},
            {"name": "FEAST_RAY_NAMESPACE", "value": NAMESPACE},
            {"name": "FEAST_RAY_SKIP_TLS", "value": "true"},
            {"name": "FEAST_RAY_AUTH_SERVER", "value": "https://kubernetes.default.svc"},
        ])
    
    # Submit Job
    print(f"🚀 Submitting Job: {job_name}")
    print(f"   Ray Address: {RAY_HEAD_ADDRESS}")
    print(f"   KubeRay Mode: {USE_KUBERAY}")
    
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": NAMESPACE,
            "labels": {"app": "sales-forecasting", "job-type": "feast-ray", "run-id": job_id}
        },
        "spec": {
            "backoffLimit": 2,
            "ttlSecondsAfterFinished": 3600,
            "template": {
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [{
                        "name": "feast-ray",
                        "image": "quay.io/modh/ray:2.35.0-py311-cu121",  # Match RayCluster version
                        "command": ["/bin/bash", "-c",
                            "pip install -q --target=/tmp/pylibs feast[postgres,ray]==0.59.0 psycopg2-binary && "
                            "export PYTHONPATH=/tmp/pylibs:$PYTHONPATH && "
                            "export PATH=/tmp/pylibs/bin:$PATH && "
                            "bash /scripts/run.sh"
                        ],
                        "env": env_vars,
                        "volumeMounts": [
                            {"name": "shared", "mountPath": "/shared"},
                            {"name": "scripts", "mountPath": "/scripts"}
                        ],
                        "resources": {
                            "requests": {"cpu": "2", "memory": "4Gi"},
                            "limits": {"cpu": "4", "memory": "8Gi"}
                        }
                    }],
                    "volumes": [
                        {"name": "shared", "persistentVolumeClaim": {"claimName": "feast-pvc"}},
                        {"name": "scripts", "configMap": {"name": "feast-ray-scripts"}}
                    ]
                }
            }
        }
    }
    batch.create_namespaced_job(NAMESPACE, job)
    
    # Wait for completion
    print("⏳ Waiting for completion...")
    for i in range(180):  # 15 minutes timeout
        j = batch.read_namespaced_job(job_name, NAMESPACE)
        if j.status.succeeded:
            print(f"\n✅ Job '{job_name}' completed!")
            print(f"   View logs: kubectl logs -n {NAMESPACE} job/{job_name}")
            sys.exit(0)
        if j.status.failed:
            print(f"\n❌ Job '{job_name}' failed!")
            print(f"   Debug: kubectl logs -n {NAMESPACE} job/{job_name}")
            sys.exit(1)
        if i % 12 == 0:
            print(f"   Running... ({i*5}s)")
        time.sleep(5)
    
    print("⏰ Timeout")
    sys.exit(1)
