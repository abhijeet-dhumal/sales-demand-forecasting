#!/usr/bin/env python3
"""
Feature Engineering with Feast CLI Pattern
Usage: python 01_feast_features.py

Following the OpenDataHub Feast quickstart pattern:
1. Create Feast project structure (feature_store.yaml + feature definitions)
2. Generate source data
3. feast apply - register features
4. feast materialize - push to online store
5. FeatureStore SDK - query features

Reference: https://github.com/opendatahub-io/feast/blob/master/examples/rhoai-quickstart
"""
import os, sys, time, json
from datetime import datetime

# Config
NAMESPACE = os.getenv("NAMESPACE", "feast-trainer-demo")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", f"feast-postgres.{NAMESPACE}.svc.cluster.local")
K8S_TOKEN = os.getenv("K8S_TOKEN")
K8S_API_SERVER = os.getenv("K8S_API_SERVER")

DATA_CONFIG = {
    "start_date": "2022-01-01",
    "weeks": 104,
    "stores": 10,
    "departments": 5,
    "seed": 42
}

# =============================================================================
# FEAST PROJECT FILES (created in cluster)
# =============================================================================

# feature_store.yaml - Feast configuration
FEATURE_STORE_YAML = '''project: sales_forecasting
provider: local

registry:
  registry_type: sql
  path: postgresql+psycopg2://feast:feast123@{postgres_host}:5432/feast

offline_store:
  type: file

online_store:
  type: postgres
  host: {postgres_host}
  port: 5432
  database: feast
  user: feast
  password: feast123

entity_key_serialization_version: 3
'''

# features.py - Feature definitions (like example_repo.py in Feast quickstart)
FEATURES_PY = '''"""
Sales Forecasting Feature Definitions

This file defines Feast objects:
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
    description="Retail store identifier (1-45)"
)

department = Entity(
    name="dept_id", 
    join_keys=["dept_id"],
    value_type=ValueType.INT64,
    description="Department within store (1-99)"
)

# =============================================================================
# FEATURE VIEWS - Feature collections with schema
# =============================================================================

sales_features = FeatureView(
    name="sales_features",
    description="Weekly sales metrics by store and department",
    entities=[store, department],
    ttl=timedelta(days=365),
    schema=[
        Field(name="weekly_sales", dtype=Float32, description="Total weekly sales ($)"),
        Field(name="is_holiday", dtype=Int64, description="Holiday week indicator (0/1)"),
        Field(name="temperature", dtype=Float32, description="Average temperature (°F)"),
        Field(name="fuel_price", dtype=Float32, description="Regional fuel price ($)"),
        Field(name="cpi", dtype=Float32, description="Consumer Price Index"),
        Field(name="unemployment", dtype=Float32, description="Unemployment rate (%)")
    ],
    source=FileSource(
        path="/shared/data/sales_features.parquet",
        timestamp_field="event_timestamp"
    ),
    online=True
)

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

sales_prediction_service = FeatureService(
    name="sales_prediction",
    description="Features for sales prediction model",
    features=[
        sales_features[["weekly_sales", "temperature", "fuel_price", "cpi", "unemployment"]],
        store_features[["store_type", "store_size"]]
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
    """Generate realistic sales data with seasonal patterns"""
    np.random.seed(config.get("seed", 42))
    base_date = datetime.fromisoformat(config["start_date"]).replace(tzinfo=timezone.utc)
    
    records = []
    for week in range(config["weeks"]):
        week_date = base_date + timedelta(weeks=week)
        week_of_year = week % 52
        
        # Realistic patterns
        seasonal = 1 + 0.3 * np.sin(2 * np.pi * week_of_year / 52)
        holiday = 1.5 if 47 <= week_of_year <= 52 else 1.0
        
        for store_id in range(1, config["stores"] + 1):
            store_base = 50000 + store_id * 5000
            
            for dept_id in range(1, config["departments"] + 1):
                dept_factor = 0.5 + dept_id * 0.2
                
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
    
    # Sales features
    sales_df = pd.DataFrame(records)
    sales_df.to_parquet(f"{data_dir}/sales_features.parquet", index=False)
    
    # Add lag features for training
    sales_df = sales_df.sort_values(["store_id", "dept_id", "event_timestamp"])
    for lag in [1, 2, 4, 8, 52]:
        sales_df[f"lag_{lag}"] = sales_df.groupby(["store_id", "dept_id"])["weekly_sales"].shift(lag)
    sales_df["rolling_mean_4w"] = sales_df.groupby(["store_id", "dept_id"])["weekly_sales"].transform(
        lambda x: x.rolling(4, min_periods=1).mean())
    sales_df.to_parquet(f"{data_dir}/features.parquet", index=False)
    
    # Store features
    stores_df = pd.DataFrame([{
        "store_id": i,
        "event_timestamp": base_date,
        "store_type": ["A", "B", "C"][i % 3],
        "store_size": 100000 + i * 10000,
        "region": f"region_{(i - 1) // 3 + 1}"
    } for i in range(1, config["stores"] + 1)])
    stores_df.to_parquet(f"{data_dir}/store_features.parquet", index=False)
    
    return len(sales_df), len(stores_df)

if __name__ == "__main__":
    import json
    config = json.loads(os.getenv("DATA_CONFIG", "{}"))
    data_dir = os.getenv("DATA_DIR", "/shared/data")
    sales, stores = generate_sales_data(config, data_dir)
    print(f"Generated {sales:,} sales records, {stores} stores")
'''

# run.sh - Main execution script
RUN_SCRIPT = '''#!/bin/bash
set -e

echo "============================================================"
echo "🍕 Feast Feature Engineering"
echo "============================================================"

DATA_DIR="${DATA_DIR:-/shared/data}"
FEATURE_REPO="${FEATURE_REPO_DIR:-/shared/feature_repo}"

# Step 1: Generate data
echo ""
echo "📊 Step 1: Generate Sales Data"
python /scripts/data_generator.py
echo "   ✅ Data saved to $DATA_DIR"

# Step 2: Setup Feast project
echo ""
echo "⚙️  Step 2: Setup Feast Project"
mkdir -p "$FEATURE_REPO"
cp /scripts/feature_store.yaml "$FEATURE_REPO/"
cp /scripts/features.py "$FEATURE_REPO/"
echo "   ✅ Project: $FEATURE_REPO"
ls -la "$FEATURE_REPO"

# Step 3: Apply features (register to registry)
echo ""
echo "📝 Step 3: feast apply (register features)"
cd "$FEATURE_REPO"
feast apply
echo "   ✅ Features registered to PostgreSQL registry"

# Step 4: Materialize to online store
echo ""
echo "🚀 Step 4: feast materialize (push to online store)"
feast materialize 2022-01-01T00:00:00 2024-01-01T00:00:00
echo "   ✅ Features materialized to online store"

# Step 5: Verify with SDK
echo ""
echo "🔍 Step 5: Verify with FeatureStore SDK"
python - << 'EOF'
from feast import FeatureStore
import pandas as pd
from datetime import datetime, timezone

store = FeatureStore(repo_path=".")

# List registered features
print("   Entities:", [e.name for e in store.list_entities()])
print("   FeatureViews:", [fv.name for fv in store.list_feature_views()])
print("   FeatureServices:", [fs.name for fs in store.list_feature_services()])

# Test online lookup
online = store.get_online_features(
    features=["sales_features:weekly_sales", "store_features:store_size"],
    entity_rows=[{"store_id": 1, "dept_id": 1}]
).to_dict()
print(f"   Online lookup: store_1/dept_1 = ${online['weekly_sales'][0]:,.0f}")
EOF

echo ""
echo "============================================================"
echo "✅ Feast Feature Engineering Complete!"
echo "============================================================"
echo "   📁 Data: $DATA_DIR"
echo "   📝 Features: $FEATURE_REPO"
echo "   🔗 Registry: PostgreSQL"
'''


# =============================================================================
# MAIN: Submit Job to Cluster
# =============================================================================

if __name__ == "__main__":
    from kubernetes import client as k8s
    
    print(f"{'='*60}")
    print("🍕 Feast Feature Engineering")
    print(f"{'='*60}")
    print("\n📋 Feast Project Structure:")
    print("   feature_store.yaml  - Registry & store config")
    print("   features.py         - Entity, FeatureView, FeatureService definitions")
    print("   data_generator.py   - Sample data generation")
    print("\n📋 Feast CLI Commands:")
    print("   feast apply         - Register features to registry")
    print("   feast materialize   - Push features to online store")

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
    job_name = f"feast-features-{job_id}"
    
    # ConfigMap with all files
    print(f"\n📦 Creating ConfigMap...")
    try: core.delete_namespaced_config_map("feast-scripts", NAMESPACE)
    except: pass
    
    core.create_namespaced_config_map(NAMESPACE, k8s.V1ConfigMap(
        metadata=k8s.V1ObjectMeta(name="feast-scripts", labels={"app": "sales-forecasting", "job-type": "feast"}),
        data={
            "feature_store.yaml": FEATURE_STORE_YAML.format(postgres_host=POSTGRES_HOST),
            "features.py": FEATURES_PY,
            "data_generator.py": DATA_GENERATOR_PY,
            "run.sh": RUN_SCRIPT
        }
    ))
    
    # Submit Job
    print(f"🚀 Submitting Job: {job_name}")
    job = {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": job_name, "namespace": NAMESPACE, 
                    "labels": {"app": "sales-forecasting", "job-type": "feast", "run-id": job_id}},
        "spec": {"backoffLimit": 0, "ttlSecondsAfterFinished": 3600,
            "template": {"spec": {"restartPolicy": "Never",
                "containers": [{
                    "name": "feast", 
                    "image": "quay.io/modh/ray:2.52.1-py312-cu128",
                    "command": ["/bin/bash", "-c", 
                               "pip install -q --target=/tmp/pylibs feast[postgres] psycopg2-binary && "
                               "export PYTHONPATH=/tmp/pylibs:$PYTHONPATH && "
                               "export PATH=/tmp/pylibs/bin:$PATH && "
                               "bash /scripts/run.sh"],
                    "env": [
                        {"name": "POSTGRES_HOST", "value": POSTGRES_HOST},
                        {"name": "DATA_DIR", "value": "/shared/data"},
                        {"name": "FEATURE_REPO_DIR", "value": "/shared/feature_repo"},
                        {"name": "DATA_CONFIG", "value": json.dumps(DATA_CONFIG)}
                    ],
                    "volumeMounts": [
                        {"name": "shared", "mountPath": "/shared"},
                        {"name": "scripts", "mountPath": "/scripts"}
                    ],
                    "resources": {"requests": {"cpu": "1", "memory": "2Gi"}, "limits": {"cpu": "2", "memory": "4Gi"}}
                }],
                "volumes": [
                    {"name": "shared", "persistentVolumeClaim": {"claimName": "feast-pvc"}},
                    {"name": "scripts", "configMap": {"name": "feast-scripts"}}
                ]
            }}
        }
    }
    batch.create_namespaced_job(NAMESPACE, job)
    
    # Wait for completion
    print("⏳ Waiting for completion...")
    for i in range(120):
        j = batch.read_namespaced_job(job_name, NAMESPACE)
        if j.status.succeeded:
            print(f"\n✅ Job '{job_name}' completed!")
            print(f"   View logs: kubectl logs -n {NAMESPACE} job/{job_name}")
            sys.exit(0)
        if j.status.failed:
            print(f"\n❌ Job '{job_name}' failed!")
            print(f"   Debug: kubectl logs -n {NAMESPACE} job/{job_name}")
            sys.exit(1)
        if i % 6 == 0: print(f"   Running... ({i*5}s)")
        time.sleep(5)
    
    print("⏰ Timeout"); sys.exit(1)
