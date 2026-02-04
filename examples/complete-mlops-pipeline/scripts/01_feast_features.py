#!/usr/bin/env python3
"""
Feature Engineering with Feast SDK
Usage: python 01_feast_features.py

This script demonstrates Feast's feature store pattern:
1. Define Entities (business objects like stores, departments)
2. Define FeatureViews (collections of related features)
3. Generate/load source data
4. Apply feature definitions to registry
5. Materialize features to online store
"""
import os, sys, time
from datetime import datetime

# Config
NAMESPACE = os.getenv("NAMESPACE", "feast-trainer-demo")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", f"feast-postgres.{NAMESPACE}.svc.cluster.local")
K8S_TOKEN = os.getenv("K8S_TOKEN")
K8S_API_SERVER = os.getenv("K8S_API_SERVER")

# =============================================================================
# FEATURE DEFINITIONS (declarative schema)
# =============================================================================

FEATURE_PROJECT = "sales_forecasting"

# Entities - the business objects we track features for
ENTITIES = {
    "store": {
        "name": "store_id",
        "description": "Retail store identifier (1-45)",
        "join_keys": ["store_id"]
    },
    "department": {
        "name": "dept_id", 
        "description": "Department within store (1-99)",
        "join_keys": ["dept_id"]
    }
}

# Feature Views - collections of related features
FEATURE_VIEWS = {
    "sales_features": {
        "description": "Weekly sales metrics by store and department",
        "entities": ["store_id", "dept_id"],
        "features": {
            "weekly_sales": {"type": "Float32", "description": "Total weekly sales ($)"},
            "is_holiday": {"type": "Int64", "description": "Holiday week indicator (0/1)"},
            "temperature": {"type": "Float32", "description": "Average temperature (°F)"},
            "fuel_price": {"type": "Float32", "description": "Regional fuel price ($)"},
            "cpi": {"type": "Float32", "description": "Consumer Price Index"},
            "unemployment": {"type": "Float32", "description": "Regional unemployment rate (%)"}
        },
        "source": "sales_features.parquet"
    },
    "store_features": {
        "description": "Static store attributes",
        "entities": ["store_id"],
        "features": {
            "store_type": {"type": "String", "description": "Store type (A=large, B=medium, C=small)"},
            "store_size": {"type": "Int64", "description": "Store size (sq ft)"},
            "region": {"type": "String", "description": "Geographic region"}
        },
        "source": "store_features.parquet"
    }
}

# Data generation config
DATA_CONFIG = {
    "start_date": "2022-01-01",
    "weeks": 104,  # 2 years
    "stores": 10,
    "departments": 5,
    "seed": 42
}


# =============================================================================
# FEAST JOB SCRIPT (runs in cluster)
# =============================================================================

FEAST_JOB_SCRIPT = '''#!/usr/bin/env python3
"""Feast feature engineering job - runs in cluster with access to PVC and Postgres"""
import os, json
import pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "/shared/data"))
FEATURE_REPO = Path(os.getenv("FEATURE_REPO_DIR", "/shared/feature_repo"))
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
CONFIG = json.loads(os.getenv("DATA_CONFIG", "{}"))

print("=" * 60)
print("🍕 Feast Feature Engineering")
print("=" * 60)

# 1. Generate training data with meaningful business logic
print("\\n📊 Step 1: Generate Sales Data")
print(f"   Config: {CONFIG['weeks']} weeks, {CONFIG['stores']} stores, {CONFIG['departments']} depts")

np.random.seed(CONFIG.get("seed", 42))
base_date = datetime.fromisoformat(CONFIG["start_date"]).replace(tzinfo=timezone.utc)
records = []

for week in range(CONFIG["weeks"]):
    week_date = base_date + timedelta(weeks=week)
    week_of_year = week % 52
    
    # Realistic seasonal patterns
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * week_of_year / 52)  # Peak in summer
    holiday_factor = 1.5 if 47 <= week_of_year <= 52 else 1.0  # Q4 holiday boost
    
    for store_id in range(1, CONFIG["stores"] + 1):
        store_base = 50000 + store_id * 5000  # Larger stores = more sales
        
        for dept_id in range(1, CONFIG["departments"] + 1):
            dept_factor = 0.5 + dept_id * 0.2  # Different dept sizes
            
            weekly_sales = max(0, store_base * dept_factor * seasonal_factor * holiday_factor 
                              + np.random.normal(0, 2000))
            
            records.append({
                "store_id": store_id,
                "dept_id": dept_id,
                "event_timestamp": week_date,
                "weekly_sales": round(weekly_sales, 2),
                "is_holiday": int(holiday_factor > 1),
                "temperature": round(60 + 20 * np.sin(2 * np.pi * week_of_year / 52) + np.random.normal(0, 5), 1),
                "fuel_price": round(3 + 0.5 * np.random.random(), 2),
                "cpi": round(220 + week * 0.1, 1),
                "unemployment": round(5 + np.random.normal(0, 0.5), 1)
            })

DATA_DIR.mkdir(parents=True, exist_ok=True)
sales_df = pd.DataFrame(records)
sales_df.to_parquet(DATA_DIR / "sales_features.parquet", index=False)
print(f"   ✅ Generated {len(sales_df):,} sales records")

# Add lag features for training
print("\\n📈 Step 2: Compute Lag Features")
sales_df = sales_df.sort_values(["store_id", "dept_id", "event_timestamp"])
for lag in [1, 2, 4, 8, 52]:
    sales_df[f"lag_{lag}"] = sales_df.groupby(["store_id", "dept_id"])["weekly_sales"].shift(lag)
sales_df["rolling_mean_4w"] = sales_df.groupby(["store_id", "dept_id"])["weekly_sales"].transform(
    lambda x: x.rolling(4, min_periods=1).mean())
sales_df.to_parquet(DATA_DIR / "features.parquet", index=False)
print(f"   ✅ Added lag features: lag_1, lag_2, lag_4, lag_8, lag_52, rolling_mean_4w")

# Store metadata
stores_df = pd.DataFrame([{
    "store_id": i,
    "event_timestamp": base_date,
    "store_type": ["A", "B", "C"][i % 3],  # A=large, B=medium, C=small
    "store_size": 100000 + i * 10000,
    "region": f"region_{(i - 1) // 3 + 1}"
} for i in range(1, CONFIG["stores"] + 1)])
stores_df.to_parquet(DATA_DIR / "store_features.parquet", index=False)
print(f"   ✅ Generated {len(stores_df)} store records")

# 2. Configure Feast
print("\\n⚙️  Step 3: Configure Feast Registry")
FEATURE_REPO.mkdir(parents=True, exist_ok=True)
(FEATURE_REPO / "feature_store.yaml").write_text(f"""project: sales_forecasting
provider: local
registry:
  registry_type: sql
  path: postgresql+psycopg2://feast:feast123@{POSTGRES_HOST}:5432/feast
offline_store:
  type: file
online_store:
  type: postgres
  host: {POSTGRES_HOST}
  port: 5432
  database: feast
  user: feast
  password: feast123
entity_key_serialization_version: 3
""")
print(f"   ✅ Registry: PostgreSQL @ {POSTGRES_HOST}")

# 3. Define and apply features using Feast SDK
print("\\n📝 Step 4: Register Features with Feast SDK")
from feast import FeatureStore, Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

store = FeatureStore(repo_path=str(FEATURE_REPO))

# Define entities
from feast.value_type import ValueType
store_entity = Entity(name="store_id", join_keys=["store_id"], value_type=ValueType.INT64, description="Retail store")
dept_entity = Entity(name="dept_id", join_keys=["dept_id"], value_type=ValueType.INT64, description="Department")

# Define feature views
sales_fv = FeatureView(
    name="sales_features",
    description="Weekly sales metrics by store and department",
    entities=[store_entity, dept_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="weekly_sales", dtype=Float32, description="Total weekly sales ($)"),
        Field(name="is_holiday", dtype=Int64, description="Holiday week indicator"),
        Field(name="temperature", dtype=Float32, description="Temperature (°F)"),
        Field(name="fuel_price", dtype=Float32, description="Fuel price ($)"),
        Field(name="cpi", dtype=Float32, description="Consumer Price Index"),
        Field(name="unemployment", dtype=Float32, description="Unemployment rate (%)")
    ],
    source=FileSource(path=str(DATA_DIR / "sales_features.parquet"), timestamp_field="event_timestamp")
)

store_fv = FeatureView(
    name="store_features",
    description="Static store attributes",
    entities=[store_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="store_type", dtype=String, description="Store type (A/B/C)"),
        Field(name="store_size", dtype=Int64, description="Store size (sq ft)"),
        Field(name="region", dtype=String, description="Geographic region")
    ],
    source=FileSource(path=str(DATA_DIR / "store_features.parquet"), timestamp_field="event_timestamp")
)

# Apply to registry
store.apply([store_entity, dept_entity, sales_fv, store_fv])
print("   ✅ Registered: store_entity, dept_entity, sales_features, store_features")

# 4. Materialize to online store
print("\\n🚀 Step 5: Materialize to Online Store")
store.materialize(
    start_date=base_date,
    end_date=base_date + timedelta(weeks=CONFIG["weeks"])
)
print("   ✅ Features materialized to Postgres online store")

# Summary
print("\\n" + "=" * 60)
print("✅ Feature Engineering Complete!")
print("=" * 60)
print(f"   📁 Data: {DATA_DIR}")
print(f"   📝 Registry: PostgreSQL")
print(f"   🔢 Records: {len(sales_df):,} sales, {len(stores_df)} stores")
print(f"   📊 Features: sales_features (6), store_features (3)")
'''


# =============================================================================
# MAIN: Submit Job to Cluster
# =============================================================================

if __name__ == "__main__":
    import json
    from kubernetes import client as k8s
    
    print(f"{'='*60}")
    print("🍕 Feast Feature Engineering")
    print(f"{'='*60}")
    print(f"\n📋 Feature Definitions:")
    for name, fv in FEATURE_VIEWS.items():
        print(f"\n   {name}: {fv['description']}")
        for fname, fdef in fv['features'].items():
            print(f"      • {fname}: {fdef['description']} ({fdef['type']})")

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
    
    # ConfigMap with script
    print(f"\n📦 Creating ConfigMap...")
    try: core.delete_namespaced_config_map("feast-scripts", NAMESPACE)
    except: pass
    core.create_namespaced_config_map(NAMESPACE, k8s.V1ConfigMap(
        metadata=k8s.V1ObjectMeta(name="feast-scripts", labels={"app": "sales-forecasting", "job-type": "feast"}),
        data={"run.py": FEAST_JOB_SCRIPT}
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
                               "export PYTHONPATH=/tmp/pylibs:$PYTHONPATH && python /scripts/run.py"],
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
