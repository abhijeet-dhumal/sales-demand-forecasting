#!/usr/bin/env python3
"""
Feature Engineering with Feast - Submits Job to Cluster
Usage: python 01_feast_features.py
"""
import os, sys, time
from datetime import datetime

# Config
NAMESPACE = os.getenv("NAMESPACE", "feast-trainer-demo")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", f"feast-postgres.{NAMESPACE}.svc.cluster.local")
K8S_TOKEN = os.getenv("K8S_TOKEN")
K8S_API_SERVER = os.getenv("K8S_API_SERVER")

DATAPREP_SCRIPT = '''#!/usr/bin/env python3
import os, pandas as pd, numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_DIR = os.getenv("DATA_DIR", "/shared/data")
FEATURE_REPO = os.getenv("FEATURE_REPO_DIR", "/shared/feature_repo")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")

print("📊 Generating data...")
np.random.seed(42)
base = datetime(2022, 1, 1, tzinfo=timezone.utc)
records = []
for w in range(104):
    d = base + timedelta(weeks=w)
    for s in range(1, 11):
        for dept in range(1, 6):
            seasonal = 1 + 0.3 * np.sin(2 * np.pi * w / 52)
            holiday = 1.5 if 47 <= (w % 52) <= 52 else 1.0
            records.append({
                "store_id": s, "dept_id": dept, "event_timestamp": d,
                "weekly_sales": max(0, (50000+s*5000)*(0.5+dept*0.2)*seasonal*holiday + np.random.normal(0, 2000)),
                "is_holiday": int(holiday > 1),
                "temperature": 60 + 20*np.sin(2*np.pi*w/52) + np.random.normal(0, 5),
                "fuel_price": 3 + 0.5*np.random.random(),
                "cpi": 220 + w*0.1,
                "unemployment": 5 + np.random.normal(0, 0.5)
            })

Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
sales_df = pd.DataFrame(records)
sales_df.to_parquet(f"{DATA_DIR}/sales_features.parquet")
sales_df.to_parquet(f"{DATA_DIR}/features.parquet")  # For training

stores = pd.DataFrame([{
    "store_id": i, "event_timestamp": base, "store_type": ["A","B","C"][i%3],
    "store_size": 100000 + i*10000, "region": f"region_{(i-1)//3+1}"
} for i in range(1, 11)])
stores.to_parquet(f"{DATA_DIR}/store_features.parquet")
print(f"✅ Saved {len(sales_df):,} sales records")

# Feast setup
print("⚙️  Configuring Feast...")
Path(FEATURE_REPO).mkdir(parents=True, exist_ok=True)
Path(f"{FEATURE_REPO}/feature_store.yaml").write_text(f"""project: sales_forecasting
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
entity_key_serialization_version: 2
""")

from feast import FeatureStore, Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64, String

store = FeatureStore(repo_path=FEATURE_REPO)
se = Entity(name="store_id", join_keys=["store_id"])
de = Entity(name="dept_id", join_keys=["dept_id"])

sales_fv = FeatureView(
    name="sales_features", entities=[se, de], ttl=timedelta(days=365),
    schema=[Field(name="weekly_sales", dtype=Float32), Field(name="is_holiday", dtype=Int64),
            Field(name="temperature", dtype=Float32), Field(name="fuel_price", dtype=Float32),
            Field(name="cpi", dtype=Float32), Field(name="unemployment", dtype=Float32)],
    source=FileSource(path=f"{DATA_DIR}/sales_features.parquet", timestamp_field="event_timestamp"))

store_fv = FeatureView(
    name="store_features", entities=[se], ttl=timedelta(days=365),
    schema=[Field(name="store_type", dtype=String), Field(name="store_size", dtype=Int64), Field(name="region", dtype=String)],
    source=FileSource(path=f"{DATA_DIR}/store_features.parquet", timestamp_field="event_timestamp"))

store.apply([se, de, sales_fv, store_fv])
print("📝 Features registered")

store.materialize(start_date=datetime(2022,1,1,tzinfo=timezone.utc), end_date=datetime(2024,1,1,tzinfo=timezone.utc))
print("✅ Done!")
'''

if __name__ == "__main__":
    from kubernetes import client as k8s
    
    print(f"{'='*60}\nFeast Feature Engineering\n{'='*60}")

    # Auth
    if K8S_TOKEN and K8S_API_SERVER:
        cfg = k8s.Configuration()
        cfg.host, cfg.verify_ssl = K8S_API_SERVER, False
        cfg.api_key = {"authorization": f"Bearer {K8S_TOKEN}"}
        k8s.Configuration.set_default(cfg)
    else:
        from kubernetes import config
        config.load_kube_config()
    
    batch = k8s.BatchV1Api()
    core = k8s.CoreV1Api()
    job_name = f"feast-dataprep-{datetime.now().strftime('%H%M%S')}"
    
    # Create ConfigMap with script
    try: core.delete_namespaced_config_map("feast-scripts", NAMESPACE)
    except: pass
    core.create_namespaced_config_map(NAMESPACE, k8s.V1ConfigMap(
        metadata=k8s.V1ObjectMeta(name="feast-scripts"), data={"run.py": DATAPREP_SCRIPT}))
    
    job = {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": job_name, "namespace": NAMESPACE},
        "spec": {"backoffLimit": 0, "ttlSecondsAfterFinished": 3600,
            "template": {"spec": {"restartPolicy": "Never",
                "containers": [{"name": "dataprep", "image": "quay.io/modh/ray:2.52.1-py312-cu128",
                    "command": ["/bin/bash", "-c", "pip install -q feast[postgres] && python /scripts/run.py"],
                    "env": [{"name": "POSTGRES_HOST", "value": POSTGRES_HOST},
                            {"name": "DATA_DIR", "value": "/shared/data"},
                            {"name": "FEATURE_REPO_DIR", "value": "/shared/feature_repo"}],
                    "volumeMounts": [{"name": "shared", "mountPath": "/shared"}, {"name": "scripts", "mountPath": "/scripts"}]}],
                "volumes": [{"name": "shared", "persistentVolumeClaim": {"claimName": "feast-pvc"}},
                           {"name": "scripts", "configMap": {"name": "feast-scripts"}}]}}}
    }
    
    batch.create_namespaced_job(NAMESPACE, job)
    print(f"✅ Job submitted: {job_name}")
    
    # Wait
    for _ in range(60):
        j = batch.read_namespaced_job(job_name, NAMESPACE)
        if j.status.succeeded: print("✅ Complete!"); sys.exit(0)
        if j.status.failed: print("❌ Failed!"); sys.exit(1)
        time.sleep(10)
    print("⏰ Timeout"); sys.exit(1)
