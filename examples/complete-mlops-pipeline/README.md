# Feast + KubeRay + Kubeflow Training on OpenShift AI

**Production MLOps: Feature Store → Distributed Training → Inference**

This example demonstrates **Feast + KubeRay** integration for distributed feature retrieval in Kubeflow TrainJobs:

```
┌─────────────────────────────────────────────────────────────────┐
│                FEAST + KUBERAY + TRAINER INTEGRATION            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐     ┌─────────────────┐     ┌──────────────┐  │
│  │   Feast     │     │  Kubeflow       │     │   KServe     │  │
│  │  Registry   │────▶│  TrainJob       │────▶│  Inference   │  │
│  │ (PostgreSQL)│     │                 │     │              │  │
│  └─────────────┘     │  ┌───────────┐  │     │ ┌──────────┐ │  │
│        │             │  │ Feast SDK │  │     │ │  Feast   │ │  │
│        ▼             │  │ get_hist_ │──┼──┐  │ │  Server  │ │  │
│  ┌─────────────┐     │  │ features()│  │  │  │ │  (REST)  │ │  │
│  │   Offline   │     │  └───────────┘  │  │  │ └──────────┘ │  │
│  │   Store     │────▶│                 │  │  │      │       │  │
│  │  (Parquet)  │     └─────────────────┘  │  │      ▼       │  │
│  └─────────────┘              │           │  │   Features   │  │
│        │                      ▼           │  └──────────────┘  │
│        │              ┌───────────────┐   │                    │
│        │              │   KUBERAY     │◀──┘                    │
│        ▼              │   CLUSTER     │                        │
│  ┌─────────────┐      │ Distributed   │                        │
│  │   Online    │      │ PIT Joins     │                        │
│  │   Store     │      └───────────────┘                        │
│  │ (PostgreSQL)│                                               │
│  └─────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

**Key Value Proposition:** One feature definition, zero train-serve skew.

> [!TIP]
> **Same Features, Everywhere**: Training calls `get_historical_features()`, inference calls `get_online_features()` - both use the same Feature Service definition.

> [!IMPORTANT]
> Tested with OpenShift AI 3.2+ - see [validation](#validation) section.

## Architecture

### Feast + Trainer Integration Flow

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  01-feast-setup  │    │  02-training     │    │  03-inference    │
│  (Notebook/Job)  │    │  (TrainJob)      │    │  (KServe)        │
├──────────────────┤    ├──────────────────┤    ├──────────────────┤
│                  │    │                  │    │                  │
│ • Generate data  │    │ • Load Feast     │    │ • Load model     │
│ • feast apply    │───▶│ • get_hist_feat()│───▶│ • Call Feast     │
│ • materialize    │    │ • Train PyTorch  │    │   Server API     │
│                  │    │ • Log to MLflow  │    │ • Return pred    │
│                  │    │                  │    │                  │
└───────┬──────────┘    └───────┬──────────┘    └───────┬──────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                     SHARED INFRASTRUCTURE                        │
├──────────────────────────────────────────────────────────────────┤
│  PostgreSQL          │  PVC (RWX)         │  MLflow Server      │
│  • Feast Registry    │  • /shared/data    │  • Experiments      │
│  • Online Store      │  • /shared/models  │  • Model Registry   │
└──────────────────────────────────────────────────────────────────┘
```

### Component Summary

| Component | Purpose | Key Integration |
|-----------|---------|-----------------|
| **Feast** | Feature Store | Training + Inference use same Feature Service |
| **Kubeflow Trainer** | Distributed Training | Calls `store.get_historical_features()` directly |
| **MLflow** | Experiment Tracking | Logs metrics, artifacts, model registry |
| **KServe** | Model Serving | Calls Feast Server for online features |
| **PostgreSQL** | Durable Storage | Feast registry + online store + MLflow backend |
| **PVC (RWX)** | Shared Storage | Data, models, feature repo shared across pods |

## Requirements

- **OpenShift AI 3.2+** with: `trainer`, `workbenches`, `mlflow` components
- **RWX Storage** (NFS-CSI or equivalent) for shared PVC
- **PostgreSQL** for Feast registry/online store and MLflow backend

## Quick Start

### Option A: Manifests (Production)

```bash
# Clone and deploy
git clone https://github.com/<your-org>/sales-demand-forecasting.git
cd sales-demand-forecasting/examples/complete-mlops-pipeline

# Apply manifests in order
kubectl apply -f manifests/00-prereqs.yaml
kubectl apply -f manifests/01-postgres.yaml
kubectl apply -f manifests/02-mlflow.yaml
kubectl apply -f manifests/04-feast-prereqs.yaml

# Wait for infrastructure
kubectl wait --for=condition=ready pod -l app=postgres -n feast-trainer-demo --timeout=120s

# Run dataprep job (generates data, applies Feast, materializes)
kubectl apply -f manifests/05-dataprep-job.yaml

# Run training job
kubectl apply -f manifests/06-trainjob.yaml
```

### Option B: Notebooks (Interactive)

1. **Create Workbench** in OpenShift AI Dashboard (PyTorch image, 50Gi RWX storage)
2. **Clone repo** and open `examples/complete-mlops-pipeline/notebooks/`
3. **Run notebooks** in order:
   - `01-feast-features.ipynb` → Setup Feast
   - `02-training.ipynb` → Submit TrainJob
   - `03-inference.ipynb` → Deploy model

## Pipeline Notebooks

| Notebook | What It Does | Feast Integration |
|----------|--------------|-------------------|
| `01-feast-features.ipynb` | Generate data, `feast apply`, materialize | Registers features to PostgreSQL |
| `02-training.ipynb` | Submit TrainJob via Kubeflow SDK | **`get_historical_features()`** in TrainJob |
| `03-inference.ipynb` | Deploy KServe + test inference | **Feast Server REST API** for online features |

### The Key Pattern: Feature Services

```python
# features.py - Single source of truth
training_features = FeatureService(
    name="training_features",
    features=[sales_history_features, store_external_features],
)

inference_features = FeatureService(
    name="inference_features", 
    features=[sales_history_features, store_external_features],
)
```

**Training (02-training.ipynb):**
```python
# Inside Kubeflow TrainJob
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=store.get_feature_service("training_features"),
).to_df()
```

**Inference (03-inference.ipynb):**
```python
# Feast Server REST API call
POST /get-online-features
{"feature_service": "inference_features", "entities": {"store_id": [1], "dept_id": [5]}}
```

## What You'll Learn

### 1. Feast ↔ Kubeflow Trainer Integration
- Call `get_historical_features()` directly inside TrainJob
- Use Feature Services for consistent feature selection
- Share feature repository via PVC between Feast and Trainer

### 2. Feast ↔ KServe Integration
- Deploy Feast Feature Server alongside model
- Fetch online features via REST API at inference time
- Eliminate train-serve skew with shared feature definitions

### 3. Kubeflow Training Operator
- Submit distributed PyTorch TrainJobs via SDK
- Configure resources, volumes, and environment
- Monitor job status and logs

### 4. MLflow Experiment Tracking
- Log metrics, parameters, and artifacts
- Track model versions
- Visualize training progress

## Configuration

### Ray Integration (Default: Enabled)

Ray is **enabled by default** for production-scale distributed processing:

| Toggle | Operation | Default | Disable For |
|--------|-----------|---------|-------------|
| `use_ray` | Training `get_historical_features()` | `true` | Quick start without Ray cluster |
| `use_ray_dataprep` | DataPrep `feast materialize` | `true` | Small datasets (<1M rows) |

**Quick Start (no Ray):** Set both to `false` for development/testing without a Ray cluster.

**Production (default):** Both `true` - leverages KubeRay for distributed PIT joins and materialization.

### Feast Two-Config Approach

We use **two Feast configs** for stability + performance:

| Config | Offline Store | Batch Engine | Used By |
|--------|---------------|--------------|---------|
| `feature_store.yaml` | `type: file` | Optional: `ray.engine` | `feast apply`, `materialize` |
| `feature_store_ray.yaml` | `type: ray` | KubeRay | `get_historical_features()` in TrainJob |

**`feature_store.yaml`** (File-based, for apply/materialize):
```yaml
project: sales_forecasting
provider: local

registry:
  registry_type: sql
  path: postgresql+psycopg://feast:feast123@postgres:5432/feast

offline_store:
  type: file  # Stable for apply/materialize

online_store:
  type: postgres
  host: postgres
  port: 5432
```

**`feature_store_ray.yaml`** (KubeRay, for training):
```yaml
project: sales_forecasting
provider: local

registry:
  registry_type: sql
  path: postgresql+psycopg://feast:feast123@postgres:5432/feast

offline_store:
  type: ray
  use_kuberay: true
  kuberay_conf:
    cluster_name: feast-ray
    namespace: feast-trainer-demo
    skip_tls: true
  broadcast_join_threshold_mb: 100
  enable_distributed_joins: true

online_store:
  type: postgres
  host: postgres
  port: 5432
```

### Kubeflow TrainJob with Feast + KubeRay

Inside the TrainJob, we:
1. Setup CodeFlare SDK auth (for KubeRay connection)
2. Copy `feature_store_ray.yaml` → `feature_store.yaml`
3. Call `get_historical_features()` which uses distributed Ray

```python
# Inside TrainJob - setup KubeRay connection
os.environ["FEAST_RAY_USE_KUBERAY"] = "true"
os.environ["FEAST_RAY_CLUSTER_NAME"] = "feast-ray"
os.environ["FEAST_RAY_NAMESPACE"] = "feast-trainer-demo"
os.environ["FEAST_RAY_SKIP_TLS"] = "true"

# In-cluster auth for CodeFlare SDK
with open("/var/run/secrets/kubernetes.io/serviceaccount/token") as f:
    os.environ["FEAST_RAY_AUTH_TOKEN"] = f.read()
os.environ["FEAST_RAY_AUTH_SERVER"] = f"https://{os.environ['KUBERNETES_SERVICE_HOST']}:{os.environ['KUBERNETES_SERVICE_PORT']}"

# Copy Ray config to feature_store.yaml
shutil.copy(f"{feature_repo}/feature_store_ray.yaml", f"{feature_repo}/feature_store.yaml")

# Now Feast uses KubeRay for distributed PIT joins!
store = FeatureStore(repo_path=feature_repo)
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=store.get_feature_service("training_features"),
).to_df()  # Distributed via KubeRay!
```

### Kubeflow Training SDK (from notebook)

```python
from kubeflow.training import TrainerClient, CustomTrainer

trainer_client.train(
    trainer=CustomTrainer(
        func=train_sales_model,
        num_nodes=1,
        resources_per_node={"cpu": 4, "memory": "8Gi"},
        # KubeRay integration requires feast[ray,postgres] + codeflare-sdk
        packages_to_install=[
            "feast[ray,postgres]==0.59.0",
            "codeflare-sdk",  # For KubeRay connection
            "mlflow>=3.0",
        ],
    ),
    runtime=runtime,
    parameters=parameters,
)
```

## Validation

Tested on OpenShift AI 3.2+ with NVIDIA GPUs:

| Configuration | Value |
|---------------|-------|
| **Dataset** | Synthetic Walmart sales (~5,000 rows) |
| **Features** | 11 (lag_1..52, rolling_mean, store_size, temp, CPI, etc.) |
| **Feast Config** | File offline store, PostgreSQL online store |
| **Model** | MLP [512, 256, 128, 64] with BatchNorm + Dropout |
| **Training** | 30 epochs, batch_size=64, lr=5e-4, AdamW |
| **Distributed** | PyTorch DDP (1-4 GPUs) |

**Results:**
```
✅ Best MAPE: ~10%
✅ Best RMSE: ~2,500
✅ Feast get_historical_features(): <2s for 5,000 rows
✅ Feast online features: <10ms per request
```

**Timing (single GPU):**
| Stage | Duration |
|-------|----------|
| Dataprep (generate + apply + materialize) | ~30s |
| Training (30 epochs) | ~2 min |
| **Total Pipeline** | **~2.5 min** |

## Directory Structure

```
complete-mlops-pipeline/
├── README.md
├── feature_repo/
│   ├── feature_store.yaml        # Feast configuration
│   └── features.py               # Feature definitions (FeatureViews, Services)
├── notebooks/
│   ├── 01-feast-features.ipynb   # Feast setup (apply, materialize)
│   ├── 02-training.ipynb         # TrainJob submission (Feast integration)
│   └── 03-inference.ipynb        # KServe deployment (Feast online serving)
├── manifests/
│   ├── 00-prereqs.yaml           # Namespace, ClusterTrainingRuntime
│   ├── 01-postgres.yaml          # PostgreSQL (Feast + MLflow backend)
│   ├── 02-mlflow.yaml            # MLflow tracking server
│   ├── 03-raycluster.yaml        # KubeRay cluster (optional, for distributed)
│   ├── 04-feast-prereqs.yaml     # PVC, ServiceAccount, RBAC
│   ├── 04b-feast-server.yaml     # Feast Feature Server (online serving)
│   ├── 05-dataprep-job.yaml      # Data generation + feast apply + materialize
│   └── 06-trainjob.yaml          # Kubeflow TrainJob (calls Feast)
└── scripts/
    ├── serve.py                  # KServe inference server (calls Feast)
    ├── setup.sh                  # Automated setup
    └── cleanup.sh                # Resource cleanup
```

## Troubleshooting

### KubeRay Connection Issues
```bash
# Check RayCluster is running
kubectl get raycluster -n feast-trainer-demo
kubectl get pods -l ray.io/cluster=feast-ray -n feast-trainer-demo

# Check Ray head service
kubectl get svc feast-ray-head-svc -n feast-trainer-demo

# From TrainJob pod - test Ray connection
kubectl exec -it <trainjob-pod> -n feast-trainer-demo -- \
  python -c "import ray; ray.init('ray://feast-ray-head-svc:10001'); print(ray.cluster_resources())"
```

### Feast Registry Connection (in TrainJob)
```bash
# Check if TrainJob can reach PostgreSQL
kubectl exec -it <trainjob-pod> -n feast-trainer-demo -- \
  python -c "from feast import FeatureStore; print(FeatureStore('/shared/feature_repo').list_feature_views())"
```

### Feast Feature Server Issues
```bash
# Check Feast Server logs
kubectl logs -l app=feast-server -n feast-trainer-demo

# Test REST API
kubectl exec -it <any-pod> -n feast-trainer-demo -- \
  curl http://feast-server:6566/health
```

### TrainJob Not Starting
```bash
# Check TrainJob status
kubectl get trainjob -n feast-trainer-demo
kubectl describe trainjob sales-forecasting-mlflow -n feast-trainer-demo

# Check for resource issues
kubectl get events -n feast-trainer-demo --sort-by='.lastTimestamp'
```

### MLflow Connection Issues
```bash
kubectl logs -l app=mlflow -n feast-trainer-demo
```

### CodeFlare SDK Auth Issues
```bash
# Check ServiceAccount token is mounted
kubectl exec -it <trainjob-pod> -n feast-trainer-demo -- \
  cat /var/run/secrets/kubernetes.io/serviceaccount/token | head -c 50

# Check RBAC permissions
kubectl auth can-i get rayclusters --as=system:serviceaccount:feast-trainer-demo:feast-sa -n feast-trainer-demo
```

## Resources

* [Feast Documentation](https://docs.feast.dev/)
* [Feast Ray Integration](https://docs.feast.dev/reference/offline-stores/ray)
* [KubeRay Documentation](https://ray-project.github.io/kuberay/)
* [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/training/)
* [MLflow Documentation](https://mlflow.org/docs/latest/)
* [OpenShift AI Documentation](https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/)

