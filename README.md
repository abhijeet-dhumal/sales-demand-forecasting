# Sales Demand Forecasting on OpenShift AI

**Distributed ML Pipeline: Feast Feature Store → PyTorch DDP → MLflow → KServe**

## Overview

End-to-end ML pipeline demonstrating production MLOps patterns on OpenShift AI:

| Component | Purpose | Key Feature |
|-----------|---------|-------------|
| **Feast** | Feature store | Distributed `get_historical_features()` via KubeRay |
| **PyTorch DDP** | Model training | Multi-node distributed training |
| **MLflow** | Experiment tracking | Params, metrics, artifacts |
| **KServe** | Model serving | Feast integration for real-time predictions |

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            PIPELINE ARCHITECTURE                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌────────────┐  │
│  │   Feast     │────▶│  TrainJob   │────▶│   MLflow    │────▶│  KServe    │  │
│  │  (KubeRay)  │     │ (PyTorch)   │     │  (Track)    │     │  (Serve)   │  │
│  └─────────────┘     └─────────────┘     └─────────────┘     └────────────┘  │
│        │                   │                   │                   │         │
│        │ Distributed       │ Multi-node        │ Experiments       │ Online  │
│        │ PIT Joins         │ DDP Training      │ & Artifacts       │ Features│
│        ▼                   ▼                   ▼                   ▼         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         Shared PVC (/shared)                         │   │
│  │  data/              feature_repo/           models/                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Option A: Manifests (Recommended)

```bash
# 1. Deploy infrastructure
kubectl apply -k manifests/

# 2. Wait for pods
kubectl wait --for=condition=ready pod -l app=postgres -n feast-trainer-demo --timeout=120s
kubectl wait --for=condition=ready pod -l ray.io/node-type=head -n feast-trainer-demo --timeout=180s

# 3. Prepare data and register features
kubectl apply -f manifests/05-dataprep-job.yaml
kubectl wait --for=condition=complete job/feast-dataprep -n feast-trainer-demo --timeout=300s

# 4. Train model
kubectl apply -f manifests/06-trainjob.yaml
kubectl logs -f -l trainer.kubeflow.org/trainjob-name=sales-training -n feast-trainer-demo
```

### Option B: Notebooks (Interactive)

In OpenShift AI Workbench, run notebooks in order:
1. `notebooks/01-feast-features.ipynb` - Generate data, register features
2. `notebooks/02-training.ipynb` - Train model with `get_historical_features()`
3. `notebooks/03-inference.ipynb` - Deploy model with KServe

---

## Feature Importance

The model uses **22 features** engineered for retail demand forecasting:

| Feature Group | Importance | Features | Why |
|---------------|------------|----------|-----|
| **Lag Features** | 35% | `lag_1`, `lag_2`, `lag_4`, `lag_8` | Recent history is most predictive |
| **Rolling Stats** | 28% | `rolling_mean_4w`, `rolling_std_4w`, `sales_vs_avg` | Trend and volatility |
| **Temporal** | 18% | `week_of_year`, `month`, `quarter` | Seasonality patterns |
| **Holiday** | 10% | `is_holiday`, `days_to_holiday` | Holiday effects |
| **Economic** | 7% | `temperature`, `fuel_price`, `cpi`, `unemployment` | External factors |
| **Store** | 2% | `store_type`, `store_size`, `region` | Store context |

**Key Insight:** 63% of predictive power comes from **lag and rolling features** - this is why time-series feature engineering is critical.

---

## Why Ray for Feature Retrieval?

| Approach | 65K rows | 1M rows | 10M rows |
|----------|----------|---------|----------|
| **Local (Pandas)** | 2 min | 30+ min | OOM crash |
| **Ray (Distributed)** | 30 sec | 3 min | 15 min |

Ray distributes point-in-time joins across the cluster, preventing memory issues and enabling scale.

---

## Architecture

### Feast Feature Store

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FEAST FEATURE STORE ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────┐         ┌─────────────────────┐                    │
│  │   DATA SOURCES      │         │      ENTITIES       │                    │
│  ├─────────────────────┤         ├─────────────────────┤                    │
│  │ sales_features.pq   │────┐    │ store_id (1-45)     │                    │
│  │ store_features.pq   │──┐ │    │ dept_id  (1-14)     │                    │
│  └─────────────────────┘  │ │    └─────────────────────┘                    │
│                           │ │              │                                │
│                           ▼ ▼              ▼                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        FEATURE VIEWS                                 │   │
│  ├──────────────────────────────┬──────────────────────────────────────┤   │
│  │  sales_features (19)         │  store_features (3)                  │   │
│  │  • weekly_sales (target)     │  • store_type (A/B/C)                │   │
│  │  • lag_1, lag_2, lag_4, lag_8│  • store_size (sqft)                 │   │
│  │  • rolling_mean_4w, std_4w   │  • region                            │   │
│  │  • week_of_year, month, qtr  │                                      │   │
│  └──────────────────────────────┴──────────────────────────────────────┘   │
│                           │                   │                             │
│                           ▼                   ▼                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      FEATURE SERVICES                                │   │
│  ├────────────────────────────────┬────────────────────────────────────┤   │
│  │  training_features             │  inference_features                │   │
│  │  22 features (incl. target)    │  21 features (excl. target)        │   │
│  │  → Model training              │  → Real-time predictions           │   │
│  └────────────────────────────────┴────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Storage Layout

```
/shared (PVC - 20Gi recommended)
├── data/                         # Parquet files (~100MB)
│   ├── sales_features.parquet    # 65K rows, 22 columns
│   └── store_features.parquet    # 630 rows, store metadata
├── feature_repo/                 # Feast configuration
│   ├── feature_store.yaml        # Ray-enabled config
│   └── features.py               # Feature definitions + auto-auth
└── models/                       # Training artifacts (~50MB)
    ├── model_best.pt             # Best model weights
    ├── model_final.pt            # Final model weights
    ├── scalers.joblib            # Feature and target scalers
    └── model_metadata.json       # Architecture, feature columns
```

### Components

| Manifest | Component | Purpose | Resources |
|----------|-----------|---------|-----------|
| 00-prereqs.yaml | Namespace, PVC | Shared storage | 20Gi PVC |
| 01-postgres.yaml | PostgreSQL | Feast registry + online store | 1 CPU, 2Gi |
| 02-mlflow.yaml | MLflow Server | Experiment tracking | 1 CPU, 2Gi |
| 03-raycluster.yaml | KubeRay | Distributed Feast operations | 4 CPU, 16Gi per node |
| 04a-feast-rbac.yaml | RBAC | ServiceAccount permissions | - |
| 04b-feast-server.yaml | Feast Server | Feature serving UI | 1 CPU, 2Gi |
| 05-dataprep-job.yaml | Data Prep | Generate & register features | 2 CPU, 4Gi |
| 06-trainjob.yaml | TrainJob | Multi-node PyTorch DDP | 4 CPU, 8Gi per node |

---

## Configuration

### GPU Support (Optional)

By default, manifests use **CPU-only** resources. To enable GPUs:

```yaml
# In 03-raycluster.yaml and 06-trainjob.yaml, uncomment:
resources:
  requests:
    cpu: "4"
    memory: "8Gi"
    # nvidia.com/gpu: "1"  # Uncomment to request GPU
```

The training script automatically detects hardware:
- **CPU**: Uses `gloo` backend for distributed training
- **GPU**: Uses `nccl` backend + AMP (Automatic Mixed Precision)

### Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server URL |
| `RAY_USE_TLS` | `1` | Enable mTLS for KubeRay |
| `NUM_EPOCHS` | `20` | Training epochs |
| `BATCH_SIZE` | `512` | Batch size |

### Images

| Component | Image |
|-----------|-------|
| Training | `quay.io/modh/training:py312-cuda128-torch280` |
| Ray | `quay.io/modh/ray:2.52.1-py312-cu128` |
| MLflow | `ghcr.io/mlflow/mlflow:v2.19.0` |

---

## Project Structure

```
.
├── feature_repo/
│   ├── feature_store.yaml        # File-based config (local)
│   ├── feature_store_ray.yaml    # KubeRay config (distributed)
│   └── features.py               # Feature definitions + CodeFlare auth
├── notebooks/
│   ├── 01-feast-features.ipynb   # Data prep + feature registration
│   ├── 02-training.ipynb         # Training with Feast + Ray
│   └── 03-inference.ipynb        # KServe deployment
├── manifests/
│   ├── kustomization.yaml        # Kustomize config
│   ├── scripts/train_ddp.py      # DDP training script
│   └── *.yaml                    # Kubernetes manifests
└── README.md
```

---

## Access URLs

```bash
# MLflow UI
echo "https://$(kubectl get route mlflow -n feast-trainer-demo -o jsonpath='{.spec.host}')"

# Feast UI
echo "https://$(kubectl get route feast-ui -n feast-trainer-demo -o jsonpath='{.spec.host}')"

# Ray Dashboard
echo "https://$(kubectl get route feast-ray-dashboard -n feast-trainer-demo -o jsonpath='{.spec.host}')"
```

---

## Results (65K rows demo)

| Metric | Value |
|--------|-------|
| Dataset | 45 stores × 14 depts × 104 weeks = 65,520 rows |
| Features | 22 (19 sales + 3 store) |
| Model | MLP [256, 128, 64] with BatchNorm + Dropout |
| MAPE | ~3-5% (excellent for retail forecasting) |
| Training Time | ~2-5 min (CPU), ~30 sec (GPU) |
| Feature Retrieval | ~30 sec via KubeRay |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `KeyError: 'auth_token'` | Ensure CodeFlare SDK env vars are set in notebook |
| `FileNotFoundError` for parquet | Run dataprep job first or check PVC mount |
| Ray connection timeout | Verify RayCluster: `kubectl get raycluster -n feast-trainer-demo` |
| PostgreSQL connection error | Check pod: `kubectl get pods -l app=postgres -n feast-trainer-demo` |
| MLflow connection refused | Check pod: `kubectl get pods -l app=mlflow -n feast-trainer-demo` |
| DDP NCCL timeout | Increase timeout or check network policies |
| InferenceService not ready | Check logs: `kubectl logs -l serving.kserve.io/inferenceservice` |

---

## Cleanup

```bash
# Delete all resources
kubectl delete namespace feast-trainer-demo

# Or delete specific components
kubectl delete trainjob sales-training -n feast-trainer-demo
kubectl delete job feast-dataprep -n feast-trainer-demo
```
