# Feast + Kubeflow Training on OpenShift AI

**Production MLOps: Feature Store → Experiment Tracking → Model Registry → Inference**

![Sequence Diagram](docs/sequence-diagram.png)


## Overview

End-to-end ML pipeline demonstrating **train-serve consistency** with full experiment tracking:

| Component | Purpose |
|-----------|---------|
| **Feast** | Feature store with `get_historical_features()` via KubeRay |
| **MLflow** | Experiment tracking (params, metrics, artifacts) |
| **Model Registry** | Production model catalog & KServe integration |
| **KServe** | Model serving with Feast online features |

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    Feast     │────▶│   MLflow     │────▶│   Model      │────▶│   KServe     │
│  (Features)  │     │ (Experiments)│     │  Registry    │     │  (Serving)   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │                    │
   KubeRay           Metrics/Params       Production         get_online_
   Distributed       per Epoch            Versioning         features()
```

---

## Quick Start

```bash
# 1. Deploy infrastructure
kubectl apply -k manifests/

# 2. Wait for pods
kubectl wait --for=condition=ready pod -l app=postgres -n feast-trainer-demo --timeout=120s
kubectl wait --for=condition=ready pod -l ray.io/node-type=head -n feast-trainer-demo --timeout=180s

# 3. Create MLflow database
kubectl exec -n feast-trainer-demo deploy/postgres -- \
  psql -U postgres -c "CREATE DATABASE mlflow OWNER feast;"

# 4. Restart MLflow to connect
kubectl rollout restart deploy/mlflow -n feast-trainer-demo
```

**Run via Manifests:**

```bash
# Prepare data and register features
kubectl apply -f manifests/05-dataprep-job.yaml
kubectl wait --for=condition=complete job/feast-dataprep -n feast-trainer-demo --timeout=300s

# Train model (logs to MLflow, registers to Model Registry)
kubectl apply -f manifests/06-trainjob.yaml
kubectl logs -f -l app=sales-training -n feast-trainer-demo
```

**Or use Notebooks** in OpenShift AI Workbench:
- `01-feast-features.ipynb` - Feature engineering via KubeRay
- `02-training.ipynb` - Training with MLflow tracking
- `03-inference.ipynb` - Model serving with KServe

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    feast-trainer-demo                        │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐             │
│  │   Feast    │  │  Training  │  │   KServe   │             │
│  │  Dataprep  │─▶│    Job     │─▶│ Inference  │             │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘             │
│        │               │               │                     │
│        ▼               ▼               ▼                     │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐              │
│  │ PostgreSQL│   │  MLflow   │   │   Feast   │              │
│  │ (Registry)│   │ (Tracking)│   │  Server   │              │
│  └───────────┘   └───────────┘   └───────────┘              │
│        │               │                                     │
│        └───────┬───────┘                                     │
│                ▼                                             │
│        ┌───────────┐        ┌────────────────┐              │
│        │ Shared PVC│        │    KubeRay     │              │
│        │ (Artifacts│        │ (Distributed   │              │
│        │  & Data)  │        │  Processing)   │              │
│        └───────────┘        └────────────────┘              │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
               ┌────────────────────────────┐
               │  rhoai-model-registries    │
               │  ┌──────────────────────┐  │
               │  │    Model Registry    │  │
               │  │  (Production Models) │  │
               │  └──────────────────────┘  │
               └────────────────────────────┘
```

## Project Structure

```
.
├── feature_repo/
│   ├── feature_store.yaml      # File-based config
│   ├── feature_store_ray.yaml  # KubeRay config (distributed)
│   └── features.py             # Feature definitions
├── notebooks/
│   ├── 01-feast-features.ipynb # Feature engineering
│   ├── 02-training.ipynb       # Training with MLflow
│   └── 03-inference.ipynb      # Model serving
└── manifests/
    ├── kustomization.yaml
    ├── 00-prereqs.yaml         # Namespace + shared PVC
    ├── 01-postgres.yaml        # PostgreSQL (Feast + MLflow)
    ├── 02-mlflow.yaml          # MLflow server
    ├── 03-raycluster.yaml      # KubeRay cluster
    ├── 04a-feast-rbac.yaml     # RBAC for Feast/Ray
    ├── 04b-feast-server.yaml   # Feast server + UI
    ├── 05-dataprep-job.yaml    # Data preparation
    ├── 06-trainjob.yaml        # Training job
    └── 08-kserve-inference.yaml
```

## Configuration

### Training Job Environment

| Variable | Value | Purpose |
|----------|-------|---------|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server |
| `MODEL_REGISTRY_URL` | `http://sales-model-registry:8080` | RHOAI Model Registry |
| `RAY_USE_TLS` | `1` | Enable mTLS for KubeRay |

### Images

| Component | Image |
|-----------|-------|
| Training | `quay.io/modh/training:py312-cuda128-torch280` |
| Ray | `quay.io/modh/ray:2.52.1-py312-cu128` |
| MLflow | `quay.io/modh/mlflow-server:v2.20.2-rhoai-24` |

## Access URLs

After deployment, access via:

```bash
# MLflow UI
echo "https://$(kubectl get route mlflow -n feast-trainer-demo -o jsonpath='{.spec.host}')"

# Feast UI
echo "https://$(kubectl get route feast-ui -n feast-trainer-demo -o jsonpath='{.spec.host}')"

# Model Registry
echo "https://$(kubectl get route sales-model-registry-http -n rhoai-model-registries -o jsonpath='{.spec.host}')"
```

## Results (65K rows demo)

| Metric | Value |
|--------|-------|
| Dataset | 45 stores × 14 depts × 104 weeks |
| Model | MLP [256, 128, 64] |
| MAPE | ~3-5% |
| Training | GPU-accelerated, ~2 min |
| Feature Retrieval | ~10-30s via KubeRay |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| MLflow connection refused | Check `kubectl get pod -l app=mlflow`, restart if needed |
| Model Registry auth error | Verify service account has access to `rhoai-model-registries` |
| Ray TLS errors | Mount `ray-worker-secret-feast-ray` and set `RAY_USE_TLS=1` |

## Cleanup

```bash
kubectl delete namespace feast-trainer-demo
```
