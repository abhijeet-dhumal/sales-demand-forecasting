# Sales Forecasting - Kubernetes Manifests

End-to-end ML pipeline with **Feast**, **Ray**, and **KServe** on OpenShift/Kubernetes.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     feast-trainer-demo namespace                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ PostgreSQL   │    │   MLflow     │    │  KubeRay     │          │
│  │ (Feast       │    │ (Experiment  │    │  Cluster     │          │
│  │  Registry)   │    │  Tracking)   │    │ (Head +      │          │
│  └──────────────┘    └──────────────┘    │  Workers)    │          │
│         │                   │            └──────┬───────┘          │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │                     ML Pipeline                          │      │
│  │                                                          │      │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌───────┐ │      │
│  │  │  Data   │───▶│ Feature │───▶│ Train   │───▶│KServe │ │      │
│  │  │  Prep   │    │  Store  │    │ Model   │    │Infer  │ │      │
│  │  └─────────┘    └─────────┘    └─────────┘    └───────┘ │      │
│  │                                                          │      │
│  └──────────────────────────────────────────────────────────┘      │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              NFS Shared Storage (feast-pvc)             │       │
│  │  /shared/data, /shared/feature_repo, /shared/models     │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Infrastructure
kubectl apply -f 01-namespace.yaml
kubectl apply -f 02-pvc-shared-storage.yaml
kubectl apply -f 05-feast-postgres.yaml
kubectl apply -f 06-mlflow.yaml
kubectl apply -f 07-kuberay-cluster.yaml
kubectl apply -f 08-feast-prereqs.yaml

# 2. Wait for Ray cluster
kubectl wait --for=condition=ready pod -l app=feast-ray -n feast-trainer-demo --timeout=120s

# 3. Data Preparation (generates data + Feast features)
kubectl apply -f 09-feast-dataprep-job.yaml

# 4. Train Model
kubectl apply -f 10-feast-train-job.yaml

# 5. Deploy KServe Inference
kubectl apply -f 11-kserve-inference.yaml

# 6. Test Inference
kubectl apply -f 11-kserve-inference.yaml  # Includes test job
```

## Manifest Files

| # | File | Description |
|---|------|-------------|
| 01 | `namespace.yaml` | Namespace: feast-trainer-demo |
| 02 | `pvc-shared-storage.yaml` | NFS storage (RWX) |
| 03 | `clustertrainingruntime.yaml` | Kubeflow runtime (alt) |
| 04 | `trainjob.yaml` | Kubeflow job (alt) |
| 05 | `feast-postgres.yaml` | PostgreSQL for Feast |
| 06 | `mlflow.yaml` | MLflow tracking |
| **07** | `kuberay-cluster.yaml` | **Ray cluster** |
| **08** | `feast-prereqs.yaml` | **SA, RBAC, PVC** |
| **09** | `feast-dataprep-job.yaml` | **Data prep + Feast** |
| **10** | `feast-train-job.yaml` | **Model training** |
| **11** | `kserve-inference.yaml` | **KServe inference** |
| 12 | `trainjob-mlflow.yaml` | Kubeflow + MLflow (alt) |

## Pipeline Details

### 1. Data Preparation (`09-feast-dataprep-job.yaml`)
- Generates 93,600 sales records
- Configures Feast with Ray offline store
- Registers feature views in PostgreSQL
- Materializes features to online store

### 2. Model Training (`10-feast-train-job.yaml`)
- Fetches historical features from Feast via Ray
- Trains PyTorch neural network
- Saves model to `/shared/models/`

### 3. KServe Inference (`11-kserve-inference.yaml`)
- Deploys model as serverless endpoint
- Auto-scales 1-3 replicas
- Supports V1 and V2 inference protocols
- Optional: Fetches online features from Feast

**Inference Endpoints:**
```
GET  /v1/models/sales-forecast         - Model metadata
GET  /v1/models/sales-forecast/ready   - Health check
POST /v1/models/sales-forecast:predict - V1 predict
POST /v2/models/sales-forecast/infer   - V2 inference
POST /v1/features                      - Feast online features
```

**Example Request:**
```bash
curl -X POST https://<route>/v1/models/sales-forecast:predict \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      [25000.0, 24000.0, 23000.0, 24500.0, 100000, 65.0, 2.75, 210.0, 5.5]
    ]
  }'
```

## Dashboards

| Service | URL |
|---------|-----|
| Ray Dashboard | https://feast-ray-dashboard-feast-trainer-demo.apps.<cluster>/ |
| MLflow | https://mlflow-feast-trainer-demo.apps.<cluster>/ |
| KServe Route | https://sales-forecast-feast-trainer-demo.apps.<cluster>/ |

## Key Technologies

- **Feast**: Feature store with PostgreSQL registry & online store
- **Ray/KubeRay**: Distributed data processing
- **KServe**: Serverless model inference
- **MLflow**: Experiment tracking (optional)
- **PyTorch**: Neural network training

## Image Versions

| Component | Image |
|-----------|-------|
| Ray | `quay.io/modh/ray:2.52.1-py312-cu128` |
| PostgreSQL | `quay.io/sclorg/postgresql-16-c9s:latest` |
