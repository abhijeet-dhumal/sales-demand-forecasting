# MLOps Pipeline with Feast, Ray, Kubeflow & MLflow on OpenShift AI

This example demonstrates a **production-grade MLOps pipeline** integrating key OpenShift AI components:

- **Feast** - Feature Store for feature management
- **Ray/KubeRay** - Distributed data processing
- **Kubeflow Training** - Distributed model training
- **MLflow** - Experiment tracking & model registry

> [!TIP]
> **One Pipeline, Four Integrations**: This quickstart shows how Feast, Ray, Kubeflow Training, and MLflow work together seamlessly on OpenShift AI.

> [!IMPORTANT]
> This example has been tested with OpenShift AI 3,2+ on configurations listed in the [validation](#validation) section.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OpenShift AI Cluster                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐                                                           │
│   │  Workbench  │──────────────────────────────────────────────┐            │
│   │ (Notebooks) │                                              │            │
│   └─────────────┘                                              │            │
│          │                                                     │            │
│          │                                                     ▼            │
│          │         ┌──────────────────────────────────────────────────┐     │
│          │         │              KubeRay Cluster                     │     │
│          │         │  ┌────────┐   ┌────────┐   ┌────────┐            │     │
│          │         │  │  Head  │   │ Worker │   │ Worker │            │     │
│          │         │  └────────┘   └────────┘   └────────┘            │     │
│          │         └──────────────────────────────────────────────────┘     │
│          │                          │                                       │
│          ▼                          ▼                                       │
│   ┌─────────────┐         ┌─────────────────┐      ┌─────────────┐          │
│   │    Feast    │────────▶│   PostgreSQL    │◀─────│   MLflow    │          │
│   │Feature Store│         │ - Feast Registry│      │  Tracking   │          │
│   └─────────────┘         │ - Online Store  │      │  Server     │          │
│          │                │ - MLflow Backend│      └─────────────┘          │
│          │                └─────────────────┘             │                 │
│          │                                                │                 │
│          ▼                                                ▼                 │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                    Kubeflow Training Operator                   │       │
│   │  ┌─────────────────────────────────────────────────────────┐    │       │
│   │  │                      TrainJob                           │    │       │
│   │  │  Node-0 ──▶ Node-1 ──▶ Node-N  (Distributed PyTorch)    │    │       │
│   │  └─────────────────────────────────────────────────────────┘    │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                    │                                        │
│                                    ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                   NFS Shared Storage (RWX)                      │       │
│   │  /data (parquet)  /feature_repo  /models  /mlflow-artifacts     │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Requirements

* OpenShift cluster with **OpenShift AI (RHOAI) 3.2+** installed:
  * Components enabled: `dashboard`, `ray`, `trainer`, `workbenches`, `mlflow`
* Worker nodes with 2+ CPUs per Ray worker
* Dynamic storage provisioner with **RWX** support (NFS-CSI recommended)
* PostgreSQL for Feast registry and MLflow backend

## Quick Start

### Option A: Automated Setup (5 minutes)

```bash
# Clone repository
git clone https://github.com/<your-org>/sales-demand-forecasting.git
cd sales-demand-forecasting/examples/feast-ray-mlops

# Run setup script
./scripts/setup.sh

# Wait for all pods
kubectl wait --for=condition=ready pod --all -n feast-mlops-demo --timeout=300s
```

### Option B: Manual Setup via Workbench

1. **Create Project** in OpenShift AI Dashboard → Data Science Projects
2. **Create Workbench** with PyTorch image, Medium size, RWX storage (50Gi)
3. **Clone this repo** and navigate to `examples/feast-ray-mlops/notebooks/`
4. **Run notebooks** in order: `01-feast-features` → `02-training` → `03-inference`

## Pipeline Notebooks

| Notebook | Description | Key Integrations |
|----------|-------------|------------------|
| `01-feast-features.ipynb` | Setup + Feature engineering | **Feast + Ray** |
| `02-training.ipynb` | Model training with tracking | **Kubeflow + MLflow** |
| `03-inference.ipynb` | Online inference | **Feast + Model** |

### Pipeline Flow

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  01-Features     │───▶│   02-Training    │───▶│  03-Inference    │
│                  │    │                  │    │                  │
│ • Verify infra   │    │ • Fetch features │    │ • Load model     │
│ • Generate data  │    │ • Train model    │    │ • Online features│
│ • Register feats │    │ • Submit TrainJob│    │ • Predict        │
│ • Materialize    │    │ • MLflow log     │    │ • Visualize      │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

## What You'll Learn

### 1. Feast Feature Store
- Configure Feast with PostgreSQL registry
- Use Ray for distributed offline feature processing
- Materialize features to online store for low-latency serving

### 2. Ray/KubeRay Integration
- Connect to KubeRay cluster from notebooks
- Distribute parquet processing across workers
- Monitor jobs via Ray Dashboard

### 3. Kubeflow Training Operator
- Submit distributed PyTorch TrainJobs
- Configure multi-node training
- Use shared storage for checkpoints

### 4. MLflow Experiment Tracking
- Log training metrics and parameters
- Track model artifacts
- Register models in model registry

## Configuration

### Feast Feature Store

```yaml
project: sales_forecasting
provider: local

registry:
  registry_type: sql
  path: postgresql+psycopg://feast:feast123@postgres:5432/feast

offline_store:
  type: ray  # Uses KubeRay cluster

online_store:
  type: postgres
  host: postgres
  port: 5432
```

### Kubeflow TrainJob

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
spec:
  trainer:
    numNodes: 2
    image: quay.io/modh/ray:2.52.1-py312-cu128
    env:
      - name: MLFLOW_TRACKING_URI
        value: "http://mlflow:5000"
```

### MLflow Tracking

```python
import mlflow
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("sales-forecasting")

with mlflow.start_run():
    mlflow.log_params({"epochs": 50, "lr": 0.001})
    mlflow.log_metrics({"rmse": 1234.56, "mae": 890.12})
    mlflow.pytorch.log_model(model, "model")
```

## Validation

This example has been validated with:

### Sales Forecasting - 2 Ray Workers - CPU

* **Infrastructure:**
  * OpenShift AI 2.19
  * 2x Ray workers (4 CPU each)
  * NFS-CSI storage class
  
* **Configuration:**
  ```yaml
  feast:
    offline_store: ray
    online_store: postgres
  ray:
    workers: 2
    cpus_per_worker: 4
  training:
    epochs: 50
    batch_size: 256
  ```

* **Results:**
  * Data generation: ~5 seconds
  * Feature registration: ~10 seconds
  * Historical retrieval (5000 rows): ~45 seconds
  * Training (50 epochs): ~2 minutes
  * Model RMSE: ~$1,500

## Directory Structure

```
feast-ray-mlops/
├── README.md                     # This file
├── docs/                         # Screenshots and diagrams
├── notebooks/
│   ├── 01-feast-features.ipynb  # Setup + Feature engineering (Feast + Ray)
│   ├── 02-training.ipynb        # Model training (Kubeflow + MLflow)
│   └── 03-inference.ipynb       # Online inference (Feast + Model)
├── manifests/
│   ├── 00-prereqs.yaml          # Namespace, storage, RBAC
│   ├── 01-postgres.yaml         # PostgreSQL (Red Hat certified)
│   ├── 02-mlflow.yaml           # MLflow server
│   ├── 03-kuberay.yaml          # KubeRay cluster
│   └── 04-trainjob.yaml         # TrainJob template
└── scripts/
    ├── setup.sh                 # Automated setup
    └── cleanup.sh               # Resource cleanup
```

## Troubleshooting

### Ray Connection Issues
```python
os.environ["FEAST_RAY_SKIP_TLS"] = "true"
os.environ["FEAST_RAY_USE_KUBERAY"] = "true"
```

### MLflow Connection Issues
```bash
kubectl get pods -n feast-mlops-demo -l app=mlflow
kubectl logs -l app=mlflow -n feast-mlops-demo
```

### Kubeflow TrainJob Pending
```bash
kubectl describe trainjob <name> -n feast-mlops-demo
kubectl get events -n feast-mlops-demo --sort-by='.lastTimestamp'
```

## Resources

* [Feast Documentation](https://docs.feast.dev/)
* [Feast Ray Integration](https://docs.feast.dev/reference/offline-stores/ray)
* [KubeRay Documentation](https://ray-project.github.io/kuberay/)
* [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/training/)
* [MLflow Documentation](https://mlflow.org/docs/latest/)
* [OpenShift AI Documentation](https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_self-managed/)

