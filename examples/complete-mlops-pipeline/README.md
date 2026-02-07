# Complete MLOps Pipeline on OpenShift AI

**Feast + KubeRay + Kubeflow Training + MLflow + Model Registry**

This example demonstrates a **production-grade MLOps pipeline** integrating key OpenShift AI components:

- **Feast** - Feature Store for feature management
- **Ray/KubeRay** - Distributed data processing
- **Kubeflow Training** - Distributed model training
- **MLflow** - Experiment tracking, model logging & artifacts
- **Model Registry** - Versioned model storage for KServe deployment

> [!TIP]
> **One Pipeline, Five Integrations**: This quickstart shows how Feast, Ray, Kubeflow Training, MLflow, and Model Registry work together seamlessly on OpenShift AI.

> [!IMPORTANT]
> This example has been tested with OpenShift AI 3,2+ on configurations listed in the [validation](#validation) section.

## Architecture

![Architecture Overview](./docs/architecture.png)

### Component Interactions

| Layer | Components | Purpose |
|-------|------------|---------|
| **Feature Store** | Feast + PostgreSQL + Ray | Feature engineering, online/offline serving |
| **Compute** | KubeRay Cluster | Distributed feature processing |
| **Training** | Kubeflow Training Operator | Distributed PyTorch DDP training |
| **Tracking** | MLflow Server | Experiment metrics, model artifacts |
| **Serving** | KServe + OpenShift Route | Low-latency model inference |
| **Storage** | NFS PVC (RWX) | Shared data, models, artifacts |

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
cd sales-demand-forecasting/examples/complete-mlops-pipeline

# Run setup script
./scripts/setup.sh

# Wait for all pods
kubectl wait --for=condition=ready pod --all -n feast-mlops-demo --timeout=300s
```

### Option B: Manual Setup via Workbench

1. **Create Project** in OpenShift AI Dashboard → Data Science Projects
2. **Create Workbench** with PyTorch image, Medium size, RWX storage (50Gi)
3. **Clone this repo** and navigate to `examples/complete-mlops-pipeline/notebooks/`
4. **Run notebooks** in order: `01-feast-features` → `02-training` → `03-inference`

## Pipeline Notebooks

| Notebook | Description | Key Integrations |
|----------|-------------|------------------|
| `01-feast-features.ipynb` | Setup + Feature engineering | **Feast + Ray** |
| `02-training.ipynb` | Model training with tracking | **Kubeflow + MLflow** |
| `03-inference.ipynb` | Online inference | **Feast + Model** |

### Feast + Ray Flow

![Feast Ray Integration](./docs/feast-ray.png)

### Training Flow

![Training Sequence](./docs/training-flow.png)

### Inference Flow

![Inference Sequence](./docs/inference-flow.png)

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

### Sales Forecasting - 4x NVIDIA GPU - Production Pattern

* **Infrastructure:**
  * OpenShift AI 2.19
  * 4x NVIDIA GPU (A100/H100)
  * 2x Ray workers
  * NFS-CSI storage class
  
* **Configuration:**
  ```yaml
  # Data
  dataset: Synthetic sales data (93,600 rows)
  features: 10 (lag features, rolling stats, store attributes)
  
  # Feature Engineering (Feast + KubeRay)
  offline_store: ray
  online_store: postgres
  compute_engine: KubeRay (2 workers)
  
  # Model
  architecture: MLP (256 → 128 → 64 → 1)
  dropout: 0.2
  batch_norm: true
  
  # Training
  epochs: 15
  batch_size: 256
  learning_rate: 1e-3
  optimizer: AdamW
  distributed: PyTorch DDP (4 GPUs)
  ```

* **Results:**
  ```
  ✅ Best MAPE: 2.3%
  ✅ Best RMSE: 500
  ✅ Best val_loss: 0.0009
  ```

* **Timing Breakdown:**
  | Component | Duration |
  |-----------|----------|
  | Dataprep (Feast + KubeRay) | ~2 min 15s |
  | Training (15 epochs, 4 GPU) | ~44s |
  | **Total Pipeline** | **~3 min** |

* **Metrics:**
  ![](./docs/mlflow-metrics.png)

## Directory Structure

```
complete-mlops-pipeline/
├── README.md                      # This file
├── docs/                          # Screenshots and diagrams
├── notebooks/
│   ├── 01-feast-features.ipynb   # Setup + Feature engineering (Feast + Ray)
│   ├── 02-training.ipynb         # Model training (Kubeflow SDK + MLflow)
│   └── 03-inference.ipynb        # Online inference (Feast + Model)
├── manifests/
│   ├── 00-prereqs.yaml           # Namespace + ClusterTrainingRuntime
│   ├── 01-postgres.yaml          # PostgreSQL (Red Hat certified)
│   ├── 02-mlflow.yaml            # MLflow tracking server
│   ├── 03-raycluster.yaml           # KubeRay cluster
│   ├── 04-feast-prereqs.yaml     # PVC, ServiceAccount, RBAC
│   ├── 05-dataprep-job.yaml      # Dataprep RayJob (Feast + KubeRay)
│   └── 06-trainjob.yaml          # TrainJob with MLflow
└── scripts/
    ├── setup.sh                  # Automated setup (runs all manifests)
    └── cleanup.sh                # Resource cleanup
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

