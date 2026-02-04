# Feature Store MLOps with Feast, KubeRay & MLflow on OpenShift AI

This example demonstrates an end-to-end MLOps pipeline for time-series forecasting using:
- **Feast** for feature management and point-in-time correct feature retrieval
- **KubeRay** for distributed data processing
- **Kubeflow Training** for distributed model training
- **MLflow** for experiment tracking and model registry

> [!TIP]
> This example showcases production-grade feature engineering patterns where feature consistency between training and inference is critical.

> [!IMPORTANT]
> This example has been tested with the configurations listed in the [validation](#validation) section.
> You need to adapt it, and validate it works as expected, with your configuration(s), on your target environment(s).

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OpenShift AI Cluster                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  PostgreSQL  │◄───│    Feast     │◄───│   KubeRay    │                   │
│  │  (Registry)  │    │  (Features)  │    │  (Compute)   │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌─────────────────────────────────────────────────────────┐                │
│  │              Shared Storage (NFS PVC)                    │                │
│  │   /data  /models  /feature_repo  /mlflow-artifacts      │                │
│  └─────────────────────────────────────────────────────────┘                │
│         │                   │                                                │
│         ▼                   ▼                                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Dataprep   │───►│   TrainJob   │───►│    MLflow    │                   │
│  │   (RayJob)   │    │  (Kubeflow)  │    │  (Tracking)  │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Requirements

* An OpenShift cluster with OpenShift AI (RHOAI) 2.17+ installed:
  * The `dashboard`, `trainingoperator`, `ray`, and `workbenches` components enabled
* Worker nodes with NVIDIA GPUs (optional, CPU training supported)
* A dynamic storage provisioner supporting RWX PVC provisioning (e.g., `nfs-csi`)
* KubeRay operator installed

## Setup

### Setup Workbench

* Access the OpenShift AI dashboard
* Log in, then go to _Data Science Projects_ and create a project (e.g., `feast-trainer-demo`)
* Create a workbench with the following settings:
    * Select the `PyTorch` notebook image
    * Select the `Medium` container size
    * Create a storage with RWX capability (50Gi recommended) using `nfs-csi` storage class
* From the workbench, clone this repository
* Navigate to the `examples/feast-kuberay-mlops` directory and open the `feast-mlops` notebook

### Apply Infrastructure Manifests

Before running the notebook, apply the infrastructure manifests:

```bash
# Create namespace
kubectl apply -f ../../v2/manifests/01-namespace.yaml

# Create ClusterTrainingRuntime (cluster-scoped)
kubectl apply -f ../../v2/manifests/02-clustertrainingruntime.yaml

# Create PostgreSQL (for Feast registry + MLflow backend)
kubectl apply -f ../../v2/manifests/03-feast-postgres.yaml

# Create MLflow tracking server
kubectl apply -f ../../v2/manifests/04-mlflow.yaml

# Create KubeRay cluster
kubectl apply -f ../../v2/manifests/05-kuberay-cluster.yaml

# Create RBAC and shared PVC
kubectl apply -f ../../v2/manifests/06-feast-prereqs.yaml
```

Wait for all pods to be ready:

```bash
kubectl wait --for=condition=available deployment/feast-postgres -n feast-trainer-demo --timeout=120s
kubectl wait --for=condition=ready pod -l ray.io/cluster=feast-ray -n feast-trainer-demo --timeout=180s
```

You can now proceed with the instructions from the notebook. Enjoy!

## Validation

This example has been validated with the following configuration:

### Sales Forecasting - MLP Model - 4x NVIDIA GPU

* Infrastructure:
  * OpenShift AI 2.19
  * 4x NVIDIA GPU (A100/H100)
  * NFS-CSI storage provisioner
* Configuration:
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
    scheduler: ReduceLROnPlateau
    distributed: PyTorch DDP (4 GPUs)
    
    # Tracking
    experiment_tracking: MLflow
    metrics: MAPE, RMSE, MAE, train_loss, val_loss
    ```
* Job:
    ```yaml
    num_nodes: 1
    gpus_per_node: 4 (auto-detected)
    resources_per_node:
      memory: 16Gi
      cpu: 8
    runtime: torch-with-storage
    storage: feast-pvc (50Gi RWX)
    ```
* Results:
    ```
    Best MAPE: 2.3%
    Best RMSE: 500
    Best val_loss: 0.0009
    Training time: ~15 seconds (15 epochs)
    ```
* Metrics:
    ![](./docs/mlflow-metrics.png)

### Timing Breakdown

| Component | Duration | Description |
|-----------|----------|-------------|
| Dataprep (Feast + KubeRay) | ~2 min 15s | Feature engineering, materialization |
| Training (4x GPU DDP) | ~44s | 15 epochs with MLflow tracking |
| **Total Pipeline** | **~3 min** | End-to-end |

### Scaling Comparison

| Data Size | Without Feast | With Feast + KubeRay |
|-----------|---------------|----------------------|
| 100K rows | 10s ✅ | 3 min |
| 10M rows | 30 min (OOM risk) | 10 min ✅ |
| 100M rows | ❌ OOM | 30 min ✅ |

## Components

### Feast Feature Store
- **Offline Store**: Ray-based for distributed feature computation
- **Online Store**: PostgreSQL for low-latency serving
- **Registry**: PostgreSQL-backed for feature metadata

### KubeRay
- **RayCluster**: Dedicated cluster for Feast compute
- **RayJob**: Dataprep job for feature engineering

### Kubeflow Training
- **TrainJob**: Distributed PyTorch training with DDP
- **ClusterTrainingRuntime**: Reusable runtime with shared storage

### MLflow
- **Experiment Tracking**: Metrics, parameters, artifacts
- **Backend Store**: PostgreSQL (shared with Feast)
- **Artifact Store**: Shared PVC

## Troubleshooting

### Common Issues

1. **PostgreSQL connection failed**
   - Ensure PostgreSQL pod is running: `kubectl get pods -n feast-trainer-demo | grep postgres`
   - Check secret exists: `kubectl get secret feast-postgres-secret -n feast-trainer-demo`

2. **Ray cluster not ready**
   - Check Ray head pod: `kubectl get pods -n feast-trainer-demo -l ray.io/node-type=head`
   - View Ray dashboard: `kubectl get route feast-ray-dashboard -n feast-trainer-demo`

3. **Training job OOM**
   - Reduce batch size in training configuration
   - Use pre-computed features (recommended pattern)

4. **MLflow metrics not showing**
   - Verify MLflow pod is running: `kubectl get pods -n feast-trainer-demo | grep mlflow`
   - Check route: `kubectl get route mlflow -n feast-trainer-demo`

