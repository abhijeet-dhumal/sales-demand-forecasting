# Model Training with Kubeflow

## Workflow

![Training Workflow](../../docs/diagrams/02-training-workflow.png)

This directory contains the distributed training notebook and scripts for the sales demand forecasting model.

## Overview

| File | Description |
|------|-------------|
| `02-training.ipynb` | Main training notebook |
| `training_script.py` | Self-contained training function for Kubeflow |

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Kubeflow       │     │  Feast          │     │  MLflow         │
│  TrainerClient  │────▶│  Remote Client  │────▶│  Tracking       │
│                 │     │  (gRPC)         │     │  + Registry     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│  TrainJob       │
│  - PyTorch DDP  │
│  - Multi-GPU    │
└─────────────────┘
```

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Features | Registered and materialized (run `01a-local.ipynb` or have Feast Operator deployed) |
| MLflow | MLflow Operator deployed (`manifests/mlflow/`) |
| Workbench | OpenShift AI workbench with Feature Store connection |
| Runtime | `torch-distributed` ClusterTrainingRuntime available |

## What the Notebook Does

| Step | Action | Component |
|------|--------|-----------|
| 1 | Configure training parameters | Notebook |
| 2 | Load training function | `training_script.py` |
| 3 | Submit TrainJob | Kubeflow TrainerClient |
| 4 | Wait for completion | Kubeflow |
| 5 | View results | MLflow |

## Training Function (`training_script.py`)

The training function is self-contained for Kubeflow's `CustomTrainer`:

```python
def train_fn(epochs, namespace, output_dir, feast_config_path):
    # 1. Feast feature retrieval (rank 0 only)
    # 2. PyTorch DDP distributed training (all ranks)
    # 3. MLflow logging & model registry (rank 0 only)
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **Feast Remote Client** | Retrieves `training_features` via gRPC |
| **PyTorch DDP** | Distributed training across GPUs |
| **MLflow** | Logs params, metrics, registers model |

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NAMESPACE` | `feast-trainer-demo` | Kubernetes namespace |
| `PVC` | `shared` | PVC for model artifacts |
| `RUNTIME` | `torch-distributed` | Training runtime |
| `EPOCHS` | `50` | Training epochs |
| `FEAST_CONFIG_PATH` | `/opt/app-root/src/feast-config/salesforecasting` | Feast config |

## Output Artifacts

After training completes, artifacts are saved to `/shared/models/`:

| File | Description |
|------|-------------|
| `best_model.pt` | PyTorch model weights |
| `scalers.joblib` | Feature and target scalers |
| `feature_cols.pkl` | Feature column names |
| `model_metadata.json` | Architecture info, MLflow run ID |

## MLflow Integration

![MLflow Workspace](../../docs/images/mlflow-workspace.png)

![MLflow Training Runs](../../docs/images/mlflow-training-runs.png)

The training automatically:
1. Creates experiment: `{namespace}/sales-forecasting`
2. Logs parameters: epochs, train_rows, val_rows, features
3. Logs metrics per epoch: train_loss, val_loss, mape, lr
4. Registers model: `sales-forecasting-model` with version tags

## Running the Notebook

1. Open `02-training.ipynb` in JupyterLab
2. Run cells sequentially
3. Monitor the TrainJob:
   ```bash
   oc logs -f -l trainer.kubeflow.org/trainjob-name=sales-forecast -n feast-trainer-demo
   ```
4. View results in MLflow UI

## Troubleshooting

### TrainJob Stuck in Pending

Check if runtime exists:
```bash
oc get clustertrainingruntime torch-distributed
```

### Feast Connection Failed

Verify ConfigMaps exist:
```bash
oc get configmap feast-salesforecasting-client -n feast-trainer-demo
oc get configmap feast-salesforecasting-client-ca -n feast-trainer-demo
```

### MLflow Authentication Error

Ensure token is valid:
```bash
oc whoami --show-token
```

### Out of Memory

Reduce batch size in `training_script.py` or request more memory:
```python
resources_per_node={'gpu': 1, 'cpu': 4, 'memory': '16Gi'}
```

## Next Steps

After training completes:
- View model in MLflow Model Registry
- Proceed to `03_inferencing/` for model deployment
