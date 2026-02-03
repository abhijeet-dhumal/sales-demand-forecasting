# Kubernetes Manifests

## Quick Start

```bash
# 1. Create namespace and storage
kubectl apply -f 01-namespace.yaml
kubectl apply -f 02-pvc-shared-storage.yaml

# 2. (Optional) Deploy Feast PostgreSQL
kubectl apply -f 05-feast-postgres.yaml

# 3. Create custom ClusterTrainingRuntime
kubectl apply -f 03-clustertrainingruntime.yaml

# 4. Submit TrainJob
kubectl apply -f 04-trainjob.yaml

# Monitor job
kubectl get trainjobs -n feast-trainer-demo
kubectl logs -f -n feast-trainer-demo -l trainer.kubeflow.org/trainjob-name=sales-forecasting-training
```

## Manifests

| File | Description |
|------|-------------|
| `01-namespace.yaml` | Namespace for training jobs |
| `02-pvc-shared-storage.yaml` | PVCs for data and models |
| `03-clustertrainingruntime.yaml` | Custom runtime with Feast support |
| `04-trainjob.yaml` | TrainJob v2 example |
| `05-feast-postgres.yaml` | PostgreSQL for Feast (optional) |
| `06-trainjob-customtrainer.yaml` | SDK-style CustomTrainer pattern |

## Using SDK Instead of Raw Manifests

The SDK is preferred over raw manifests:

```python
from kubeflow.trainer import TrainerClient, CustomTrainer

client = TrainerClient()
job = client.train(
    runtime="torch-feast",  # Uses our custom runtime
    trainer=CustomTrainer(
        func=train_function,
        num_nodes=2,
        resources_per_node={"cpu": 8, "memory": "32Gi"},
    ),
)
```

The SDK handles:
- Function serialization
- Package installation
- Environment setup
- TrainJob creation

## TrainJob v2 vs PyTorchJob

| Aspect | PyTorchJob (v1) | TrainJob (v2) |
|--------|-----------------|---------------|
| API Group | `kubeflow.org/v1` | `trainer.kubeflow.org/v1alpha1` |
| Runtime Config | Embedded | `ClusterTrainingRuntime` (reusable) |
| Initializers | Manual | Built-in (model, dataset) |
| Checkpointing | Manual | JIT checkpoint support |
| Progression | None | HTTP metrics endpoint |

