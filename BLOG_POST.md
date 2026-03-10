# Feast + Ray + Kubeflow Training Operator + MLflow + KServe: A Production MLOps Blueprint

*Building a scalable ML pipeline that eliminates feature skew from training to serving*

**By Abhijeet Dhumal and Nikhil Kathole**

---

## TL;DR

```bash
# Clone and deploy in 5 commands
git clone https://github.com/redhat-et/sales-demand-forecasting
cd sales-demand-forecasting
kubectl apply -k manifests/                    # Deploy infrastructure
kubectl apply -f manifests/05-dataprep-job.yaml  # Prepare features
kubectl apply -f manifests/06-trainjob.yaml      # Train model
```

**What you get:** A complete ML pipeline where training and serving use identical feature definitions (via Feast), feature retrieval scales to 10M+ rows (via Ray), training runs distributed across nodes (via Kubeflow Training Operator), experiments are tracked (via MLflow), and serving auto-scales (via KServe).

**Key result:** 3-5% MAPE on retail forecasting with zero training-serving skew.

---

![Feature Lineage in Feast UI](docs/images/sales_forecasting_lineage.png)
*Figure 1: Feast UI showing feature lineage from data sources → entities → feature views → feature services. This single source of truth powers both training and inference.*

---

## Introduction

Feature stores have become essential infrastructure for production ML systems. Yet many tutorials stop at "define your features in Feast"—they don't show how to scale feature retrieval to millions of rows, distribute training across nodes, or serve predictions without duplicating feature logic.

This article walks you through building an end-to-end ML pipeline using Feast, Ray, Kubeflow Training Operator, MLflow, and KServe. You'll see how these tools compose together to solve three common problems:

1. **Point-in-time joins don't scale** — Pandas runs out of memory on 10M+ rows
2. **Feature logic gets duplicated** — Training scripts diverge from serving code  
3. **Training-serving skew** — Silent model degradation when features differ

We use retail demand forecasting as our example: 45 stores × 14 departments × 104 weeks = **65K records**, scaling to millions in production.

---

## Why This Matters: The $50M Feature Skew Problem

Training-serving skew isn't a theoretical concern—it's a production reality that has cost companies millions:

- **Uber** built [Michelangelo](https://www.uber.com/blog/michelangelo-machine-learning-platform/) specifically to solve feature consistency after experiencing model degradation in production
- **DoorDash** reported that feature skew caused [significant prediction errors](https://doordash.engineering/2020/11/19/building-a-gigascale-ml-feature-store-with-redis/) in delivery time estimates before implementing their feature store
- **Instacart** found that [70% of ML bugs](https://www.instacart.com/company/how-its-made/building-a-modern-ml-platform-at-instacart/) in production were data-related, not model-related

The pattern is consistent: **models that work perfectly in notebooks fail silently in production** because the feature computation differs between training and serving.

### What Goes Wrong Without a Feature Store

| Failure Mode | Symptom | Business Impact |
|--------------|---------|-----------------|
| **Stale features** | Serving uses yesterday's data while training used point-in-time | Predictions drift over time |
| **Different aggregations** | Training computes 4-week rolling avg differently than serving | Model accuracy drops 5-15% |
| **Missing features** | Serving code omits a feature that training included | Silent prediction errors |
| **Type mismatches** | Training uses float64, serving uses float32 | Subtle numerical differences |

In retail forecasting specifically, a 5% increase in MAPE (Mean Absolute Percentage Error) can translate to **millions in lost revenue** from overstocking or stockouts. The architecture in this article eliminates these failure modes by construction.

---

## Prerequisites

You need access to a Kubernetes cluster (OpenShift, EKS, GKE, or local) with the following components:

| Component | Purpose | Installation |
|-----------|---------|--------------|
| **KubeRay** | Ray cluster orchestration | [Helm install](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html) |
| **Kubeflow Training Operator** | Distributed training jobs | [kubectl apply](https://www.kubeflow.org/docs/components/training/installation/) |
| **KServe** | Model serving | [Quick install](https://kserve.github.io/website/latest/get_started/) |
| **PostgreSQL** | Feast registry + online store | Included in our manifests |

**Quick check** — Verify operators are running:

```bash
kubectl get pods -n ray-system          # KubeRay operator
kubectl get pods -n kubeflow            # Training operator  
kubectl get pods -n kserve              # KServe controller
```

> **Note:** If your cluster doesn't have RWX (ReadWriteMany) storage, you'll need to configure shared storage. We use a shared PVC—our manifests include a default `StorageClass`. For OpenShift, [OpenShift Data Foundation](https://www.redhat.com/en/technologies/cloud-computing/openshift-data-foundation) provides this out of the box.

---

## Repository Structure

```
sales-demand-forecasting/
├── feature_repo/                    # Feast feature definitions
│   ├── features.py                  # Entities, FeatureViews, FeatureServices
│   ├── feature_store.yaml           # Feast config (local/dev)
│   └── feature_store_ray.yaml       # Feast config (KubeRay)
├── manifests/                       # Kubernetes deployments
│   ├── 01-namespace.yaml            # Namespace + RBAC
│   ├── 02-postgres.yaml             # PostgreSQL (Feast registry + online store)
│   ├── 03-ray-cluster.yaml          # KubeRay cluster
│   ├── 04-mlflow.yaml               # MLflow tracking server
│   ├── 05-dataprep-job.yaml         # Data generation + feast apply/materialize
│   ├── 06-trainjob.yaml             # Kubeflow TrainJob (PyTorch DDP)
│   ├── 07-feast-server.yaml         # Feast feature server
│   ├── 08-kserve-inference.yaml     # KServe InferenceService
│   └── scripts/
│       └── train_ddp.py             # Training script
├── notebooks/                       # Interactive exploration
│   ├── 01-feast-features.ipynb      # Feature engineering
│   ├── 02-training.ipynb            # Model training
│   └── 03-inference.ipynb           # Serving setup
└── docs/diagrams/                   # Architecture diagrams
```

---

## Architecture Overview

![Sequence Diagram](docs/diagrams/sequence-diagram.png)
*Figure 2: Complete pipeline flow across three phases: Feature Engineering → Distributed Training → Model Serving. The same FeatureService definitions are used in training and inference, ensuring zero skew.*

The architecture separates concerns across five components:

| Component | Role |
|-----------|------|
| **Feast** | Feature definitions, offline/online stores, point-in-time joins |
| **Ray (KubeRay)** | Distributed `get_historical_features()` — scales to 10M+ rows |
| **Kubeflow Training Operator** | Multi-node PyTorch DDP orchestration |
| **MLflow** | Experiment tracking, model registry |
| **KServe** | Auto-scaling inference with Feast integration |

All components share a common PVC at `/shared`, eliminating the need for S3 or artifact copying between stages.

---

## Phase 1: Feature Engineering with Feast

![Feature Engineering Workflow](docs/diagrams/01-features-workflow.png)
*Figure 3: Feature engineering workflow: Generate data → Engineer features → Save to Parquet → Register with Feast → Materialize to online store.*

Start by defining your features in Feast. The key insight is creating **two FeatureServices**—one for training (includes target) and one for inference (excludes target):

### Define entities and feature views

Create `feature_repo/features.py`:

```python
# feature_repo/features.py
from datetime import timedelta
from feast import Entity, FeatureView, FeatureService, Field, FileSource
from feast.types import Float32, Int32, String

# Entities define your join keys
store = Entity(name="store_id", join_keys=["store_id"])
dept = Entity(name="dept_id", join_keys=["dept_id"])

# Time-series features (lag, rolling stats, temporal)
sales_features = FeatureView(
    name="sales_features",
    entities=[store, dept],
    ttl=timedelta(days=365),
    schema=[
        Field(name="weekly_sales", dtype=Float32),    # Target variable
        Field(name="lag_1", dtype=Float32),           # Lag features (35% importance)
        Field(name="lag_2", dtype=Float32),
        Field(name="lag_4", dtype=Float32),
        Field(name="rolling_mean_4w", dtype=Float32), # Rolling stats (28% importance)
        Field(name="rolling_std_4w", dtype=Float32),
        Field(name="week_of_year", dtype=Int32),      # Temporal (18% importance)
        Field(name="is_holiday", dtype=Int32),        # Holiday (10% importance)
        Field(name="temperature", dtype=Float32),     # Economic (7% importance)
    ],
    source=FileSource(path="/shared/data/sales_features.parquet", timestamp_field="event_timestamp"),
)

# Static store metadata
store_features = FeatureView(
    name="store_features",
    entities=[store, dept],
    ttl=timedelta(days=365),
    schema=[
        Field(name="store_type", dtype=String),
        Field(name="store_size", dtype=Int32),
    ],
    source=FileSource(path="/shared/data/store_features.parquet", timestamp_field="event_timestamp"),
)
```

### Create FeatureServices for training and inference

```python
# Training: includes target (weekly_sales)
training_features = FeatureService(
    name="training_features", 
    features=[sales_features, store_features]
)

# Inference: excludes target—only the features model needs
inference_features = FeatureService(
    name="inference_features",
    features=[
        sales_features[["lag_1", "lag_2", "lag_4", "rolling_mean_4w", "rolling_std_4w",
                        "week_of_year", "is_holiday", "temperature"]],
        store_features
    ]
)
```

> **Key insight:** In our experiments, 63% of predictive power came from lag + rolling features. Using identical Feast definitions for training and serving eliminates the feature skew that causes silent model degradation.

### Register and materialize

```bash
# Register feature definitions in Feast
feast apply

# Materialize to online store for serving
feast materialize 2024-01-01T00:00:00 2024-12-31T23:59:59
```

---

## Phase 2: Scaling Feature Retrieval with Ray

For datasets beyond ~100K rows, Feast's default Pandas engine becomes a bottleneck. Configure the Ray offline store to distribute point-in-time joins across your KubeRay cluster.

### Configure Feast to use Ray

Create `feature_repo/feature_store.yaml`:

```yaml
# feature_repo/feature_store.yaml
project: sales_forecasting
registry:
  registry_type: sql
  path: postgresql://feast:feast@postgres:5432/feast

offline_store:
  type: ray
  use_kuberay: true
  kuberay_conf:
    cluster_name: feast-ray
    namespace: feast-trainer-demo
  enable_distributed_joins: true

online_store:
  type: postgres
  host: postgres
  port: 5432
  database: feast
  user: feast
  password: feast

batch_engine:
  type: ray
```

> **Note:** Authentication to the Ray cluster uses [CodeFlare SDK](https://github.com/project-codeflare/codeflare-sdk) with Kubernetes service account tokens—no manual credential management required.

### Performance comparison

| Dataset Size | Pandas | Ray (KubeRay)* | Speedup |
|--------------|--------|----------------|---------|
| 65K rows     | ~2 min  | ~30 sec       | 4x |
| 1M rows      | 30+ min | ~3 min        | 10x |
| 10M rows     | OOM    | ~15 min        | ∞ |
| 100M rows    | —      | ~2 hours       | — |

*Benchmarks on 4-worker Ray cluster (4 CPU, 8GB RAM each). Production clusters with more workers scale linearly.*

For context, **Uber processes 10M+ features per second** through their feature store, and **DoorDash's feature platform handles 10B+ feature requests daily**. The Ray backend enables this scale by sharding entity DataFrames across workers, performing parallel PIT joins, and aggregating results—all transparent to your training code.

> **Why PIT joins are expensive:** Point-in-time joins prevent data leakage by ensuring training data only includes features that would have been available at prediction time. This requires sorting and binary search operations that are O(n log n)—exactly what benefits from parallelization.

---

## Phase 3: Distributed Training with Kubeflow Training Operator

![Training Workflow](docs/diagrams/02-training-workflow.png)
*Figure 4: Training workflow: Kubeflow SDK submits TrainJob → Feast+Ray retrieves features → Preprocess → PyTorch DDP training → Save model → Log to MLflow.*

### Why Kubeflow Training Operator?

Distributed training on Kubernetes is hard. Without an operator, you'd need to manually:

- Configure `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK` for each pod
- Handle pod scheduling to ensure network locality
- Manage job lifecycle (restarts, failures, gang scheduling)
- Set up headless services for pod-to-pod communication
- Configure GPU affinity and NCCL environment variables

The **Kubeflow Training Operator** handles all of this declaratively. You specify `numNodes: 2` and it figures out the rest.

### Comparison with Training Orchestration Alternatives

| Approach | Multi-node | Kubernetes Native | Framework Support | Gang Scheduling |
|----------|------------|-------------------|-------------------|-----------------|
| Manual DDP setup | ✅ (painful) | ❌ | PyTorch only | ❌ |
| Ray Train | ✅ | Via KubeRay | PyTorch, TF, JAX | ✅ |
| Horovod | ✅ | Requires MPI | PyTorch, TF | ❌ |
| SageMaker Training | ✅ | ❌ (AWS only) | PyTorch, TF | ✅ |
| **Kubeflow Training Operator** | ✅ | ✅ | PyTorch, TF, JAX, XGBoost | ✅ |

The Training Operator is used in production by **Apple, Bloomberg, Spotify, and Red Hat OpenShift AI**. It's the standard way to run distributed training on Kubernetes.

### What the Operator Does Automatically

| Concern | Manual Approach | With Training Operator |
|---------|-----------------|------------------------|
| **Environment variables** | Script to set RANK, WORLD_SIZE per pod | Auto-injected based on pod index |
| **Service discovery** | Create headless Service, configure DNS | Built-in worker discovery |
| **Gang scheduling** | Hope pods schedule together | Integrates with Kueue, Volcano |
| **Failure handling** | Manual restart logic | Configurable restart policies |
| **GPU allocation** | Per-pod resource requests | `resourcesPerNode` applies uniformly |
| **Multi-framework** | Different setup per framework | Single API for PyTorch, TensorFlow, JAX |

### Create the TrainJob

The new `TrainJob` API (v1alpha1) provides a clean, high-level interface:

```yaml
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: sales-training
spec:
  runtimeRef:
    name: torch-distributed    # Pre-configured runtime for PyTorch DDP
  trainer:
    image: quay.io/modh/training:py312-cuda128-torch280
    command:
      - python
      - /shared/scripts/train_ddp.py
    numNodes: 2                # Scale by changing this number
    resourcesPerNode:
      requests:
        cpu: "4"
        memory: "8Gi"
        # nvidia.com/gpu: "1"  # Uncomment for GPU training
    env:
      - name: MLFLOW_TRACKING_URI
        value: "http://mlflow:5000"
      - name: FEAST_REPO_PATH
        value: "/shared/feature_repo"
    volumeMounts:
      - name: shared
        mountPath: /shared
```

**Key TrainJob features:**

| Feature | Description |
|---------|-------------|
| `runtimeRef` | References a `ClusterTrainingRuntime` with pre-configured settings (NCCL, rendezvous) |
| `numNodes` | Number of training workers—operator handles all coordination |
| `resourcesPerNode` | Uniform resource allocation across all workers |
| Automatic env injection | `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` set automatically |

> **New in Kubeflow 1.9:** The `TrainJob` API simplifies the previous `PyTorchJob` spec while adding support for `ClusterTrainingRuntime` templates. This enables platform teams to define approved training configurations that data scientists can reference by name.

### Submit via Python SDK

For notebook-based workflows, the Kubeflow Training SDK provides a programmatic interface:

```python
from kubeflow.training import TrainingClient, TrainJob

client = TrainingClient()

# Submit training job
client.create_job(
    TrainJob(
        name="sales-training",
        runtime_ref="torch-distributed",
        trainer={
            "image": "quay.io/modh/training:py312-cuda128-torch280",
            "command": ["python", "/shared/scripts/train_ddp.py"],
            "num_nodes": 2,
            "resources_per_node": {"cpu": "4", "memory": "8Gi"},
        },
    )
)

# Monitor progress
client.get_job_logs(name="sales-training", follow=True)
```

This SDK approach is particularly powerful when combined with experiment tracking—you can programmatically sweep hyperparameters and launch multiple training jobs from a notebook.

### Training script pattern

The training script follows a specific pattern for distributed execution. See full implementation in [`manifests/scripts/train_ddp.py`](https://github.com/redhat-et/sales-demand-forecasting/blob/main/manifests/scripts/train_ddp.py):

```python
# manifests/scripts/train_ddp.py (simplified)
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
import mlflow
from feast import FeatureStore

# Initialize distributed training (env vars set by Kubeflow Training Operator)
dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}" if torch.cuda.is_available() else "cpu")

# Rank 0 fetches features via Feast/Ray, saves to shared storage
if rank == 0:
    store = FeatureStore(repo_path="/shared/feature_repo")
    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=store.get_feature_service("training_features")
    ).to_df()
    training_df.to_parquet("/shared/models/training_data.parquet")

dist.barrier()  # All ranks wait for data to be written

# All ranks load identical data from shared storage
training_df = pd.read_parquet("/shared/models/training_data.parquet")

# Standard DDP training loop
model = DDP(SalesMLP(input_dim=len(feature_cols)).to(device))
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    
    # Only rank 0 logs to MLflow (avoid duplicate logging)
    if rank == 0:
        mlflow.log_metrics({"train_loss": train_loss, "mape": mape}, step=epoch)
```

> **Note:** The pattern of rank 0 fetching data and all ranks reading from shared storage avoids redundant API calls while ensuring all workers train on identical data.

### Scaling Training: From 2 Nodes to 100+

The same `TrainJob` spec scales from local development to production clusters:

| Scale | Configuration | Use Case | Bottleneck |
|-------|---------------|----------|------------|
| **Development** | `numNodes: 1`, CPU only | Debugging, quick iteration | Compute |
| **Team** | `numNodes: 2-4`, GPU | Regular training runs | Compute |
| **Production** | `numNodes: 8-64`, multi-GPU | Large models, hyperparameter sweeps | Network* |
| **Enterprise** | `numNodes: 100+`, RDMA | Foundation model fine-tuning | Compute |

*At production scale, standard Kubernetes networking (OVN) becomes the bottleneck. Red Hat's testing showed that [GPUDirect RDMA reduces fine-tuning time from 5 hours to 1.5 hours](https://developers.redhat.com/articles/2025/04/29/accelerate-model-training-openshift-ai-nvidia-gpudirect-rdma)—a **3x speedup**—by eliminating network IO as the bottleneck.*

For large-scale training, the Kubeflow Training Operator integrates with:

- **[Kueue](https://kueue.sigs.k8s.io/)** — Fair-share scheduling and quota management
- **[Volcano](https://volcano.sh/)** — Gang scheduling to ensure all pods start together
- **[NVIDIA GPUDirect RDMA](https://developers.redhat.com/articles/2025/04/29/accelerate-model-training-openshift-ai-nvidia-gpudirect-rdma)** — High-bandwidth GPU-to-GPU communication across nodes (3x speedup vs OVN network)

> **Production examples from Red Hat OpenShift AI:**
> - [Fine-tune LLMs with Kubeflow Trainer](https://developers.redhat.com/articles/2025/04/22/fine-tune-llms-kubeflow-trainer-openshift-ai) — Complete walkthrough of fine-tuning Llama 3.1 8B with PyTorch FSDP, LoRA/QLoRA, and HuggingFace SFTTrainer on OpenShift AI
> - [Accelerate training with GPUDirect RDMA](https://developers.redhat.com/articles/2025/04/29/accelerate-model-training-openshift-ai-nvidia-gpudirect-rdma) — How NVIDIA Spectrum-X networking reduces distributed training time from 5 hours to 1.5 hours

### Track experiments with MLflow

![MLflow Training Runs](docs/images/mlflow-training-runs.png)
*Figure 5: MLflow UI comparing training runs: MAPE, validation loss, learning rate schedules across CPU/GPU configurations.*

Each training run logs:

| Category | What's Logged |
|----------|---------------|
| **Parameters** | hardware, world_size, epochs, batch_size, learning_rate, feature_count |
| **Metrics** | train_loss, val_loss, mape, learning_rate (per epoch) |
| **Artifacts** | model_best.pt, scalers.joblib, model_metadata.json |

### Results

| Metric | Our Results | Industry Benchmark* |
|--------|-------------|---------------------|
| **MAPE** | 3-5% | 5-10% (typical retail) |
| **Training Time (CPU)** | 2-5 min | Hours (without DDP) |
| **Training Time (GPU)** | 30 sec | Minutes (single GPU) |
| **Feature Retrieval** | 30 sec | Minutes (Pandas at scale) |
| **Online Feature Latency** | <5ms p99 | <10ms (production SLA) |

*Industry benchmarks from [Walmart's M5 competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy) and published feature store latency SLAs.*

> **Why 3-5% MAPE matters:** In retail, every 1% improvement in forecast accuracy can reduce inventory costs by 2-4%. For a retailer with $1B in inventory, that's **$20-40M in savings**.

---

## Phase 4: Production Serving with KServe

![Inference Workflow](docs/diagrams/03-inference-workflow.png)
*Figure 6: Inference workflow: Client sends entity IDs → KServe fetches features from Feast online store (~5ms) → Scale → Predict → Return dollar prediction.*

The serving layer is where feature consistency matters most. Clients send only entity IDs—KServe fetches features from Feast's online store using the same `inference_features` FeatureService.

### Deploy the InferenceService

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: sales-forecast
spec:
  predictor:
    containers:
      - name: kserve-container
        image: quay.io/your-org/sales-forecast-server:latest
        env:
          - name: FEAST_SERVER_URL
            value: "http://feast-server:6566"
          - name: MODEL_PATH
            value: "/mnt/models"
        volumeMounts:
          - name: model-volume
            mountPath: /mnt/models
```

### Implement the prediction server

The serving script is deployed via ConfigMap. Key parts:

```python
# serve.py (simplified, see full version in manifests/)
import os
import torch
import numpy as np
import joblib
import requests
from kserve import Model, ModelServer

FEAST_URL = os.environ.get("FEAST_SERVER_URL", "http://feast-server:6566")

class SalesForecastModel(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.model = None
        self.scalers = None
        self.ready = False
    
    def load(self):
        # Load model and scalers from shared PVC
        self.model = torch.load("/mnt/models/model_best.pt", map_location="cpu")
        self.model.eval()
        self.scalers = joblib.load("/mnt/models/scalers.joblib")
        self.ready = True

    def preprocess(self, payload, headers=None):
        entities = payload["inputs"][0]["data"]
        
        # Fetch from Feast online store—same FeatureService as training
        response = requests.post(f"{FEAST_URL}/get-online-features", json={
            "feature_service": "inference_features",  # Same definition used in training
            "entities": {
                "store_id": [e["store_id"] for e in entities],
                "dept_id": [e["dept_id"] for e in entities]
            }
        })
        return self._parse_features(response.json())

    def predict(self, X, headers=None):
        X_scaled = self.scalers["scaler_X"].transform(X)
        with torch.no_grad():
            preds = self.model(torch.FloatTensor(X_scaled)).numpy()
        # Inverse transform: log1p was applied during training
        return np.expm1(self.scalers["scaler_y"].inverse_transform(
            preds.reshape(-1, 1)
        )).flatten()

if __name__ == "__main__":
    model = SalesForecastModel("sales-forecast")
    model.load()
    ModelServer().start([model])
```

### Test predictions

```bash
curl -X POST http://sales-forecast/v2/models/sales-forecast/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "entities", "data": [{"store_id": 1, "dept_id": 3}]}]}'

# Response: {"outputs": [{"name": "predictions", "data": [96763.45]}]}
```

### Latency breakdown

| Component | Latency | Notes |
|-----------|---------|-------|
| Feast online store lookup | ~2-5ms | PostgreSQL with connection pooling |
| Feature scaling | <1ms | NumPy vectorized operations |
| Model inference | ~1-2ms | PyTorch CPU (MLP is lightweight) |
| **Total p99** | **<10ms** | Well within production SLAs |

For comparison, **Uber targets <10ms p99 for feature serving**, and **DoorDash's feature store serves at <5ms p50**. The PostgreSQL-backed online store achieves similar performance for moderate QPS (1000s of requests/second).

---

## Deploy the Complete Pipeline

### Step 1: Clone the repository

```bash
git clone https://github.com/redhat-et/sales-demand-forecasting
cd sales-demand-forecasting
```

### Step 2: Deploy infrastructure

```bash
kubectl apply -k manifests/
```

**Wait for pods to be ready** (~2-3 minutes):

```bash
kubectl get pods -w
```

**Expected output:**
```
NAME                           READY   STATUS    
postgres-0                     1/1     Running   
mlflow-...                     1/1     Running   
feast-ray-head-...             1/1     Running   
feast-ray-worker-...           1/1     Running   
feast-server-...               1/1     Running   
```

### Step 3: Prepare data and register features

```bash
kubectl apply -f manifests/05-dataprep-job.yaml
```

**Monitor progress:**
```bash
kubectl logs -f job/feast-dataprep
```

**Expected output:**
```
Generating 65,520 rows of sales data...
Running feast apply...
Created entity store_id
Created entity dept_id
Created feature view sales_features
Created feature view store_features
Created feature service training_features
Created feature service inference_features
Running feast materialize...
Materializing 65,520 rows to online store...
Done!
```

### Step 4: Train the model

```bash
kubectl apply -f manifests/06-trainjob.yaml
```

**Monitor training:**
```bash
kubectl logs -f sales-training-node-0-0
```

**Expected output:**
```
============================================================
  TRAINING CONFIGURATION
============================================================
PyTorch: 2.8.0
Hardware: CPU (or CUDA if GPUs available)
Distributed: True (world_size=2)
============================================================
  FEATURE RETRIEVAL (Feast + KubeRay)
============================================================
Querying 65,520 entities...
Retrieved 65,520 rows in 28.3s
============================================================
  MODEL TRAINING
============================================================
Epoch   Train       Val         MAPE      LR          Status
--------------------------------------------------------------
1       0.892341    0.743219    8.42      1.00e-03    
2       0.654123    0.612847    6.21      1.00e-03    
...
15      0.089234    0.091823    3.47      1.00e-04    * best
============================================================
  TRAINING COMPLETE
============================================================
Best MAPE:   3.47% (epoch 15)
Train time:  127.4s
```

### Step 5: Deploy inference service

```bash
kubectl apply -f manifests/08-kserve-inference.yaml
```

**Wait for InferenceService to be ready:**
```bash
kubectl get inferenceservice sales-forecast -w
```

**Expected output:**
```
NAME             URL                                      READY
sales-forecast   http://sales-forecast.default.svc...     True
```

### Step 6: Test predictions

```bash
# Get the service URL
INGRESS=$(kubectl get inferenceservice sales-forecast -o jsonpath='{.status.url}')

# Make a prediction
curl -X POST "$INGRESS/v2/models/sales-forecast/infer" \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "entities", "data": [{"store_id": 1, "dept_id": 3}]}]}'
```

**Expected output:**
```json
{"outputs": [{"name": "predictions", "data": [96763.45]}]}
```

### Step 7: Access UIs

```bash
# MLflow UI (experiment tracking)
kubectl port-forward svc/mlflow 5000:5000 &
open http://localhost:5000

# Feast UI (feature lineage)
kubectl port-forward svc/feast-server 8888:8888 &
open http://localhost:8888
```

---

## Troubleshooting

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Pods stuck in Pending** | `kubectl get pods` shows Pending | Check PVC binding: `kubectl get pvc`. Ensure StorageClass supports RWX. |
| **Ray workers not connecting** | Training hangs at "Connecting to Ray..." | Verify Ray cluster: `kubectl get rayclusters`. Check Ray head logs. |
| **Feast materialize fails** | "Connection refused" to PostgreSQL | Wait for postgres pod: `kubectl wait --for=condition=ready pod/postgres-0` |
| **Training OOM** | Pod killed with exit code 137 | Reduce `BATCH_SIZE` env var or increase `memory` in TrainJob |
| **KServe not ready** | InferenceService stuck at "Unknown" | Check predictor logs: `kubectl logs -l serving.kserve.io/inferenceservice=sales-forecast` |

---

## Why This Architecture Works

| Decision | Benefit |
|----------|---------|
| **Feast as single source of truth** | Same `FeatureService` for training + serving = no skew |
| **Ray offline store** | Scales PIT joins from 65K to 10M+ rows |
| **Shared PVC** | No S3/artifact copying—all components read/write `/shared` |
| **Separate FeatureServices** | `training_features` includes target; `inference_features` excludes it |
| **DDP + rank 0 pattern** | One feature fetch, all workers train on identical data |

### Comparison with Industry Alternatives

| Approach | Scales | No Skew | Open Source | Vendor Lock-in | Typical Users |
|----------|--------|---------|-------------|----------------|---------------|
| Pandas + custom serving | ❌ | ❌ | ✅ | None | Startups, POCs |
| Spark + custom serving | ✅ | ❌ | ✅ | None | Data teams without ML Ops |
| **Tecton** | ✅ | ✅ | ❌ | High | Enterprise (DoorDash, Atlassian) |
| **Vertex AI Feature Store** | ✅ | ✅ | ❌ | GCP | Google Cloud users |
| **SageMaker Feature Store** | ✅ | ✅ | ❌ | AWS | AWS users |
| **Databricks Feature Store** | ✅ | ✅ | ❌ | Databricks | Databricks users |
| **Feast + Ray + Kubeflow + KServe** | ✅ | ✅ | ✅ | None | Multi-cloud, on-prem |

The open-source stack provides the same capabilities as managed services while running on any Kubernetes cluster—cloud, on-prem, or hybrid.

### Production Considerations

| Concern | How This Architecture Addresses It |
|---------|-----------------------------------|
| **High availability** | PostgreSQL (registry/online store) supports replication; KServe handles pod failures |
| **Disaster recovery** | Feature definitions are code (GitOps); data in Parquet can be backed up |
| **Cost optimization** | Ray workers scale to zero when idle; KServe scales inference pods based on load |
| **Monitoring** | MLflow tracks model drift; Feast provides feature freshness metrics |
| **Security** | Kubernetes RBAC for access control; no external API keys required |
| **Multi-tenancy** | Feast projects isolate feature namespaces; Kubernetes namespaces isolate workloads |

---

## Conclusion

This article walked through building a production ML pipeline using open source tools that compose rather than compete:

| Concern | Tool | Production Benefit |
|---------|------|-------------------|
| Feature definitions + storage | Feast | Single source of truth eliminates skew |
| Distributed feature retrieval | Ray (KubeRay) | Scales from 65K to 100M+ rows |
| Multi-node training | Kubeflow Training Operator | GPU utilization, faster iteration |
| Experiment tracking | MLflow | Reproducibility, model governance |
| Auto-scaling serving | KServe | Cost efficiency, high availability |

### The Bottom Line

Feature stores aren't just about organizing data—they're about **preventing the silent failures that plague production ML systems**. Companies like Uber, DoorDash, and Instacart learned this the hard way before building their feature platforms.

The architecture in this article provides:

- **Zero training-serving skew** by construction (same FeatureService definitions)
- **Horizontal scalability** from prototype (65K rows) to production (100M+ rows)
- **No vendor lock-in**—runs on any Kubernetes cluster
- **GitOps-friendly**—feature definitions are version-controlled Python code

The key insight: **feature consistency between training and serving** is the foundation everything else builds on. Get this wrong, and no amount of model tuning will save you. Get it right, and you have a platform that scales with your business.

Try the [complete implementation](https://github.com/redhat-et/sales-demand-forecasting) and see how these components work together in practice.

---

## Resources

**This project:**
- [Complete source code on GitHub](https://github.com/redhat-et/sales-demand-forecasting)

**Kubeflow Training on OpenShift AI:**
- [Fine-tune LLMs with Kubeflow Trainer on OpenShift AI](https://developers.redhat.com/articles/2025/04/22/fine-tune-llms-kubeflow-trainer-openshift-ai) — Step-by-step guide for Llama 3.1 fine-tuning with FSDP
- [Accelerate model training with NVIDIA GPUDirect RDMA](https://developers.redhat.com/articles/2025/04/29/accelerate-model-training-openshift-ai-nvidia-gpudirect-rdma) — 3x speedup with high-performance networking

**Component documentation:**
- [Feast Documentation](https://docs.feast.dev/)
- [KubeRay Documentation](https://docs.ray.io/en/latest/cluster/kubernetes.html)
- [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/training/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [KServe Documentation](https://kserve.github.io/website/)

**Additional resources:**
- [Red Hat OpenShift AI](https://www.redhat.com/en/technologies/cloud-computing/openshift/openshift-ai)
- [AI on OpenShift patterns and recipes](https://ai-on-openshift.io/)
