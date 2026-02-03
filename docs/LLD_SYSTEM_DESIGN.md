# Low-Level Design: Kubeflow Trainer + SDK + Feast Integration

**Version**: 1.0  
**Date**: February 2026  
**Status**: Reference Architecture  

## References

- [ODH Trainer](https://github.com/opendatahub-io/trainer) - Kubeflow Trainer v2.0 fork for OpenShift AI
  - **API**: `trainer.kubeflow.org/v1alpha1` TrainJob (NOT PyTorchJob)
  - **CRDs**: TrainJob, ClusterTrainingRuntime, TrainingRuntime
- [ODH Kubeflow SDK](https://github.com/opendatahub-io/kubeflow-sdk) - Python SDK v0.2.1+rhai2
  - **Trainers**: CustomTrainer, BuiltinTrainer, TransformersTrainer, TrainingHubTrainer
  - **Module**: `kubeflow.trainer`
- [Feast Feature Store](https://github.com/feast-dev/feast) - v0.59.0

---

## 1. Executive Summary

This document defines the standard integration pattern for distributed ML training on OpenShift AI using:

| Component | Version | Purpose |
|-----------|---------|---------|
| **Kubeflow Trainer** | v2.0 (ODH fork) | Distributed training via **TrainJob v2 API** |
| **Kubeflow SDK** | v0.2.1+rhai2 | `TrainerClient`, built-in trainers (Custom, Transformers, TrainingHub) |
| **Feast** | v0.59.0 | Feature management & serving |
| **OpenShift AI** | 2.17+ | Platform (RHOAI) |

### Use Case: Sales Demand Forecasting
- **Input**: Historical sales, external factors (weather, economy, promotions)
- **Output**: Weekly sales predictions per store-department
- **Scale**: 421K records, 45 stores × 99 departments

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OpenShift AI Platform                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │  Workbench  │───▶│  Kubeflow SDK   │───▶│    Kubeflow Trainer         │ │
│  │  (Jupyter)  │    │  TrainerClient  │    │    Controller               │ │
│  └─────────────┘    └─────────────────┘    └─────────────────────────────┘ │
│         │                                              │                    │
│         │                                              ▼                    │
│         │           ┌─────────────────────────────────────────────────────┐│
│         │           │              Training Pod(s)                        ││
│         │           │  ┌────────────┐  ┌────────────┐  ┌────────────┐    ││
│         │           │  │  Master    │  │  Worker-0  │  │  Worker-N  │    ││
│         │           │  │  (Rank 0)  │  │  (Rank 1)  │  │  (Rank N)  │    ││
│         │           │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘    ││
│         │           │        │               │               │            ││
│         │           │        └───────────────┼───────────────┘            ││
│         │           │                        │ PyTorch DDP                ││
│         │           └────────────────────────┼────────────────────────────┘│
│         │                                    │                              │
│         ▼                                    ▼                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         Feast Feature Store                             ││
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 ││
│  │  │  Registry   │    │   Offline   │    │   Online    │                 ││
│  │  │ (PostgreSQL)│    │   Store     │    │   Store     │                 ││
│  │  │             │    │ (PostgreSQL)│    │ (PostgreSQL)│                 ││
│  │  └─────────────┘    └─────────────┘    └─────────────┘                 ││
│  │         │                  │                  │                         ││
│  │         └──────────────────┼──────────────────┘                         ││
│  │                            │                                            ││
│  │                   ┌────────┴────────┐                                   ││
│  │                   │  Ray Compute    │                                   ││
│  │                   │  Engine         │                                   ││
│  │                   └─────────────────┘                                   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    Persistent Storage (PVCs)                            ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         ││
│  │  │  shared-storage │  │  model-storage  │  │  checkpoint-pvc │         ││
│  │  │  (feature_repo) │  │  (trained model)│  │  (training ckpt)│         ││
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Design

### 3.1 Feast Feature Store Layer

#### 3.1.1 Data Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Feature Store Schema                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ENTITIES                                                           │
│  ┌─────────────┐     ┌─────────────┐                               │
│  │   store     │     │    dept     │                               │
│  │  (Int64)    │     │   (Int64)   │                               │
│  │  PK: 1-45   │     │  PK: 1-99   │                               │
│  └─────────────┘     └─────────────┘                               │
│                                                                     │
│  FEATURE VIEWS                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  sales_history_features                                      │   │
│  │  ├── weekly_sales (Float64) ─── TARGET (exclude from input)  │   │
│  │  ├── is_holiday (Int64)                                      │   │
│  │  ├── week_of_year (Int64)                                    │   │
│  │  ├── month, quarter (Int64)                                  │   │
│  │  ├── sales_lag_1, sales_lag_2, sales_lag_4 (Float64)        │   │
│  │  └── sales_rolling_mean_4, sales_rolling_mean_12 (Float64)   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  store_external_features                                     │   │
│  │  ├── temperature, fuel_price, cpi, unemployment (Float64)    │   │
│  │  ├── markdown1-5, total_markdown, has_markdown               │   │
│  │  └── store_type (String), store_size (Int64)                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ON-DEMAND FEATURE VIEWS (computed at retrieval time)              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  feature_transformations                                     │   │
│  │  ├── temperature_normalized ─── (temp - 5) / 95              │   │
│  │  ├── holiday_markdown_interaction ─── is_holiday * markdown  │   │
│  │  ├── markdown_momentum ─── markdown / (sales_lag_1 + 1)      │   │
│  │  └── seasonal_sine, seasonal_cosine ─── cyclical encoding    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  temporal_transformations                                    │   │
│  │  ├── sales_velocity ─── (lag1 - lag2) / (lag2 + 1)          │   │
│  │  ├── sales_acceleration ─── velocity change                  │   │
│  │  └── demand_stability_score ─── 1 - CV                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 3.1.2 Feature Store Configuration

```yaml
# feature_store.yaml - Production Configuration
project: sales_demand_forecasting

# Registry: Feature definitions & metadata
registry:
  registry_type: sql
  path: postgresql+psycopg://feast:${FEAST_PASSWORD}@feast-postgres:5432/feast_registry
  cache_ttl_seconds: 60
  sqlalchemy_config_kwargs:
    pool_pre_ping: true
    pool_size: 10

# Offline Store: Historical feature retrieval (training)
offline_store:
  type: postgres
  host: feast-postgres.${NAMESPACE}.svc.cluster.local
  port: 5432
  database: feast_offline
  user: feast
  password_secret: feast-credentials  # K8s Secret reference

# Batch Engine: Distributed compute for joins/transforms
batch_engine:
  type: ray.engine
  max_workers: 8
  enable_optimization: true
  enable_distributed_joins: true
  target_partition_size_mb: 64

# Online Store: Low-latency serving (inference)
online_store:
  type: postgres
  host: feast-postgres.${NAMESPACE}.svc.cluster.local
  database: feast_online
  user: feast
  password_secret: feast-credentials

entity_key_serialization_version: 3
```

#### 3.1.3 Feature Retrieval Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Training Feature Retrieval                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Entity DataFrame (from PostgreSQL)                                       │
│     ┌────────────────────────────────────────────────────────────────┐      │
│     │  SELECT DISTINCT store, dept, date FROM sales_features         │      │
│     │  ORDER BY date, store, dept                                    │      │
│     │  [Optional: LIMIT N for sampling]                              │      │
│     └────────────────────────────────────────────────────────────────┘      │
│                              │                                               │
│                              ▼                                               │
│  2. Feast Historical Feature Retrieval                                       │
│     ┌────────────────────────────────────────────────────────────────┐      │
│     │  store.get_historical_features(                                │      │
│     │      entity_df=entity_df,                                      │      │
│     │      features=store.get_feature_service('demand_forecasting')  │      │
│     │  )                                                             │      │
│     └────────────────────────────────────────────────────────────────┘      │
│                              │                                               │
│                              ▼                                               │
│  3. Ray Compute Engine (parallel processing)                                 │
│     ┌────────────────────────────────────────────────────────────────┐      │
│     │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │      │
│     │  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │  │ Worker N │       │      │
│     │  │ Joins    │  │ Joins    │  │ Joins    │  │ Joins    │       │      │
│     │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │      │
│     │       │             │             │             │              │      │
│     │       └─────────────┴─────────────┴─────────────┘              │      │
│     │                           │                                    │      │
│     │                    Combined Result                             │      │
│     └───────────────────────────┼────────────────────────────────────┘      │
│                                 │                                            │
│                                 ▼                                            │
│  4. On-Demand Transformations (computed during retrieval)                    │
│     ┌────────────────────────────────────────────────────────────────┐      │
│     │  feature_transformations() + temporal_transformations()         │      │
│     │  → sales_velocity, markdown_momentum, seasonal_encoding, etc.   │      │
│     └────────────────────────────────────────────────────────────────┘      │
│                                 │                                            │
│                                 ▼                                            │
│  5. Output: training_df (DataFrame with all features)                        │
│     ┌────────────────────────────────────────────────────────────────┐      │
│     │  Columns: store, dept, date, [27 features], weekly_sales       │      │
│     │  Rows: 421,570 (full) or N (sampled)                           │      │
│     └────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 3.2 Kubeflow SDK Layer

#### 3.2.1 SDK Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Kubeflow SDK Architecture                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      TrainerClient                                   │    │
│  │  (opendatahub-io/kubeflow-sdk v0.2.1+rhai2)                         │    │
│  │                                                                      │    │
│  │  Methods:                                                            │    │
│  │  ├── create_job(job_kind, name, train_func, parameters, ...)        │    │
│  │  ├── get_job(name, namespace, job_kind)                             │    │
│  │  ├── get_job_logs(name, namespace, follow=True)                     │    │
│  │  ├── delete_job(name, namespace)                                    │    │
│  │  └── wait_for_job_conditions(name, expected_conditions)             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              │ K8s API                                       │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   Kubernetes Custom Resources                        │    │
│  │                                                                      │    │
│  │  ┌────────────────────────────────────────────────────────────────┐ │    │
│  │  │  PyTorchJob (kubeflow.org/v1)                                  │ │    │
│  │  │  ├── spec.pytorchReplicaSpecs.Master                           │ │    │
│  │  │  ├── spec.pytorchReplicaSpecs.Worker                           │ │    │
│  │  │  └── spec.runPolicy (backoffLimit, cleanPodPolicy)             │ │    │
│  │  └────────────────────────────────────────────────────────────────┘ │    │
│  │                                                                      │    │
│  │  ┌────────────────────────────────────────────────────────────────┐ │    │
│  │  │  TrainJob (trainer.kubeflow.org/v1alpha1) - NEW in v2.0        │ │    │
│  │  │  ├── spec.trainer (image, command, args)                       │ │    │
│  │  │  ├── spec.runtimeRef (ClusterTrainingRuntime)                  │ │    │
│  │  │  └── spec.modelConfig, datasetConfig                           │ │    │
│  │  └────────────────────────────────────────────────────────────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 3.2.2 Job Submission Contract

> **IMPORTANT**: ODH Trainer uses **TrainJob v2 API** (`trainer.kubeflow.org/v1alpha1`), NOT the legacy PyTorchJob. The v2 API is a unified, declarative abstraction for distributed training.

```python
# SDK Job Submission Pattern (TrainJob v2)
from kubeflow.trainer import TrainerClient, CustomTrainer

# Initialize client (uses KubernetesBackendConfig by default)
client = TrainerClient()

# List available ClusterTrainingRuntimes
for runtime in client.list_runtimes():
    print(f"Runtime: {runtime.name}, Framework: {runtime.trainer.framework}")

# Option 1: CustomTrainer - User-defined Python function
def train_sales_forecast():
    """Self-contained training function with all imports inside."""
    import os
    import torch
    import torch.distributed as dist
    from feast import FeatureStore
    
    # DDP setup (SDK handles MASTER_ADDR, WORLD_SIZE, RANK env vars)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    
    # Feature loading from Feast
    store = FeatureStore(repo_path="/shared/feature_repo")
    features_df = store.get_historical_features(
        entity_df=entity_df,
        features=["sales_features:weekly_sales", "sales_features:lag_1"]
    ).to_df()
    
    # Training loop with DDP
    model = torch.nn.parallel.DistributedDataParallel(model)
    # ... training code ...
    
    dist.destroy_process_group()

# Submit TrainJob with CustomTrainer
job_name = client.train(
    runtime="torch-distributed",  # ClusterTrainingRuntime name
    trainer=CustomTrainer(
        func=train_sales_forecast,
        num_nodes=2,                          # Number of training pods
        resources_per_node={
            "nvidia.com/gpu": 1,
            "memory": "32Gi",
            "cpu": 8,
        },
        packages_to_install=[
            "feast[postgres,ray]==0.59.0",
            "pandas==2.2.3",
            "scikit-learn==1.6.1",
        ],
        env={"FEAST_REPO_PATH": "/shared/feature_repo"},
    ),
)

# Stream logs
for log in client.get_job_logs(job_name, follow=True):
    print(log, end='')
```

#### 3.2.3 SDK Built-in Trainers

The ODH SDK provides several trainer types for different use cases:

| Trainer Type | Use Case | Key Features |
|-------------|----------|--------------|
| `CustomTrainer` | User-defined Python function | DDP auto-setup, packages_to_install |
| `CustomTrainerContainer` | Pre-built container image | Direct container execution |
| `BuiltinTrainer` | TorchTune LLM fine-tuning | LoRA, QLoRA, DoRA configs |
| `TransformersTrainer` | HuggingFace Transformers/TRL | Auto-instrumentation, JIT checkpointing |
| `TrainingHubTrainer` | RHAI Training Hub | SFT, OSFT algorithms |

```python
# Option 2: TransformersTrainer - HuggingFace with auto-instrumentation
from kubeflow.trainer.rhai import TransformersTrainer

def hf_sft_training():
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
    from feast import FeatureStore
    
    # Load features and train HuggingFace model
    store = FeatureStore(repo_path="/shared/feature_repo")
    # ... HuggingFace training code ...

job_name = client.train(
    trainer=TransformersTrainer(
        func=hf_sft_training,
        num_nodes=2,
        resources_per_node={"nvidia.com/gpu": 1},
        enable_progression_tracking=True,     # HTTP metrics server
        output_dir="pvc://checkpoints/output", # PVC-backed checkpoints
    ),
)

# Option 3: TrainingHubTrainer - RHAI Training Hub algorithms
from kubeflow.trainer.rhai import TrainingHubTrainer, TrainingHubAlgorithms

job_name = client.train(
    trainer=TrainingHubTrainer(
        algorithm=TrainingHubAlgorithms.SFT,   # SFT | OSFT | LORA_SFT
        func_args={
            "data_path": "/data/train.jsonl",
            "ckpt_output_dir": "/checkpoints",
            "num_epochs": 3,
        },
        resources_per_node={"nvidia.com/gpu": 2},
        enable_progression_tracking=True,
    ),
)
```

---

### 3.3 Training Operator Layer

> **NOTE**: ODH Trainer v2.0+ uses `TrainJob` CRD (not `PyTorchJob`). The SDK abstracts this, but here's the underlying resource.

#### 3.3.1 TrainJob v2 Resource Specification

```yaml
# TrainJob v2 API - trainer.kubeflow.org/v1alpha1
apiVersion: trainer.kubeflow.org/v1alpha1
kind: TrainJob
metadata:
  name: sales-forecasting-training
  namespace: ml-training
spec:
  # Reference to ClusterTrainingRuntime (provides ML framework defaults)
  runtimeRef:
    name: torch-distributed
    apiGroup: trainer.kubeflow.org
    kind: ClusterTrainingRuntime
  
  # Optional: Model/Dataset initializers (for HuggingFace, S3, etc.)
  initializer:
    dataset:
      storageUri: "hf://tatsu-lab/alpaca"  # or s3://bucket/dataset
    model:
      storageUri: "hf://meta-llama/Llama-3.1-8B"
  
  # Trainer configuration (SDK generates this from CustomTrainer/etc.)
  trainer:
    image: quay.io/modh/training:py311-cuda124-torch251
    command: ["torchrun"]
    args: ["--nnodes=2", "--nproc_per_node=1", "train.py"]
    env:
    - name: FEAST_REPO_PATH
      value: "/shared/feature_repo"
    - name: NCCL_DEBUG
      value: "INFO"
    numNodes: 2
    numProcPerNode: 1
    resourcesPerNode:
      limits:
        nvidia.com/gpu: 1
        memory: 32Gi
        cpu: 8
  
  # Pod-level overrides (volumes, tolerations, etc.)
  podTemplateOverrides:
  - targetJob:
      name: trainer
    containers:
    - name: trainer
      volumeMounts:
      - name: shared-storage
        mountPath: /shared
    volumes:
    - name: shared-storage
      persistentVolumeClaim:
        claimName: shared-storage
```

#### 3.3.2 ClusterTrainingRuntime (Pre-configured ML Framework)

```yaml
# ClusterTrainingRuntime defines reusable training configurations
apiVersion: trainer.kubeflow.org/v1alpha1
kind: ClusterTrainingRuntime
metadata:
  name: torch-distributed-feast
spec:
  mlPolicy:
    torch:
      numProcPerNode: auto  # auto-detect GPUs
  template:
    spec:
      containers:
      - name: trainer
        image: quay.io/modh/training:py311-cuda124-torch251
        env:
        - name: NCCL_DEBUG
          value: INFO
        - name: TORCH_DISTRIBUTED_DEBUG
          value: DETAIL
      # Init container for Feast feature pre-loading
      initContainers:
      - name: feast-init
        image: quay.io/modh/feast:0.59.0
        command: ["feast", "materialize-incremental"]
        env:
        - name: FEAST_REPO_PATH
          value: /shared/feature_repo
```

#### 3.3.3 Distributed Training Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Distributed Training Execution Flow                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 1: Initialization                                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Master (Rank 0)                    Worker (Rank 1)                   │  │
│  │  ┌─────────────────────┐            ┌─────────────────────┐          │  │
│  │  │ 1. ddp_setup()      │            │ 1. ddp_setup()      │          │  │
│  │  │ 2. detect_device()  │            │ 2. detect_device()  │          │  │
│  │  │ 3. init_process_    │◀──NCCL───▶│ 3. init_process_    │          │  │
│  │  │    group()          │            │    group()          │          │  │
│  │  └─────────────────────┘            └─────────────────────┘          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  PHASE 2: Feature Loading (Rank 0 only)                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Master (Rank 0)                                                      │  │
│  │  ┌─────────────────────────────────────────────────────────┐         │  │
│  │  │ 1. Initialize Ray (8 workers)                           │         │  │
│  │  │ 2. Connect to PostgreSQL                                │         │  │
│  │  │ 3. Feast get_historical_features()                      │         │  │
│  │  │ 4. Save chunks to /shared/models/chunks/                │         │  │
│  │  │ 5. Write completion marker                              │         │  │
│  │  │ 6. Shutdown Ray                                         │         │  │
│  │  └─────────────────────────────────────────────────────────┘         │  │
│  │                                                                       │  │
│  │  Worker (Rank 1)                                                      │  │
│  │  ┌─────────────────────────────────────────────────────────┐         │  │
│  │  │ Wait for completion marker (.feast_load_complete)       │         │  │
│  │  └─────────────────────────────────────────────────────────┘         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  PHASE 3: Data Preparation (All Ranks)                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  ┌─────────────────────┐            ┌─────────────────────┐          │  │
│  │  │ Master: Fit scaler  │            │ Worker: Wait        │          │  │
│  │  │ Encode categoricals │            │ for scaler.pkl      │          │  │
│  │  │ Save metadata.pkl   │            │                     │          │  │
│  │  └─────────────────────┘            └─────────────────────┘          │  │
│  │                    │                           │                      │  │
│  │                    └───────dist.barrier()──────┘                      │  │
│  │                                   │                                   │  │
│  │                    ┌──────────────┴──────────────┐                   │  │
│  │                    │ All: Load scaler, metadata  │                   │  │
│  │                    └─────────────────────────────┘                   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  PHASE 4: Training Loop                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  for epoch in range(num_epochs):                                      │  │
│  │                                                                       │  │
│  │    ┌─────────────────────┐       ┌─────────────────────┐             │  │
│  │    │ Master: chunks 0,2,4│       │ Worker: chunks 1,3,5│             │  │
│  │    │ forward/backward    │       │ forward/backward    │             │  │
│  │    └──────────┬──────────┘       └──────────┬──────────┘             │  │
│  │               │                              │                        │  │
│  │               └──────────AllReduce───────────┘                        │  │
│  │                          (gradients)                                  │  │
│  │                              │                                        │  │
│  │               ┌──────────────┴──────────────┐                        │  │
│  │               │ Synchronized optimizer.step()│                        │  │
│  │               └──────────────────────────────┘                        │  │
│  │                                                                       │  │
│  │    Validation:                                                        │  │
│  │    ┌─────────────────────────────────────────────────────────────┐   │  │
│  │    │ All ranks compute val_loss → AllReduce(AVG) → check early   │   │  │
│  │    │ stopping → Rank 0 saves best_model.pt                       │   │  │
│  │    └─────────────────────────────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  PHASE 5: Cleanup                                                            │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Rank 0: Save final artifacts                                         │  │
│  │  ├── /shared/models/best_model.pt                                     │  │
│  │  ├── /shared/models/scaler.pkl                                        │  │
│  │  ├── /shared/models/target_scaler.pkl                                 │  │
│  │  └── /shared/models/metadata.pkl                                      │  │
│  │                                                                       │  │
│  │  All Ranks: dist.destroy_process_group()                              │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow Design

### 4.1 Training Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Training Data Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Stage 1: Raw Data Ingestion                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Kaggle Dataset → CSV → Feature Engineering → PostgreSQL             │    │
│  │                                                                      │    │
│  │  train.csv ─────┐                                                    │    │
│  │  features.csv ──┼──▶ ETL (Notebook 01) ──▶ sales_features table     │    │
│  │  stores.csv ────┘                     └──▶ store_features table     │    │
│  │                                                                      │    │
│  │  Transformations:                                                    │    │
│  │  • Lag features: sales_lag_1, sales_lag_2, sales_lag_4              │    │
│  │  • Rolling stats: rolling_mean_4, rolling_mean_12, rolling_std_4    │    │
│  │  • Temporal: week_of_year, month, quarter                           │    │
│  │  • Derived: total_markdown, has_markdown                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Stage 2: Feature Store Registration                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  feast apply                                                         │    │
│  │  ├── Register entities (store, dept)                                 │    │
│  │  ├── Register feature views (sales_history, store_external)          │    │
│  │  ├── Register on-demand views (transformations)                      │    │
│  │  └── Register feature service (demand_forecasting_service)           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Stage 3: Training Feature Retrieval                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                      │    │
│  │  PostgreSQL ──▶ Ray Compute ──▶ On-Demand Transforms ──▶ Chunks     │    │
│  │  (offline)      (parallel)      (feature_transformations)  (parquet)│    │
│  │                                                                      │    │
│  │  Output Schema (27 features):                                        │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │ store, dept, date (identifiers - excluded from training)    │    │    │
│  │  │ is_holiday, week_of_year, month, quarter                    │    │    │
│  │  │ sales_lag_1, sales_lag_2, sales_lag_4                       │    │    │
│  │  │ sales_rolling_mean_4, sales_rolling_mean_12, sales_rolling_ │    │    │
│  │  │ temperature, fuel_price, cpi, unemployment                  │    │    │
│  │  │ markdown1-5, total_markdown, has_markdown                   │    │    │
│  │  │ store_type_encoded, store_size                              │    │    │
│  │  │ temperature_normalized, holiday_markdown_interaction        │    │    │
│  │  │ markdown_momentum, seasonal_sine, seasonal_cosine           │    │    │
│  │  │ sales_velocity, sales_acceleration, demand_stability_score  │    │    │
│  │  │ weekly_sales (TARGET)                                       │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Stage 4: Train/Validation Split (TEMPORAL)                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                      │    │
│  │  ⚠️  CRITICAL: Split by DATE, not random                            │    │
│  │                                                                      │    │
│  │  Timeline: 2010-02-05 ──────────────────────────── 2012-10-26       │    │
│  │                                                                      │    │
│  │  Training: 2010-02-05 to 2012-03-01 (80%)                           │    │
│  │  └──────────────────────────────────────────┘                        │    │
│  │                                                                      │    │
│  │  Validation: 2012-03-01 to 2012-10-26 (20%)                         │    │
│  │                                              └────────────────┘      │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Inference Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Inference Data Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Option A: Batch Inference (Offline)                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                      │    │
│  │  Entity DataFrame ──▶ Feast Offline Store ──▶ Model ──▶ Predictions │    │
│  │  (store, dept, date)   (get_historical_features)                    │    │
│  │                                                                      │    │
│  │  Use Case: Weekly batch forecasting for all store-departments        │    │
│  │  Latency: Minutes (acceptable for batch)                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Option B: Real-Time Inference (Online)                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                      │    │
│  │  1. Materialize features to online store:                            │    │
│  │     feast materialize 2010-02-05 2012-10-26                         │    │
│  │                                                                      │    │
│  │  2. Real-time retrieval:                                             │    │
│  │     store.get_online_features(                                       │    │
│  │         features=['sales_history_features:*', 'store_external:*'],   │    │
│  │         entity_rows=[{'store': 1, 'dept': 1}]                        │    │
│  │     )                                                                │    │
│  │                                                                      │    │
│  │  3. Apply on-demand transforms (client-side)                         │    │
│  │                                                                      │    │
│  │  4. Model prediction                                                 │    │
│  │                                                                      │    │
│  │  Latency: <100ms (required for real-time)                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. API Contracts

### 5.1 Training Function Interface

```python
def training_func(parameters: dict = None) -> None:
    """
    Self-contained distributed training function.
    
    Injected by Kubeflow SDK into PyTorchJob pods.
    Must be fully self-contained (all imports inside function).
    
    Parameters:
    -----------
    parameters : dict
        Configuration dictionary with keys:
        
        # Data Source
        - data_source: str          # "postgres_ray" | "feast_file" | "direct"
        - feast_repo_path: str      # Path to Feast repository
        - sample_size: int | None   # Rows to sample (None = all)
        - chunk_size: int           # Rows per chunk file
        
        # Model Architecture
        - model_type: str           # "mlp" | "tft"
        - hidden_dims: list[int]    # MLP layer sizes
        - dropout: float            # Dropout rate
        
        # Training
        - num_epochs: int           # Training epochs
        - batch_size: int           # Batch size per GPU
        - learning_rate: float      # Initial learning rate
        - weight_decay: float       # L2 regularization
        - use_amp: bool             # Mixed precision training
        - grad_clip_norm: float     # Gradient clipping
        
        # Validation
        - val_size: float           # Validation split (0.0-1.0)
        - early_stopping_patience: int | None
        
        # Distributed
        - backend: str              # "nccl" | "gloo" | "auto"
        
        # Output
        - model_output_dir: str     # Model artifact path
        - checkpoint_every: int | None
    
    Outputs (written to model_output_dir):
    --------------------------------------
    - best_model.pt         # Model checkpoint with state_dict
    - scaler.pkl            # Feature scaler (StandardScaler)
    - target_scaler.pkl     # Target scaler
    - metadata.pkl          # Feature columns, categorical encoders
    
    Environment Variables (set by Training Operator):
    ------------------------------------------------
    - MASTER_ADDR           # Master pod hostname
    - MASTER_PORT           # Communication port (29500)
    - WORLD_SIZE            # Total number of processes
    - RANK                  # Global rank of this process
    - LOCAL_RANK            # Local rank on this node
    """
    pass
```

### 5.2 Feature Store Interface

```python
# Feature Definition Contract
from feast import Entity, FeatureView, Field, FeatureService
from feast.on_demand_feature_view import on_demand_feature_view

# Entity Definition
entity = Entity(
    name="store",
    value_type=Int64,
    description="Store identifier",
)

# Feature View Definition
feature_view = FeatureView(
    name="sales_history_features",
    entities=[store_entity, dept_entity],
    ttl=timedelta(days=730),
    schema=[
        Field(name="sales_lag_1", dtype=Float64),
        # ... more fields
    ],
    source=PostgreSQLSource(...),
    online=True,  # Enable for online serving
)

# On-Demand Feature View
@on_demand_feature_view(
    sources=[feature_view],
    schema=[Field(name="computed_feature", dtype=Float64)],
)
def compute_feature(inputs: pd.DataFrame) -> pd.DataFrame:
    """Transform must be deterministic and stateless."""
    return pd.DataFrame({"computed_feature": inputs["x"] * 2})

# Feature Service (groups features for training/inference)
feature_service = FeatureService(
    name="demand_forecasting_service",
    features=[feature_view, on_demand_view],
)
```

### 5.3 Model Checkpoint Format

```python
# Checkpoint Structure (saved by training, loaded by inference)
checkpoint = {
    # Model State
    "MODEL_STATE": OrderedDict,  # model.state_dict()
    "OPTIMIZER_STATE": dict,     # optimizer.state_dict()
    
    # Training Metadata
    "EPOCH": int,                # Last completed epoch
    "VAL_LOSS": float,           # Best validation loss
    
    # Architecture Config (for reconstruction)
    "MODEL_CONFIG": {
        "model_type": str,       # "mlp" | "tft"
        "input_dim": int,        # Number of input features
        "hidden_dims": list,     # For MLP
        "dropout": float,
        # TFT-specific
        "tft_hidden_size": int,
        "tft_num_heads": int,
        "tft_lstm_layers": int,
    }
}

# Loading Pattern
def load_model(checkpoint_path: str) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['MODEL_CONFIG']
    
    if config['model_type'] == 'mlp':
        model = SalesForecastingMLP(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            dropout=config['dropout']
        )
    elif config['model_type'] == 'tft':
        model = TemporalFusionTransformer(...)
    
    model.load_state_dict(checkpoint['MODEL_STATE'])
    return model
```

---

## 6. Sequence Diagrams

### 6.1 End-to-End Training Flow

```
┌─────────┐  ┌─────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────┐
│Workbench│  │   SDK   │  │   Training   │  │   Master     │  │  Worker  │  │  Feast   │
│(User)   │  │ Client  │  │  Operator    │  │   Pod        │  │  Pod     │  │  Store   │
└────┬────┘  └────┬────┘  └──────┬───────┘  └──────┬───────┘  └────┬─────┘  └────┬─────┘
     │            │               │                 │               │             │
     │ 1. create_job()            │                 │               │             │
     │───────────▶│               │                 │               │             │
     │            │               │                 │               │             │
     │            │ 2. Create PyTorchJob CR         │               │             │
     │            │──────────────▶│                 │               │             │
     │            │               │                 │               │             │
     │            │               │ 3. Reconcile    │               │             │
     │            │               │────────────────▶│               │             │
     │            │               │                 │               │             │
     │            │               │ 4. Create Pods  │               │             │
     │            │               │─────────────────┼──────────────▶│             │
     │            │               │                 │               │             │
     │            │               │                 │ 5. DDP Init   │             │
     │            │               │                 │◀─────────────▶│             │
     │            │               │                 │   (NCCL)      │             │
     │            │               │                 │               │             │
     │            │               │                 │ 6. get_historical_features()│
     │            │               │                 │───────────────┼────────────▶│
     │            │               │                 │               │             │
     │            │               │                 │               │  7. Ray     │
     │            │               │                 │               │  parallel   │
     │            │               │                 │               │  joins      │
     │            │               │                 │               │             │
     │            │               │                 │◀──────────────┼─────────────│
     │            │               │                 │ 8. training_df              │
     │            │               │                 │               │             │
     │            │               │                 │ 9. Save chunks              │
     │            │               │                 │──────▶ PVC    │             │
     │            │               │                 │               │             │
     │            │               │                 │ 10. Barrier   │             │
     │            │               │                 │◀─────────────▶│             │
     │            │               │                 │               │             │
     │            │               │                 │ 11. Training Loop           │
     │            │               │                 │◀═════════════▶│             │
     │            │               │                 │  (epochs)     │             │
     │            │               │                 │  AllReduce    │             │
     │            │               │                 │               │             │
     │            │               │                 │ 12. Save best_model.pt      │
     │            │               │                 │──────▶ PVC    │             │
     │            │               │                 │               │             │
     │            │               │ 13. Update Status               │             │
     │            │               │◀────────────────│               │             │
     │            │               │                 │               │             │
     │ 14. get_job_logs()         │                 │               │             │
     │───────────▶│               │                 │               │             │
     │            │──────────────▶│                 │               │             │
     │◀───────────│               │                 │               │             │
     │   (logs)   │               │                 │               │             │
     │            │               │                 │               │             │
```

### 6.2 Inference Flow

```
┌─────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐
│  Client │  │    Model     │  │    Feast     │  │PostgreSQL│
│  App    │  │   Service    │  │   Online     │  │  Online  │
└────┬────┘  └──────┬───────┘  └──────┬───────┘  └────┬─────┘
     │              │                 │               │
     │ 1. POST /predict               │               │
     │   {store:1, dept:1}            │               │
     │─────────────▶│                 │               │
     │              │                 │               │
     │              │ 2. get_online_features()        │
     │              │────────────────▶│               │
     │              │                 │               │
     │              │                 │ 3. SELECT     │
     │              │                 │──────────────▶│
     │              │                 │◀──────────────│
     │              │                 │               │
     │              │◀────────────────│               │
     │              │  raw_features   │               │
     │              │                 │               │
     │              │ 4. Apply on-demand transforms   │
     │              │   (client-side computation)     │
     │              │                 │               │
     │              │ 5. Scale features               │
     │              │   scaler.transform()            │
     │              │                 │               │
     │              │ 6. Model inference              │
     │              │   model(features)               │
     │              │                 │               │
     │              │ 7. Inverse transform            │
     │              │   target_scaler.inverse()       │
     │              │                 │               │
     │◀─────────────│                 │               │
     │ {prediction: $24,500}          │               │
     │              │                 │               │
```

---

## 7. Storage Design

### 7.1 PVC Layout

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Persistent Volume Claims                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  shared-storage (RWX - ReadWriteMany)                                        │
│  ├── feature_repo/                                                           │
│  │   ├── feature_store.yaml           # Feast configuration                  │
│  │   ├── features.py                  # Feature definitions                  │
│  │   └── data/                                                               │
│  │       ├── sales_features.parquet   # 18.5 MB (421K rows)                  │
│  │       └── store_features.parquet   # 0.4 MB                               │
│  │                                                                           │
│  └── models/                                                                 │
│      ├── postgres_ray_chunks/         # Training data chunks                 │
│      │   ├── chunk_0000.parquet                                              │
│      │   ├── chunk_0001.parquet                                              │
│      │   └── .feast_postgres_ray_complete  # Completion marker               │
│      │                                                                       │
│      ├── best_model.pt                # 0.2 MB (51K params)                  │
│      ├── scaler.pkl                   # Feature scaler                       │
│      ├── target_scaler.pkl            # Target scaler                        │
│      ├── metadata.pkl                 # Feature columns, encoders            │
│      └── encoders.pkl                 # Categorical encoders                 │
│                                                                              │
│  Storage Class: nfs-csi (for RWX support)                                    │
│  Size: 100Gi                                                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 PostgreSQL Schema

```sql
-- Database: feast_registry
-- Purpose: Feast metadata and feature definitions
CREATE TABLE feast_registry.feature_views (
    name VARCHAR PRIMARY KEY,
    spec JSONB,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Database: feast_offline
-- Purpose: Historical feature data (training)
CREATE TABLE sales_features (
    store INT NOT NULL,
    dept INT NOT NULL,
    date TIMESTAMP NOT NULL,
    weekly_sales FLOAT,
    is_holiday INT,
    week_of_year INT,
    month INT,
    quarter INT,
    sales_lag_1 FLOAT,
    sales_lag_2 FLOAT,
    sales_lag_4 FLOAT,
    sales_rolling_mean_4 FLOAT,
    sales_rolling_mean_12 FLOAT,
    sales_rolling_std_4 FLOAT,
    PRIMARY KEY (store, dept, date)
);

CREATE INDEX idx_sales_date ON sales_features(date);
CREATE INDEX idx_sales_store_dept ON sales_features(store, dept);

CREATE TABLE store_features (
    store INT NOT NULL,
    dept INT NOT NULL,
    date TIMESTAMP NOT NULL,
    temperature FLOAT,
    fuel_price FLOAT,
    cpi FLOAT,
    unemployment FLOAT,
    markdown1 FLOAT,
    markdown2 FLOAT,
    markdown3 FLOAT,
    markdown4 FLOAT,
    markdown5 FLOAT,
    total_markdown FLOAT,
    has_markdown INT,
    store_type VARCHAR(1),
    store_size INT,
    PRIMARY KEY (store, dept, date)
);

-- Database: feast_online
-- Purpose: Materialized features for real-time serving
-- (Managed by Feast, schema auto-generated)
```

---

## 8. Error Handling & Fault Tolerance

### 8.1 Training Failure Scenarios

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Error Handling Matrix                                    │
├───────────────────┬─────────────────────────┬───────────────────────────────┤
│ Failure Mode      │ Detection               │ Recovery Strategy             │
├───────────────────┼─────────────────────────┼───────────────────────────────┤
│ Pod OOMKilled     │ K8s event, exit code    │ Restart with backoffLimit     │
│                   │ 137                     │ Increase memory request       │
├───────────────────┼─────────────────────────┼───────────────────────────────┤
│ NCCL Timeout      │ Process hang, timeout   │ Retry with gloo backend       │
│                   │ error in logs           │ Check network policy          │
├───────────────────┼─────────────────────────┼───────────────────────────────┤
│ Feast Connection  │ psycopg.OperationalError│ Retry with exponential        │
│ Failure           │                         │ backoff (3 attempts)          │
├───────────────────┼─────────────────────────┼───────────────────────────────┤
│ GPU Memory Error  │ CUDA OOM                │ Reduce batch_size             │
│                   │                         │ Enable gradient checkpointing │
├───────────────────┼─────────────────────────┼───────────────────────────────┤
│ Checkpoint        │ IOError on save         │ Retry save, verify PVC        │
│ Save Failure      │                         │ storage availability          │
├───────────────────┼─────────────────────────┼───────────────────────────────┤
│ Worker Divergence │ NaN loss, exploding     │ Early stopping, reduce LR     │
│                   │ gradients               │ Enable grad_clip_norm         │
├───────────────────┼─────────────────────────┼───────────────────────────────┤
│ Rank 0 Crash      │ Other ranks timeout     │ All ranks restart together    │
│                   │ waiting                 │ (OnFailure policy)            │
└───────────────────┴─────────────────────────┴───────────────────────────────┘
```

### 8.2 Retry Configuration

```yaml
# PyTorchJob Retry Configuration
spec:
  runPolicy:
    backoffLimit: 3                    # Max retries before failure
    cleanPodPolicy: None               # Keep pods for debugging
    ttlSecondsAfterFinished: 86400     # Cleanup after 24h
  
  pytorchReplicaSpecs:
    Master:
      restartPolicy: OnFailure         # Restart on non-zero exit
    Worker:
      restartPolicy: OnFailure
```

---

## 9. Security Considerations

### 9.1 Secrets Management

```yaml
# PostgreSQL Credentials Secret
apiVersion: v1
kind: Secret
metadata:
  name: feast-credentials
  namespace: ml-training
type: Opaque
stringData:
  username: feast
  password: ${FEAST_PASSWORD}  # Injected from vault

---
# HuggingFace Token (if using gated models)
apiVersion: v1
kind: Secret
metadata:
  name: hf-token
type: Opaque
stringData:
  token: ${HF_TOKEN}
```

### 9.2 Network Policies

```yaml
# Allow training pods to access Feast PostgreSQL
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: training-to-feast
spec:
  podSelector:
    matchLabels:
      training.kubeflow.org/job-role: master
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: feast-postgres
    ports:
    - protocol: TCP
      port: 5432

---
# Allow inter-worker communication (NCCL)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: training-inter-worker
spec:
  podSelector:
    matchLabels:
      training.kubeflow.org/job-name: sales-forecasting
  ingress:
  - from:
    - podSelector:
        matchLabels:
          training.kubeflow.org/job-name: sales-forecasting
    ports:
    - protocol: TCP
      port: 29500  # MASTER_PORT
```

---

## 10. Performance Optimization

### 10.1 Training Optimizations

| Optimization | Setting | Impact |
|--------------|---------|--------|
| **Mixed Precision (AMP)** | `use_amp: true` | 2x speedup on NVIDIA GPUs |
| **Gradient Accumulation** | `accumulation_steps: 2` | Effective batch 2048 |
| **Data Loader Workers** | `num_workers: 4` | Keep GPU fed |
| **Pin Memory** | `pin_memory: True` | Faster GPU transfer |
| **Prefetch Factor** | `prefetch_factor: 2` | Overlap I/O and compute |
| **Chunk Size** | `chunk_size: 50000` | Balance memory vs I/O |

### 10.2 Feast Optimizations

| Optimization | Setting | Impact |
|--------------|---------|--------|
| **Ray Workers** | `max_workers: 8` | Parallel joins |
| **Partition Size** | `target_partition_size_mb: 64` | Efficient Ray tasks |
| **Broadcast Join** | `threshold_mb: 100` | Small table broadcast |
| **Connection Pool** | `pool_size: 10` | Reduce connection overhead |
| **Index Usage** | Composite indexes | Fast lookups |

---

## 11. Monitoring & Observability

### 11.1 Key Metrics

```
# Training Metrics (logged per epoch)
training_loss          # MSE loss (scaled)
validation_loss        # MSE loss (scaled)
validation_rmse_actual # RMSE in dollars
mape                   # Mean Absolute Percentage Error
learning_rate          # Current LR
epoch_duration_sec     # Time per epoch
gpu_utilization        # % GPU usage
memory_allocated_gb    # GPU memory used

# Feast Metrics
feast_retrieval_duration_sec    # Time to get features
feast_rows_retrieved            # Number of rows
ray_worker_utilization          # Ray cluster usage

# Infrastructure Metrics
pod_cpu_usage
pod_memory_usage
pvc_storage_used
network_bytes_transmitted       # NCCL traffic
```

### 11.2 Logging Pattern

```python
# Structured logging format
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Rank %(rank)s] %(levelname)-8s %(message)s"
)

# Key log points
logger.info(f"[Rank {rank}] Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
logger.info(f"[Rank {rank}] MAPE: {mape:.2f}% | RMSE: ${rmse:,.0f}")
logger.info(f"[Rank 0] Saved best model (val_loss={val_loss:.4f})")
```

---

## 12. Migration Notes

### 12.1 From kubeflow-training v1 to ODH Trainer v2

```python
# Old (kubeflow-training 1.9.x)
from kubeflow.training import PyTorchJobClient
client = PyTorchJobClient()
client.create(pytorchjob_spec)

# New (opendatahub-io/kubeflow-sdk 0.2.1+)
from kubeflow.training import TrainerClient
client = TrainerClient()
client.create_job(
    job_kind="PyTorchJob",
    train_func=training_func,  # Function-based API
    parameters=params,
    # ... simplified configuration
)
```

### 12.2 Version Compatibility Matrix

| Component | Minimum Version | Recommended | Notes |
|-----------|----------------|-------------|-------|
| OpenShift AI | 2.17 | 2.18+ | Training Operator included |
| Kubeflow Trainer | 2.0 | ODH fork | opendatahub-io/trainer |
| Kubeflow SDK | 0.2.0 | 0.2.1+rhai2 | opendatahub-io/kubeflow-sdk |
| Feast | 0.55+ | 0.59.0 | PostgreSQL + Ray support |
| PyTorch | 2.0+ | 2.5.1 | DDP improvements |
| Python | 3.10+ | 3.11 | Performance improvements |

---

## 13. Known Issues & Workarounds

### 13.1 Data Leakage (CRITICAL)

**Issue**: On-demand features use `weekly_sales` (target) as input  
**Impact**: Model sees answer during training, inflated metrics  
**Fix**: Remove features derived from `weekly_sales`:

```python
# REMOVE these from feature_transformations:
# - sales_normalized
# - sales_per_sqft  
# - markdown_efficiency

# KEEP these (use lagged values only):
# - sales_velocity (uses sales_lag_*, not weekly_sales)
# - demand_stability_score
```

### 13.2 Non-Temporal Train/Val Split

**Issue**: Random chunk split leaks future data  
**Fix**: Split by date:

```python
# In training function
all_chunks = sorted(glob.glob(f"{chunks_dir}/chunk_*.parquet"))
# Load first chunk to get date range
sample = pd.read_parquet(all_chunks[0])
cutoff_date = sample['date'].quantile(0.8)  # 80% for training

train_chunks = [c for c in all_chunks if get_max_date(c) < cutoff_date]
val_chunks = [c for c in all_chunks if get_min_date(c) >= cutoff_date]
```

### 13.3 Syntax Error in torch_training.py

**Issue**: Missing indentation on line 1047  
**Fix**:
```python
# Before
if data_source == "direct":
logger.info(f"Data path: {data_path}")

# After
if data_source == "direct":
    logger.info(f"Data path: {data_path}")
```

---

## 14. References

- [ODH Trainer GitHub](https://github.com/opendatahub-io/trainer)
- [ODH Kubeflow SDK GitHub](https://github.com/opendatahub-io/kubeflow-sdk)
- [Feast Feature Store](https://github.com/feast-dev/feast)
- [Kubeflow Trainer v2.0 Blog Post](https://www.kubeflow.org/docs/components/training)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Feast PostgreSQL + Ray Architecture](https://docs.feast.dev/)

---

## Appendix A: Quick Reference Commands

```bash
# Create namespace
oc new-project ml-training

# Deploy PostgreSQL for Feast
oc apply -f postgres-feast-registry-openshift.yaml

# Initialize Feast
cd /shared/feature_repo && feast apply

# Submit training job (from notebook)
# See Section 3.2.2

# Monitor job
oc get pytorchjob -n ml-training
oc logs -f sales-forecasting-training-master-0

# Materialize features for inference
feast materialize 2010-02-05 2012-10-26

# Test online features
python -c "from feast import FeatureStore; store = FeatureStore('.'); print(store.get_online_features(features=['sales_history_features:sales_lag_1'], entity_rows=[{'store': 1, 'dept': 1}]).to_dict())"
```

---

*Document generated: February 2026*  
*Architecture Version: 1.0*

