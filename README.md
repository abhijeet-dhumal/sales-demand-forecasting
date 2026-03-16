# Sales Demand Forecasting on OpenShift AI

*A production ML pipeline that eliminates training-serving skew using Feast, Kubeflow, MLflow, and KServe*

---

## Overview

This project demonstrates **train-serve consistency** using Feast Feature Services:

- **Training**: `get_historical_features()` via KubeRay (distributed PIT joins)
- **Inference**: `get_online_features()` via Redis (low-latency lookups)
- **Same features, zero skew**: Both use identical FeatureService definitions

```
Time
  │                    Monolithic (OOM)
  │                         X
  │                       /
  │     ────────────────/──── Feast+Ray
  │   /                /
  │  / Monolithic    /
  │/               /
  └─────────────────────────────► Data Size
    100K   1M   10M   100M   1B
                 ↑
         Crossover (~5-10M rows)
```

**Real-World Comparison (100M rows, 50 features):**

| Approach | Time | Feasibility |
|----------|------|-------------|
| Pandas | OOM | ❌ Impossible |
| Spark | ~45 min | ✅ Works |
| Feast + Ray (4 nodes) | ~30 min | ✅ Works |

**Hidden Benefits:** Feature versioning, train-serve consistency, cached materialization, MLflow tracking.

---

## The Problem

Machine learning models that work in notebooks often fail silently in production due to **training-serving skew** — features computed differently during training vs inference.

Google's [Hidden Technical Debt in ML Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) established that most ML failures stem from data inconsistencies. Companies like [Uber](https://www.uber.com/blog/michelangelo-machine-learning-platform/), [DoorDash](https://doordash.engineering/2020/11/19/building-a-gigascale-ml-feature-store-with-redis/), and [Airbnb](https://medium.com/airbnb-engineering/chronon-a-declarative-feature-engineering-framework-b7b8ce796e04) built feature platforms to solve this.

| Failure Mode | Symptom | Impact |
|--------------|---------|--------|
| Stale features | Serving uses old data | Predictions drift |
| Different aggregations | Inconsistent rolling averages | Accuracy drops |
| Missing features | Serving omits a feature | Silent errors |
| Type mismatches | float64 vs float32 | Numerical differences |

---

## Use Case: Retail Demand Forecasting

Predicting weekly sales for store-department combinations. According to [IHL Group](https://www.ihlservices.com/news/analyst-corner/2025/09/retail-inventory-crisis-persists-despite-172-billion-in-improvements/), retailers lose **\$1.77 trillion annually** due to inventory distortion.

| Metric | Value |
|--------|-------|
| Dataset | 65,520 samples (45 stores × 14 depts × 104 weeks) |
| Features | 22 (lag, rolling, temporal, economic, store) |
| Model | MLP [512, 256, 128, 64] with BatchNorm + Dropout |
| MAPE | ~3-5% |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Phase 1             Phase 2                  Phase 3                │
│  ┌────────────┐      ┌─────────────────┐      ┌─────────────────┐    │
│  │ Feast      │─────▶│ Kubeflow        │─────▶│ KServe          │    │
│  │ Apply +    │      │ TrainJob        │      │ InferenceService│    │
│  │ Materialize│      │ (PyTorch DDP)   │      │ + Feast SDK     │    │
│  └─────┬──────┘      └────────┬────────┘      └────────┬────────┘    │
│        │                      │                        │             │
│        ▼                      ▼                        ▼             │
│  ┌───────────┐         ┌───────────┐            ┌───────────┐        │
│  │PostgreSQL │         │ MLflow    │            │ Redis     │        │
│  │ Registry  │         │ Tracking  │            │ Online    │        │
│  └───────────┘         └───────────┘            └───────────┘        │
│        │                      │                        │             │
│        └──────────────────────┴────────────────────────┘             │
│                               │                                      │
│                        ┌──────┴──────┐                               │
│                        │   KubeRay   │                               │
│                        │ Offline PIT │                               │
│                        └─────────────┘                               │
└──────────────────────────────────────────────────────────────────────┘
```

![Pipeline Sequence](docs/diagrams/sequence-diagram.png)

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Feature Store** | Feast Operator | PostgreSQL registry, Redis online, Ray offline |
| **Training** | Kubeflow Trainer | Multi-node PyTorch DDP orchestration |
| **Experiment Tracking** | MLflow Operator | Workspace isolation, model registry |
| **Model Serving** | KServe | Auto-scaling with Feast SDK integration |
| **Platform** | OpenShift AI | Managed ML infrastructure |

---

## Quick Start

### 1. Deploy Infrastructure

```bash
# Clone repository
git clone https://github.com/abhijeet-dhumal/sales-demand-forecasting.git
cd sales-demand-forecasting

# Deploy all components
oc apply -k manifests/

# Wait for pods
oc wait --for=condition=ready pod -l app=postgres -n feast-trainer-demo --timeout=120s
oc wait --for=condition=ready pod -l ray.io/node-type=head -n feast-trainer-demo --timeout=180s
```

### 2. Verify Deployment

```bash
oc get pods -n feast-trainer-demo
```

Expected output:
```
NAME                                    READY   STATUS    AGE
feast-ray-head-xxxxx                    1/1     Running   2m
feast-ray-worker-xxxxx                  1/1     Running   2m
feast-salesforecasting-server-xxxxx     4/4     Running   2m
postgres-xxxxx                          1/1     Running   2m
redis-xxxxx                             1/1     Running   2m
mlflow-xxxxx                            1/1     Running   2m
```

### 3. Create Workbench in OpenShift AI

1. **Access OpenShift AI** from console app launcher
   ![Access OpenShift AI](docs/images/access-openshift-ai.png)

2. **Create Data Science Project**: `feast-trainer-demo`
   ![Create Project](docs/images/create-project.png)

3. **Create Workbench**:
   - Image: `PyTorch | CUDA | Python 3.12`
   - Storage: Attach `shared` PVC with RWX
   - Feature Store: Connect to `salesforecasting`
   
   ![Workbench Image](docs/images/workbench-image.png)
   ![Feature Store Connection](docs/images/feature-store-connection.png)

### 4. Run Notebooks

| Notebook | Purpose | Time |
|----------|---------|------|
| `01_feature_store/01a-local.ipynb` | Generate data → Feast apply → Materialize | ~2 min |
| `02_training/02-training.ipynb` | Distributed training with Kubeflow | ~3 min |
| `03_inferencing/03-inference.ipynb` | Deploy model, test predictions | ~1 min |

---

## Phase Details

### Phase 1: Feature Engineering

![Feature Engineering](docs/diagrams/01-features-workflow.png)

**What Feast Operator manages:**

| Component | Purpose |
|-----------|---------|
| PostgreSQL Registry | Durable metadata for feature definitions |
| Redis Online Store | Low-latency serving (~5ms) |
| Ray Offline Store | Distributed historical queries |
| Client ConfigMaps | Auto-generated configuration |

**Two FeatureServices for consistency:**

| Service | Use Case | Includes Target? |
|---------|----------|------------------|
| `training_features` | Historical retrieval for training | ✅ Yes |
| `inference_features` | Real-time lookup for predictions | ❌ No |

### Phase 2: Distributed Training

![Training](docs/diagrams/02-training-workflow.png)

**Why Kubeflow Trainer:**

| Capability | Description |
|------------|-------------|
| Multi-node | Scale across 2, 4, 8+ nodes |
| Multi-GPU | Utilize all GPUs per node |
| Multi-accelerator | NVIDIA (CUDA) and AMD (ROCm) |
| Auto-coordination | Environment variables handled automatically |

**MLflow Integration:**

![MLflow Workspace](docs/images/mlflow-workspace.png)

### Phase 3: Model Serving

![Inference](docs/diagrams/03-inference-workflow.png)

**The serving pattern:**

| Step | Action |
|------|--------|
| 1 | Client sends entity IDs (`store_id`, `dept_id`) |
| 2 | KServe receives request |
| 3 | Feast SDK fetches features from Redis |
| 4 | Model predicts |
| 5 | Return result |

**Why this matters:**

| Approach | Client Sends | Skew Risk |
|----------|--------------|-----------|
| Without Feast | All features | ⚠️ High |
| With Feast | Entity IDs only | ✅ Zero |

---

## Project Structure

```
sales-demand-forecasting/
├── manifests/
│   ├── kustomization.yaml      # oc apply -k manifests/
│   ├── base/                   # Namespace, PVC
│   ├── databases/              # PostgreSQL, Redis
│   ├── ray/                    # RayCluster
│   ├── feast/                  # FeatureStore CR
│   └── mlflow/                 # MLflow Operator
├── notebooks/
│   ├── 01_feature_store/       # Feature engineering
│   │   ├── 01a-local.ipynb     # Admin: register features
│   │   └── 01b-remote.ipynb    # User: use existing features
│   ├── 02_training/            # Distributed training
│   └── 03_inferencing/         # Model serving
├── feature_repo/
│   └── features.py             # Feature definitions
└── docs/
    ├── diagrams/               # Architecture diagrams
    └── images/                 # Screenshots
```

---

## Configuration

| Setting | Default | Notes |
|---------|---------|-------|
| Namespace | `feast-trainer-demo` | All components deployed here |
| Training nodes | 2 | PyTorch DDP workers |
| GPUs per node | 1 | Configurable in TrainJob |
| Redis latency | ~5ms | Online feature lookup |
| Ray workers | 2 | Distributed offline queries |

---

## Results (Demo Dataset)

| Metric | Value |
|--------|-------|
| Dataset | 45 stores × 14 depts × 104 weeks |
| Features | 22 engineered features |
| Model | MLP [512, 256, 128, 64] |
| MAPE | ~3-5% |
| Training time | ~45s (2 nodes × 1 GPU) |
| Inference latency | ~50ms (including feature fetch) |

---

## Why This Architecture

| Decision | Benefit |
|----------|---------|
| Feast as single source of truth | Same definitions for training + serving = no skew |
| Ray offline store | Scales PIT joins from thousands to millions of rows |
| Kubeflow Trainer | Declarative distributed training, no manual coordination |
| KServe + Feast SDK | Consistent feature retrieval at inference time |
| MLflow Operator | Workspace isolation, model versioning |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Feature Store connection not found | Edit workbench → Add Feature Store → Restart |
| Pods not starting | `oc describe pod <name> -n feast-trainer-demo` |
| PVC not bound | Check storage class supports RWX |
| DDP timeout | Increase `RDZV_TIMEOUT` in TrainJob |
| Ray FileNotFoundError | Restart Ray cluster |

---

## Cleanup

```bash
oc delete namespace feast-trainer-demo
```

---

## Resources

**This Project:**
- [Detailed Blog Post](BLOG_POST.md) — Full technical write-up with code examples

**Documentation:**
- [Feast](https://docs.feast.dev/) | [Kubeflow Trainer](https://www.kubeflow.org/docs/components/training/) | [KServe](https://kserve.github.io/website/) | [MLflow](https://mlflow.org/docs/latest/)
- [OpenShift AI](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/)

**Industry References:**
- [Hidden Technical Debt in ML Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) — Google NIPS 2015
- [Uber Michelangelo](https://www.uber.com/blog/michelangelo-machine-learning-platform/) | [DoorDash Feature Store](https://doordash.engineering/2020/11/19/building-a-gigascale-ml-feature-store-with-redis/) | [Airbnb Chronon](https://medium.com/airbnb-engineering/chronon-a-declarative-feature-engineering-framework-b7b8ce796e04)

---

## License

Apache 2.0
