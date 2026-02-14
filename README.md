# Feast + Kubeflow Training on OpenShift AI

**Production MLOps: Feature Store → Distributed Training → Inference**

![Sequence Diagram](docs/sequence-diagram.png)


## Overview

This example demonstrates **train-serve consistency** using Feast Feature Services:

- **Training**: `get_historical_features()` via KubeRay (distributed PIT joins)
- **Inference**: `get_online_features()` via Feast Server (low-latency REST API)
- **Same features, zero skew**: Both use the same Feature Service definitions

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

**Real-World (100M rows, 50 features):**

| Approach | Time | Feasibility |
|----------|------|-------------|
| Pandas | OOM | ❌ Impossible |
| Spark | ~45 min | ✅ Works |
| Feast + Ray (4 nodes) | ~30 min | ✅ Works |

**Hidden Benefits:** Feature versioning, train-serve consistency, cached materialization, MLflow tracking.

---

## Quick Start

```bash
# 1. Deploy infrastructure
kubectl apply -k manifests/

# 2. Wait for pods
kubectl wait --for=condition=ready pod -l app=postgres -n feast-trainer-demo --timeout=120s
kubectl wait --for=condition=ready pod -l ray.io/node-type=head -n feast-trainer-demo --timeout=180s
```

**3. Create Workbench** in OpenShift AI:
- Image: PyTorch (CUDA)
- PVC Name: `shared`

**4. Run Notebooks:**

| Notebook | Purpose | Time |
|----------|---------|------|
| `01-feast-features.ipynb` | Data → Feast apply → Materialize | ~2 min |
| `02-training.ipynb` | Distributed training (Kubeflow) | ~3 min |
| `03-inference.ipynb` | Model serving test | ~1 min |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  01-features       02-training         03-inference      │
│  ┌────────────┐    ┌────────────┐     ┌────────────┐     │
│  │ Feast      │───▶│ Kubeflow   │────▶│ KServe     │     │
│  │ Apply      │    │ TrainJob   │     │ + Feast    │     │
│  └─────┬──────┘    │ (DDP)      │     │ Server     │     │
│        │           └─────┬──────┘     └─────┬──────┘     │
│        ▼                 │                  │            │
│  ┌───────────┐           │                  │            │
│  │ PostgreSQL│◀──────────┴──────────────────┘            │
│  │ (Registry │                                           │
│  │ + Online) │     ┌──────────┐    ┌──────────┐          │
│  └───────────┘     │ KubeRay  │    │ MLflow   │          │
│                    │ (PIT)    │    │ Tracking │          │
│                    └──────────┘    └──────────┘          │
└──────────────────────────────────────────────────────────┘
```

## Project Structure

```
.
├── feature_repo/
│   ├── feature_store.yaml      # File-based config
│   ├── feature_store_ray.yaml  # KubeRay config
│   └── features.py             # Feature definitions
├── notebooks/
│   ├── 01-feast-features.ipynb
│   ├── 02-training.ipynb
│   └── 03-inference.ipynb
└── manifests/
    ├── kustomization.yaml      # kubectl apply -k manifests/
    ├── 01-postgres.yaml
    ├── 02-mlflow.yaml
    ├── 03-raycluster.yaml
    └── 04b-feast-server.yaml
```

## Configuration

| Setting | Default | Notes |
|---------|---------|-------|
| `USE_RAY` | `True` | Distributed PIT joins |
| `RDZV_TIMEOUT` | `1800` | 30 min for DDP sync |
| Path handling | Auto | Symlink created on Ray pods |

## Results (65K rows demo)

| Metric | Value |
|--------|-------|
| Dataset | 45 stores × 14 depts × 104 weeks |
| Model | MLP [512, 256, 128, 64] |
| MAPE | ~3-5% |
| Training | 2 nodes × 1 GPU, ~45s |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `FileNotFoundError` on Ray | Restart Ray cluster (symlink created on startup) |
| DDP timeout | Already handled: `RDZV_TIMEOUT=1800` |

## Cleanup

```bash
kubectl delete namespace feast-trainer-demo
```
