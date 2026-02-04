# Sales Demand Forecasting - Industry-Grade Quickstart

An end-to-end ML pipeline demonstrating **Kubeflow Trainer + Feast Feature Store** integration on OpenShift AI / Red Hat AI.

## 🎯 Key Results

| Metric | Value |
|--------|-------|
| **Model MAPE** | 9.9% |
| **Improvement** | 87.5% vs naive baseline |
| **Training Time** | ~30 seconds (4 GPUs) |
| **Feature Retrieval** | PostgreSQL online store |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     OpenShift AI Cluster                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │ Data Prep Job   │    │ Feast PostgreSQL│                     │
│  │ (Synthetic Data)│───▶│    (Registry,   │                     │
│  │ Feature Eng.    │    │  Online Store)  │                     │
│  │ feast apply     │    └────────┬────────┘                     │
│  └────────┬────────┘             │                              │
│           │                      │                              │
│           ▼                      ▼                              │
│  ┌─────────────────────────────────────────────────┐            │
│  │              Shared PVC (NFS)                   │            │
│  │  /shared/data     - Feature parquet files      │            │
│  │  /shared/models   - Trained models, scalers    │            │
│  │  /shared/feature_repo - Feast config           │            │
│  └───────────────────────────────────────────────┬─┘            │
│                                                  │              │
│  ┌──────────────────────┐     ┌──────────────────┴──────┐       │
│  │    TrainJob v2       │     │    Inference Job        │       │
│  │  ┌───────────────┐   │     │  ┌───────────────┐      │       │
│  │  │ Feast SDK     │   │     │  │ Feast SDK     │      │       │
│  │  │ get_historical│   │     │  │ get_online    │      │       │
│  │  │ _features()   │   │     │  │ _features()   │      │       │
│  │  └───────┬───────┘   │     │  └───────┬───────┘      │       │
│  │          │           │     │          │              │       │
│  │  ┌───────▼───────┐   │     │  ┌───────▼───────┐      │       │
│  │  │ PyTorch DDP   │   │     │  │ Model Predict │      │       │
│  │  │ (4 GPUs)      │   │     │  │ Compare Base  │      │       │
│  │  └───────────────┘   │     │  └───────────────┘      │       │
│  └──────────────────────┘     └─────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Feature Store** | Feast + PostgreSQL | Feature registry, online serving |
| **Training** | Kubeflow Trainer v2 | Distributed PyTorch training |
| **Model** | MLP (256→128→64→1) | Sales forecasting |
| **Storage** | NFS PVC (RWX) | Shared data/models |
| **Runtime** | torch-with-storage | Custom ClusterTrainingRuntime |

## 🚀 Quick Start

### Prerequisites

- OpenShift AI / RHOAI cluster
- `kubectl` configured
- Namespace: `feast-trainer-demo`

### Deploy

```bash
# 1. Create namespace and storage
kubectl create namespace feast-trainer-demo
kubectl apply -f v2/manifests/02-pvc-shared-storage.yaml
kubectl apply -f v2/manifests/03-clustertrainingruntime.yaml

# 2. Deploy Feast PostgreSQL
kubectl apply -f v2/manifests/05-feast-postgres.yaml
kubectl wait --for=condition=available deployment/feast-postgres -n feast-trainer-demo --timeout=120s

# 3. Prepare data & register features
kubectl apply -f v2/manifests/data-prep-job.yaml
kubectl wait --for=condition=complete job/feast-data-prep -n feast-trainer-demo --timeout=300s

# 4. Train model (fetches features via Feast)
kubectl apply -f v2/manifests/04-trainjob.yaml
kubectl wait --for=jsonpath='{.status.state}'=Complete trainjob/sales-forecasting -n feast-trainer-demo --timeout=600s

# 5. Run inference
kubectl apply -f v2/manifests/07-inference-job.yaml
kubectl logs -n feast-trainer-demo -l job-name=feast-inference -f
```

## 📊 Features Used

### Sales Features (Historical)
- `lag_1`, `lag_2`, `lag_4`, `lag_8`, `lag_52` - Past sales
- `rolling_mean_4w`, `rolling_std_4w` - 4-week rolling stats
- `rolling_mean_8w`, `rolling_std_8w` - 8-week rolling stats
- `rolling_mean_52w` - 52-week (YoY) rolling mean

### Store Features (External)
- `store_size`, `temperature`, `fuel_price`, `cpi`, `unemployment`
- `markdown1` - `markdown5` - Promotion markdowns
- `is_holiday`, `week_of_year`, `month` - Calendar features

### No Data Leakage ✅
- All lag features use `.shift(1)` before rolling
- Target (`weekly_sales`) never used as input feature
- Temporal train/val split (2010-2011 train, 2012 val)

## 🔧 Configuration

### Feast Feature Store
```yaml
# feature_store.yaml
project: sales_forecasting
registry:
  registry_type: sql
  path: postgresql+psycopg://feast:feast123@feast-postgres:5432/feast
offline_store:
  type: file  # Parquet files
online_store:
  type: postgres  # Real-time serving
```

### Training Parameters
```yaml
# 04-trainjob.yaml
numNodes: 1
numProcPerNode: 4  # 4 GPUs
epochs: 10
batch_size: 256
learning_rate: 1e-3
```

## 📁 File Structure

```
v2/
├── manifests/
│   ├── 01-namespace.yaml           # feast-trainer-demo
│   ├── 02-pvc-shared-storage.yaml  # NFS PVCs
│   ├── 03-clustertrainingruntime.yaml  # torch-with-storage
│   ├── 04-trainjob.yaml            # Training with Feast
│   ├── 05-feast-postgres.yaml      # PostgreSQL + init
│   ├── 07-inference-job.yaml       # Inference with Feast
│   └── data-prep-job.yaml          # Data + feast apply
├── feature_repo/
│   ├── feature_store.yaml          # Feast config
│   └── features.py                 # Feature definitions
└── README.md
```

## 🎓 Key Learnings

1. **Feast Integration**: Use `get_historical_features()` for training (point-in-time join), `get_online_features()` for real-time inference
2. **TrainJob v2**: Use `ClusterTrainingRuntime` for shared storage, not inline volume mounts
3. **OpenShift**: Use `nfs-csi` storage class for RWX access
4. **DDP Training**: `torchrun` with `PET_*` environment variables from TrainJob status

## 📈 Results

```
Model                            MAPE           RMSE            MAE
-------------------------------------------------------------------
Base (Random)                   62.0%         31,627         24,985
Naive (Mean)                    78.6%         28,376         23,847
Trained (Feast)                  9.9%          6,471          4,608

✅ Improvement vs Naive: 87.5%
```

## 🔗 References

- [Kubeflow Trainer](https://github.com/opendatahub-io/trainer)
- [Kubeflow SDK](https://github.com/opendatahub-io/kubeflow-sdk)
- [Feast Feature Store](https://github.com/feast-dev/feast)
- [Red Hat AI Quickstarts](https://github.com/rh-ai-quickstarts)
