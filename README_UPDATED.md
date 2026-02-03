# Sales Demand Forecasting with Feast + Kubeflow Training Operator

**Production ML Pipeline with PostgreSQL + Ray Architecture**

## Overview

This project demonstrates enterprise-grade ML workflows for Walmart sales forecasting, combining:
- **Feast 0.56.0** (Feature Store with PostgreSQL + Ray)
- **Kubeflow Training Operator** (Distributed Training on OpenShift AI)
- **PyTorch DDP** (Data Parallel Training)
- **Temporal Fusion Transformer** (State-of-the-art time-series model)

### Business Impact
- **10.5% MAPE** forecast error (vs 15-20% industry baseline)
- **10x faster** feature loading (PostgreSQL + Ray vs file-based)
- **~10-12 min** distributed training (100K sample quickstart)
- **Production-ready** architecture (PostgreSQL storage, Ray compute)

---

## Architecture

### Data Flow
```
Raw Data (Kaggle) → PostgreSQL (feast_offline)
                          ↓
            Feast + Ray Compute Engine
            (distributed joins & transforms)
                          ↓
              Distributed PyTorch Training
              (2 workers, DDP, mixed precision)
                          ↓
                    Trained Model
                          ↓
            Feature Materialization → PostgreSQL (feast_online)
                          ↓
                Real-time Inference
```

### Technology Stack

#### Feature Store (Feast 0.56.0)
- **Storage Layer**: PostgreSQL
  - `feast_registry`: Feature definitions & metadata (SQL-queryable)
  - `feast_offline`: Raw feature storage (ACID-compliant, indexed)
  - `feast_online`: Low-latency serving (materialized features)
  
- **Compute Layer**: Ray
  - Automatic parallelization of joins & transformations
  - 8 workers for distributed processing
  - Optimized for 421K row dataset

#### Training Infrastructure
- **Kubeflow Training Operator**: Manages PyTorchJob on OpenShift
- **PyTorch DDP**: Data parallel training across 2 workers
- **Mixed Precision**: 2x speedup with AMP (Automatic Mixed Precision)
- **Shared Storage**: PVC for models and data

---

## Quickstart (100K Sample)

### 1. Deploy PostgreSQL

```bash
# Create namespace
oc new-project kft-feast-quickstart

# Deploy PostgreSQL with Feast databases
oc apply -f postgres-feast-registry-openshift.yaml

# Wait for PostgreSQL to be ready
oc wait --for=condition=ready pod -l app=feast-postgres --timeout=300s

# Verify databases created
oc exec -it feast-postgres-0 -- psql -U feast -d feast_registry -c "\l"
# Should show: feast_registry, feast_offline, feast_online
```

### 2. Data Preparation (Notebook 01)

```python
# Notebook: 01_data_preparation_feast_setup.ipynb

# Downloads Walmart dataset from Kaggle
# Performs feature engineering (lags, rolling stats, on-demand transforms)
# Loads to PostgreSQL (sales_features, store_features tables)
# Registers features with Feast

# Expected time: ~5-7 minutes
# Output:
#   - PostgreSQL tables: sales_features (421K rows), store_features (421K rows)
#   - Feast registry: 2 FeatureViews, 2 OnDemandFeatureViews, 1 FeatureService
```

### 3. Distributed Training (Notebook 02)

```python
# Notebook: 02_distributed_training_kubeflow.ipynb

# Creates PyTorchJob using kubeflow-training SDK
# Loads features using Feast + PostgreSQL + Ray (1-2 min)
# Trains Temporal Fusion Transformer (8-10 min)

training_parameters = {
    "data_source": "postgres_ray",  # PostgreSQL + Ray
    "feast_repo_path": "/shared/feature_repo",
    "sample_size": 100000,  # QUICKSTART
    "num_epochs": 5,
    "model_type": "tft",
}

# Expected time: ~10-12 minutes
# Output:
#   - Trained model: /shared/models/best_model.pt
#   - Scalers: scaler.pkl, target_scaler.pkl
#   - Training logs with MAPE metrics
```

### 4. Model Evaluation (Notebook 03)

```python
# Notebook: 03_model_evaluation_inference.ipynb

# Loads trained model
# Evaluates on test set (MAPE, MAE, RMSE)
# Materializes features to PostgreSQL online store
# Demonstrates real-time inference

# Expected time: ~3-5 minutes
# Output:
#   - Test MAPE: ~10.5%
#   - Feature vectors materialized to feast_online
#   - Real-time predictions (<100ms latency)
```

---

## Full Dataset (421K Rows)

Update `training_parameters` in Notebook 02:

```yaml
sample_size: null  # Use full dataset
num_epochs: 20     # More epochs for better convergence
```

**Expected Time**: ~30-40 min (feature loading: 3-5 min, training: 25-35 min)

**Expected MAPE**: ~9-10% (better than 10.5% with 100K sample)

---

## Performance Comparison

### Before: File-Based Feast (0.54.0)
```
Architecture: Parquet files + FileSource
Feature loading: 15-18 min (100K rows)
Bottleneck: Single-threaded file I/O
Total time: ~25-30 min
```

### After: PostgreSQL + Ray (0.56.0)
```
Architecture: PostgreSQL storage + Ray compute
Feature loading: 1-2 min (100K rows)
Improvement: 10x faster (distributed processing)
Total time: ~10-12 min
```

---

## Model Performance

### Temporal Fusion Transformer (TFT)
- **Test MAPE**: 10.5% (quickstart), 9-10% (full dataset)
- **Architecture**: 64-dim hidden, 4 attention heads, 2 LSTM layers
- **Training**: 5 epochs (quickstart), 20 epochs (production)
- **Inference**: <100ms per prediction (materialized features)

### Feature Importance
1. **sales_lag_1** (last week's sales) - strongest predictor
2. **sales_rolling_mean_12** (12-week trend)
3. **is_holiday** (holiday indicator)
4. **holiday_markdown_interaction** (promotional impact)
5. **seasonal_sine/cosine** (cyclical patterns)

---

## Project Structure

```
sales-demand-forecasting/
├── 01_data_preparation_feast_setup.ipynb     # ETL + Feast setup
├── 02_distributed_training_kubeflow.ipynb    # Distributed training
├── 03_model_evaluation_inference.ipynb       # Evaluation + serving
├── torch_training.py                         # PyTorch training logic
├── feature_repo/
│   ├── feature_store.yaml                    # Feast config (PostgreSQL + Ray)
│   └── features.py                           # Feature definitions
├── postgres-feast-registry-openshift.yaml    # PostgreSQL StatefulSet
├── init_postgres.sql                         # Database initialization
└── README.md                                 # This file
```

---

## Configuration Files

### feature_store.yaml (PostgreSQL + Ray)

```yaml
project: sales_demand_forecasting

# PostgreSQL registry (feature metadata)
registry:
  registry_type: sql
  path: postgresql+psycopg://feast:feast_password@feast-postgres:5432/feast_registry
  cache_ttl_seconds: 60

# PostgreSQL offline store (feature storage)
offline_store:
  type: postgres
  host: feast-postgres.kft-feast-quickstart.svc.cluster.local
  database: feast_offline
  user: feast
  password: feast_password

# Ray compute engine (distributed processing)
batch_engine:
  type: ray.engine
  max_workers: 8
  enable_optimization: true
  enable_distributed_joins: true

# PostgreSQL online store (low-latency serving)
online_store:
  type: postgres
  host: feast-postgres.kft-feast-quickstart.svc.cluster.local
  database: feast_online
  user: feast
  password: feast_password
```

### features.py (PostgreSQL Sources)

```python
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource

# PostgreSQL source for sales features
sales_source = PostgreSQLSource(
    name="sales_source",
    query="SELECT * FROM sales_features",
    timestamp_field="date",
)

# PostgreSQL source for store features
store_external_source = PostgreSQLSource(
    name="store_external_source",
    query="SELECT * FROM store_features",
    timestamp_field="date",
)
```

---

## Troubleshooting

### PostgreSQL Connection Issues

```bash
# Check PostgreSQL pod status
oc get pods -l app=feast-postgres

# View PostgreSQL logs
oc logs feast-postgres-0

# Test connection from within cluster
oc run -it --rm debug --image=postgres:15 --restart=Never -- \
  psql -h feast-postgres.kft-feast-quickstart.svc.cluster.local -U feast -d feast_offline
```

### Feature Loading Slow

```bash
# Check Ray worker utilization
# In training logs, look for:
#   "✓ Offline store: postgres"
#   "✓ Batch engine: ray.engine"

# Verify indexes created
oc exec -it feast-postgres-0 -- psql -U feast -d feast_offline -c "\d sales_features"
# Should show indexes: idx_sales_date, idx_sales_store_dept, idx_sales_store_dept_date
```

### Training Job Stuck

```bash
# Check PyTorchJob status
oc get pytorchjob -n kft-feast-quickstart

# View worker logs
oc logs -f <pytorch-job-name>-worker-0

# Common issues:
#   - PVC not mounted: Check /shared directory exists
#   - GPU allocation: Verify nvidia.com/gpu resource requests
#   - NCCL timeout: Check network connectivity between workers
```

---

## Next Steps

1. **Scale to Full Dataset**: Set `sample_size: null` for 421K rows
2. **Hyperparameter Tuning**: Adjust TFT architecture (hidden_size, num_heads)
3. **Feature Engineering**: Add external data (weather, promotions, competitors)
4. **Production Deployment**: 
   - Set up CI/CD pipeline for model retraining
   - Deploy Feast online serving endpoint
   - Monitor feature drift and model performance

---

## References

- [Feast Documentation](https://docs.feast.dev/)
- [Kubeflow Training Operator](https://www.kubeflow.org/docs/components/training/)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Temporal Fusion Transformer Paper](https://arxiv.org/abs/1912.09363)
- [Walmart Sales Forecasting Dataset](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)

---

## License

MIT License - See LICENSE file

---

## Authors

Red Hat AI/ML Engineering Team

**Questions?** Open an issue or contact the maintainers.



