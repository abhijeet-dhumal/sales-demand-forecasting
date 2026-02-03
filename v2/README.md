# Sales Demand Forecasting v2

**Using TrainJob v2 API with Kubeflow SDK**

## Key Improvements over v1

| Issue | v1 (Legacy) | v2 (Fixed) |
|-------|-------------|------------|
| **Training API** | `PyTorchJob` | **`TrainJob v2`** (`trainer.kubeflow.org/v1alpha1`) |
| **SDK** | Manual job creation | **`TrainerClient` + `CustomTrainer`** |
| **Data Leakage** | Features derived from target | **Only lag/historical features** |
| **Train/Val Split** | Random chunks | **Temporal split by date** |
| **Feast Version** | 0.54.0 | **0.59.0** |

## Directory Structure

```
v2/
├── feature_repo/           # Feast feature definitions
│   ├── feature_store.yaml  # Feast configuration
│   └── features.py         # Feature views (no data leakage)
├── training/
│   └── train.py            # Standalone training script
├── notebooks/
│   ├── 01_data_preparation.ipynb  # Data prep + Feast setup
│   ├── 02_training.ipynb          # TrainJob submission
│   └── 03_inference.ipynb         # Model evaluation
├── manifests/              # Kubernetes manifests (optional)
├── requirements.txt        # Pinned dependencies
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# Or install ODH SDK from source:
pip install git+https://github.com/opendatahub-io/kubeflow-sdk.git
```

### 2. Prepare Data

```bash
cd notebooks
jupyter lab 01_data_preparation.ipynb
```

### 3. Submit Training Job

```python
from kubeflow.trainer import TrainerClient, CustomTrainer

client = TrainerClient()

job_name = client.train(
    runtime="torch-distributed",
    trainer=CustomTrainer(
        func=train_sales_forecast,  # Self-contained function
        num_nodes=2,
        resources_per_node={"cpu": 8, "memory": "32Gi"},
        packages_to_install=["torch", "feast==0.59.0", "pandas"],
    ),
)

# Stream logs
for log in client.get_job_logs(job_name, follow=True):
    print(log, end="")
```

### 4. Local Testing (No Cluster)

```python
from kubeflow.trainer import TrainerClient, CustomTrainer, LocalProcessBackendConfig

client = TrainerClient(backend_config=LocalProcessBackendConfig())
job_name = client.train(
    trainer=CustomTrainer(func=train_sales_forecast, ...),
)
```

## SDK Trainer Options

| Trainer | Use Case |
|---------|----------|
| `CustomTrainer` | User-defined Python function |
| `CustomTrainerContainer` | Pre-built container image |
| `BuiltinTrainer` | TorchTune LLM fine-tuning |
| `TransformersTrainer` | HuggingFace Transformers/TRL |
| `TrainingHubTrainer` | RHAI Training Hub (SFT/OSFT) |

## Feature Engineering (No Data Leakage)

**Safe features (used in v2):**
- `lag_1`, `lag_2`, `lag_4`, `lag_8`, `lag_52` (historical sales)
- `rolling_mean_4w`, `rolling_mean_8w` (computed from past data)
- `temperature`, `fuel_price`, `cpi` (external factors)
- `is_holiday`, `week_of_year` (calendar features)

**Removed features (caused data leakage in v1):**
- ❌ `sales_normalized` (derived from current week's sales)
- ❌ `sales_per_sqft` (derived from current week's sales)
- ❌ `markdown_efficiency` (derived from current week's sales)

## Temporal Split

```
2010-02  ──────────────────  2011-12  ──  2012-01  ────────  2012-10
          TRAINING SET                      VALIDATION SET
```

**NOT** random 80/20 splits which leak future data into training.

## References

- [ODH Trainer](https://github.com/opendatahub-io/trainer) - TrainJob v2 CRD
- [ODH Kubeflow SDK](https://github.com/opendatahub-io/kubeflow-sdk) - Python SDK
- [Feast](https://github.com/feast-dev/feast) - Feature Store v0.59.0

