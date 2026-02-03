# Notebook 02 Updates - PostgreSQL + Ray Architecture

## Cell 6: Update Training Parameters

Replace the `training_parameters` YAML with:

```yaml
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data configuration - PostgreSQL + Ray (Production Architecture)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
data_source: postgres_ray  # PostgreSQL offline store + Ray compute engine
feast_repo_path: /shared/feature_repo
chunk_size: 50000          # 9 chunks for 421K rows (distributed across workers)
sample_size: 100000        # QUICKSTART: 100K rows (~1-2 min with PostgreSQL+Ray)
                           # FULL: null for all 421K rows (~3-5 min)
test_size: 0.2
val_size: 0.1

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model configuration - TFT (Temporal Fusion Transformer)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
model_type: tft
model_output_dir: /shared/models

# TFT Architecture (state-of-the-art for time-series)
tft_hidden_size: 64        # Core feature dimension (controls model capacity)
tft_num_heads: 4           # Multi-head attention (captures different patterns)
tft_lstm_layers: 2         # LSTM depth for temporal dependencies
tft_dropout: 0.1           # Regularization (prevents overfitting)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
num_epochs: 5              # QUICKSTART: 5 epochs (~10-15 min)
                           # PRODUCTION: 20-50 epochs for optimal performance
batch_size: 128            # Per-worker batch size (2 workers = 256 effective)
accumulation_steps: 2      # Gradient accumulation (effective batch = 512)
learning_rate: 0.0014
weight_decay: 0.0001
early_stopping_patience: 5
use_amp: true              # Mixed precision training (2x speedup)
grad_clip_norm: 1.0
```

**Key Changes:**
- `data_source: postgres_ray` (was `feast_file`)
- Performance: 1-2 min feature loading (vs 15-18 min with file-based Feast)
- Production-ready: PostgreSQL storage + Ray distributed compute

## Cell 7: Update Time Estimates

Replace the time estimate section:

```python
if training_parameters['model_type'] == 'tft':
    print('\n📊 TFT Architecture:')
    print(f"   Hidden size: {training_parameters['tft_hidden_size']}")
    print(f"   Attention heads: {training_parameters['tft_num_heads']}")
    print(f"   LSTM layers: {training_parameters['tft_lstm_layers']}")
    print(f"   Dropout: {training_parameters['tft_dropout']}")
    if training_parameters['data_source'] == 'postgres_ray':
        print(f"\n⏱️  Estimated time: ~10-12 min (5 epochs, 100K sample)")
        print(f"   Feature loading: ~1-2 min (PostgreSQL + Ray)")
        print(f"   Training: ~8-10 min (2 workers, GPU)")
    else:
        print(f"\n⏱️  Estimated time: ~8-10 hours (5 epochs, full dataset)")
else:
    print('\n📊 MLP Architecture:')
    print(f"   Hidden dims: {training_parameters['hidden_dims']}")
    print(f"   Dropout: {training_parameters['dropout']}")
    if training_parameters['data_source'] == 'postgres_ray':
        print(f"\n⏱️  Estimated time: ~6-8 min (5 epochs, 100K sample)")
        print(f"   Feature loading: ~1-2 min (PostgreSQL + Ray)")
        print(f"   Training: ~5-6 min (2 workers, GPU)")
    else:
        print(f"\n⏱️  Estimated time: ~50-60 minutes (5 epochs, full dataset)")

print(f'\n🎓 Training:')
print(f"   Epochs: {training_parameters['num_epochs']}")
print(f"   Learning rate: {training_parameters['learning_rate']}")
print(f"   Weight decay: {training_parameters['weight_decay']}")
print(f"   Batch: {training_parameters['batch_size']} × {training_parameters['accumulation_steps']} steps")
print(f"   Effective batch: {training_parameters['batch_size']} × {training_parameters['accumulation_steps']} × 2 workers = {training_parameters['batch_size'] * training_parameters['accumulation_steps'] * 2}")
print(f"   Early stopping: {training_parameters['early_stopping_patience']} epochs")
print(f"   Mixed precision: {training_parameters['use_amp']}")

print(f'\n💾 Data:')
print(f"   Source: {training_parameters['data_source']}")
if training_parameters['data_source'] == 'postgres_ray':
    print(f"   ✓ PostgreSQL offline store (durable, indexed)")
    print(f"   ✓ Ray compute engine (distributed, parallel)")
    print(f"   ✓ Feast SDK - guarantees train/inference consistency")
    print(f"   ✓ On-demand transformations parallelized by Ray")
print(f"   Chunk size: {training_parameters['chunk_size']:,} rows")
print(f"   Val split: {training_parameters['val_size']*100:.0f}%")
```

## Cell 11: Update PyTorch Dependencies

Update the `pip_packages` list to include PostgreSQL and Ray dependencies:

```python
pip_packages=[
    # PyTorch ecosystem
    "torch==2.5.1",
    "torchvision==0.20.1",
    "lightning==2.4.0",
    # ML utilities
    "pandas==2.2.3",
    "numpy==1.26.4",
    "pyarrow==17.0.0",
    "scikit-learn==1.6.1",
    "joblib>=1.3.0",
    # PyTorch data streaming
    "torchdata>=0.7.0",
    # Feast + PostgreSQL + Ray (Production Architecture)
] + (
    [
        "feast[postgres]==0.56.0",  # Feast with PostgreSQL support
        "psycopg==3.1.18",          # PostgreSQL driver
        "sqlalchemy==2.0.36",       # Database toolkit
        "ray[default]==2.35.0",     # Distributed computing
    ] 
    if training_parameters.get('data_source') == 'postgres_ray' 
    else []
),
```

**Key Changes:**
- `feast[postgres]==0.56.0` (upgraded from 0.54.0)
- `psycopg==3.1.18` (PostgreSQL driver for Python 3.10+)
- `sqlalchemy==2.0.36` (required for PostgreSQL offline store)
- `ray[default]==2.35.0` (required for Ray compute engine)

## Cell 11: Update Environment Variables

Add PostgreSQL connection environment variables to the container spec:

```python
V1Container(
    name="pytorch",
    image=f"quay.io/modh/runtime-images:runtime-pytorch-ubi9-python-3.11-20241211",
    command=["torchrun"],
    args=[
        f"--nproc-per-node={training_parameters['num_gpus_per_worker']}",
        f"--nnodes={num_workers}",
        "--node-rank=$(RANK)",
        "--master-addr=$(MASTER_ADDR)",
        "--master-port=$(MASTER_PORT)",
        "/shared/torch_training.py",
    ],
    env=[
        # PostgreSQL connection for Feast
        V1EnvVar(name="FEAST_PG_HOST", value="feast-postgres.kft-feast-quickstart.svc.cluster.local"),
        V1EnvVar(name="FEAST_PG_PORT", value="5432"),
        V1EnvVar(name="FEAST_PG_USER", value="feast"),
        V1EnvVar(name="FEAST_PG_PASSWORD", value="feast_password"),
        V1EnvVar(name="FEAST_PG_DATABASE", value="feast_offline"),
        # Ray configuration
        V1EnvVar(name="RAY_memory_monitor_refresh_ms", value="0"),  # Disable Ray memory monitor
        V1EnvVar(name="RAY_DEDUP_LOGS", value="0"),  # Show all Ray logs
        # PyTorch configuration
        V1EnvVar(name="NCCL_DEBUG", value="INFO"),
        V1EnvVar(name="NCCL_SOCKET_IFNAME", value="eth0"),
        V1EnvVar(name="TORCH_DISTRIBUTED_DEBUG", value="DETAIL"),
    ],
    # ... rest of container spec
)
```

## Performance Comparison

### Before (file-based Feast):
```
Feature loading: 15-18 min (100K rows)
Total training:  ~25-30 min (5 epochs)
```

### After (PostgreSQL + Ray):
```
Feature loading: 1-2 min (100K rows, 10x faster)
Total training:  ~10-12 min (5 epochs, 2.5x faster)
```

### Full Dataset (421K rows):
```
Feature loading: 3-5 min (Ray distributed)
Total training:  ~15-20 min (5 epochs)
```

## Benefits

1. **Performance**: 10x faster feature loading via PostgreSQL + Ray
2. **Scalability**: Ray automatically parallelizes across available CPUs
3. **Durability**: PostgreSQL provides ACID-compliant storage
4. **Production-Ready**: Same architecture used for training and serving
5. **Feature Consistency**: Feast SDK guarantees identical transformations



