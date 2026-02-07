# E2E MLOps Pipeline Workflow (With Ray)

> **⚠️ Experimental** - Ray integration with Feast is a contributed plugin, not production-ready.
> Use this for learning distributed computing patterns, not for production deployments.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE E2E FLOW (With Ray)                            │
└─────────────────────────────────────────────────────────────────────────────────┘

INFRASTRUCTURE (kubectl apply -k manifests/)
═══════════════════════════════════════════

    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │  PostgreSQL     │   │  RayCluster     │   │  MLflow         │
    │  postgres │   │  feast-ray      │   │  mlflow         │
    │                 │   │                 │   │                 │
    │  • Registry     │   │  • 1 head       │   │  • Tracking     │
    │  • Offline Store│   │  • 2 workers    │   │  • Artifacts    │
    │  • Online Store │   │    (CPU-only)   │   │                 │
    └─────────────────┘   └─────────────────┘   └─────────────────┘
           ▲                      ▲                     ▲
           │                      │                     │
           └──────────────────────┴─────────────────────┘
                          Always Running


═══════════════════════════════════════════════════════════════════════════════════
STEP 1: FEATURE ENGINEERING (01_feast_features.py) - Run as K8s Job or locally
═══════════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 1a. GENERATE SYNTHETIC DATA                                             │
    └─────────────────────────────────────────────────────────────────────────┘
    
         generate_sales_data()
              │
              ▼
         ┌─────────────────────────────────────────────────────────┐
         │  DataFrame with columns:                                │
         │  store, dept, date, weekly_sales, is_holiday,          │
         │  temperature, fuel_price, cpi, unemployment            │
         │  (143 stores × 81 depts × 52 weeks = ~600K rows)       │
         └─────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 1b. WRITE TO POSTGRESQL (Offline Store)                                 │
    └─────────────────────────────────────────────────────────────────────────┘
    
         from sqlalchemy import create_engine
         engine = create_engine(POSTGRES_URI)
         df.to_sql("sales_features", engine, if_exists="replace")
              │
              ▼
         ┌─────────────────────────────────────────────────────────┐
         │  PostgreSQL: sales_features table                       │
         │  ┌─────────────────────────────────────────────────────┐│
         │  │ store | dept | date       | weekly_sales | ...     ││
         │  │ 1     | 1    | 2024-01-05 | 24924.50     | ...     ││
         │  │ 1     | 1    | 2024-01-12 | 46039.49     | ...     ││
         │  │ ...   | ...  | ...        | ...          | ...     ││
         │  └─────────────────────────────────────────────────────┘│
         └─────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 1c. FEAST APPLY (Register Feature Definitions)                          │
    └─────────────────────────────────────────────────────────────────────────┘
    
         store = FeatureStore(repo_path="feature_repo/")
         store.apply([entity, feature_view, feature_service])
              │
              ▼
         ┌─────────────────────────────────────────────────────────┐
         │  PostgreSQL: feast_registry table (metadata)            │
         │  Stores: Entity, FeatureView, FeatureService definitions│
         └─────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 1d. FEAST MATERIALIZE (Sync Offline → Online via Ray)                   │
    └─────────────────────────────────────────────────────────────────────────┘
    
         store.materialize(start_date, end_date)
              │
              │  Uses Ray compute engine (from feature_store.yaml):
              │  batch_engine:
              │    type: ray.RayMaterializationEngine
              │    
              ▼
         ┌─────────────────────────────────────────────────────────┐
         │                    RayCluster                           │
         │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
         │  │ Ray Worker 1│  │ Ray Worker 2│  │ Ray Head    │     │
         │  │             │  │             │  │             │     │
         │  │ Partition 1 │  │ Partition 2 │  │ Coordinator │     │
         │  │ (stores 1-50)  │ (stores 51+)│  │             │     │
         │  └──────┬──────┘  └──────┬──────┘  └─────────────┘     │
         │         │                │                              │
         │         └────────────────┴──────────────┐               │
         │                                         ▼               │
         │                              Write to Online Store      │
         └─────────────────────────────────────────────────────────┘
              │
              ▼
         ┌─────────────────────────────────────────────────────────┐
         │  PostgreSQL: feast_online_store table                   │
         │  Key-value format: (entity_key) → (latest features)     │
         │  ┌─────────────────────────────────────────────────────┐│
         │  │ entity_key    | feature_values       | event_ts    ││
         │  │ store=1,dept=1| {sales:24924, ...}   | 2024-12-28  ││
         │  │ store=1,dept=2| {sales:12345, ...}   | 2024-12-28  ││
         │  └─────────────────────────────────────────────────────┘│
         └─────────────────────────────────────────────────────────┘

    ⏱️ Step 1 Total Time: ~5-8 minutes (Ray startup + materialization)


═══════════════════════════════════════════════════════════════════════════════════
STEP 2: TRAINING (02_training.py) - Submitted as Kubeflow TrainJob
═══════════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 2a. SUBMIT TRAINJOB                                                     │
    └─────────────────────────────────────────────────────────────────────────┘
    
         TrainerClient().train(
             trainer=CustomTrainer(func=train_sales_model, ...),
             runtime=PyTorchRuntime(num_nodes=1),
             ...
         )
              │
              ▼
         ┌─────────────────────────────────────────────────────────┐
         │  Kubeflow Creates TrainJob Pod                          │
         │  ┌─────────────────────────────────────────────────────┐│
         │  │ Training Pod                                        ││
         │  │   • Mounts: /mnt/shared (PVC with feature_repo)     ││
         │  │   • Env: MLFLOW_TRACKING_URI, POSTGRES_* etc        ││
         │  └─────────────────────────────────────────────────────┘│
         └─────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 2b. INSIDE TRAINING POD: FETCH TRAINING DATA VIA RAY                    │
    └─────────────────────────────────────────────────────────────────────────┘
    
         # Initialize Feast inside training function
         store = FeatureStore(repo_path="/mnt/shared/feature_repo")
         
         # Create entity_df for point-in-time join
         entity_df = pd.DataFrame({
             "store": [...],
             "dept": [...], 
             "event_timestamp": [...]
         })
         
         # Fetch historical features (uses Ray for distributed joins)
         training_df = store.get_historical_features(
             entity_df=entity_df,
             features=["sales_features:weekly_sales", "sales_features:temperature", ...]
         ).to_df()
              │
              │  Feast connects to RayCluster for distributed join:
              │  
              ▼
         ┌─────────────────────────────────────────────────────────┐
         │                    RayCluster                           │
         │  ┌─────────────────────────────────────────────────────┐│
         │  │  Point-in-Time Join (distributed across workers)    ││
         │  │                                                     ││
         │  │  entity_df  ⋈  sales_features (offline store)       ││
         │  │                                                     ││
         │  │  For each (store, dept, event_timestamp):           ││
         │  │    → Find latest feature values <= event_timestamp  ││
         │  └─────────────────────────────────────────────────────┘│
         └─────────────────────────────────────────────────────────┘
              │
              ▼
         ┌─────────────────────────────────────────────────────────┐
         │  training_df: Complete training dataset                 │
         │  ┌─────────────────────────────────────────────────────┐│
         │  │ store | dept | event_ts   | weekly_sales | temp | ..││
         │  │ 1     | 1    | 2024-01-05 | 24924.50     | 42.3 | ..││
         │  │ 1     | 1    | 2024-01-12 | 46039.49     | 38.5 | ..││
         │  └─────────────────────────────────────────────────────┘│
         └─────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 2c. TRAIN MODEL                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
    
         X = training_df[FEATURE_COLUMNS]
         y = training_df["weekly_sales"]
         
         X_train, X_test, y_train, y_test = train_test_split(...)
         
         # PyTorch training loop
         model = SalesForecastModel()
         for epoch in range(num_epochs):
             ...
              │
              ▼
         ┌─────────────────────────────────────────────────────────┐
         │  Trained model weights                                  │
         └─────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 2d. SAVE MODEL & LOG TO MLFLOW                                          │
    └─────────────────────────────────────────────────────────────────────────┘
    
         torch.save(model.state_dict(), "/mnt/shared/models/model.pt")
         
         mlflow.log_params({"epochs": 50, "lr": 0.001, ...})
         mlflow.log_metrics({"test_mse": 0.15, "test_r2": 0.87})
         mlflow.log_artifact("/mnt/shared/models/model.pt")
              │
              ▼
         ┌─────────────────────────────────────────────────────────┐
         │  MLflow                                                 │
         │  ┌─────────────────────────────────────────────────────┐│
         │  │ Run: sales-training-20240207-123456                 ││
         │  │ Params: {epochs: 50, lr: 0.001, ...}                ││
         │  │ Metrics: {test_mse: 0.15, test_r2: 0.87}            ││
         │  │ Artifacts: model.pt, scaler.pkl                     ││
         │  └─────────────────────────────────────────────────────┘│
         └─────────────────────────────────────────────────────────┘

    ⏱️ Step 2 Total Time: ~5-10 minutes


═══════════════════════════════════════════════════════════════════════════════════
STEP 3: DEPLOYMENT & INFERENCE (03_inference.py)
═══════════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 3a. DEPLOY KSERVE INFERENCE SERVICE                                     │
    └─────────────────────────────────────────────────────────────────────────┘
    
         KServeClient().create(inference_service)
              │
              ▼
         ┌─────────────────────────────────────────────────────────┐
         │  KServe InferenceService Pod                            │
         │  ┌─────────────────────────────────────────────────────┐│
         │  │ Model Server                                        ││
         │  │   • Loads model.pt from PVC or S3                   ││
         │  │   • Mounts feature_repo for Feast config            ││
         │  │   • Env: POSTGRES_* for online store access         ││
         │  └─────────────────────────────────────────────────────┘│
         └─────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 3b. INFERENCE REQUEST (Real-time) - NO RAY NEEDED                       │
    └─────────────────────────────────────────────────────────────────────────┘
    
         # Client sends request with just entity keys
         POST /v1/models/sales-model:predict
         {
             "instances": [
                 {"store": 1, "dept": 3},
                 {"store": 5, "dept": 12}
             ]
         }
              │
              ▼
         ┌─────────────────────────────────────────────────────────┐
         │  Model Server: Fetch features from Online Store         │
         │                                                         │
         │  store = FeatureStore(repo_path="/mnt/feature_repo")    │
         │  features = store.get_online_features(                  │
         │      entity_rows=[{"store": 1, "dept": 3}, ...],        │
         │      features=["sales_features:temperature", ...]       │
         │  ).to_dict()                                            │
         └─────────────────────────────────────────────────────────┘
              │
              │  Direct lookup (NO RAY - just key-value fetch)
              │  Latency: ~5-10ms
              │
              ▼
         ┌─────────────────────────────────────────────────────────┐
         │  PostgreSQL Online Store                                │
         │  SELECT feature_values FROM feast_online_store          │
         │  WHERE entity_key IN ('store=1,dept=3', ...)            │
         └─────────────────────────────────────────────────────────┘
              │
              ▼
         ┌─────────────────────────────────────────────────────────┐
         │  Model Prediction                                       │
         │                                                         │
         │  X = prepare_features(features)                         │
         │  predictions = model(X)                                 │
         │                                                         │
         │  return {"predictions": [24500.0, 18200.0]}             │
         └─────────────────────────────────────────────────────────┘

    ⏱️ Step 3 Deploy: ~2-3 minutes
    ⏱️ Each inference: ~20-50ms


═══════════════════════════════════════════════════════════════════════════════════
RAY USAGE SUMMARY
═══════════════════════════════════════════════════════════════════════════════════

    ONE PERSISTENT RAYCLUSTER (feast-ray) serves TWO purposes:

    ┌──────────────────────────────────────────────────────────────────────────┐
    │                                                                          │
    │   RAY USAGE #1: Materialization (Step 1d)                               │
    │   ───────────────────────────────────────                               │
    │   When: During feature engineering                                       │
    │   What: store.materialize() uses Ray to read from offline store         │
    │         and write to online store in parallel                           │
    │   Why:  Large data volumes benefit from distributed processing          │
    │                                                                          │
    │   RAY USAGE #2: Historical Feature Retrieval (Step 2b)                  │
    │   ─────────────────────────────────────────────────────                 │
    │   When: During training                                                  │
    │   What: store.get_historical_features() uses Ray for distributed        │
    │         point-in-time joins                                             │
    │   Why:  Point-in-time joins are expensive; Ray parallelizes them        │
    │                                                                          │
    │   NO RAY for Inference:                                                 │
    │   ─────────────────────                                                 │
    │   get_online_features() is a simple key-value lookup                    │
    │   Direct PostgreSQL query - no computation needed                       │
    │                                                                          │
    └──────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════
SUMMARY
═══════════════════════════════════════════════════════════════════════════════════

    COMPONENTS NEEDED:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │  ✅ PostgreSQL         (Feast registry + offline + online store)        │
    │  ✅ RayCluster         (Distributed compute for Feast)                  │
    │  ✅ MLflow             (Experiment tracking)                            │
    │  ✅ PVC                (Shared storage)                                 │
    │  ✅ Kubeflow Trainer   (Training jobs)                                  │
    │  ✅ KServe             (Model serving)                                  │
    └──────────────────────────────────────────────────────────────────────────┘

    TOTAL E2E TIME: ~15-20 minutes (first run)
    
    FILES:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │  01_feast_features.py  → Generate data, write to PG, materialize via Ray│
    │  02_training.py        → TrainJob, get_historical_features via Ray      │
    │  03_inference.py       → KServe, get_online_features (no Ray)           │
    │  feature_store.yaml    → PostgreSQL + Ray batch_engine config           │
    └──────────────────────────────────────────────────────────────────────────┘
```

## feature_store.yaml (With Ray)

```yaml
project: sales_forecasting
provider: local

registry:
  registry_type: sql
  path: postgresql://feast:feast@postgres:5432/feast

offline_store:
  type: postgres
  host: postgres
  port: 5432
  database: feast
  user: feast
  password: feast

online_store:
  type: postgres
  host: postgres
  port: 5432
  database: feast
  user: feast
  password: feast

# Ray compute engine for materialization and historical features
batch_engine:
  type: ray.RayMaterializationEngine
  ray_cluster:
    address: ray://feast-ray-head-svc:10001

entity_key_serialization_version: 2
```

## ⚠️ Important Caveats

1. **Ray is a "contrib" plugin** - Not officially supported for production
2. **Adds complexity** - More infrastructure to manage
3. **Longer startup time** - Ray cluster needs to be running
4. **Use for learning** - Great for understanding distributed patterns

## Pros & Cons

| Aspect | Rating | Notes |
|--------|--------|-------|
| Complexity | ⭐⭐ | More moving parts, harder to debug |
| Demo Speed | ⭐⭐ | ~15-20 min total |
| Production Ready | ⭐⭐ | Ray is experimental in Feast |
| Scalability | ⭐⭐⭐⭐⭐ | Designed for large scale |
| Learning Value | ⭐⭐⭐⭐⭐ | Shows distributed patterns |

## When to Use Ray

- Dataset > 10GB
- Complex feature transformations
- Learning distributed computing
- Already have Ray infrastructure

## When NOT to Use Ray

- Quick demos
- Dataset < 1GB  
- Production deployments (use Redis + BigQuery instead)
- Limited cluster resources

