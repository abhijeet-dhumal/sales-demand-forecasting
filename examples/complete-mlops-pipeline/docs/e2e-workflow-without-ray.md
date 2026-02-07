# E2E MLOps Pipeline Workflow (Without Ray)

> **Recommended for demos and production** - Uses PostgreSQL for all Feast stores.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    E2E MLOps PIPELINE (No Ray)                                  │
└─────────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════════
INFRASTRUCTURE (kubectl apply -k manifests/)
═══════════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────┐       ┌─────────────────────┐       ┌─────────────────┐
    │     PostgreSQL      │       │       MLflow        │       │       PVC       │
    │   postgres    │       │       mlflow        │       │    feast-pvc    │
    │                     │       │                     │       │                 │
    │ • Registry (metadata)│      │ • Experiment tracking│      │ • feature_repo/ │
    │ • Offline Store     │       │ • Model artifacts   │       │ • models/       │
    │ • Online Store      │       │                     │       │                 │
    └─────────────────────┘       └─────────────────────┘       └─────────────────┘
    
    NO RAY CLUSTER NEEDED ✓


═══════════════════════════════════════════════════════════════════════════════════
STEP 1: FEATURE ENGINEERING (01_feast_features.py)
═══════════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ 1a. GENERATE SYNTHETIC DATA                                                 │
    └─────────────────────────────────────────────────────────────────────────────┘

         df = generate_sales_data()
         # 143 stores × 81 depts × 52 weeks ≈ 600K rows
              │
              ▼
         ┌───────────────────────────────────────────────────────────┐
         │  pandas DataFrame                                         │
         │  store | dept | date       | weekly_sales | temperature  │
         │  1     | 1    | 2024-01-05 | 24924.50     | 42.31        │
         │  ...                                                      │
         └───────────────────────────────────────────────────────────┘


    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ 1b. WRITE TO POSTGRESQL (Offline Store)                                     │
    └─────────────────────────────────────────────────────────────────────────────┘

         from sqlalchemy import create_engine
         engine = create_engine("postgresql://feast:feast@postgres:5432/feast")
         df.to_sql("sales_features", engine, if_exists="replace", index=False)
              │
              ▼
         ┌───────────────────────────────────────────────────────────┐
         │  PostgreSQL: sales_features table                         │
         │  (This is your offline store - historical data)           │
         └───────────────────────────────────────────────────────────┘


    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ 1c. FEAST APPLY (Register Feature Definitions)                              │
    └─────────────────────────────────────────────────────────────────────────────┘

         store = FeatureStore(repo_path="/mnt/shared/feature_repo")
         store.apply([sales_entity, sales_fv, sales_service])
              │
              ▼
         ┌───────────────────────────────────────────────────────────┐
         │  PostgreSQL: feast_registry table                         │
         │  (Stores Entity, FeatureView, FeatureService metadata)    │
         └───────────────────────────────────────────────────────────┘


    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ 1d. FEAST MATERIALIZE (Offline → Online)                                    │
    └─────────────────────────────────────────────────────────────────────────────┘

         store.materialize(
             start_date=datetime(2024, 1, 1),
             end_date=datetime.now()
         )
              │
              │  Native Feast materialization (NO RAY)
              │  Reads from PostgreSQL offline store
              │  Writes to PostgreSQL online store
              │
              ▼
         ┌───────────────────────────────────────────────────────────┐
         │  PostgreSQL: feast_online_store table                     │
         │                                                           │
         │  entity_key          | feature_values      | event_ts    │
         │  store=1__dept=1     | {sales:24924,...}   | 2024-12-28  │
         │  store=1__dept=2     | {sales:18234,...}   | 2024-12-28  │
         │  ...                                                      │
         │                                                           │
         │  (Key-value format for fast lookups at inference time)    │
         └───────────────────────────────────────────────────────────┘

    ⏱️ Step 1 Total Time: ~2-3 minutes


═══════════════════════════════════════════════════════════════════════════════════
STEP 2: TRAINING (02_training.py) - Submitted as Kubeflow TrainJob
═══════════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ 2a. SUBMIT TRAINJOB FROM LOCAL MACHINE                                      │
    └─────────────────────────────────────────────────────────────────────────────┘

         python 02_training.py
              │
              │  TrainerClient().train(
              │      trainer=CustomTrainer(func=train_sales_model),
              │      runtime=PyTorchRuntime(num_nodes=1),
              │      ...
              │  )
              │
              ▼
         ┌───────────────────────────────────────────────────────────┐
         │  Kubeflow creates TrainJob Pod                            │
         │  ┌───────────────────────────────────────────────────────┐│
         │  │ Pod: sales-training-xxxxx                             ││
         │  │   Mounts: /mnt/shared (PVC)                           ││
         │  │   Env: POSTGRES_HOST, MLFLOW_TRACKING_URI             ││
         │  └───────────────────────────────────────────────────────┘│
         └───────────────────────────────────────────────────────────┘


    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ 2b. INSIDE POD: LOAD TRAINING DATA                                          │
    └─────────────────────────────────────────────────────────────────────────────┘

         # Option A: Direct SQL (simpler, recommended for demo)
         import pandas as pd
         from sqlalchemy import create_engine
         
         engine = create_engine(POSTGRES_URI)
         df = pd.read_sql("SELECT * FROM sales_features", engine)
              │
              │  OR
              │
         # Option B: Feast get_historical_features (production pattern)
         store = FeatureStore(repo_path="/mnt/shared/feature_repo")
         df = store.get_historical_features(
             entity_df=entity_df,
             features=["sales_fv:weekly_sales", "sales_fv:temperature", ...]
         ).to_df()
              │
              ▼
         ┌───────────────────────────────────────────────────────────┐
         │  Training DataFrame ready                                 │
         │  ~600K rows with all features                             │
         └───────────────────────────────────────────────────────────┘


    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ 2c. TRAIN MODEL                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘

         X = df[FEATURE_COLUMNS]
         y = df["weekly_sales"]
         X_train, X_test, y_train, y_test = train_test_split(X, y)
         
         scaler = StandardScaler()
         X_train_scaled = scaler.fit_transform(X_train)
         
         model = SalesForecastModel(input_dim=len(FEATURE_COLUMNS))
         
         for epoch in range(50):
             # Training loop
             loss = criterion(model(X_batch), y_batch)
             loss.backward()
             optimizer.step()
              │
              ▼
         ┌───────────────────────────────────────────────────────────┐
         │  Trained PyTorch model                                    │
         └───────────────────────────────────────────────────────────┘


    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ 2d. SAVE & LOG                                                              │
    └─────────────────────────────────────────────────────────────────────────────┘

         # Save to PVC
         torch.save(model.state_dict(), "/mnt/shared/models/model.pt")
         joblib.dump(scaler, "/mnt/shared/models/scaler.pkl")
         
         # Log to MLflow
         mlflow.set_tracking_uri(MLFLOW_URI)
         with mlflow.start_run(run_name="sales-training"):
             mlflow.log_params({"epochs": 50, "lr": 0.001})
             mlflow.log_metrics({"test_mse": 0.15, "test_r2": 0.87})
             mlflow.log_artifact("/mnt/shared/models/model.pt")
              │
              ▼
         ┌───────────────────────────────────────────────────────────┐
         │  PVC: /mnt/shared/models/                                 │
         │    • model.pt                                             │
         │    • scaler.pkl                                           │
         │                                                           │
         │  MLflow: Run logged with metrics & artifacts              │
         └───────────────────────────────────────────────────────────┘

    ⏱️ Step 2 Total Time: ~3-5 minutes


═══════════════════════════════════════════════════════════════════════════════════
STEP 3: DEPLOY & INFERENCE (03_inference.py)
═══════════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ 3a. DEPLOY KSERVE INFERENCE SERVICE                                         │
    └─────────────────────────────────────────────────────────────────────────────┘

         KServeClient().create(V1beta1InferenceService(
             metadata={"name": "sales-predictor"},
             spec=V1beta1InferenceServiceSpec(
                 predictor=V1beta1PredictorSpec(
                     containers=[V1Container(
                         image="custom-predictor:latest",
                         volumeMounts=["/mnt/shared"]
                     )]
                 )
             )
         ))
              │
              ▼
         ┌───────────────────────────────────────────────────────────┐
         │  KServe InferenceService Pod running                      │
         │    • Loads model.pt from PVC                              │
         │    • Has access to feature_repo for Feast config          │
         │    • Can connect to PostgreSQL for online features        │
         └───────────────────────────────────────────────────────────┘


    ┌─────────────────────────────────────────────────────────────────────────────┐
    │ 3b. INFERENCE REQUEST                                                       │
    └─────────────────────────────────────────────────────────────────────────────┘

         # Client sends request with entity keys only
         POST https://sales-predictor.namespace.svc/v1/models/sales:predict
         {
             "instances": [
                 {"store": 1, "dept": 3},
                 {"store": 5, "dept": 12}
             ]
         }
              │
              ▼
         ┌───────────────────────────────────────────────────────────┐
         │  Model Server: Fetch features from Online Store           │
         │                                                           │
         │  store = FeatureStore(repo_path="/mnt/shared/feature_repo")│
         │  features = store.get_online_features(                    │
         │      entity_rows=[{"store": 1, "dept": 3}, ...],          │
         │      features=["sales_fv:temperature", ...]               │
         │  ).to_dict()                                              │
         └───────────────────────────────────────────────────────────┘
              │
              │  Direct PostgreSQL lookup (NO RAY)
              │  Latency: ~5-15ms
              │
              ▼
         ┌───────────────────────────────────────────────────────────┐
         │  features = {                                             │
         │      "temperature": [42.3, 38.5],                         │
         │      "fuel_price": [3.45, 3.42],                          │
         │      "is_holiday": [0, 1],                                │
         │      ...                                                  │
         │  }                                                        │
         └───────────────────────────────────────────────────────────┘
              │
              ▼
         ┌───────────────────────────────────────────────────────────┐
         │  Model Prediction                                         │
         │                                                           │
         │  X = scaler.transform(features)                           │
         │  predictions = model(torch.tensor(X))                     │
         │                                                           │
         │  Response: {"predictions": [24500.0, 18200.0]}            │
         └───────────────────────────────────────────────────────────┘

    ⏱️ Step 3 Deploy: ~1-2 minutes
    ⏱️ Each inference: ~20-50ms


═══════════════════════════════════════════════════════════════════════════════════
SUMMARY
═══════════════════════════════════════════════════════════════════════════════════

    COMPONENTS NEEDED:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │  ✅ PostgreSQL         (Feast registry + offline + online store)        │
    │  ✅ MLflow             (Experiment tracking)                            │
    │  ✅ PVC                (Shared storage)                                 │
    │  ✅ Kubeflow Trainer   (Training jobs)                                  │
    │  ✅ KServe             (Model serving)                                  │
    │  ❌ Ray                (NOT NEEDED)                                     │
    └──────────────────────────────────────────────────────────────────────────┘

    TOTAL E2E TIME: ~8-12 minutes (first run)
    
    FILES:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │  01_feast_features.py  → Generate data, write to PG, feast apply/materialize│
    │  02_training.py        → Kubeflow TrainJob, read from PG, train, save   │
    │  03_inference.py       → Deploy KServe, get_online_features, predict    │
    │  feature_store.yaml    → PostgreSQL config (no Ray)                     │
    └──────────────────────────────────────────────────────────────────────────┘
```

## feature_store.yaml (No Ray)

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

entity_key_serialization_version: 2
```

## Pros & Cons

| Aspect | Rating | Notes |
|--------|--------|-------|
| Complexity | ⭐⭐⭐⭐⭐ | Simple, fewer moving parts |
| Demo Speed | ⭐⭐⭐⭐⭐ | ~8-12 min total |
| Production Ready | ⭐⭐⭐⭐ | PostgreSQL is proven |
| Scalability | ⭐⭐⭐ | Limited by single PostgreSQL |
| Inference Latency | ⭐⭐⭐⭐ | ~5-15ms |

