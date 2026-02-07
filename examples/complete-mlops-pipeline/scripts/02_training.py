#!/usr/bin/env python3
"""
Sales Forecasting Training with Feast + Ray Feature Retrieval

Usage: python 02_training.py

This script trains a sales forecasting model using:
1. Feast get_historical_features() with Ray compute engine for distributed PIT joins
2. Kubeflow Trainer SDK for distributed PyTorch training
3. MLflow for experiment tracking and model registry

Architecture:
- Feature Retrieval: Feast + Ray (distributed PIT joins)
- Training: Kubeflow TrainJob (DDP distributed training)
- Tracking: MLflow (metrics, params, artifacts)
"""
import os
import sys
import urllib3
from datetime import datetime, timedelta, timezone

urllib3.disable_warnings()

# =============================================================================
# CONFIGURATION
# =============================================================================

NAMESPACE = os.getenv("NAMESPACE", "feast-trainer-demo")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"http://mlflow.{NAMESPACE}.svc.cluster.local:5000")
SHARED_PVC = os.getenv("SHARED_PVC", "feast-pvc")
RUNTIME = os.getenv("RUNTIME_NAME", "torch-distributed")
K8S_TOKEN = os.getenv("K8S_TOKEN")
K8S_API_SERVER = os.getenv("K8S_API_SERVER")

# Training hyperparameters
EPOCHS = os.getenv("EPOCHS", "50")
BATCH_SIZE = os.getenv("BATCH_SIZE", "256")
LR = os.getenv("LEARNING_RATE", "0.001")


def train_sales_model():
    """
    Training function - executed on cluster workers.
    
    Flow:
    1. Load entity DataFrame from parquet (store_id, dept_id, event_timestamp)
    2. Call Feast get_historical_features() → uses Ray for distributed PIT joins
    3. Preprocess features (scale, split)
    4. Train PyTorch model with DDP
    5. Log to MLflow
    """
    import os
    import json
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from pathlib import Path
    from datetime import datetime, timedelta, timezone

    # Config from env
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    feature_repo = os.getenv("FEATURE_REPO", "/mnt/shared/feature_repo")
    data_dir = os.getenv("DATA_DIR", "/mnt/shared/data")
    model_dir = os.getenv("MODEL_DIR", "/mnt/shared/models")
    epochs = int(os.getenv("EPOCHS", 50))
    batch_size = int(os.getenv("BATCH_SIZE", 256))
    lr = float(os.getenv("LEARNING_RATE", 0.001))

    # =========================================================================
    # DISTRIBUTED SETUP
    # =========================================================================
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    if rank == 0:
        print("=" * 70)
        print("SALES FORECASTING TRAINING WITH FEAST + RAY")
        print("=" * 70)
        print(f"   Distributed: world_size={world_size}, device={device}")
        print(f"   Feature Repo: {feature_repo}")
        print(f"   Data Dir: {data_dir}")
        print(f"   Hyperparameters: epochs={epochs}, batch_size={batch_size}, lr={lr}")

    # =========================================================================
    # MODEL DEFINITION
    # =========================================================================
    class SalesMLP(nn.Module):
        def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.2):
            super().__init__()
            layers = []
            prev_dim = input_dim
            for dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = dim
            layers.append(nn.Linear(prev_dim, 1))
            self.net = nn.Sequential(*layers)
            self.hidden_dims = hidden_dims
            self.dropout = dropout
            
        def forward(self, x):
            return self.net(x).squeeze(-1)

    # =========================================================================
    # FEAST: LOAD TRAINING DATA VIA get_historical_features()
    # This uses Ray compute engine for distributed PIT joins!
    # =========================================================================
    if rank == 0:
        print("\n" + "=" * 70)
        print("STEP 1: LOAD TRAINING DATA VIA FEAST + RAY")
        print("=" * 70)
        print("   Using Feast get_historical_features() with Ray compute engine")
        print("   Ray performs distributed point-in-time joins across the cluster")
        
        from feast import FeatureStore
        import time
        
        # Initialize Feast store (connects to Ray cluster via feature_store.yaml)
        store = FeatureStore(repo_path=feature_repo)
        
        # Load entity DataFrame from parquet
        # This contains (store_id, dept_id, event_timestamp) for training samples
        entities_path = f"{data_dir}/entities.parquet"
        
        if os.path.exists(entities_path):
            entity_df = pd.read_parquet(entities_path)
            # Rename 'date' to 'event_timestamp' if needed
            if 'date' in entity_df.columns and 'event_timestamp' not in entity_df.columns:
                entity_df = entity_df.rename(columns={'date': 'event_timestamp'})
        else:
            # Fallback: Create entity DataFrame from sales data
            print(f"   ⚠️  entities.parquet not found, creating from sales data...")
            sales_df = pd.read_parquet(f"{data_dir}/sales_features.parquet")
            entity_df = sales_df[['store_id', 'dept_id', 'event_timestamp']].copy()
        
        # Ensure timezone-aware timestamps
        if entity_df['event_timestamp'].dt.tz is None:
            entity_df['event_timestamp'] = pd.to_datetime(entity_df['event_timestamp']).dt.tz_localize('UTC')
        
        print(f"   Entity DataFrame: {len(entity_df):,} rows")
        print(f"   Columns: {list(entity_df.columns)}")
        print(f"   Date range: {entity_df['event_timestamp'].min()} to {entity_df['event_timestamp'].max()}")
        
        # Sample for large datasets (optional - for faster training iterations)
        MAX_SAMPLES = int(os.getenv("MAX_TRAINING_SAMPLES", 100000))
        if len(entity_df) > MAX_SAMPLES:
            print(f"   Sampling {MAX_SAMPLES:,} from {len(entity_df):,} rows...")
            entity_df = entity_df.sample(n=MAX_SAMPLES, random_state=42)
        
        # =====================================================================
        # FEAST GET_HISTORICAL_FEATURES - USES RAY FOR DISTRIBUTED PIT JOINS!
        # =====================================================================
        print("\n   🚀 Calling Feast get_historical_features()...")
        print("   Ray compute engine will perform distributed PIT joins.")
        print("   Check Ray Dashboard for job visibility.")
        
        start_time = time.time()
        
        # Define features to retrieve
        features = [
            # Sales lag features (critical for time-series)
            "sales_features:weekly_sales",
            "sales_features:lag_1",
            "sales_features:lag_2",
            "sales_features:lag_4",
            "sales_features:lag_8",
            "sales_features:lag_52",
            "sales_features:rolling_mean_4w",
            # External factors
            "sales_features:temperature",
            "sales_features:fuel_price",
            "sales_features:cpi",
            "sales_features:unemployment",
            # Store attributes
            "store_features:store_size",
        ]
        
        # Get historical features (distributed via Ray!)
        training_df = store.get_historical_features(
            entity_df=entity_df,
            features=features
        ).to_df()
        
        elapsed = time.time() - start_time
        
        print(f"   ✅ Retrieved {len(training_df):,} samples in {elapsed:.2f}s")
        print(f"   Columns: {list(training_df.columns)}")
        print(f"   Non-null counts:")
        for col in training_df.columns:
            non_null = training_df[col].notna().sum()
            print(f"      {col}: {non_null:,}")
        
        # Prepare training data
        # Target: weekly_sales
        # Features: All lag features + external factors
        target_col = "weekly_sales"
        exclude_cols = ["store_id", "dept_id", "event_timestamp", target_col]
        feature_cols = [c for c in training_df.columns 
                       if c not in exclude_cols 
                       and training_df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
        
        print(f"\n   Feature columns ({len(feature_cols)}): {feature_cols}")
        
        # Drop rows with NaN in features or target
        training_df = training_df.dropna(subset=feature_cols + [target_col])
        print(f"   After dropping NaN: {len(training_df):,} samples")
        
        X = training_df[feature_cols].values
        y = training_df[target_col].values
        
        # Scale features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        print(f"   Train: {len(X_train):,}, Test: {len(X_test):,}")
        
        # Save for other ranks
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        np.save(f"{model_dir}/.X_train.npy", X_train)
        np.save(f"{model_dir}/.y_train.npy", y_train)
        np.save(f"{model_dir}/.X_test.npy", X_test)
        np.save(f"{model_dir}/.y_test.npy", y_test)
        np.save(f"{model_dir}/.y_test_orig.npy", y_test)
        np.save(f"{model_dir}/.input_dim.npy", np.array([len(feature_cols)]))
        
        # Save scalers and feature columns
        import joblib
        joblib.dump(scaler_X, f"{model_dir}/scaler_X.pkl")
        joblib.dump(scaler_y, f"{model_dir}/scaler_y.pkl")
        joblib.dump(feature_cols, f"{model_dir}/feature_cols.pkl")
        # Also save combined format for serve.py compatibility
        joblib.dump({"scaler_X": scaler_X, "scaler_y": scaler_y}, f"{model_dir}/scalers.joblib")
        
        print("   ✅ Data preprocessing complete")

    dist.barrier()

    # Load data on all ranks
    X_train = np.load(f"{model_dir}/.X_train.npy")
    y_train = np.load(f"{model_dir}/.y_train.npy")
    X_test = np.load(f"{model_dir}/.X_test.npy")
    y_test = np.load(f"{model_dir}/.y_test.npy")
    input_dim = int(np.load(f"{model_dir}/.input_dim.npy")[0])

    dist.barrier()

    # =========================================================================
    # TRAINING
    # =========================================================================
    if rank == 0:
        print("\n" + "=" * 70)
        print("STEP 2: DISTRIBUTED TRAINING")
        print("=" * 70)

    # Setup model
    HIDDEN_DIMS = [256, 128, 64]
    DROPOUT = 0.2
    
    model = SalesMLP(input_dim, HIDDEN_DIMS, DROPOUT).to(device)
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank] if torch.cuda.is_available() else None
        )

    # Setup data loaders
    train_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=2
    )
    
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)

    # Setup optimizer and scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    best_state = None

    # MLflow setup (rank 0 only)
    if rank == 0:
        try:
            import mlflow
            os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
            run_name = os.getenv("RUN_NAME", f"sales-forecast-{datetime.now().strftime('%m%d-%H%M')}")
            mlflow.set_experiment("sales-forecasting")
            mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "world_size": world_size,
                "input_dim": input_dim,
                "hidden_dims": str(HIDDEN_DIMS),
                "dropout": DROPOUT,
                "feature_source": "feast+ray",
            })
            print(f"   MLflow run: {run_name}")
        except Exception as e:
            print(f"   ⚠️  MLflow setup failed: {e}")

    # Training loop
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t)
            test_loss = criterion(test_pred, y_test_t).item()

        if test_loss < best_loss:
            best_loss = test_loss
            if rank == 0:
                best_state = (model.module if hasattr(model, 'module') else model).state_dict().copy()

        if rank == 0:
            # Log to MLflow
            try:
                import mlflow
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "learning_rate": optimizer.param_groups[0]['lr']
                }, step=epoch)
            except:
                pass
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Test: {test_loss:.6f}")

    # =========================================================================
    # SAVE MODEL AND ARTIFACTS
    # =========================================================================
    dist.barrier()
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("STEP 3: SAVE MODEL AND ARTIFACTS")
        print("=" * 70)
        
        # Save model
        torch.save(best_state, f"{model_dir}/best_model.pt")
        print(f"   ✅ Model saved to {model_dir}/best_model.pt")
        
        # Save metadata
        import joblib
        feature_cols = joblib.load(f"{model_dir}/feature_cols.pkl")
        
        metadata = {
            "model_type": "SalesMLP",
            "input_dim": input_dim,
            "hidden_dims": HIDDEN_DIMS,
            "dropout": DROPOUT,
            "feature_columns": feature_cols,
            "best_test_loss": float(best_loss),
            "training_epochs": epochs,
            "feature_source": "feast+ray",
            "feast_features": [
                "sales_features:lag_1", "sales_features:lag_2", "sales_features:lag_4",
                "sales_features:lag_8", "sales_features:lag_52", "sales_features:rolling_mean_4w",
                "sales_features:temperature", "sales_features:fuel_price", 
                "sales_features:cpi", "sales_features:unemployment",
                "store_features:store_size"
            ]
        }
        
        with open(f"{model_dir}/model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✅ Metadata saved to {model_dir}/model_metadata.json")
        
        # Log to MLflow
        try:
            import mlflow
            mlflow.log_metric("best_test_loss", best_loss)
            mlflow.log_artifact(f"{model_dir}/best_model.pt")
            mlflow.log_artifact(f"{model_dir}/scaler_X.pkl")
            mlflow.log_artifact(f"{model_dir}/scaler_y.pkl")
            mlflow.log_artifact(f"{model_dir}/feature_cols.pkl")
            mlflow.log_artifact(f"{model_dir}/model_metadata.json")
            mlflow.end_run()
            print("   ✅ Artifacts logged to MLflow")
        except Exception as e:
            print(f"   ⚠️  MLflow logging failed: {e}")
        
        # Cleanup temp files
        for f in [".X_train.npy", ".y_train.npy", ".X_test.npy", ".y_test.npy", ".y_test_orig.npy", ".input_dim.npy"]:
            try:
                os.remove(f"{model_dir}/{f}")
            except:
                pass
        
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETE!")
        print("=" * 70)
        print(f"   Best test loss: {best_loss:.6f}")
        print(f"   Model: {model_dir}/best_model.pt")
        print(f"   Features: Loaded via Feast + Ray distributed PIT joins")

    dist.barrier()
    dist.destroy_process_group()


# =============================================================================
# SUBMIT TO CLUSTER
# =============================================================================

if __name__ == "__main__":
    from kubernetes import client as k8s
    from kubeflow.trainer import TrainerClient, CustomTrainer
    from kubeflow.common.types import KubernetesBackendConfig
    from kubeflow.trainer.options import (
        PodTemplateOverrides, PodTemplateOverride, 
        PodSpecOverride, ContainerOverride, Labels, Annotations
    )

    print(f"{'='*70}")
    print("SALES FORECASTING TRAINING WITH FEAST + RAY")
    print(f"{'='*70}")
    print(f"   Namespace: {NAMESPACE}")
    print(f"   MLflow: {MLFLOW_URI}")
    print(f"   Runtime: {RUNTIME}")
    print(f"   Hyperparameters: epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}")

    # Auth with namespace
    cfg = k8s.Configuration()
    if K8S_TOKEN and K8S_API_SERVER:
        cfg.host, cfg.verify_ssl = K8S_API_SERVER, False
        cfg.api_key = {"authorization": f"Bearer {K8S_TOKEN}"}
    
    client = TrainerClient(KubernetesBackendConfig(
        namespace=NAMESPACE,
        client_configuration=cfg if K8S_TOKEN else None
    ))

    runtime = client.get_runtime(RUNTIME)
    print(f"\n🚀 Submitting to runtime: {runtime.name}")

    job_id = datetime.now().strftime("%m%d-%H%M")
    job = client.train(
        trainer=CustomTrainer(
            func=train_sales_model,
            num_nodes=2,
            resources_per_node={"cpu": 4, "memory": "16Gi", "nvidia.com/gpu": 1},
            packages_to_install=[
                "scikit-learn", "pandas", "pyarrow", "joblib", "mlflow",
                "feast[postgres,ray]==0.59.0", "psycopg2-binary"
            ],
            env={
                "MLFLOW_TRACKING_URI": MLFLOW_URI,
                "DATA_DIR": "/mnt/shared/data",
                "MODEL_DIR": "/mnt/shared/models",
                "FEATURE_REPO": "/mnt/shared/feature_repo",
                "EPOCHS": EPOCHS,
                "BATCH_SIZE": BATCH_SIZE,
                "LEARNING_RATE": LR,
                "RUN_NAME": f"sales-forecast-{job_id}",
                "MAX_TRAINING_SAMPLES": "100000",
            },
        ),
        runtime=runtime,
        options=[
            Labels({"app": "sales-forecasting", "job-type": "training", "run-id": job_id}),
            Annotations({"description": f"Sales forecasting with Feast+Ray - {job_id}"}),
            PodTemplateOverrides(PodTemplateOverride(
                target_jobs=["node"],
                spec=PodSpecOverride(
                    volumes=[{"name": "shared", "persistentVolumeClaim": {"claimName": SHARED_PVC}}],
                    containers=[ContainerOverride(
                        name="node", 
                        volume_mounts=[{"name": "shared", "mountPath": "/mnt/shared"}]
                    )]
                )
            )),
        ],
    )
    
    print(f"✅ Job: {job} (run-id: {job_id})")
    
    client.wait_for_job_status(name=job, status={"Running"}, timeout=300)
    print("🏃 Running...")
    
    client.wait_for_job_status(name=job, status={"Complete", "Failed"}, timeout=3600)
    
    status = client.get_job(name=job).status
    print(f"📊 Status: {status}")
    
    if status == "Failed":
        sys.exit(1)
    
    print("\n📜 Logs (last 30 lines):")
    for line in list(client.get_job_logs(name=job))[-30:]:
        print(line)
