#!/usr/bin/env python3
"""
Sales Forecasting Training with Kubeflow Trainer SDK
Usage: python 02_training.py
"""
import os, sys, urllib3
from datetime import datetime
urllib3.disable_warnings()

# Config
NAMESPACE = os.getenv("NAMESPACE", "feast-trainer-demo")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"http://mlflow.{NAMESPACE}.svc.cluster.local:5000")
SHARED_PVC = os.getenv("SHARED_PVC", "feast-pvc")
RUNTIME = os.getenv("RUNTIME_NAME", "torch-distributed")
K8S_TOKEN = os.getenv("K8S_TOKEN")
K8S_API_SERVER = os.getenv("K8S_API_SERVER")
EPOCHS = os.getenv("EPOCHS", "50")
BATCH_SIZE = os.getenv("BATCH_SIZE", "256")
LR = os.getenv("LEARNING_RATE", "0.001")


def train_sales_model():
    """Training function - serialized and executed on cluster workers."""
    import os, json, torch, torch.nn as nn, torch.distributed as dist
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from pathlib import Path
    from datetime import datetime

    # Config from env
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    feature_repo = os.getenv("FEATURE_REPO", "/mnt/shared/feature_repo")
    model_dir = os.getenv("MODEL_DIR", "/mnt/shared/models")
    epochs = int(os.getenv("EPOCHS", 50))
    batch_size = int(os.getenv("BATCH_SIZE", 256))
    lr = float(os.getenv("LEARNING_RATE", 0.001))

    # Distributed init
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.set_device(local_rank)
    if rank == 0: print(f"Distributed: world_size={world_size}, device={device}")

    # Model
    class SalesMLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 1))
        def forward(self, x): return self.net(x).squeeze(-1)

    # === FEAST: Load training data via get_historical_features() ===
    if rank == 0: print(f"Loading features from Feast (repo: {feature_repo})...")
    from feast import FeatureStore
    store = FeatureStore(repo_path=feature_repo)
    
    # Create entity DataFrame (all store-dept combinations for training window)
    from sqlalchemy import create_engine, text
    pg_url = "postgresql://feast:feast123@feast-postgres:5432/feast"
    engine = create_engine(pg_url)
    with engine.connect() as conn:
        entity_df = pd.read_sql("SELECT store, dept, date as event_timestamp FROM sales_features", conn)
    entity_df = entity_df.dropna()
    if rank == 0: print(f"Entity DataFrame: {len(entity_df):,} rows")
    
    # Fetch training data via Feast (distributed via Ray!)
    df = store.get_historical_features(
        entity_df=entity_df,
        features=store.get_feature_service("demand_forecasting_service")
    ).to_df()
    if rank == 0: print(f"✅ Fetched {len(df):,} samples with {len(df.columns)} features from Feast")
    
    # Feature columns (exclude entities and target)
    exclude = ["store", "dept", "event_timestamp", "weekly_sales"]
    cols = [c for c in df.columns if c not in exclude and df[c].dtype in ["float64", "int64", "float32", "int32"]]
    df = df.dropna(subset=cols + ["weekly_sales"])
    X, y = df[cols].values, df["weekly_sales"].values
    if rank == 0: print(f"Training data: {len(df):,} samples, {len(cols)} features: {cols[:5]}...")

    # Scale & split
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_scaled, y_scaled = scaler_X.fit_transform(X), scaler_y.fit_transform(y.reshape(-1,1)).flatten()
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Setup
    model = SalesMLP(len(cols)).to(device)
    if world_size > 1: model = nn.parallel.DistributedDataParallel(model)
    
    train_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    X_test_t, y_test_t = torch.FloatTensor(X_test).to(device), torch.FloatTensor(y_test).to(device)
    
    criterion, optimizer = nn.MSELoss(), torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_loss, best_state = float('inf'), None

    # MLflow
    if rank == 0:
        try:
            import mlflow
            os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
            run_name = os.getenv("RUN_NAME", f"sales-forecast-{datetime.now().strftime('%m%d-%H%M')}")
            mlflow.set_experiment("sales-forecasting")
            mlflow.start_run(run_name=run_name)
            mlflow.log_params({"epochs": epochs, "lr": lr, "world_size": world_size, "run_name": run_name})
        except: pass

    # Train
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); optimizer.step()
        scheduler.step()
        
        model.eval()
        with torch.no_grad(): test_loss = criterion(model(X_test_t), y_test_t).item()
        if test_loss < best_loss:
            best_loss = test_loss
            if rank == 0: best_state = (model.module if hasattr(model,'module') else model).state_dict().copy()
        if rank == 0 and (epoch+1) % 10 == 0: print(f"Epoch {epoch+1}/{epochs} | Loss: {test_loss:.6f}")

    # Save
    dist.barrier()
    if rank == 0:
        import joblib
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        torch.save(best_state, f"{model_dir}/best_model.pt")
        joblib.dump({"scaler_X": scaler_X, "scaler_y": scaler_y, "feature_cols": cols}, f"{model_dir}/scalers.joblib")
        with open(f"{model_dir}/model_metadata.json", "w") as f:
            json.dump({"input_dim": len(cols), "hidden_dims": [256,128,64], "feature_columns": cols}, f)
        try: import mlflow; mlflow.log_metric("best_loss", best_loss); mlflow.end_run()
        except: pass
        print(f"✅ Training complete! Best loss: {best_loss:.6f}")
    
    dist.barrier(); dist.destroy_process_group()


# === Submit to cluster ===
if __name__ == "__main__":
    from kubernetes import client as k8s
    from kubeflow.trainer import TrainerClient, CustomTrainer
    from kubeflow.common.types import KubernetesBackendConfig
    from kubeflow.trainer.options import PodTemplateOverrides, PodTemplateOverride, PodSpecOverride, ContainerOverride, Labels, Annotations

    print(f"{'='*60}\nSales Forecasting Training\nNamespace: {NAMESPACE}\n{'='*60}")

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
    print(f"🚀 Submitting to runtime: {runtime.name}")

    job_id = datetime.now().strftime("%m%d-%H%M")
    job = client.train(
        trainer=CustomTrainer(
            func=train_sales_model,
            num_nodes=2,
            resources_per_node={"cpu": 4, "memory": "16Gi", "nvidia.com/gpu": 1},
            packages_to_install=["scikit-learn", "pandas", "pyarrow", "joblib", "mlflow"],
            env={"MLFLOW_TRACKING_URI": MLFLOW_URI, "DATA_DIR": "/mnt/shared/data", 
                 "MODEL_DIR": "/mnt/shared/models", "EPOCHS": EPOCHS, "BATCH_SIZE": BATCH_SIZE, "LEARNING_RATE": LR,
                 "RUN_NAME": f"sales-forecast-{job_id}"},
        ),
        runtime=runtime,
        options=[
            Labels({"app": "sales-forecasting", "job-type": "training", "run-id": job_id}),
            Annotations({"description": f"Sales forecasting model training - {job_id}"}),
            PodTemplateOverrides(PodTemplateOverride(target_jobs=["node"],
                spec=PodSpecOverride(
                    volumes=[{"name": "shared", "persistentVolumeClaim": {"claimName": SHARED_PVC}}],
                    containers=[ContainerOverride(name="node", volume_mounts=[{"name": "shared", "mountPath": "/mnt/shared"}])]))),
        ],
    )
    print(f"✅ Job: {job} (run-id: {job_id})")
    
    client.wait_for_job_status(name=job, status={"Running"}, timeout=300)
    print("🏃 Running...")
    client.wait_for_job_status(name=job, status={"Complete", "Failed"}, timeout=3600)
    
    status = client.get_job(name=job).status
    print(f"📊 Status: {status}")
    if status == "Failed": sys.exit(1)
    print("\n📜 Logs:")
    for line in list(client.get_job_logs(name=job))[-20:]: print(line)
