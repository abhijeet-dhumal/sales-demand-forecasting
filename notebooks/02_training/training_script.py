"""
Training Function for Sales Forecasting

Self-contained training function for Kubeflow CustomTrainer.
All imports and class definitions are inside the function for cloudpickle serialization.

Usage in notebook:
    from training_script import train_fn
    trainer.train(trainer=CustomTrainer(func=train_fn, ...))
"""


def train_fn(
    epochs=50,
    namespace='feast-trainer-demo',
    output_dir='/shared/models',
    feast_config_path='/opt/app-root/src/feast-config/salesforecasting'
):
    """
    Distributed training function for sales forecasting.
    
    This function runs inside Kubeflow TrainJob pods and handles:
    - Feast feature retrieval (remote client via gRPC)
    - PyTorch DDP distributed training
    - MLflow experiment tracking and model registry
    
    Args:
        epochs: Number of training epochs
        namespace: Kubernetes namespace (used for MLflow workspace)
        output_dir: Directory for model artifacts
        feast_config_path: Path to Feast remote client config (operator-provided)
    """
    # All imports inside function for cloudpickle serialization
    import os
    import json
    import time
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    import joblib
    import mlflow
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, Dataset, DistributedSampler
    from sklearn.preprocessing import StandardScaler
    from datetime import datetime, timezone, timedelta
    
    # =========================================================================
    # Model and Dataset classes (defined inside function for serialization)
    # =========================================================================
    class MLP(nn.Module):
        """Multi-layer perceptron for sales forecasting."""
        def __init__(self, inp, hidden=[512, 256, 128, 64], drop=0.3):
            super().__init__()
            layers = []
            for h in hidden:
                layers.extend([nn.Linear(inp, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop)])
                inp = h
            layers.append(nn.Linear(inp, 1))
            self.net = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.net(x).squeeze(-1)
    
    class DS(Dataset):
        """PyTorch Dataset for sales data."""
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, i):
            return self.X[i], self.y[i]
    
    # =========================================================================
    # Initialize distributed training
    # =========================================================================
    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu")
    print(f"[Rank {rank}] Device: {device}, World size: {world}")
    
    OUT = output_dir
    EPOCHS = epochs
    
    # =========================================================================
    # RANK 0: Data preparation, Feast retrieval, MLflow setup
    # =========================================================================
    if rank == 0:
        os.makedirs(OUT, exist_ok=True)
        
        # Configure RHOAI MLflow with workspace and token auth
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
        mlflow_workspace = os.getenv('MLFLOW_WORKSPACE', namespace)
        mlflow_token = os.getenv('MLFLOW_TRACKING_TOKEN')
        
        if mlflow_token:
            os.environ['MLFLOW_TRACKING_TOKEN'] = mlflow_token
        if os.getenv('MLFLOW_TRACKING_INSECURE_TLS'):
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(f'{mlflow_workspace}/sales-forecasting')
        mlflow.start_run(run_name=f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        
        # Use Feast remote client (operator-provided config)
        from feast import FeatureStore
        print(f"Loading Feast config from: {feast_config_path}")
        store = FeatureStore(repo_path=feast_config_path)
        
        # Build entity DataFrame (45 stores x 14 depts x 104 weeks = 65,520 rows)
        entity_rows = [
            {'store_id': s, 'dept_id': d, 'event_timestamp': datetime(2022, 1, 1, tzinfo=timezone.utc) + timedelta(weeks=w)}
            for w in range(104) for s in range(1, 46) for d in range(1, 15)
        ]
        entity_df = pd.DataFrame(entity_rows)
        print(f"Entity DF: {len(entity_df):,} rows")
        
        # Retrieve historical features via Feast gRPC
        t0 = time.time()
        df = store.get_historical_features(
            entity_df=entity_df,
            features=store.get_feature_service('training_features')
        ).to_df()
        print(f"✅ Feast: {len(df):,} rows in {time.time()-t0:.1f}s")
        
        # Prepare data
        df = df.dropna(subset=['weekly_sales']).sort_values('event_timestamp')
        split = int(len(df) * 0.8)
        train_df, val_df = df.iloc[:split], df.iloc[split:]
        
        # Select numeric feature columns
        exclude = ['store_id', 'dept_id', 'event_timestamp', 'date', 'weekly_sales']
        feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
        print(f"Features ({len(feat_cols)}): {feat_cols[:5]}...")
        
        # Extract arrays
        X_train, y_train = train_df[feat_cols].fillna(0).values, train_df['weekly_sales'].values
        X_val, y_val = val_df[feat_cols].fillna(0).values, val_df['weekly_sales'].values
        
        # Scale features and target
        scaler, y_scaler = StandardScaler(), StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        y_train_s = y_scaler.fit_transform(np.log1p(y_train).reshape(-1, 1)).flatten()
        y_val_s = y_scaler.transform(np.log1p(y_val).reshape(-1, 1)).flatten()
        
        # Save scalers and share data with other ranks
        joblib.dump({'scaler_X': scaler, 'scaler_y': y_scaler, 'use_log_transform': True}, f'{OUT}/scalers.joblib')
        joblib.dump(feat_cols, f'{OUT}/feature_cols.pkl')
        np.savez(f'{OUT}/.data.npz', X_train=X_train, y_train=y_train_s, X_val=X_val, y_val=y_val_s, y_val_orig=y_val)
        np.save(f'{OUT}/.dim.npy', [X_train.shape[1]])
        
        mlflow.log_params({
            'epochs': EPOCHS,
            'train_rows': len(train_df),
            'val_rows': len(val_df),
            'features': len(feat_cols),
            'world_size': world
        })
    
    # =========================================================================
    # ALL RANKS: Load shared data and create DataLoaders
    # =========================================================================
    dist.barrier()
    data = np.load(f'{OUT}/.data.npz')
    X_train, y_train, X_val, y_val, y_val_orig = data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['y_val_orig']
    inp_dim = int(np.load(f'{OUT}/.dim.npy')[0])
    dist.barrier()
    
    train_ds = DS(X_train, y_train)
    val_ds = DS(X_val, y_val)
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank)
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=64)
    
    # =========================================================================
    # Initialize model with DDP
    # =========================================================================
    model = DDP(
        MLP(inp_dim).to(device),
        device_ids=[int(os.environ.get('LOCAL_RANK', 0))] if torch.cuda.is_available() else None
    )
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    crit = nn.MSELoss()
    
    # =========================================================================
    # Training loop
    # =========================================================================
    best_loss, best_mape = float('inf'), float('inf')
    for ep in range(EPOCHS):
        sampler.set_epoch(ep)
        model.train()
        train_loss = 0.0
        
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss = crit(model(X_b), y_b)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            preds = np.concatenate([model(X.to(device)).cpu().numpy() for X, _ in val_loader])
        val_loss = np.mean((preds - y_val) ** 2)
        sched.step(val_loss)
        
        # Log metrics (rank 0 only)
        if rank == 0:
            y_sc = joblib.load(f'{OUT}/scalers.joblib')['scaler_y']
            pred_orig = np.expm1(y_sc.inverse_transform(preds.reshape(-1, 1)).flatten())
            mask = y_val_orig > 1000
            mape = np.mean(np.abs((y_val_orig[mask] - pred_orig[mask]) / y_val_orig[mask])) * 100
            
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'mape': mape,
                'lr': opt.param_groups[0]['lr']
            }, step=ep)
            
            if (ep + 1) % 10 == 0:
                print(f"Ep {ep+1}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | MAPE: {mape:.1f}%")
            
            if val_loss < best_loss:
                best_loss, best_mape = val_loss, mape
                torch.save(model.module.state_dict(), f'{OUT}/best_model.pt')
        
        dist.barrier()
    
    # =========================================================================
    # RANK 0: Save final model and register in MLflow Model Registry
    # =========================================================================
    if rank == 0:
        print(f"\n✅ DONE: Best MAPE {best_mape:.1f}%")
        mlflow.log_metrics({'best_mape': best_mape, 'best_val_loss': best_loss})
        
        # Load best model for logging
        m = MLP(inp_dim)
        m.load_state_dict(torch.load(f'{OUT}/best_model.pt', weights_only=True))
        m.eval()
        
        # Log model to MLflow and register in Model Registry
        model_info = mlflow.pytorch.log_model(m, 'model')
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        
        # Register model in MLflow Model Registry
        model_name = "sales-forecasting-model"
        try:
            registered_model = mlflow.register_model(model_uri, model_name)
            print(f"✅ Registered model: {model_name} v{registered_model.version}")
            
            # Tag the model version with metrics
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            client.set_model_version_tag(model_name, registered_model.version, "mape", f"{best_mape:.2f}")
            client.set_model_version_tag(model_name, registered_model.version, "workspace", namespace)
        except Exception as e:
            print(f"⚠️ Model registration skipped: {e}")
        
        # Save metadata locally for KServe deployment
        feat_cols = joblib.load(f'{OUT}/feature_cols.pkl')
        metadata = {
            'model_type': 'SalesMLP',
            'input_dim': inp_dim,
            'hidden_dims': [512, 256, 128, 64],
            'dropout': 0.3,
            'best_mape': float(best_mape),
            'feature_columns': feat_cols,
            'mlflow_run_id': run_id,
            'mlflow_model_uri': model_uri
        }
        json.dump(metadata, open(f'{OUT}/model_metadata.json', 'w'))
        print(f"📁 Model saved to: {OUT}")
        print(f"📊 MLflow model URI: {model_uri}")
        
        mlflow.end_run()
        
        # Cleanup temp files
        for f in ['.data.npz', '.dim.npy']:
            try:
                os.remove(f'{OUT}/{f}')
            except:
                pass
    
    dist.destroy_process_group()
