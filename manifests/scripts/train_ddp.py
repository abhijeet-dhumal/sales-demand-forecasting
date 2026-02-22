#!/usr/bin/env python3
"""
Flexible DDP Training Script for Sales Forecasting

Supports:
  - Single node / Multi-node
  - Single GPU / Multi-GPU
  - CPU only
  - NVIDIA CUDA
  - AMD ROCm

Usage:
  # Single GPU
  python train_ddp.py

  # Multi-GPU single node (4 GPUs)
  torchrun --nproc_per_node=4 train_ddp.py

  # Multi-node (2 nodes, 2 GPUs each)
  torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 --master_addr=<IP> train_ddp.py

  # CPU only
  CUDA_VISIBLE_DEVICES="" python train_ddp.py

Environment variables:
  - RANK, WORLD_SIZE, LOCAL_RANK: Set by torchrun for distributed
  - OUTPUT_DIR, FEATURE_REPO: Data paths
  - NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE: Hyperparameters
  - EARLY_STOP_PATIENCE: Early stopping patience (default: 5)
  - USE_AMP: Enable mixed precision (auto-detected if not set)
  - SKIP_FEAST: Skip Feast and use cached data if available
  - MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME: MLflow config
"""
import os
import sys
import json
import logging
import time
import shutil
import warnings
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from sklearn.preprocessing import StandardScaler
import joblib

warnings.filterwarnings('ignore')

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("train")


# =============================================================================
# HARDWARE DETECTION
# =============================================================================
def detect_hardware():
    """Detect available hardware: CUDA, ROCm, or CPU."""
    if torch.cuda.is_available():
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return "rocm"
        return "cuda"
    return "cpu"


def get_device_info():
    """Get detailed device information."""
    hw = detect_hardware()
    info = {"type": hw, "count": 0, "names": []}
    
    if hw in ("cuda", "rocm"):
        info["count"] = torch.cuda.device_count()
        info["names"] = [torch.cuda.get_device_name(i) for i in range(info["count"])]
    else:
        info["count"] = 1
        info["names"] = ["CPU"]
    
    return info


# =============================================================================
# DISTRIBUTED SETUP
# =============================================================================
def setup_distributed():
    """Initialize distributed training if applicable."""
    # Check if launched with torchrun/distributed
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    hw = detect_hardware()
    distributed = world_size > 1
    
    if distributed:
        # Select backend based on hardware
        if hw == "cuda":
            backend = "nccl"
        elif hw == "rocm":
            backend = "nccl"  # ROCm uses RCCL which is NCCL-compatible
        else:
            backend = "gloo"
        
        dist.init_process_group(backend=backend)
        
        if hw in ("cuda", "rocm"):
            torch.cuda.set_device(local_rank)
    
    # Determine device
    if hw in ("cuda", "rocm"):
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    
    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "distributed": distributed,
        "device": device,
        "hardware": hw,
        "backend": backend if distributed else None
    }


def cleanup_distributed(ctx):
    """Cleanup distributed training."""
    if ctx["distributed"]:
        dist.destroy_process_group()


# =============================================================================
# MIXED PRECISION
# =============================================================================
def setup_amp(ctx):
    """Setup automatic mixed precision if supported."""
    use_amp = os.environ.get("USE_AMP", "auto").lower()
    
    if use_amp == "false" or use_amp == "0":
        return {"enabled": False, "scaler": None, "dtype": torch.float32}
    
    # AMP supported on CUDA and ROCm
    if ctx["hardware"] in ("cuda", "rocm"):
        return {
            "enabled": True,
            "scaler": torch.amp.GradScaler(device=ctx["hardware"]),
            "dtype": torch.float16
        }
    
    return {"enabled": False, "scaler": None, "dtype": torch.float32}


# =============================================================================
# LOGGING HELPERS
# =============================================================================
def log_section(title, ctx):
    """Log a section header (rank 0 only)."""
    if ctx["rank"] == 0:
        logger.info("=" * 60)
        logger.info(f"  {title}")
        logger.info("=" * 60)


def log_info(msg, ctx):
    """Log info (rank 0 only)."""
    if ctx["rank"] == 0:
        logger.info(msg)


# =============================================================================
# MODEL DEFINITION
# =============================================================================
class MLP(nn.Module):
    """Multi-layer perceptron for regression."""
    def __init__(self, input_dim, hidden_dims=None, dropout=0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
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

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SalesDataset(Dataset):
    """Dataset for sales forecasting."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_epoch(model, loader, criterion, optimizer, ctx, amp_ctx, sampler=None, epoch=0):
    """Train for one epoch."""
    if sampler:
        sampler.set_epoch(epoch)
    
    model.train()
    total_loss = 0
    
    for xb, yb in loader:
        xb, yb = xb.to(ctx["device"]), yb.to(ctx["device"])
        optimizer.zero_grad()
        
        if amp_ctx["enabled"]:
            with torch.amp.autocast(device_type=ctx["hardware"]):
                loss = criterion(model(xb), yb)
            amp_ctx["scaler"].scale(loss).backward()
            amp_ctx["scaler"].unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_ctx["scaler"].step(optimizer)
            amp_ctx["scaler"].update()
        else:
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, ctx, amp_ctx, y_scaler, y_val):
    """Validate model and compute metrics."""
    model.eval()
    total_loss = 0
    preds = []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(ctx["device"]), yb.to(ctx["device"])
            
            if amp_ctx["enabled"]:
                with torch.amp.autocast(device_type=ctx["hardware"]):
                    p = model(xb)
                    total_loss += criterion(p, yb).item()
            else:
                p = model(xb)
                total_loss += criterion(p, yb).item()
            
            preds.extend(p.cpu().numpy())
    
    val_loss = total_loss / len(loader)
    
    # MAPE on original scale
    preds = np.expm1(y_scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten())
    mask = y_val > 1000
    mape = np.mean(np.abs((y_val[mask] - preds[mask]) / y_val[mask])) * 100
    
    return val_loss, mape


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Setup distributed
    ctx = setup_distributed()
    amp_ctx = setup_amp(ctx)
    device_info = get_device_info()
    
    log_section("TRAINING CONFIGURATION", ctx)
    log_info(f"PyTorch: {torch.__version__}", ctx)
    log_info(f"Hardware: {ctx['hardware'].upper()}", ctx)
    log_info(f"Devices: {device_info['count']} x {device_info['names'][0] if device_info['names'] else 'N/A'}", ctx)
    log_info(f"Distributed: {ctx['distributed']} (world_size={ctx['world_size']})", ctx)
    log_info(f"Mixed Precision: {amp_ctx['enabled']}", ctx)
    if ctx['distributed']:
        log_info(f"Backend: {ctx['backend']}", ctx)
        log_info(f"Rank: {ctx['rank']}/{ctx['world_size']}, Local Rank: {ctx['local_rank']}", ctx)
    
    # Configuration
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/shared/models")
    FEATURE_REPO = os.environ.get("FEATURE_REPO", "/shared/feature_repo")
    NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", 20))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 512))
    LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))
    EARLY_STOP_PATIENCE = int(os.environ.get("EARLY_STOP_PATIENCE", "5"))
    TEMP_DIR = "/tmp/artifacts"
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    if ctx["rank"] == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    log_info(f"Config: epochs={NUM_EPOCHS}, batch={BATCH_SIZE}, lr={LEARNING_RATE}, early_stop={EARLY_STOP_PATIENCE}", ctx)
    
    # MLflow setup
    mlflow_enabled = False
    mlflow_run = None
    if ctx["rank"] == 0:
        try:
            import mlflow
            uri = os.environ.get("MLFLOW_TRACKING_URI")
            if uri:
                mlflow.set_tracking_uri(uri)
                mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "sales-forecasting"))
                mlflow_enabled = True
                log_info(f"MLflow: {uri}", ctx)
        except Exception as e:
            log_info(f"MLflow disabled: {e}", ctx)
    
    # Feature retrieval (or load from existing file)
    training_df = None
    feature_time = 0
    data_file = f"{OUTPUT_DIR}/training_data.parquet"
    
    if ctx["rank"] == 0:
        # Check if data already exists (skip Feast if so)
        if os.path.exists(data_file) and os.environ.get("SKIP_FEAST", "").lower() in ("1", "true"):
            log_section("LOADING CACHED DATA", ctx)
            training_df = pd.read_parquet(data_file)
            log_info(f"Loaded {len(training_df):,} rows from {data_file}", ctx)
        else:
            log_section("FEATURE RETRIEVAL (Feast + KubeRay)", ctx)
            try:
                from feast import FeatureStore
                ray_cfg = f"{FEATURE_REPO}/feature_store_ray.yaml"
                if os.path.exists(ray_cfg):
                    shutil.copy(ray_cfg, f"{FEATURE_REPO}/feature_store.yaml")
                store = FeatureStore(repo_path=FEATURE_REPO)
                
                rows = []
                base = datetime(2022, 1, 1, tzinfo=timezone.utc)
                for w in range(104):
                    ts = base + timedelta(weeks=w)
                    for s in range(1, 46):
                        for d in range(1, 15):
                            rows.append({"store_id": s, "dept_id": d, "event_timestamp": ts})
                entity_df = pd.DataFrame(rows)
                log_info(f"Querying {len(entity_df):,} entities (45 stores x 14 depts x 104 weeks)", ctx)
                
                t0 = time.time()
                training_df = store.get_historical_features(
                    entity_df=entity_df,
                    features=store.get_feature_service("training_features")
                ).to_df()
                feature_time = time.time() - t0
                log_info(f"Retrieved {len(training_df):,} rows in {feature_time:.1f}s", ctx)
                training_df.to_parquet(data_file)
            except ImportError:
                # Feast not available, try loading existing data
                if os.path.exists(data_file):
                    log_info("Feast not available, loading cached data...", ctx)
                    training_df = pd.read_parquet(data_file)
                else:
                    raise RuntimeError(f"No training data found at {data_file} and Feast not available")
    
    # Sync ranks
    feat_cols_file = f"{OUTPUT_DIR}/feat_cols.json"
    if ctx["rank"] == 0:
        log_section("DATA PREPARATION", ctx)
        training_df = training_df.dropna(subset=["weekly_sales"]).sort_values("event_timestamp")
        exclude = ["store_id", "dept_id", "date", "event_timestamp", "weekly_sales"]
        feat_cols = [c for c in training_df.columns
                     if c not in exclude and training_df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
        with open(feat_cols_file, "w") as f:
            json.dump(feat_cols, f)
        log_info(f"Features: {feat_cols}", ctx)
    
    if ctx["distributed"]:
        dist.barrier()
    
    if ctx["rank"] != 0:
        training_df = pd.read_parquet(data_file)
        training_df = training_df.dropna(subset=["weekly_sales"]).sort_values("event_timestamp")
    
    with open(feat_cols_file, "r") as f:
        feat_cols = json.load(f)
    
    # Split data
    split = int(len(training_df) * 0.8)
    train_df, val_df = training_df.iloc[:split], training_df.iloc[split:]
    
    X_train, y_train = train_df[feat_cols].fillna(0).values, train_df["weekly_sales"].values
    X_val, y_val = val_df[feat_cols].fillna(0).values, val_df["weekly_sales"].values
    
    scaler = StandardScaler()
    X_train, X_val = scaler.fit_transform(X_train), scaler.transform(X_val)
    
    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(np.log1p(y_train).reshape(-1, 1)).flatten()
    y_val_s = y_scaler.transform(np.log1p(y_val).reshape(-1, 1)).flatten()
    
    log_info(f"Train: {len(train_df):,}, Val: {len(val_df):,}", ctx)
    
    # MLflow run
    if mlflow_enabled and ctx["rank"] == 0:
        import mlflow
        mlflow_run = mlflow.start_run(run_name=f"{ctx['hardware']}-{ctx['world_size']}x-{datetime.now().strftime('%H%M%S')}")
        mlflow.log_params({
            "hardware": ctx["hardware"],
            "world_size": ctx["world_size"],
            "mixed_precision": amp_ctx["enabled"],
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "features": len(feat_cols)
        })
        mlflow.log_metric("feature_time", feature_time)
    
    # Data loaders
    train_ds = SalesDataset(X_train, y_train_s)
    val_ds = SalesDataset(X_val, y_val_s)
    
    sampler = DistributedSampler(train_ds, ctx["world_size"], ctx["rank"]) if ctx["distributed"] else None
    
    # Optimized DataLoader settings
    loader_kwargs = {
        "batch_size": BATCH_SIZE,
        "num_workers": 2,
        "pin_memory": ctx["hardware"] in ("cuda", "rocm"),
        "persistent_workers": True,
        "prefetch_factor": 2
    }
    
    train_loader = DataLoader(train_ds, shuffle=(sampler is None), sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, **loader_kwargs)
    
    # Model
    model = MLP(X_train.shape[1]).to(ctx["device"])
    num_params = sum(p.numel() for p in model.parameters())
    
    if ctx["distributed"]:
        model = DDP(model, device_ids=[ctx["local_rank"]] if ctx["hardware"] in ("cuda", "rocm") else None)
    
    log_section("MODEL TRAINING", ctx)
    log_info(f"Architecture: MLP ({X_train.shape[1]} -> 256 -> 128 -> 64 -> 1)", ctx)
    log_info(f"Parameters: {num_params:,}", ctx)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=LEARNING_RATE * 0.01
    )
    criterion = nn.MSELoss()
    
    best_loss, best_mape, best_epoch = float('inf'), float('inf'), 0
    patience_counter = 0
    
    log_info(f"{'Epoch':<8}{'Train':<12}{'Val':<12}{'MAPE':<10}{'LR':<12}{'Status'}", ctx)
    log_info("-" * 62, ctx)
    
    # Training loop
    t0 = time.time()
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, ctx, amp_ctx, sampler, epoch)
        val_loss, mape = validate(model, val_loader, criterion, ctx, amp_ctx, y_scaler, y_val)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Track best
        status = ""
        if val_loss < best_loss:
            best_loss, best_mape, best_epoch = val_loss, mape, epoch + 1
            patience_counter = 0
            status = "* best"
            if ctx["rank"] == 0:
                state = model.module.state_dict() if ctx["distributed"] else model.state_dict()
                torch.save(state, f"{TEMP_DIR}/model_best.pt")
        else:
            patience_counter += 1
        
        log_info(f"{epoch+1:<8}{train_loss:<12.6f}{val_loss:<12.6f}{mape:<10.2f}{current_lr:<12.2e}{status}", ctx)
        
        if mlflow_enabled and ctx["rank"] == 0:
            import mlflow
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss, "mape": mape, "lr": current_lr}, step=epoch)
        
        # Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            log_info(f"Early stopping at epoch {epoch+1}", ctx)
            break
    
    train_time = time.time() - t0
    
    # Save artifacts
    if ctx["rank"] == 0:
        state = model.module.state_dict() if ctx["distributed"] else model.state_dict()
        torch.save(state, f"{TEMP_DIR}/model_final.pt")
        joblib.dump({"scaler_X": scaler, "scaler_y": y_scaler}, f"{TEMP_DIR}/scalers.joblib")
        joblib.dump(feat_cols, f"{TEMP_DIR}/feature_cols.pkl")
        
        with open(f"{TEMP_DIR}/metadata.json", "w") as f:
            json.dump({
                "mape": best_mape,
                "best_epoch": best_epoch,
                "hardware": ctx["hardware"],
                "world_size": ctx["world_size"],
                "mixed_precision": amp_ctx["enabled"],
                "train_time": train_time,
                "features": len(feat_cols)
            }, f)
        
        run_id = "local"
        if mlflow_enabled:
            import mlflow
            mlflow.log_metrics({"best_mape": best_mape, "train_time": train_time})
            for fname in ["model_best.pt", "model_final.pt", "scalers.joblib", "feature_cols.pkl", "metadata.json"]:
                fpath = f"{TEMP_DIR}/{fname}"
                if os.path.exists(fpath):
                    mlflow.log_artifact(fpath)
            run_id = mlflow_run.info.run_id
            mlflow.end_run()
        
        log_section("TRAINING COMPLETE", ctx)
        log_info(f"Best MAPE:   {best_mape:.2f}% (epoch {best_epoch})", ctx)
        log_info(f"Train time:  {train_time:.1f}s", ctx)
        log_info(f"Hardware:    {ctx['hardware'].upper()} x {ctx['world_size']}", ctx)
        log_info(f"MLflow run:  {run_id}", ctx)
    
    cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
