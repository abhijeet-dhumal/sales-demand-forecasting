"""
Sales Demand Forecasting - Training Script for Kubeflow Trainer v2

This script is designed to be used with the Kubeflow SDK's CustomTrainer.
It handles:
- Distributed training with PyTorch DDP
- Feature loading from Feast
- Temporal train/validation split (NO random splits for time series)
- Mixed precision training (AMP)
- Checkpointing
"""
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# MODEL DEFINITION
# =============================================================================


class SalesForecastingMLP(nn.Module):
    """Multi-layer perceptron for sales forecasting."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [256, 128, 64],
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# =============================================================================
# DATASET
# =============================================================================


class SalesDataset(Dataset):
    """PyTorch Dataset for sales forecasting features."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


# =============================================================================
# DATA LOADING
# =============================================================================


def load_features_from_feast(
    feast_repo_path: str,
    entity_df: pd.DataFrame,
    feature_service: str = "sales_forecasting_v2",
) -> pd.DataFrame:
    """Load features from Feast feature store."""
    from feast import FeatureStore

    logger.info(f"Loading features from Feast: {feast_repo_path}")
    store = FeatureStore(repo_path=feast_repo_path)

    features_df = store.get_historical_features(
        entity_df=entity_df,
        features=store.get_feature_service(feature_service),
    ).to_df()

    logger.info(f"Loaded {len(features_df)} rows with {len(features_df.columns)} columns")
    return features_df


def load_features_from_parquet(data_path: str) -> pd.DataFrame:
    """Load pre-computed features from Parquet (fallback)."""
    logger.info(f"Loading features from Parquet: {data_path}")
    return pd.read_parquet(data_path)


def temporal_train_val_split(
    df: pd.DataFrame,
    date_column: str = "date",
    val_start_date: str = "2012-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally for time series forecasting.
    
    CRITICAL: Random splits cause data leakage in time series.
    We train on past data and validate on future data.
    
    Args:
        df: DataFrame with features and target
        date_column: Column containing dates
        val_start_date: First date of validation set
        
    Returns:
        (train_df, val_df) tuple
    """
    df[date_column] = pd.to_datetime(df[date_column])
    val_date = pd.to_datetime(val_start_date)

    train_df = df[df[date_column] < val_date].copy()
    val_df = df[df[date_column] >= val_date].copy()

    logger.info(f"Temporal split: train={len(train_df)}, val={len(val_df)}")
    logger.info(f"Train date range: {train_df[date_column].min()} to {train_df[date_column].max()}")
    logger.info(f"Val date range: {val_df[date_column].min()} to {val_df[date_column].max()}")

    return train_df, val_df


# =============================================================================
# TRAINING FUNCTION (for CustomTrainer)
# =============================================================================


def train_sales_forecast(
    # Data config
    data_source: str = "parquet",  # "feast" or "parquet"
    feast_repo_path: Optional[str] = None,
    data_path: Optional[str] = None,
    # Model config
    hidden_dims: list[int] = [256, 128, 64],
    dropout: float = 0.2,
    # Training config
    num_epochs: int = 10,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    use_amp: bool = True,
    grad_clip_norm: float = 1.0,
    # Temporal split
    val_start_date: str = "2012-01-01",
    # Output
    output_dir: str = "/shared/models",
):
    """
    Main training function for Kubeflow SDK CustomTrainer.
    
    This function is self-contained and handles:
    - DDP initialization
    - Feature loading from Feast or Parquet
    - Temporal train/val split
    - Model training with AMP
    - Checkpointing
    """
    # =========================================================================
    # DDP Setup (SDK provides MASTER_ADDR, WORLD_SIZE, RANK env vars)
    # =========================================================================
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    logger.info(f"DDP initialized: rank={rank}, world_size={world_size}, backend={backend}")

    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # =========================================================================
    # Data Loading (only rank 0 loads, then broadcasts)
    # =========================================================================
    if rank == 0:
        if data_source == "feast" and feast_repo_path:
            # Create entity DataFrame for Feast
            # In production, this would come from your entity source
            entity_df = pd.read_parquet(f"{data_path}/entities.parquet")
            df = load_features_from_feast(feast_repo_path, entity_df)
        else:
            df = load_features_from_parquet(f"{data_path}/features.parquet")

        # Temporal split
        train_df, val_df = temporal_train_val_split(df, val_start_date=val_start_date)

        # Define feature columns (excluding target and metadata)
        exclude_cols = {"date", "store_id", "dept_id", "weekly_sales", "created_timestamp"}
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]
        target_col = "weekly_sales"

        # Prepare features
        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df[target_col].values
        X_val = val_df[feature_cols].fillna(0).values
        y_val = val_df[target_col].values

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Save scaler for inference
        os.makedirs(output_dir, exist_ok=True)
        import joblib
        joblib.dump(scaler, f"{output_dir}/scaler.pkl")
        joblib.dump(feature_cols, f"{output_dir}/feature_cols.pkl")

        input_dim = X_train.shape[1]
        logger.info(f"Features: {len(feature_cols)} columns, input_dim={input_dim}")

        # Save to shared storage for other ranks
        np.save(f"{output_dir}/.X_train.npy", X_train)
        np.save(f"{output_dir}/.y_train.npy", y_train)
        np.save(f"{output_dir}/.X_val.npy", X_val)
        np.save(f"{output_dir}/.y_val.npy", y_val)

    # Synchronize
    dist.barrier()

    # Other ranks load from shared storage
    if rank != 0:
        X_train = np.load(f"{output_dir}/.X_train.npy")
        y_train = np.load(f"{output_dir}/.y_train.npy")
        X_val = np.load(f"{output_dir}/.X_val.npy")
        y_val = np.load(f"{output_dir}/.y_val.npy")
        input_dim = X_train.shape[1]

    dist.barrier()

    # =========================================================================
    # Create DataLoaders
    # =========================================================================
    train_dataset = SalesDataset(X_train, y_train)
    val_dataset = SalesDataset(X_val, y_val)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # =========================================================================
    # Model Setup
    # =========================================================================
    model = SalesForecastingMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.MSELoss()
    scaler = GradScaler() if use_amp and torch.cuda.is_available() else None

    # =========================================================================
    # Training Loop
    # =========================================================================
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0

        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            if scaler:
                with autocast():
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(features)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            train_loss += loss.item()

        # Average train loss
        train_loss /= len(train_loader)

        # =====================================================================
        # Validation
        # =====================================================================
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)

                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Calculate MAPE (only on rank 0)
        if rank == 0:
            preds = np.array(all_preds)
            actuals = np.array(all_targets)
            # Filter out near-zero actuals to avoid division issues
            mask = np.abs(actuals) > 1000  # Only consider meaningful sales
            if mask.sum() > 0:
                mape = np.mean(np.abs((actuals[mask] - preds[mask]) / actuals[mask])) * 100
            else:
                mape = float("nan")

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val MAPE: {mape:.2f}%"
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    model.module.state_dict(),
                    f"{output_dir}/best_model.pt",
                )
                logger.info(f"Saved best model with val_loss={val_loss:.4f}")

        dist.barrier()

    # =========================================================================
    # Cleanup
    # =========================================================================
    if rank == 0:
        # Clean up temp files
        for f in [".X_train.npy", ".y_train.npy", ".X_val.npy", ".y_val.npy"]:
            try:
                os.remove(f"{output_dir}/{f}")
            except:
                pass

        logger.info(f"Training complete! Best val_loss: {best_val_loss:.4f}")
        logger.info(f"Model saved to: {output_dir}/best_model.pt")

    dist.destroy_process_group()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-source", default="parquet")
    parser.add_argument("--feast-repo-path", default=None)
    parser.add_argument("--data-path", default="/shared/data")
    parser.add_argument("--output-dir", default="/shared/models")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-start-date", default="2012-01-01")
    args = parser.parse_args()

    train_sales_forecast(
        data_source=args.data_source,
        feast_repo_path=args.feast_repo_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_start_date=args.val_start_date,
    )

