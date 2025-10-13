def training_func(parameters=None):
    """
    Complete training function for distributed sales forecasting.
    
    Args:
        parameters: Dict with training configuration including:
            - feast_repo_path: Path to Feast repository
            - model_output_dir: Directory for model outputs
            - num_epochs: Number of training epochs
            - batch_size: Training batch size
            - learning_rate: Learning rate
            - hidden_dims: List of hidden layer dimensions
            - chunk_size: Rows per chunk for streaming
            - sample_size: Sample N rows for testing
            - backend: DDP backend ('nccl' or 'gloo')
    """
    import os
    import logging
    import glob
    from pathlib import Path
    
    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.utils.data import DataLoader, IterableDataset
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    try:
        from torchdata.stateful_dataloader import StatefulDataLoader
        TORCHDATA_AVAILABLE = True
    except ImportError:
        StatefulDataLoader = DataLoader
        TORCHDATA_AVAILABLE = False
    
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import joblib
    import time
    import fcntl
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    logger = logging.getLogger(__name__)
    
    def detect_device():
        """
        Auto-detect available compute device (NVIDIA GPU, AMD GPU, or CPU).
        
        Returns:
            tuple: (device_type, device_count, backend)
                - device_type: 'cuda' (NVIDIA), 'hip' (AMD ROCm), or 'cpu'
                - device_count: number of devices available
                - backend: 'nccl' (NVIDIA GPU), 'gloo' (AMD GPU/CPU)
        """
        # Check for NVIDIA CUDA GPUs
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            logger.info(f"Detected NVIDIA GPU: {device_name} (count: {device_count})")
            return 'cuda', device_count, 'nccl'
        
        # Check for AMD ROCm GPUs (PyTorch with ROCm support)
        try:
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                # ROCm-enabled PyTorch
                device_count = torch.cuda.device_count()  # ROCm uses torch.cuda API
                logger.info(f"Detected AMD ROCm GPU (count: {device_count})")
                return 'hip', device_count, 'gloo'  # Use gloo for AMD
        except Exception:
            pass
        
        # Fallback to CPU
        logger.info("No GPU detected, using CPU")
        return 'cpu', 0, 'gloo'
    
    def ddp_setup(backend="auto"):
        """
        Setup for Distributed Data Parallel with auto device detection.
        
        Args:
            backend: 'auto' (detect), 'nccl' (NVIDIA), or 'gloo' (CPU/AMD)
        """
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_type, device_count, recommended_backend = detect_device()
        
        # Auto-select backend if requested
        if backend == "auto":
            backend = recommended_backend
            logger.info(f"Auto-selected backend: {backend} for device type: {device_type}")
        
        # Initialize process group first to get global rank
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        node_rank = int(os.environ.get("GROUP_RANK", 0))
        
        # Setup device-specific configuration
        # CRITICAL: Use global rank (not local_rank) to handle same-node multi-GPU scheduling
        if device_type in ['cuda', 'hip'] and device_count > 0:
            # Use global rank modulo GPU count to assign unique GPU per rank
            # This handles both multi-node (1 GPU/node) and same-node (multiple GPUs) scenarios
            device_id = rank % device_count
            torch.cuda.set_device(device_id)  # ROCm also uses torch.cuda API
            logger.info(f"[Global Rank {rank}] [Local Rank {local_rank}] Using {device_type.upper()} device {device_id}")
        else:
            # CPU only
            logger.info(f"[Global Rank {rank}] [Local Rank {local_rank}] Using CPU")
        
        logger.info(f"[Global Rank {rank}/{world_size-1}] [Local Rank {local_rank}] [Node {node_rank}] Backend: {backend}")
        return backend, device_type
    
    class StreamingSalesDataset(IterableDataset):
        """Memory-efficient streaming dataset."""
        
        def __init__(self, cache_dir, feature_cols, target_col, scaler, target_scaler,
                     rank, world_size, chunk_files=None, shuffle=False, seed=42):
            super().__init__()
            self.cache_dir = cache_dir
            self.feature_cols = feature_cols
            self.target_col = target_col
            self.scaler = scaler
            self.target_scaler = target_scaler
            self.rank = rank
            self.world_size = world_size
            self.shuffle = shuffle
            self.seed = seed
            
            if chunk_files is not None:
                self.chunk_files = sorted(chunk_files)
            else:
                self.chunk_files = sorted(glob.glob(f"{cache_dir}/chunk_*.parquet"))
            
            if not self.chunk_files:
                raise ValueError(f"No chunk files found in {cache_dir}")
        
        def __iter__(self):
            worker_info = torch.utils.data.get_worker_info()
            worker_id = 0 if worker_info is None else worker_info.id
            num_workers = 1 if worker_info is None else worker_info.num_workers
            
            total_workers = self.world_size * num_workers
            global_worker_id = self.rank * num_workers + worker_id
            
            my_chunks = [
                self.chunk_files[i] 
                for i in range(global_worker_id, len(self.chunk_files), total_workers)
            ]
            
            if self.shuffle:
                rng = np.random.RandomState(self.seed + global_worker_id)
                rng.shuffle(my_chunks)
            
            for chunk_file in my_chunks:
                df = pd.read_parquet(chunk_file, columns=self.feature_cols + [self.target_col])
                
                # Handle NaN and inf values
                X = df[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
                y = df[self.target_col].values.astype(np.float32)
                
                if self.scaler:
                    X = self.scaler.transform(X).astype(np.float32)
                
                # Scale target to reasonable range for training
                if self.target_scaler:
                    y = self.target_scaler.transform(y.reshape(-1, 1)).flatten().astype(np.float32)
                
                if self.shuffle:
                    indices = np.arange(len(X))
                    rng.shuffle(indices)
                    X = X[indices]
                    y = y[indices]
                
                for i in range(len(X)):
                    yield torch.from_numpy(X[i]), torch.tensor(y[i])
                
                del df, X, y
    
    class SalesForecastingMLP(nn.Module):
        """Multi-Layer Perceptron for sales forecasting."""
        
        def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout=0.3):
            super().__init__()
            
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    def load_features_feast_file(feast_repo_path, data_path, output_cache_dir, chunk_size, sample_size):
        """Load features using Feast with file-based offline store - RANK 0 ONLY."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        completion_marker = f"{output_cache_dir}/.feast_load_complete"
        
        if rank == 0:
            logger.info("=" * 70)
            logger.info(f"LOADING WITH FEAST FILE OFFLINE STORE [Rank 0 of {world_size}]")
            logger.info("=" * 70)
            
            os.makedirs(output_cache_dir, exist_ok=True)
            
            if os.path.exists(completion_marker):
                logger.info(f"Found existing features at {output_cache_dir}, skipping retrieval")
            else:
                from feast import FeatureStore
                
                logger.info(f"Initializing Feast from: {feast_repo_path}")
                store = FeatureStore(repo_path=feast_repo_path)
                
                # Load entity data
                sales_path = f"{data_path}/sales_features.parquet"
                if not os.path.exists(sales_path):
                    raise FileNotFoundError(f"Sales features not found: {sales_path}")
                
                logger.info(f"Loading entity data from: {sales_path}")
                entity_df = pd.read_parquet(sales_path)[['store', 'dept', 'date']]
                
                # Sample if requested
                if sample_size and sample_size < len(entity_df):
                    logger.info(f"Sampling {sample_size:,} entities from {len(entity_df):,}")
                    entity_df = entity_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                
                total_rows = len(entity_df)
                num_chunks = (total_rows + chunk_size - 1) // chunk_size
                logger.info(f"Processing {total_rows:,} rows in {num_chunks} chunks")
                
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, total_rows)
                    
                    entity_chunk = entity_df.iloc[start_idx:end_idx].copy()
                    entity_chunk = entity_chunk.rename(columns={'date': 'event_timestamp'})
                    
                    logger.info(f"  Retrieving chunk {chunk_idx+1}/{num_chunks} ({start_idx}:{end_idx})...")
                    
                    # Use Feast to get features (file offline store, no Ray)
                    features_chunk = store.get_historical_features(
                        entity_df=entity_chunk,
                        features=[
                            "sales_history_features:weekly_sales",
                            "sales_history_features:is_holiday",
                            "sales_history_features:sales_lag_1",
                            "sales_history_features:sales_lag_2",
                            "sales_history_features:sales_lag_4",
                            "sales_history_features:sales_rolling_mean_4",
                            "sales_history_features:sales_rolling_mean_12",
                            "sales_history_features:sales_rolling_std_4",
                            "store_external_features:temperature",
                            "store_external_features:fuel_price",
                            "store_external_features:cpi",
                            "store_external_features:unemployment",
                            "store_external_features:markdown1",
                            "store_external_features:markdown2",
                            "store_external_features:markdown3",
                            "store_external_features:markdown4",
                            "store_external_features:markdown5",
                            "store_external_features:total_markdown",
                            "store_external_features:has_markdown",
                            "store_external_features:store_type",
                            "store_external_features:store_size",
                        ],
                    ).to_df()
                    
                    logger.info(f"  Retrieved {len(features_chunk)} rows, computing transformations...")
                    
                    # Compute on-demand transformations manually
                    features_chunk["sales_normalized"] = features_chunk["weekly_sales"].clip(0, 200000) / 200000
                    features_chunk["temperature_normalized"] = ((features_chunk["temperature"] - 5) / 95).clip(0, 1)
                    features_chunk["sales_per_sqft"] = features_chunk["weekly_sales"] / (features_chunk["store_size"] + 1)
                    features_chunk["markdown_efficiency"] = features_chunk["weekly_sales"] / (features_chunk["total_markdown"] + 1)
                    
                    features_chunk["sales_velocity"] = (
                        (features_chunk["sales_lag_1"] - features_chunk["sales_lag_2"]) / (features_chunk["sales_lag_2"] + 1)
                    )
                    velocity_prev = (features_chunk["sales_lag_2"] - features_chunk["sales_lag_4"]) / (features_chunk["sales_lag_4"] + 1)
                    features_chunk["sales_acceleration"] = features_chunk["sales_velocity"] - velocity_prev
                    features_chunk["demand_stability_score"] = 1 - (
                        features_chunk["sales_rolling_std_4"] / (features_chunk["sales_rolling_mean_4"] + 1)
                    ).clip(0, 1)
                    
                    chunk_file = f"{output_cache_dir}/chunk_{chunk_idx:04d}.parquet"
                    features_chunk.to_parquet(chunk_file, index=False)
                    logger.info(f"  Chunk {chunk_idx+1}/{num_chunks}: {len(features_chunk):,} rows -> {chunk_file}")
                
                with open(completion_marker, 'w') as f:
                    f.write(f"completed_at={time.time()}\nnum_chunks={num_chunks}\n")
                
                logger.info(f"Feast file retrieval complete: {num_chunks} chunks saved")
        else:
            logger.info(f"[Rank {rank}/{world_size}] Waiting for Rank 0 to complete feature retrieval...")
            
            timeout = 1800
            start_time = time.time()
            while not os.path.exists(completion_marker):
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Rank {rank} timed out waiting for features")
                time.sleep(5)
            
            logger.info(f"[Rank {rank}] Feature retrieval complete marker found")
        
        if dist.is_initialized():
            logger.info(f"[Rank {rank}] Synchronizing at barrier...")
            # Don't use device_ids for multi-node - causes "invalid device ordinal"
            dist.barrier()
            logger.info(f"[Rank {rank}] Barrier passed")
        
        return output_cache_dir
    
    def load_features_direct(data_path, output_cache_dir, chunk_size, sample_size):
        """Load features directly from parquet files without Feast - RANK 0 ONLY."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        completion_marker = f"{output_cache_dir}/.data_load_complete"
        
        if rank == 0:
            logger.info("=" * 70)
            logger.info(f"LOADING FROM PARQUET FILES [Rank 0 of {world_size}]")
            logger.info("=" * 70)
            
            os.makedirs(output_cache_dir, exist_ok=True)
            
            if os.path.exists(completion_marker):
                logger.info(f"Found existing data at {output_cache_dir}, skipping load")
            else:
                # Load sales and store features
                sales_path = f"{data_path}/sales_features.parquet"
                store_path = f"{data_path}/store_features.parquet"
                
                if not os.path.exists(sales_path):
                    raise FileNotFoundError(f"Sales features not found: {sales_path}")
                if not os.path.exists(store_path):
                    raise FileNotFoundError(f"Store features not found: {store_path}")
                
                logger.info(f"Loading sales features: {sales_path}")
                sales_df = pd.read_parquet(sales_path)
                logger.info(f"Loading store features: {store_path}")
                store_df = pd.read_parquet(store_path)
                
                # Merge on store, dept, date
                logger.info("Merging sales and store features...")
                merged_df = pd.merge(
                    sales_df, store_df,
                    on=['store', 'dept', 'date'],
                    how='inner'
                )
                logger.info(f"Merged dataset: {len(merged_df):,} rows, {len(merged_df.columns)} columns")
                
                # Sample if requested
                if sample_size and sample_size < len(merged_df):
                    logger.info(f"Sampling {sample_size:,} rows from {len(merged_df):,}")
                    merged_df = merged_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                
                # Compute on-demand transformations
                logger.info("Computing on-demand transformations...")
                merged_df["sales_normalized"] = merged_df["weekly_sales"].clip(0, 200000) / 200000
                merged_df["temperature_normalized"] = ((merged_df["temperature"] - 5) / 95).clip(0, 1)
                merged_df["sales_per_sqft"] = merged_df["weekly_sales"] / (merged_df["store_size"] + 1)
                merged_df["markdown_efficiency"] = merged_df["weekly_sales"] / (merged_df["total_markdown"] + 1)
                
                merged_df["sales_velocity"] = (
                    (merged_df["sales_lag_1"] - merged_df["sales_lag_2"]) / (merged_df["sales_lag_2"] + 1)
                )
                velocity_prev = (merged_df["sales_lag_2"] - merged_df["sales_lag_4"]) / (merged_df["sales_lag_4"] + 1)
                merged_df["sales_acceleration"] = merged_df["sales_velocity"] - velocity_prev
                merged_df["demand_stability_score"] = 1 - (
                    merged_df["sales_rolling_std_4"] / (merged_df["sales_rolling_mean_4"] + 1)
                ).clip(0, 1)
                
                logger.info(f"Computed 7 on-demand features")
                
                # Save in chunks
                num_chunks = (len(merged_df) + chunk_size - 1) // chunk_size
                logger.info(f"Saving {num_chunks} chunks...")
                
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, len(merged_df))
                    chunk_df = merged_df.iloc[start_idx:end_idx]
                    
                    chunk_file = f"{output_cache_dir}/chunk_{chunk_idx:04d}.parquet"
                    chunk_df.to_parquet(chunk_file, index=False)
                    logger.info(f"  Chunk {chunk_idx+1}/{num_chunks}: {len(chunk_df):,} rows -> {chunk_file}")
                
                with open(completion_marker, 'w') as f:
                    f.write(f"completed_at={time.time()}\nnum_chunks={num_chunks}\n")
                
                logger.info(f"Data loading complete: {num_chunks} chunks saved")
        else:
            logger.info(f"[Rank {rank}/{world_size}] Waiting for Rank 0 to complete data loading...")
            
            timeout = 1800
            start_time = time.time()
            while not os.path.exists(completion_marker):
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Rank {rank} timed out waiting for data load")
                time.sleep(5)
            
            logger.info(f"[Rank {rank}] Data load complete marker found")
        
        if dist.is_initialized():
            logger.info(f"[Rank {rank}] Synchronizing at barrier...")
            # Don't use device_ids for multi-node - causes "invalid device ordinal"
            dist.barrier()
            logger.info(f"[Rank {rank}] Barrier passed")
        
        return output_cache_dir
    
    def load_features_chunked(feast_repo_path, output_cache_dir, chunk_size, sample_size):
        """Load features from Feast using Ray - RANK 0 ONLY with file-based locking."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Configure Ray for controlled resource usage (not disabling, but limiting)
        os.environ.update({
            "FEAST_USAGE": "false",  # Disable telemetry
            "RAY_DEDUP_LOGS": "0",  # Show all logs
        })
        
        completion_marker = f"{output_cache_dir}/.feast_load_complete"
        lock_file = f"{output_cache_dir}/.feast_lock"
        
        if rank == 0:
            logger.info("=" * 70)
            logger.info(f"CHUNKED FEAST FEATURE RETRIEVAL [Rank 0 of {world_size}]")
            logger.info("=" * 70)
            
            os.makedirs(output_cache_dir, exist_ok=True)
            
            if os.path.exists(completion_marker):
                logger.info(f"Found existing features at {output_cache_dir}, skipping Feast retrieval")
            else:
                logger.info("Acquiring exclusive lock for Feast access...")
                lock_fd = None
                try:
                    lock_fd = open(lock_file, 'w')
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    logger.info("Lock acquired - proceeding with Feast retrieval")
                    logger.info("Using Ray offline store for distributed feature retrieval")
                    
                    # Initialize Ray for this process (if not already running)
                    try:
                        import ray
                        if not ray.is_initialized():
                            logger.info("Initializing Ray for Feast feature retrieval...")
                            ray.init(
                                num_cpus=2,
                                object_store_memory=2 * 1024 * 1024 * 1024,  # 2GB
                                _memory=4 * 1024 * 1024 * 1024,  # 4GB
                                ignore_reinit_error=True,
                                log_to_driver=True,
                            )
                            logger.info(f"Ray initialized: {ray.cluster_resources()}")
                        else:
                            logger.info(f"Ray already initialized: {ray.cluster_resources()}")
                    except Exception as e:
                        logger.warning(f"Ray initialization skipped: {e}")
                    
                    # Use PostgreSQL registry cache
                    os.environ["FEAST_REGISTRY_CACHE_TTL_SECONDS"] = "3600"
                    
                    # Monkey-patch Ray 2.9+ random_shuffle to fix Feast compatibility
                    try:
                        import ray.data
                        _original_random_shuffle = ray.data.Dataset.random_shuffle
                        
                        def _patched_random_shuffle(self, seed=None, num_blocks=None, **kwargs):
                            """Patched random_shuffle that ignores deprecated num_blocks parameter."""
                            if num_blocks is not None:
                                logger.warning(f"Ignoring deprecated num_blocks={num_blocks} in random_shuffle")
                                # Call repartition first, then shuffle (Ray 2.9+ compatible)
                                return self.repartition(num_blocks=num_blocks).random_shuffle(seed=seed, **kwargs)
                            return _original_random_shuffle(self, seed=seed, **kwargs)
                        
                        ray.data.Dataset.random_shuffle = _patched_random_shuffle
                        logger.info("Applied Ray 2.9+ random_shuffle compatibility patch")
                    except Exception as e:
                        logger.warning(f"Could not patch Ray random_shuffle: {e}")
                    
                    # Monkey-patch to fix Python 3.11 + dill serialization issue with on-demand features
                    import dill
                    
                    _original_dumps = dill.dumps
                    def _patched_dumps(obj, *args, **kwargs):
                        try:
                            return _original_dumps(obj, *args, **kwargs)
                        except IndexError as e:
                            if "tuple index out of range" in str(e):
                                logger.warning(f"Skipping dill.dumps due to Python 3.11 bytecode issue")
                                return b"SKIP_PYTHON311"  # Marker for empty/invalid UDF
                            raise
                    dill.dumps = _patched_dumps
                    
                    _original_loads = dill.loads
                    def _patched_loads(bytes_obj, *args, **kwargs):
                        if bytes_obj == b"SKIP_PYTHON311" or not bytes_obj:
                            logger.warning(f"Skipping dill.loads - UDF was not serialized due to Python 3.11")
                            return lambda x: x  # Return identity function as placeholder
                        try:
                            return _original_loads(bytes_obj, *args, **kwargs)
                        except EOFError as e:
                            logger.warning(f"Skipping dill.loads due to EOFError: {e}")
                            return lambda x: x  # Return identity function as placeholder
                    dill.loads = _patched_loads
                    
                    logger.info("Applied dill.dumps/loads patches for Python 3.11 compatibility")
                    
                    from feast import FeatureStore
                    
                    data_dir = Path(feast_repo_path).parent / "feature_data"
                    if not data_dir.exists():
                        data_dir = Path(feast_repo_path) / "data"
                    
                    sales_file = data_dir / "sales_features.parquet"
                    entity_df = pd.read_parquet(sales_file, columns=['store', 'dept', 'date'])
                    
                    if sample_size:
                        entity_df = entity_df.sample(n=min(sample_size, len(entity_df)), random_state=42)
                    
                    # CRITICAL: Sort by join keys + timestamp to avoid slow manual join fallback
                    logger.info("Sorting entity data by (store, dept, date) for efficient Ray merge_asof...")
                    entity_df = entity_df.sort_values(['store', 'dept', 'date'], ascending=True).reset_index(drop=True)
                    logger.info("Entity data sorted successfully")
                    
                    total_rows = len(entity_df)
                    num_chunks = (total_rows + chunk_size - 1) // chunk_size
                    logger.info(f"Processing {total_rows:,} rows in {num_chunks} chunks")
                    
                    max_retries = 3
                    retry_delay = 2
                    
                    for attempt in range(max_retries):
                        try:
                            logger.info(f"Initializing Feast store (attempt {attempt+1}/{max_retries})...")
                            store = FeatureStore(repo_path=feast_repo_path)
                            logger.info("Feast store initialized successfully")
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                logger.warning(f"Feast init failed: {e}, retrying in {retry_delay}s...")
                                time.sleep(retry_delay)
                                retry_delay *= 2
                            else:
                                logger.error(f"Failed to initialize Feast after {max_retries} attempts")
                                raise
                    
                    for chunk_idx in range(num_chunks):
                        start_idx = chunk_idx * chunk_size
                        end_idx = min(start_idx + chunk_size, total_rows)
                        
                        entity_chunk = entity_df.iloc[start_idx:end_idx].copy()
                        entity_chunk = entity_chunk.rename(columns={'date': 'event_timestamp'})
                        
                        logger.info(f"  Retrieving chunk {chunk_idx+1}/{num_chunks} ({start_idx}:{end_idx})...")
                        
                        # Fetch base features only (on-demand features don't work on Python 3.11 due to dill serialization)
                        features_chunk = store.get_historical_features(
                            entity_df=entity_chunk,
                            features=[
                                "sales_history_features:weekly_sales",
                                "sales_history_features:is_holiday",
                                "sales_history_features:sales_lag_1",
                                "sales_history_features:sales_lag_2",
                                "sales_history_features:sales_lag_4",
                                "sales_history_features:sales_rolling_mean_4",
                                "sales_history_features:sales_rolling_mean_12",
                                "sales_history_features:sales_rolling_std_4",
                                "store_external_features:temperature",
                                "store_external_features:fuel_price",
                                "store_external_features:cpi",
                                "store_external_features:unemployment",
                                "store_external_features:markdown1",
                                "store_external_features:markdown2",
                                "store_external_features:markdown3",
                                "store_external_features:markdown4",
                                "store_external_features:markdown5",
                                "store_external_features:total_markdown",
                                "store_external_features:has_markdown",
                                "store_external_features:store_type",
                                "store_external_features:store_size",
                            ],
                        ).to_df()
                        
                        logger.info(f"  Retrieved {len(features_chunk)} rows, computing transformations...")
                        
                        # Manually compute on-demand transformations (Python 3.11 compatible)
                        features_chunk["sales_normalized"] = features_chunk["weekly_sales"].clip(0, 200000) / 200000
                        features_chunk["temperature_normalized"] = ((features_chunk["temperature"] - 5) / 95).clip(0, 1)
                        features_chunk["sales_per_sqft"] = features_chunk["weekly_sales"] / (features_chunk["store_size"] + 1)
                        features_chunk["markdown_efficiency"] = features_chunk["weekly_sales"] / (features_chunk["total_markdown"] + 1)
                        
                        features_chunk["sales_velocity"] = (
                            (features_chunk["sales_lag_1"] - features_chunk["sales_lag_2"]) / (features_chunk["sales_lag_2"] + 1)
                        )
                        velocity_prev = (features_chunk["sales_lag_2"] - features_chunk["sales_lag_4"]) / (features_chunk["sales_lag_4"] + 1)
                        features_chunk["sales_acceleration"] = features_chunk["sales_velocity"] - velocity_prev
                        features_chunk["demand_stability_score"] = 1 - (
                            features_chunk["sales_rolling_std_4"] / (features_chunk["sales_rolling_mean_4"] + 1)
                        ).clip(0, 1)
                        
                        logger.info(f"  Computed 7 on-demand features manually")
                        
                        chunk_file = f"{output_cache_dir}/chunk_{chunk_idx:04d}.parquet"
                        features_chunk.to_parquet(chunk_file, index=False)
                        logger.info(f"  Chunk {chunk_idx+1}/{num_chunks}: {len(features_chunk):,} rows -> {chunk_file}")
                        
                        del features_chunk, entity_chunk
                    
                    with open(completion_marker, 'w') as f:
                        f.write(f"completed_at={time.time()}\nnum_chunks={num_chunks}\n")
                    
                    logger.info(f"Saved {num_chunks} chunks to {output_cache_dir}")
                    
                except BlockingIOError:
                    logger.warning("Another process holds the lock, waiting at barrier...")
                except Exception as e:
                    logger.error(f"Error during Feast retrieval: {e}")
                    raise
                finally:
                    if lock_fd:
                        try:
                            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                            lock_fd.close()
                        except:
                            pass
                        try:
                            os.remove(lock_file)
                        except:
                            pass
        else:
            logger.info(f"[Rank {rank}/{world_size}] Waiting for Rank 0 to complete Feast retrieval...")
            timeout = 600
            start_time = time.time()
            while not os.path.exists(completion_marker):
                time.sleep(5)
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Rank {rank} timed out waiting for Rank 0 feature loading")
            logger.info(f"[Rank {rank}] Feature loading complete, proceeding...")
        
        if dist.is_initialized():
            logger.info(f"[Rank {rank}] Synchronizing at barrier...")
            # Don't use device_ids for multi-node - causes "invalid device ordinal"
            dist.barrier()
            logger.info(f"[Rank {rank}] Barrier passed")
        
        return output_cache_dir
    
    def prepare_features(chunks_dir, cache_dir):
        """Prepare features with consistent encoding - RANK 0 ONLY."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        if rank == 0:
            logger.info("=" * 70)
            logger.info(f"PREPARING FEATURES AND SCALER [Rank 0 of {world_size}]")
            logger.info("=" * 70)
            
            first_chunk = sorted(glob.glob(f"{chunks_dir}/chunk_*.parquet"))[0]
            df_sample = pd.read_parquet(first_chunk)
            
            target_col = 'weekly_sales'
            categorical_cols = []
            
            for col in df_sample.columns:
                if df_sample[col].dtype == 'object' or df_sample[col].dtype.name == 'category':
                    categorical_cols.append(col)
            
            encoders = {}
            for col in categorical_cols:
                if col in ['store', 'dept', 'date', 'event_timestamp', target_col]:
                    continue
                
                if 'type' in col.lower() or col == 'store_type':
                    encoders[col] = {'mapping': {'A': 0, 'B': 1, 'C': 2}, 'type': 'map'}
                    df_sample[f'{col}_encoded'] = df_sample[col].map(encoders[col]['mapping']).fillna(0).astype(int)
                else:
                    le = LabelEncoder()
                    le.fit(df_sample[col].dropna().astype(str))
                    encoders[col] = {'encoder': le, 'type': 'label'}
                    df_sample[f'{col}_encoded'] = le.transform(df_sample[col].fillna('missing').astype(str))
            
            joblib.dump(encoders, f"{cache_dir}/encoders.pkl")
            
            all_chunks = sorted(glob.glob(f"{chunks_dir}/chunk_*.parquet"))
            for chunk_file in all_chunks:
                df_chunk = pd.read_parquet(chunk_file)
                
                for col, encoder_info in encoders.items():
                    if encoder_info['type'] == 'map':
                        df_chunk[f'{col}_encoded'] = df_chunk[col].map(encoder_info['mapping']).fillna(0).astype(int)
                    else:
                        le = encoder_info['encoder']
                        encoded_vals = []
                        for val in df_chunk[col].fillna('missing').astype(str):
                            try:
                                encoded_vals.append(le.transform([val])[0])
                            except ValueError:
                                encoded_vals.append(0)
                        df_chunk[f'{col}_encoded'] = encoded_vals
                
                df_chunk.to_parquet(chunk_file, index=False)
            
            logger.info(f"Encoded {len(all_chunks)} chunks")
            
            df_sample = pd.read_parquet(first_chunk)
            exclude_cols = ['store', 'dept', 'date', 'event_timestamp', target_col] + categorical_cols
            feature_cols = [col for col in df_sample.columns if col not in exclude_cols]
            
            # Handle NaN and inf values before fitting scaler
            X_sample = df_sample[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype(np.float32)
            scaler = StandardScaler()
            scaler.fit(X_sample)
            
            # Fit target scaler (CRITICAL for reasonable loss values)
            y_sample = df_sample[target_col].values.reshape(-1, 1).astype(np.float32)
            target_scaler = StandardScaler()
            target_scaler.fit(y_sample)
            logger.info(f"Target scaling: mean={target_scaler.mean_[0]:.2f}, std={target_scaler.scale_[0]:.2f}")
            
            metadata = {
                'feature_cols': feature_cols,
                'target_col': target_col,
                'categorical_cols': categorical_cols,
            }
            joblib.dump(metadata, f"{cache_dir}/metadata.pkl")
            joblib.dump(scaler, f"{cache_dir}/scaler.pkl")
            joblib.dump(target_scaler, f"{cache_dir}/target_scaler.pkl")
            logger.info(f"Fitted scaler on {len(X_sample):,} samples with {len(feature_cols)} features")
        else:
            logger.info(f"[Rank {rank}/{world_size}] Waiting for Rank 0 to complete feature preparation...")
        
        if dist.is_initialized():
            logger.info(f"[Rank {rank}] Synchronizing at barrier...")
            # Don't use device_ids for multi-node - causes "invalid device ordinal"
            dist.barrier()
            logger.info(f"[Rank {rank}] Barrier passed - loading artifacts...")
        
        metadata = joblib.load(f"{cache_dir}/metadata.pkl")
        scaler = joblib.load(f"{cache_dir}/scaler.pkl")
        target_scaler = joblib.load(f"{cache_dir}/target_scaler.pkl")
        logger.info(f"[Rank {rank}] Loaded metadata and scaler")
        
        return metadata['feature_cols'], metadata['target_col'], scaler, target_scaler
    
    def train_model(model, train_loader, val_loader, optimizer, epochs, snapshot_path, 
                    use_amp, grad_clip_norm, device, backend, early_stopping_patience=None,
                    checkpoint_every=None):
        """Training loop with AMP, checkpointing, and early stopping."""
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        rank = dist.get_rank()
        
        use_amp = use_amp and torch.cuda.is_available() and backend == "nccl"
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            batch_count = 0
            
            # CRITICAL: Track batches to detect DDP synchronization issues
            max_batches_per_epoch = 100000  # Safety limit
            
            for batch_idx, (source, targets) in enumerate(train_loader):
                if batch_idx >= max_batches_per_epoch:
                    logger.warning(f"[Rank {rank}] Hit max batch limit {max_batches_per_epoch}, stopping epoch")
                    break
                source = source.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        output = model(source).squeeze()
                        loss = criterion(output, targets)
                    scaler.scale(loss).backward()
                    if grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output = model(source).squeeze()
                    loss = criterion(output, targets)
                    loss.backward()
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                # Log from all ranks, but less frequently for non-rank-0
                if batch_idx % 50 == 0:
                    if rank == 0:
                        logger.info(f"[Rank {rank}/{dist.get_world_size()-1}] Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
                    elif batch_idx % 200 == 0:  # Worker ranks log every 200 batches
                        logger.info(f"[Rank {rank}/{dist.get_world_size()-1}] Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
            logger.info(f"[Rank {rank}] Epoch {epoch} training complete, {batch_count} batches processed")
            
            # Aggregate training loss across all ranks
            if dist.is_initialized():
                avg_loss_tensor = torch.tensor(avg_loss, device=device)
                dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
                avg_loss = avg_loss_tensor.item()
            logger.info(f"[Rank {rank}] Training loss aggregated: {avg_loss:.4f}")
            
            if val_loader is not None:
                logger.info(f"[Rank {rank}] Starting validation...")
                model.eval()
                val_loss = 0.0
                val_count = 0
                with torch.no_grad():
                    for val_batch_idx, (source, targets) in enumerate(val_loader):
                        if val_batch_idx % 50 == 0 and rank == 0:
                            logger.info(f"[Rank {rank}] Validation batch {val_batch_idx}")
                        source = source.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                        output = model(source).squeeze()
                        loss = criterion(output, targets)
                        val_loss += loss.item()
                        val_count += 1
                
                val_loss = val_loss / val_count if val_count > 0 else 0.0
                logger.info(f"[Rank {rank}] Validation complete, {val_count} batches processed")
                
                # Aggregate validation loss across all ranks
                if dist.is_initialized():
                    logger.info(f"[Rank {rank}] Aggregating validation loss across ranks...")
                    val_loss_tensor = torch.tensor(val_loss, device=device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                    val_loss = val_loss_tensor.item()
                    logger.info(f"[Rank {rank}] Validation loss aggregated: {val_loss:.4f}")
                
                if rank == 0:
                    logger.info(f"[Rank 0] Epoch {epoch} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
                
                # Early stopping logic - check on all ranks
                should_stop = False
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    if rank == 0:
                        model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                        torch.save({
                            "MODEL_STATE": model_state,
                            "OPTIMIZER_STATE": optimizer.state_dict(),
                            "EPOCH": epoch,
                            "VAL_LOSS": val_loss,
                        }, snapshot_path)
                        logger.info(f"Saved best model (val_loss={val_loss:.4f})")
                else:
                    epochs_without_improvement += 1
                    if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                        should_stop = True
                        if rank == 0:
                            logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
                
                # Periodic checkpoints (rank 0 only)
                if rank == 0 and checkpoint_every and (epoch + 1) % checkpoint_every == 0:
                    checkpoint_path = snapshot_path.replace('.pt', f'_epoch{epoch+1}.pt')
                    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                    torch.save({
                        "MODEL_STATE": model_state,
                        "OPTIMIZER_STATE": optimizer.state_dict(),
                        "EPOCH": epoch,
                        "VAL_LOSS": val_loss,
                    }, checkpoint_path)
                    logger.info(f"Periodic checkpoint saved: epoch {epoch+1}")
                
                # Synchronize early stopping across all ranks
                if dist.is_initialized():
                    should_stop_tensor = torch.tensor(int(should_stop), device=device)
                    dist.all_reduce(should_stop_tensor, op=dist.ReduceOp.MAX)
                    should_stop = bool(should_stop_tensor.item())
                
                if should_stop:
                    break
            else:
                # No validation loader - just log training loss and save periodically
                if rank == 0:
                    logger.info(f"[Rank 0] Epoch {epoch} | Train Loss: {avg_loss:.4f}")
                    
                    # Save model periodically when no validation
                    if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
                        checkpoint_path = snapshot_path.replace('.pt', f'_epoch{epoch+1}.pt')
                        model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                        torch.save({
                            "MODEL_STATE": model_state,
                            "OPTIMIZER_STATE": optimizer.state_dict(),
                            "EPOCH": epoch,
                        }, checkpoint_path)
                        logger.info(f"Checkpoint saved: epoch {epoch+1}")
        
        # Save final model if no validation was used
        if val_loader is None and rank == 0:
            model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            torch.save({
                "MODEL_STATE": model_state,
                "OPTIMIZER_STATE": optimizer.state_dict(),
                "EPOCH": epochs - 1,
            }, snapshot_path)
            logger.info(f"Saved final model after {epochs} epochs")
        
        return best_val_loss
    
    logger.info("=" * 70)
    logger.info("STARTING DISTRIBUTED TRAINING")
    logger.info("=" * 70)
    
    if parameters is None:
        parameters = {}
    
    # Data loading configuration
    # Options: 
    #   "feast_ray"    - Feast with Ray offline store (distributed, complex)
    #   "feast_file"   - Feast with File offline store (simple, no Ray/PostgreSQL)
    #   "direct"       - Direct parquet loading without Feast (simplest)
    data_source = parameters.get("data_source", "feast_ray")
    data_path = parameters.get("data_path", "/shared/feature_repo/data")  # For feast_file/direct
    feast_repo_path = parameters.get("feast_repo_path", "/shared/feature_repo")  # For feast_ray
    
    # Training configuration
    model_output_dir = parameters.get("model_output_dir", "/shared/models")
    epochs = parameters.get("num_epochs", parameters.get("epochs", 50))
    batch_size = parameters.get("batch_size", 256)
    learning_rate = parameters.get("learning_rate", 0.001)
    weight_decay = parameters.get("weight_decay", 0.0001)
    hidden_dims = parameters.get("hidden_dims", [256, 128, 64, 32])
    dropout = parameters.get("dropout", 0.3)
    chunk_size = parameters.get("chunk_size", 50000)
    sample_size = parameters.get("sample_size", None)
    val_size = parameters.get("val_size", 0.2)
    backend = parameters.get("backend", "nccl")
    use_amp = parameters.get("use_amp", True)
    grad_clip_norm = parameters.get("grad_clip_norm", 1.0)
    early_stopping_patience = parameters.get("early_stopping_patience", None)
    checkpoint_every = parameters.get("checkpoint_every", None)
    
    # Input validation
    if data_source not in ["feast_ray", "feast_file", "direct"]:
        raise ValueError(f"data_source must be 'feast_ray', 'feast_file', or 'direct', got {data_source}")
    if data_source in ["feast_ray", "feast_file"] and not os.path.exists(feast_repo_path):
        raise ValueError(f"Feast repo not found: {feast_repo_path}")
    if data_source in ["feast_file", "direct"] and not os.path.exists(data_path):
        raise ValueError(f"Data path not found: {data_path}")
    if not 0.0 < val_size < 1.0:
        raise ValueError(f"val_size must be between 0 and 1, got {val_size}")
    
    logger.info(f"Config: epochs={epochs}, batch={batch_size}, lr={learning_rate}")
    logger.info(f"Data source: {data_source}")
    if data_source in ["feast_ray", "feast_file"]:
        logger.info(f"Feast repo: {feast_repo_path}")
    if data_source in ["feast_file", "direct"]:
        logger.info(f"Data path: {data_path}")
    logger.info(f"Output dir: {model_output_dir}")
    logger.info(f"Train/Val split: {1-val_size:.0%}/{val_size:.0%}")
    
    backend, device_type = ddp_setup(backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    cache_dir = Path(model_output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = str(cache_dir / f"{data_source}_chunks")
    snapshot_path = str(cache_dir / "best_model.pt")
    
    # Load data based on source
    if data_source == "feast_ray":
        # Option 1: Feast with Ray offline store (PostgreSQL registry + Ray)
        load_features_chunked(feast_repo_path, chunks_dir, chunk_size, sample_size)
    elif data_source == "feast_file":
        # Option 2: Feast with file offline store (simple parquet, no Ray/PostgreSQL)
        load_features_feast_file(feast_repo_path, data_path, chunks_dir, chunk_size, sample_size)
    elif data_source == "direct":
        # Option 3: Direct parquet loading without Feast
        load_features_direct(data_path, chunks_dir, chunk_size, sample_size)
    
    feature_cols, target_col, scaler, target_scaler = prepare_features(chunks_dir, str(cache_dir))
    
    logger.info(f"Features: {len(feature_cols)} total (base + on-demand transformations)")
    
    all_chunk_files = sorted(glob.glob(f"{chunks_dir}/chunk_*.parquet"))
    split_idx = int(len(all_chunk_files) * (1 - val_size))
    train_chunks = all_chunk_files[:split_idx]
    val_chunks = all_chunk_files[split_idx:]
    
    # CRITICAL: Ensure even distribution for DDP - trim to multiple of world_size
    train_chunks = train_chunks[:len(train_chunks) // world_size * world_size]
    val_chunks = val_chunks[:len(val_chunks) // world_size * world_size]
    
    logger.info(f"Data: {len(train_chunks)} train chunks, {len(val_chunks)} val chunks (trimmed for even DDP distribution)")
    
    train_dataset = StreamingSalesDataset(
        cache_dir=chunks_dir,
        feature_cols=feature_cols,
        target_col=target_col,
        scaler=scaler,
        target_scaler=target_scaler,
        rank=rank,
        world_size=world_size,
        chunk_files=train_chunks,
        shuffle=True,
        seed=42,
    )
    
    val_dataset = StreamingSalesDataset(
        cache_dir=chunks_dir,
        feature_cols=feature_cols,
        target_col=target_col,
        scaler=scaler,
        target_scaler=target_scaler,
        rank=rank,
        world_size=world_size,
        chunk_files=val_chunks,
        shuffle=False,
        seed=42,
    )
    
    # Setup device based on detected hardware
    use_gpu = device_type in ['cuda', 'hip']
    # CRITICAL: num_workers=0 to avoid DataLoader worker deadlock with IterableDataset
    # DDP already provides parallelism across ranks/GPUs
    num_workers = 0
    
    # AMP only supported on NVIDIA GPUs (not AMD ROCm or CPU)
    use_amp = use_amp and device_type == 'cuda'
    if parameters.get("use_amp", False) and device_type != 'cuda':
        logger.info(f"AMP requested but not supported on {device_type.upper()}, disabling")
    
    DataLoaderClass = StatefulDataLoader if TORCHDATA_AVAILABLE else DataLoader
    # Simple DataLoader config with num_workers=0 (no multiprocessing, DDP handles parallelism)
    train_loader = DataLoaderClass(
        train_dataset,
        batch_size=batch_size,
        pin_memory=use_gpu,
    )
    
    val_loader = DataLoaderClass(
        val_dataset,
        batch_size=batch_size,
        pin_memory=use_gpu,
    )
    
    input_dim = len(feature_cols)
    model = SalesForecastingMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    
    # Setup device and DDP
    # CRITICAL: Use global rank % GPU count (set in ddp_setup) to avoid duplicate GPU error
    if use_gpu:
        # Get the device ID that was already set in ddp_setup
        current_device = torch.cuda.current_device()
        if device_type == 'cuda':
            device = torch.device(f'cuda:{current_device}')
        else:  # hip (AMD ROCm)
            device = torch.device(f'cuda:{current_device}')  # ROCm uses cuda API
        model = DDP(model.to(device), device_ids=[current_device])
    else:
        device = torch.device('cpu')
        model = DDP(model.to(device))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    logger.info(f"[Global Rank {rank}/{world_size-1}] [Local Rank {local_rank}] Model: {input_dim} -> {hidden_dims} -> 1")
    logger.info(f"[Global Rank {rank}/{world_size-1}] Device: {device} ({device_type.upper()}), AMP: {use_amp}")
    if early_stopping_patience and rank == 0:
        logger.info(f"Early stopping: patience={early_stopping_patience}")
    if checkpoint_every and rank == 0:
        logger.info(f"Periodic checkpoints: every {checkpoint_every} epochs")
    
    best_loss = train_model(
        model, train_loader, val_loader, optimizer, epochs, snapshot_path,
        use_amp, grad_clip_norm, device, backend, early_stopping_patience,
        checkpoint_every
    )
    
    # Final validation test
    if rank == 0 and val_loader is not None:
        logger.info("=" * 70)
        logger.info("FINAL VALIDATION TEST")
        logger.info("=" * 70)
        
        model.eval()
        final_val_loss = 0.0
        final_val_count = 0
        with torch.no_grad():
            for source, targets in val_loader:
                source = source.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                output = model(source).squeeze()
                loss = nn.MSELoss()(output, targets)
                final_val_loss += loss.item()
                final_val_count += 1
        
        final_val_loss = final_val_loss / final_val_count if final_val_count > 0 else 0.0
        logger.info(f"[Rank 0] Final Validation Loss: {final_val_loss:.4f}")
    
    if rank == 0:
        logger.info("=" * 70)
        logger.info(f"TRAINING COMPLETE | Best Val Loss: {best_loss:.4f}")
        logger.info(f"Model saved to: {snapshot_path}")
        logger.info(f"Feature scaler: {cache_dir}/scaler.pkl")
        logger.info(f"Target scaler: {cache_dir}/target_scaler.pkl")
        logger.info("=" * 70)
    
    dist.destroy_process_group()


__all__ = ['training_func']


if __name__ == "__main__":
    print("Testing self-contained training function...")
    print("Note: This requires Feast setup and distributed environment.")
    
    params = {
        'num_epochs': 5,
        'batch_size': 128,
        'sample_size': 10000,
    }
    
    training_func(params)
