"""Self-contained worker function submitted by Kubeflow's TransformersTrainer."""


def train_fn(epochs, use_ray, ray_cluster, namespace, output_dir, feature_repo, checkpoint_dir):
    """Retrieve Feast features, run the backtest, and train the forecast MLP."""
    import json
    import os
    import shutil
    import sys
    import tempfile
    import time
    from contextlib import nullcontext
    from pathlib import Path

    import joblib
    import numpy as np
    import pandas as pd
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import sklearn
    from sklearn.preprocessing import StandardScaler
    from transformers import Trainer, TrainingArguments, __version__ as transformers_version

    def signed_log1p(values):
        return np.sign(values) * np.log1p(np.abs(values))

    def signed_expm1(values):
        return np.sign(values) * np.expm1(np.abs(values))

    seed = int(os.environ["RANDOM_SEED"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    out = Path(output_dir)
    repo = Path(feature_repo)
    sys.path.insert(0, str(repo))
    from forecast_contract import MODEL_FEATURE_COLUMNS, chronological_split

    sys.path.insert(0, os.path.join(os.environ["POD_SHARED_ROOT"], "scripts"))
    from evaluate_forecasts import evaluate_forecasts

    hidden_dims = [int(value) for value in os.environ["MODEL_HIDDEN_DIMS"].split(",")]
    dropout = float(os.environ["MODEL_DROPOUT"])
    batch_size = int(os.environ["TRAINING_BATCH_SIZE"])
    learning_rate = float(os.environ["TRAINING_LEARNING_RATE"])
    min_mape_target = float(os.environ["TRAINING_MAPE_MIN_TARGET"])

    class SalesMLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            layers = []
            width = input_dim
            for next_width in hidden_dims:
                layers.extend(
                    [nn.Linear(width, next_width), nn.BatchNorm1d(next_width), nn.ReLU(), nn.Dropout(dropout)]
                )
                width = next_width
            layers.append(nn.Linear(width, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, features, labels=None):
            logits = self.net(features).squeeze(-1)
            loss = nn.functional.mse_loss(logits, labels) if labels is not None else None
            return {"loss": loss, "logits": logits}

    class SalesDataset(torch.utils.data.Dataset):
        def __init__(self, features, labels):
            self.features = torch.tensor(features, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.float32)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, index):
            return {"features": self.features[index], "labels": self.labels[index]}

    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)

        # Load database credentials from the mounted Secret, never from TrainJob metadata.
        secret_dir = Path("/var/run/secrets/feast-db")
        for key in ("POSTGRES_URL", "POSTGRES_PY_URL", "POSTGRES_PASSWORD"):
            secret_file = secret_dir / key
            if secret_file.is_file():
                os.environ[key] = secret_file.read_text().strip()

        if use_ray:
            os.environ["FEAST_RAY_USE_KUBERAY"] = "true"
            os.environ["FEAST_RAY_CLUSTER_NAME"] = ray_cluster
            os.environ["FEAST_RAY_NAMESPACE"] = namespace
            os.environ["FEAST_RAY_SKIP_TLS"] = "false"
            token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
            if os.path.exists(token_path):
                os.environ["FEAST_RAY_AUTH_TOKEN"] = Path(token_path).read_text().strip()
                os.environ["FEAST_RAY_AUTH_SERVER"] = (
                    f"https://{os.environ['KUBERNETES_SERVICE_HOST']}:"
                    f"{os.environ['KUBERNETES_SERVICE_PORT']}"
                )
        ray_config = repo / "feature_store_ray.yaml"
        if use_ray:
            if not ray_config.is_file():
                raise FileNotFoundError(f"Ray Feast configuration is missing: {ray_config}")
            # Use a private per-job config directory. Never overwrite the
            # shared Feast repo config: concurrent jobs and Workbench
            # notebooks must be able to use the file-based store safely.
            config_context = tempfile.TemporaryDirectory(prefix="feast-ray-config-")
        else:
            config_context = nullcontext(str(repo))

        with config_context as config_dir:
            if use_ray:
                shutil.copy2(ray_config, Path(config_dir) / "feature_store.yaml")
            from feast import FeatureStore

            store = FeatureStore(repo_path=str(config_dir))
            entity_path = Path(os.environ["FEAST_DATA_ROOT"]) / "sales_features.parquet"
            entity_df = pd.read_parquet(
                entity_path,
                columns=["store_id", "dept_id", "event_timestamp", "weekly_sales"],
            )
            entity_df = entity_df.loc[entity_df["weekly_sales"].notna()].drop(
                columns="weekly_sales"
            )
            if entity_df.empty:
                raise ValueError(f"No labeled training entities found in {entity_path}")
            entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"], utc=True)

            started = time.time()
            frame = store.get_historical_features(
                entity_df=entity_df,
                features=store.get_feature_service("training_features"),
            ).to_df()
        print(f"Feast returned {len(frame):,} rows in {time.time() - started:.1f}s")

        backtest_config = {
            "validation_start_week": int(os.environ["BACKTEST_VALIDATION_START_WEEK"]),
            "validation_end_week": int(os.environ["BACKTEST_VALIDATION_END_WEEK"]),
            "holdout_start_week": int(os.environ["BACKTEST_HOLDOUT_START_WEEK"]),
            "holdout_end_week": int(os.environ["BACKTEST_HOLDOUT_END_WEEK"]),
            "max_iter": int(os.environ["BACKTEST_MAX_ITER"]),
            "min_wape_improvement": float(os.environ["BACKTEST_MIN_WAPE_IMPROVEMENT"]),
            "max_abs_weighted_bias": float(os.environ["BACKTEST_MAX_ABS_WEIGHTED_BIAS"]),
            "random_state": seed,
            "resource_profile": {
                "backtest_executor": "rank-0 CPU",
                "trainer_world_size": world,
                "cpu_request_per_node": os.environ.get("TRAINING_CPU_REQUEST"),
                "cpu_limit_per_node": os.environ.get("TRAINING_CPU_LIMIT"),
                "memory_request_per_node": os.environ.get("TRAINING_MEMORY_REQUEST"),
                "memory_limit_per_node": os.environ.get("TRAINING_MEMORY_LIMIT"),
                "visible_os_cpu_count": os.cpu_count(),
            },
        }
        frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"], utc=True)
        target_weeks = sorted(frame["event_timestamp"].dropna().unique())
        window_weeks = int(os.environ["BACKTEST_HOLDOUT_END_WEEK"])
        if len(target_weeks) < window_weeks:
            raise ValueError(
                f"Backtest needs {window_weeks} recent target weeks; found {len(target_weeks)}"
            )
        window_start = pd.Timestamp(target_weeks[len(target_weeks) - window_weeks])
        evaluation = evaluate_forecasts(
            frame.loc[frame["event_timestamp"] >= window_start].copy(),
            MODEL_FEATURE_COLUMNS,
            backtest_config,
        )
        (out / "rolling_origin_report.json").write_text(json.dumps(evaluation, indent=2))
        print(
            "Backtest candidate gates passed: "
            f"{evaluation['decision']['candidate_gates_passed']}; no model promoted"
        )

        holdout_start_index = (
            len(target_weeks) - window_weeks + backtest_config["holdout_start_week"] - 1
        )
        holdout_start = pd.Timestamp(target_weeks[holdout_start_index])
        frame = frame.loc[frame["event_timestamp"] < holdout_start]
        frame = frame.dropna(subset=["weekly_sales", *MODEL_FEATURE_COLUMNS]).sort_values(
            "event_timestamp"
        )
        train_frame, val_frame = chronological_split(
            frame, float(os.environ["TRAINING_VALIDATION_FRACTION"])
        )
        feature_columns = list(MODEL_FEATURE_COLUMNS)
        x_train = train_frame[feature_columns].fillna(0).to_numpy(dtype=np.float32)
        x_val = val_frame[feature_columns].fillna(0).to_numpy(dtype=np.float32)
        y_train_original = train_frame["weekly_sales"].to_numpy(dtype=np.float64)
        y_val_original = val_frame["weekly_sales"].to_numpy(dtype=np.float64)

        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        x_train = scaler_x.fit_transform(x_train).astype(np.float32)
        x_val = scaler_x.transform(x_val).astype(np.float32)
        y_train = scaler_y.fit_transform(
            signed_log1p(y_train_original).reshape(-1, 1)
        ).reshape(-1).astype(np.float32)
        y_val = scaler_y.transform(
            signed_log1p(y_val_original).reshape(-1, 1)
        ).reshape(-1).astype(np.float32)

        joblib.dump(
            {"scaler_X": scaler_x, "scaler_y": scaler_y, "target_transform": "signed_log1p"},
            out / "scalers.joblib",
        )
        joblib.dump(feature_columns, out / "feature_cols.pkl")
        np.savez(
            out / ".data.npz",
            X_train=x_train,
            y_train=y_train,
            X_val=x_val,
            y_val=y_val,
            y_val_orig=y_val_original,
        )
        np.save(out / ".dim.npy", [x_train.shape[1]])

    if dist.is_initialized():
        dist.barrier()

    data = np.load(out / ".data.npz")
    x_train, y_train = data["X_train"], data["y_train"]
    x_val, y_val, y_val_original = data["X_val"], data["y_val"], data["y_val_orig"]
    input_dim = int(np.load(out / ".dim.npy")[0])
    scaler_y = joblib.load(out / "scalers.joblib")["scaler_y"]

    def compute_metrics(eval_prediction):
        predicted_scaled = np.asarray(eval_prediction.predictions).reshape(-1, 1)
        actual_scaled = np.asarray(eval_prediction.label_ids).reshape(-1, 1)
        predicted = signed_expm1(scaler_y.inverse_transform(predicted_scaled).reshape(-1))
        actual = signed_expm1(scaler_y.inverse_transform(actual_scaled).reshape(-1))
        denominator = max(float(np.abs(actual).sum()), 1e-12)
        mape_mask = np.abs(actual) >= min_mape_target
        if not mape_mask.any():
            raise ValueError("Validation data has no sales values above TRAINING_MAPE_MIN_TARGET")
        mape = float(
            np.mean(np.abs((actual[mape_mask] - predicted[mape_mask]) / actual[mape_mask])) * 100
        )
        return {
            "mape": mape,
            "wape": float(np.abs(actual - predicted).sum() / denominator),
            "weighted_bias": float((predicted - actual).sum() / denominator),
        }

    training_args = TrainingArguments(
        output_dir="/tmp/sales-forecast-checkpoints",
        num_train_epochs=int(epochs),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=float(os.environ["TRAINING_WEIGHT_DECAY"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=int(os.environ.get("TRAINING_LOGGING_STEPS", "100")),
        report_to="none",
        seed=seed,
        dataloader_num_workers=int(os.environ["TRAINING_NUM_WORKERS"]),
        dataloader_drop_last=True,
        remove_unused_columns=False,
        disable_tqdm=True,
    )
    hf_trainer = Trainer(
        model=SalesMLP(input_dim),
        args=training_args,
        train_dataset=SalesDataset(x_train, y_train),
        eval_dataset=SalesDataset(x_val, y_val),
        compute_metrics=compute_metrics,
    )
    hf_trainer.train()

    if hf_trainer.is_world_process_zero():
        unwrapped_model = hf_trainer.accelerator.unwrap_model(hf_trainer.model)
        torch.save(unwrapped_model.state_dict(), out / "best_model.pt")

        history_by_epoch = {}
        for record in hf_trainer.state.log_history:
            if "epoch" not in record:
                continue
            epoch = int(round(float(record["epoch"])))
            row = history_by_epoch.setdefault(
                epoch,
                {"epoch": epoch, "train_loss": None, "val_loss": None, "mape": None, "lr": None},
            )
            if "loss" in record:
                row["train_loss"] = float(record["loss"])
            if "eval_loss" in record:
                row["val_loss"] = float(record["eval_loss"])
            if "eval_mape" in record:
                row["mape"] = float(record["eval_mape"])
            if "learning_rate" in record:
                row["lr"] = float(record["learning_rate"])

        history = [history_by_epoch[key] for key in sorted(history_by_epoch)]
        valid_eval = [row for row in history if row["val_loss"] is not None]
        if not valid_eval:
            raise RuntimeError("Hugging Face Trainer did not record validation metrics")
        best_epoch = min(valid_eval, key=lambda row: row["val_loss"])
        best_val_loss = float(best_epoch["val_loss"])
        best_mape = float(best_epoch["mape"])
        feature_columns = joblib.load(out / "feature_cols.pkl")
        metadata = {
            "model_type": "SalesMLP",
            "training_api": "RHOAI TransformersTrainer + Hugging Face Trainer",
            "training_stack": {
                "torch": str(torch.__version__),
                "transformers": transformers_version,
                "scikit_learn": sklearn.__version__,
                "world_size": world,
            },
            "checkpoint_dir": checkpoint_dir,
            "model_role": "integration_example_not_selected",
            "selection_status": "not_selected",
            "rolling_origin_report": "rolling_origin_report.json",
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "best_mape": best_mape,
            "best_val_loss": best_val_loss,
            "feature_columns": feature_columns,
            "target_transform": "signed_log1p",
        }
        (out / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
        (out / "training_metrics.json").write_text(
            json.dumps(
                {
                    "best_mape": best_mape,
                    "best_val_loss": best_val_loss,
                    "epochs": history,
                    "checkpoint_dir": checkpoint_dir,
                },
                indent=2,
            )
        )
        for temporary_file in (out / ".data.npz", out / ".dim.npy"):
            temporary_file.unlink(missing_ok=True)
        print(f"Training complete; best validation MAPE: {best_mape:.2f}%")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
