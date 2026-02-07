#!/usr/bin/env python3
"""
Custom Flask server for sales forecasting model with Feast online feature support.

Architecture:
- Training: Feast get_historical_features() with Ray (distributed PIT joins)
- Inference: Feast get_online_features() from PostgreSQL (low-latency, no Ray)

Endpoints:
- GET  /health                              - Health check
- GET  /v1/models/sales-forecast           - Model info
- POST /v1/models/sales-forecast:predict   - Predict (raw features or Feast entities)
- POST /v1/models/sales-forecast:explain   - Feature importance
"""
import os
import json
import logging
import torch
import torch.nn as nn
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Flask app
from flask import Flask, request, jsonify
app = Flask(__name__)

# Global model state
model = None
scaler_X = None
scaler_y = None
feature_cols = None
feast_store = None
metadata = None


class SalesMLP(nn.Module):
    """Sales forecasting MLP model (must match training architecture)"""
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
        
    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_model():
    """Load model, scalers, and optionally Feast store"""
    global model, scaler_X, scaler_y, feature_cols, feast_store, metadata
    
    model_dir = os.getenv("MODEL_DIR", "/mnt/models")
    feature_repo = os.getenv("FEATURE_REPO", "/mnt/feature_repo")
    
    logger.info(f"Loading model from {model_dir}")
    
    # Load model metadata
    with open(f"{model_dir}/model_metadata.json") as f:
        metadata = json.load(f)
    
    input_dim = metadata["input_dim"]
    hidden_dims = metadata.get("hidden_dims", [256, 128, 64])
    dropout = metadata.get("dropout", 0.2)
    feature_cols = metadata["feature_columns"]
    
    logger.info(f"Model: input_dim={input_dim}, hidden_dims={hidden_dims}, features={len(feature_cols)}")
    
    # Load model weights
    model = SalesMLP(input_dim, hidden_dims, dropout)
    model.load_state_dict(torch.load(f"{model_dir}/best_model.pt", map_location="cpu", weights_only=True))
    model.eval()
    logger.info("Model weights loaded")
    
    # Load scalers - support both old format (scalers.joblib) and new format (scaler_X.pkl, scaler_y.pkl)
    import joblib
    
    if os.path.exists(f"{model_dir}/scalers.joblib"):
        # Old format: combined scalers dict
        scalers = joblib.load(f"{model_dir}/scalers.joblib")
        scaler_X = scalers["scaler_X"]
        scaler_y = scalers["scaler_y"]
        logger.info("Loaded scalers from scalers.joblib")
    elif os.path.exists(f"{model_dir}/scaler_X.pkl"):
        # New format: separate files
        scaler_X = joblib.load(f"{model_dir}/scaler_X.pkl")
        scaler_y = joblib.load(f"{model_dir}/scaler_y.pkl")
        logger.info("Loaded scalers from scaler_X.pkl, scaler_y.pkl")
    elif os.path.exists(f"{model_dir}/scaler.pkl"):
        # Alternative format from manifest training
        scaler_X = joblib.load(f"{model_dir}/scaler.pkl")
        scaler_y = joblib.load(f"{model_dir}/y_scaler.pkl")
        logger.info("Loaded scalers from scaler.pkl, y_scaler.pkl")
    else:
        logger.warning("No scalers found - predictions will be unscaled!")
        scaler_X = None
        scaler_y = None
    
    # Load Feast store for online features (optional)
    if os.path.exists(f"{feature_repo}/feature_store.yaml"):
        try:
            from feast import FeatureStore
            feast_store = FeatureStore(repo_path=feature_repo)
            logger.info(f"Feast store loaded from {feature_repo}")
            logger.info(f"  Entities: {[e.name for e in feast_store.list_entities()]}")
            logger.info(f"  FeatureViews: {[fv.name for fv in feast_store.list_feature_views()]}")
        except Exception as e:
            logger.warning(f"Feast store not available: {e}")
            feast_store = None
    else:
        logger.info(f"Feast repo not found at {feature_repo} - online features disabled")
        feast_store = None
    
    logger.info("=" * 50)
    logger.info("SERVER READY")
    logger.info("=" * 50)
    logger.info(f"  Features: {feature_cols}")
    logger.info(f"  Feast online: {'enabled' if feast_store else 'disabled'}")


def parse_instances(instances):
    """Convert instances to numpy array - supports both list and dict formats"""
    result = []
    for inst in instances:
        if isinstance(inst, dict):
            # Named features: {"lag_1": 25000, "temperature": 65.0, ...}
            result.append([float(inst.get(col, 0)) for col in feature_cols])
        elif isinstance(inst, list):
            # Array format: [25000, 24000, ...]
            result.append([float(x) for x in inst])
        else:
            raise ValueError(f"Unsupported instance format: {type(inst)}")
    return np.array(result, dtype=np.float32)


def get_feast_features(entities):
    """
    Fetch features from Feast online store.
    
    This is the key integration point - Feast online store (PostgreSQL)
    provides low-latency feature serving without Ray.
    
    Args:
        entities: List of entity dicts, e.g., [{"store_id": 1, "dept_id": 1}, ...]
    
    Returns:
        numpy array of features aligned with feature_cols
    """
    if feast_store is None:
        raise RuntimeError("Feast store not configured")
    
    # Define features to fetch (must match what was used in training)
    # These are the inference features from the Feast feature service
    feast_features = metadata.get("feast_features", [
        "sales_features:lag_1",
        "sales_features:lag_2",
        "sales_features:lag_4",
        "sales_features:lag_8",
        "sales_features:lag_52",
        "sales_features:rolling_mean_4w",
        "sales_features:temperature",
        "sales_features:fuel_price",
        "sales_features:cpi",
        "sales_features:unemployment",
        "store_features:store_size",
    ])
    
    logger.info(f"Fetching online features for {len(entities)} entities")
    
    # Call Feast get_online_features - this queries PostgreSQL, NOT Ray
    online_features = feast_store.get_online_features(
        features=feast_features,
        entity_rows=entities
    ).to_dict()
    
    # Convert to numpy array matching feature_cols order
    result = []
    for i in range(len(entities)):
        row = []
        for col in feature_cols:
            # Map feature column name to Feast feature name
            # e.g., "lag_1" -> "sales_features:lag_1" or just "lag_1"
            value = None
            
            # Try exact match first
            if col in online_features:
                value = online_features[col][i]
            else:
                # Try with prefix
                for feast_col in online_features:
                    if feast_col.endswith(f":{col}") or feast_col == col:
                        value = online_features[feast_col][i]
                        break
            
            row.append(float(value) if value is not None else 0.0)
        result.append(row)
    
    return np.array(result, dtype=np.float32)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "model": "sales-forecast",
        "feast_online": feast_store is not None
    })


@app.route("/v1/models/sales-forecast", methods=["GET"])
def model_info():
    """Model metadata endpoint"""
    return jsonify({
        "name": "sales-forecast",
        "version": "1",
        "input_dim": len(feature_cols) if feature_cols else 0,
        "feature_columns": feature_cols or [],
        "hidden_dims": metadata.get("hidden_dims", [256, 128, 64]) if metadata else [],
        "feast_online_enabled": feast_store is not None,
        "example_raw": {col: 0.0 for col in (feature_cols or [])},
        "example_feast": {"store_id": 1, "dept_id": 1}
    })


@app.route("/v1/models/sales-forecast:predict", methods=["POST"])
def predict():
    """
    Prediction endpoint - supports two modes:
    
    1. Raw features (instances):
       {"instances": [{"lag_1": 25000, "lag_2": 24000, ...}]}
       
    2. Feast online features (entities):
       {"entities": [{"store_id": 1, "dept_id": 1}], "use_feast": true}
    """
    try:
        data = request.json
        
        # Mode 1: Feast online features
        if data.get("use_feast") and data.get("entities"):
            if feast_store is None:
                return jsonify({
                    "error": "Feast online store not configured",
                    "hint": "Deploy with FEATURE_REPO pointing to Feast repo with feature_store.yaml"
                }), 400
            
            entities = data["entities"]
            logger.info(f"Feast mode: {len(entities)} entities")
            
            # Fetch features from Feast online store (PostgreSQL)
            X = get_feast_features(entities)
            
        # Mode 2: Raw features
        elif data.get("instances"):
            instances = data["instances"]
            logger.info(f"Raw mode: {len(instances)} instances")
            X = parse_instances(instances)
            
        else:
            return jsonify({
                "error": "Invalid request",
                "usage": {
                    "raw_features": {"instances": [{"lag_1": 25000, ...}]},
                    "feast_online": {"entities": [{"store_id": 1, "dept_id": 1}], "use_feast": True}
                }
            }), 400
        
        # Scale features
        if scaler_X is not None:
            X_scaled = scaler_X.transform(X)
        else:
            X_scaled = X
        
        # Predict
        with torch.no_grad():
            preds_scaled = model(torch.FloatTensor(X_scaled)).numpy()
        
        # Inverse scale predictions
        if scaler_y is not None:
            preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        else:
            preds = preds_scaled
        
        return jsonify({
            "predictions": preds.tolist(),
            "mode": "feast_online" if data.get("use_feast") else "raw_features"
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/v1/models/sales-forecast:explain", methods=["POST"])
def explain():
    """Feature importance based on first layer weights"""
    try:
        weights = model.net[0].weight.abs().mean(dim=0).detach().numpy()
        importance = {f: float(w) / weights.sum() for f, w in zip(feature_cols, weights)}
        return jsonify({
            "feature_importance": dict(sorted(importance.items(), key=lambda x: -x[1]))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_model()
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port)
