#!/usr/bin/env python3
"""
Sales Forecasting Inference Server

Production Pattern:
1. Receives entity keys (store_id, dept_id)
2. Calls Feast Feature Server REST API for online features
3. Runs model inference
4. Returns prediction
"""
import os
import json
import requests
import torch
import torch.nn as nn
import joblib
import numpy as np
from flask import Flask, request, jsonify
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
model, scalers, feature_cols, metadata = None, None, None, None
FEAST_SERVER_URL = os.getenv("FEAST_SERVER_URL", "http://feast-server:6566")


class SalesMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_model():
    global model, scalers, feature_cols, metadata
    model_dir = os.getenv("MODEL_DIR", "/shared/models")

    logger.info(f"Loading model from {model_dir}...")

    with open(f"{model_dir}/model_metadata.json") as f:
        metadata = json.load(f)

    hidden_dims = metadata.get("hidden_dims", [256, 128, 64])
    dropout = metadata.get("dropout", 0.2)
    model = SalesMLP(metadata["input_dim"], hidden_dims, dropout)
    model.load_state_dict(torch.load(f"{model_dir}/best_model.pt", map_location="cpu", weights_only=True))
    model.eval()

    scalers = joblib.load(f"{model_dir}/scalers.joblib")
    feature_cols = metadata.get("feature_columns", joblib.load(f"{model_dir}/feature_cols.pkl"))
    logger.info(f"Model loaded: {len(feature_cols)} features, arch={hidden_dims}")
    logger.info(f"Feast Feature Server: {FEAST_SERVER_URL}")


def get_features_from_feast(entity_rows):
    """Call Feast Feature Server REST API for online features."""
    payload = {
        "feature_service": "inference_features",
        "entities": {
            "store_id": [row["store_id"] for row in entity_rows],
            "dept_id": [row["dept_id"] for row in entity_rows],
        }
    }
    response = requests.post(f"{FEAST_SERVER_URL}/get-online-features", json=payload, timeout=10)
    response.raise_for_status()
    result = response.json()

    feature_names = result.get("metadata", {}).get("feature_names", [])
    results = result.get("results", [])

    features = []
    for i in range(len(entity_rows)):
        row = {}
        for j, name in enumerate(feature_names):
            if name in feature_cols and j < len(results):
                val = results[j].get("values", [None])[i] if i < len(results[j].get("values", [])) else None
                row[name] = val if val is not None else 0
        features.append(row)
    return features


@app.route("/health", methods=["GET"])
def health():
    feast_ok = False
    try:
        feast_ok = requests.get(f"{FEAST_SERVER_URL}/health", timeout=2).status_code == 200
    except:
        pass
    return jsonify({
        "status": "healthy" if feast_ok else "degraded",
        "feast_server": "connected" if feast_ok else "disconnected"
    })


@app.route("/v1/models/sales-forecast", methods=["GET"])
def model_info():
    return jsonify({
        "name": "sales-forecast",
        "features": feature_cols,
        "hidden_dims": metadata.get("hidden_dims"),
        "best_mape": metadata.get("best_mape")
    })


@app.route("/v1/models/sales-forecast:predict", methods=["POST"])
def predict():
    """Predict with full features or entity keys (fetches from Feast)"""
    try:
        instances = request.json.get("instances", [])
        first = instances[0] if instances else {}

        if "store_id" in first and "dept_id" in first and len(first) <= 2:
            # Entity keys only - fetch from Feast
            logger.info(f"Fetching features from Feast for {len(instances)} entities")
            feature_rows = get_features_from_feast(instances)
            X = np.array([[row.get(c, 0) for c in feature_cols] for row in feature_rows])
            source = "feast_feature_server"
        else:
            # Full features provided
            X = np.array([[inst.get(c, 0) for c in feature_cols] for inst in instances])
            source = "request_payload"

        X_scaled = scalers["scaler_X"].transform(X)
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_scaled)).numpy()

        predictions = scalers["scaler_y"].inverse_transform(preds.reshape(-1, 1)).flatten()
        return jsonify({
            "predictions": predictions.tolist(),
            "features_source": source
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/v1/models/sales-forecast:explain", methods=["POST"])
def explain():
    """Return feature importance from first layer weights"""
    weights = model.net[0].weight.abs().mean(dim=0).detach().numpy()
    importance = {f: float(w) / weights.sum() for f, w in zip(feature_cols, weights)}
    return jsonify({
        "feature_importance": dict(sorted(importance.items(), key=lambda x: -x[1]))
    })


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=8080, threaded=True)
