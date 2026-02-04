#!/usr/bin/env python3
"""Custom Flask server for sales forecasting model"""
import os, json, torch, torch.nn as nn, joblib, numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
model, scalers, feature_cols = None, None, None


class SalesMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def load_model():
    global model, scalers, feature_cols
    model_dir = os.getenv("MODEL_DIR", "/mnt/models")
    with open(f"{model_dir}/model_metadata.json") as f:
        meta = json.load(f)
    model = SalesMLP(meta["input_dim"])
    model.load_state_dict(torch.load(f"{model_dir}/best_model.pt", map_location="cpu", weights_only=True))
    model.eval()
    scalers = joblib.load(f"{model_dir}/scalers.joblib")
    feature_cols = meta["feature_columns"]
    print(f"Model loaded: {len(feature_cols)} features: {feature_cols}")


def parse_instances(instances):
    """Convert instances to numpy array - supports both list and dict formats"""
    result = []
    for inst in instances:
        if isinstance(inst, dict):
            # Named features: {"lag_1": 25000, "temperature": 65.0, ...}
            result.append([inst.get(col, 0) for col in feature_cols])
        else:
            # Array format: [25000, 24000, ...]
            result.append(inst)
    return np.array(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": "sales-forecast"})


@app.route("/v1/models/sales-forecast", methods=["GET"])
def model_info():
    return jsonify({
        "name": "sales-forecast", 
        "version": "1", 
        "input_dim": len(feature_cols), 
        "features": feature_cols,
        "example": {col: 0 for col in feature_cols}
    })


@app.route("/v1/models/sales-forecast:predict", methods=["POST"])
def predict():
    instances = request.json.get("instances", [])
    X = parse_instances(instances)
    X_scaled = scalers["scaler_X"].transform(X)
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_scaled)).numpy()
    return jsonify({"predictions": scalers["scaler_y"].inverse_transform(preds.reshape(-1, 1)).flatten().tolist()})


@app.route("/v1/models/sales-forecast:explain", methods=["POST"])
def explain():
    weights = model.net[0].weight.abs().mean(dim=0).detach().numpy()
    importance = {f: float(w)/weights.sum() for f, w in zip(feature_cols, weights)}
    return jsonify({"feature_importance": dict(sorted(importance.items(), key=lambda x: -x[1]))})


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=8080)
