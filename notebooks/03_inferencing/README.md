# Model Serving with KServe

## Workflow

![Inference Workflow](../../docs/diagrams/03-inference-workflow.png)

This directory contains the inference notebook for deploying the trained model as a KServe InferenceService.

## Overview

| File | Description |
|------|-------------|
| `03-inference.ipynb` | Deploys model and tests predictions |
| `serving_script.py` | KServe Model server implementation |

## Architecture

```
┌─────────────┐     ┌─────────────────────────────────────┐     ┌─────────────┐
│   Client    │     │        KServe InferenceService      │     │   Feast     │
│             │────▶│  ┌─────────────────────────────┐    │────▶│   Online    │
│  {store_id, │     │  │  1. Receive entity IDs      │    │     │   Store     │
│   dept_id}  │     │  │  2. Fetch features (gRPC)   │    │     │   (Redis)   │
│             │◀────│  │  3. Scale + Inference       │    │     │             │
│ {prediction}│     │  │  4. Return prediction       │    │     │             │
└─────────────┘     │  └─────────────────────────────┘    │     └─────────────┘
                    └─────────────────────────────────────┘
```

## Prerequisites

| Requirement | Details |
|-------------|---------|
| Model | Training completed (`02-training.ipynb`) |
| Artifacts | `/shared/models/` contains `best_model.pt`, `scalers.joblib`, etc. |
| Feast | Online store materialized with features |
| KServe | KServe operator installed on cluster |

## What the Notebook Does

| Step | Action | Component |
|------|--------|-----------|
| 1 | Load serving script | From `serving_script.py` |
| 2 | Create ConfigMap | Store serving script |
| 3 | Deploy InferenceService | KServe |
| 4 | Test predictions | V2 Inference Protocol |

## Feast Integration

![Feature Store for Inference](../../docs/images/FeatureServices.png)

The serving script (`serving_script.py`) uses **Feast SDK directly** (not REST API):

- Mounts `feast-salesforecasting-client` ConfigMap
- Mounts `feast-salesforecasting-client-ca` for TLS
- Calls `store.get_online_features()` via gRPC

This matches how training retrieves features, ensuring consistency.

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MODEL_NAME` | `sales-forecast` | InferenceService name |
| `MODEL_DIR` | `/shared/models` | Model artifacts path |
| `FEAST_CONFIG_PATH` | `/opt/app-root/src/feast-config/salesforecasting` | Feast config |
| `TRAINER_IMAGE` | `quay.io/modh/training:py312-cuda128-torch280` | Container image |

## InferenceService Spec

| Setting | Value | Purpose |
|---------|-------|---------|
| `minReplicas` | 1 | Always available |
| `maxReplicas` | 3 | Auto-scale under load |
| `deploymentMode` | RawDeployment | Standard K8s deployment |
| `readinessProbe` | `/v2/health/ready` | Health check |

## Volume Mounts

| Volume | Mount Path | Source |
|--------|------------|--------|
| `model-storage` | `/shared` | PVC `shared` |
| `feast-config` | `/opt/app-root/src/feast-config/salesforecasting` | ConfigMap |
| `feast-ca` | `/etc/pki/tls/custom-certs` | CA ConfigMap |
| `scripts` | `/scripts` | Serving script ConfigMap |

## Making Predictions

### V2 Inference Protocol

Request format:
```json
{
  "inputs": [
    {
      "name": "entities",
      "shape": [2],
      "datatype": "BYTES",
      "data": [
        {"store_id": 1, "dept_id": 3},
        {"store_id": 10, "dept_id": 5}
      ]
    }
  ]
}
```

Response format:
```json
{
  "outputs": [
    {
      "name": "predictions",
      "shape": [2],
      "datatype": "FP32",
      "data": [96763.45, 84521.30]
    }
  ]
}
```

### Using curl

```bash
ENDPOINT="http://sales-forecast-predictor.feast-trainer-demo.svc.cluster.local:8080"

curl -X POST "$ENDPOINT/v2/models/sales-forecast/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "entities",
      "shape": [1],
      "datatype": "BYTES",
      "data": [{"store_id": 1, "dept_id": 3}]
    }]
  }'
```

## Running the Notebook

1. Ensure training completed and artifacts exist in `/shared/models/`
2. Open `03-inference.ipynb` in JupyterLab
3. Run cells sequentially
4. Wait for InferenceService to be ready (~2 minutes first time)
5. Test predictions

## Troubleshooting

### InferenceService Not Ready

Check pod logs:
```bash
oc logs -l app=sales-forecasting -n feast-trainer-demo
```

Common issues:
- Missing model files in `/shared/models/`
- Feast ConfigMap not mounted
- pip install failing (network issues)

### Feast Connection Failed

Verify ConfigMaps:
```bash
oc get configmap feast-salesforecasting-client -n feast-trainer-demo -o yaml
```

### Model Load Error

Check artifacts exist:
```bash
ls -la /shared/models/
```

Expected files:
- `best_model.pt`
- `scalers.joblib`
- `feature_cols.pkl`
- `model_metadata.json`

### Prediction Returns Null Features

Ensure online store is materialized:
```bash
# Check Feast online store pod
oc get pods -l feast.dev/service-type=online -n feast-trainer-demo
```

## Performance

| Metric | Expected |
|--------|----------|
| Cold start | ~60-90 seconds |
| Warm inference | <100ms |
| Feast lookup | <50ms |
| Batch (16 entities) | <200ms |

## Next Steps

- Monitor with Prometheus/Grafana
- Add Istio for traffic management
- Configure autoscaling policies
- Set up A/B testing with traffic splitting
