# Sales Demand Forecasting - MLOps Pipeline

End-to-end ML pipeline for retail demand forecasting using **Feast**, **Kubeflow Training**, **KServe**, and **Ray** on OpenShift AI.

## 🎯 Quick Start

```bash
cd examples/complete-mlops-pipeline
```

See [examples/complete-mlops-pipeline/README.md](examples/complete-mlops-pipeline/README.md) for full documentation.

## 📊 Business Value

| Metric | Value | Impact |
|--------|-------|--------|
| **MAPE** | 10.5% | 40% better than industry baseline (15-20%) |
| **Inventory Savings** | 20% | Reduced holding costs |
| **Stockout Reduction** | 15% | Fewer lost sales |
| **Payroll Optimization** | 10-15% | Right staffing levels |

## 🏗️ Architecture

### End-to-End Pipeline

![Architecture](examples/complete-mlops-pipeline/docs/architecture.png)

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Feature Store** | Feast + PostgreSQL + Ray | Feature engineering & serving |
| **Training** | Kubeflow Training Operator | Distributed PyTorch training |
| **Experiment Tracking** | MLflow | Metrics, params, artifacts |
| **Model Serving** | KServe | Low-latency inference |
| **Compute** | KubeRay | Distributed feature processing |

### Feast + Ray Integration

![Feast Ray](examples/complete-mlops-pipeline/docs/feast-ray.png)

### Training Flow

![Training](examples/complete-mlops-pipeline/docs/training-flow.png)

### Inference Flow

![Inference](examples/complete-mlops-pipeline/docs/inference-flow.png)

## 📁 Project Structure

```
sales-demand-forecasting/
├── examples/
│   └── complete-mlops-pipeline/     # ← Main example
│       ├── docs/                    # Architecture diagrams
│       ├── feature_repo/            # Feast definitions
│       ├── manifests/               # K8s resources
│       ├── notebooks/               # Interactive notebooks
│       │   ├── 01-feast-features.ipynb
│       │   ├── 02-training.ipynb
│       │   └── 03-inference.ipynb
│       ├── scripts/                 # Python scripts
│       │   ├── 01_feast_features.py
│       │   ├── 02_training.py
│       │   └── 03_inference.py
│       └── README.md
├── LICENSE
└── README.md
```

## 🚀 Deployment Options

### Option 1: Notebooks (Interactive)
Run in OpenShift AI workbench:
1. `01-feast-features.ipynb` - Generate data, register features
2. `02-training.ipynb` - Train model with Kubeflow SDK
3. `03-inference.ipynb` - Deploy and test with KServe

### Option 2: Scripts (Automated)
```bash
# Setup infrastructure
kubectl apply -k examples/complete-mlops-pipeline/manifests/

# Run pipeline
cd examples/complete-mlops-pipeline/scripts
python 01_feast_features.py  # Feature engineering
python 02_training.py        # Model training
python 03_inference.py       # Deploy & test
```

## 📚 References

- [Kubeflow Training Operator](https://github.com/kubeflow/training-operator)
- [Feast Feature Store](https://feast.dev)
- [KServe](https://kserve.github.io/website/)
- [OpenShift AI](https://www.redhat.com/en/technologies/cloud-computing/openshift/openshift-ai)

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE)
