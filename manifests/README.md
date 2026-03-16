# Kubernetes Manifests

This directory contains Kubernetes/OpenShift manifests for deploying the ML pipeline infrastructure.

**Note**: The training and serving manifests are synchronized with the notebook implementations:
- `serve-configmap.yaml` matches `notebooks/03_inferencing/serving_script.py`
- `training-script-configmap.yaml` mirrors the logic in `notebooks/02_training/training_script.py`
- Both use Feast SDK (gRPC) via the Feast Operator's client ConfigMaps

## Quick Start

Deploy all infrastructure with Kustomize:

```bash
oc apply -k manifests/
```

## Directory Structure

| Directory | Purpose | Components |
|-----------|---------|------------|
| `base/` | Core infrastructure | Namespace, shared PVC, config |
| `databases/` | Data stores | PostgreSQL, Redis |
| `ray/` | Distributed compute | RayCluster, RBAC |
| `feast/` | Feature store | FeatureStore CR, secrets |
| `mlflow/` | Experiment tracking | MLflow workspace |
| `observability/` | Monitoring | Network policies |
| `training/` | Training jobs | DataPrep, TrainJob (manual) |
| `serving/` | Model serving | KServe InferenceService (manual) |

## Component Details

### Base (`base/`)

| File | Resource | Description |
|------|----------|-------------|
| `namespace.yaml` | Namespace | `feast-trainer-demo` namespace |
| `shared-pvc.yaml` | PVC | Shared storage for models/data |
| `config.yaml` | ConfigMap | Common configuration |

### Databases (`databases/`)

| File | Resource | Description |
|------|----------|-------------|
| `postgres.yaml` | PostgreSQL | Feast registry, offline store |
| `redis.yaml` | Redis | Feast online store |

### Ray (`ray/`)

| File | Resource | Description |
|------|----------|-------------|
| `rbac.yaml` | ServiceAccount, Role | Permissions for Ray |
| `raycluster.yaml` | RayCluster | Distributed compute for Feast offline |

### Feast (`feast/`)

| File | Resource | Description |
|------|----------|-------------|
| `data-stores-secret.yaml` | Secret | Database credentials |
| `featurestore.yaml` | FeatureStore | Feast Operator CR |
| `image-build.yaml` | BuildConfig | Custom Feast image (optional) |
| `dataprep-scripts-configmap.yaml` | ConfigMap | Feast scripts (alternative to `01a-local.ipynb`) |
| `dataprep-job.yaml` | Job | Data preparation job |

### MLflow (`mlflow/`)

| File | Resource | Description |
|------|----------|-------------|
| `mlflow.yaml` | MLflow | RHOAI MLflow workspace |

### Observability (`observability/`)

| File | Resource | Description |
|------|----------|-------------|
| `policies.yaml` | NetworkPolicy | Network access rules |

### Training (`training/`) - Manual Apply

| File | Resource | Description |
|------|----------|-------------|
| `training-script-configmap.yaml` | ConfigMap | Training script for TrainJob |
| `trainjob.yaml` | TrainJob | Kubeflow training (alternative to `02-training.ipynb`) |

### Serving (`serving/`) - Manual Apply

| File | Resource | Description |
|------|----------|-------------|
| `serve-configmap.yaml` | ConfigMap | Serving script (matches `serving_script.py`) |
| `kserve-inference.yaml` | InferenceService | KServe deployment (alternative to `03-inference.ipynb`) |

## Deployment Order

The Kustomize file handles dependencies automatically. For manual deployment:

```bash
# 1. Base infrastructure
oc apply -f manifests/base/

# 2. Databases (wait for pods to be ready)
oc apply -f manifests/databases/

# 3. Ray cluster
oc apply -f manifests/ray/

# 4. Feature store (requires databases)
oc apply -f manifests/feast/

# 5. MLflow
oc apply -f manifests/mlflow/

# 6. Network policies
oc apply -f manifests/observability/
```

## Training & Serving Jobs

The `training/` and `serving/` directories contain manifest alternatives to the notebooks. These are **not included** in the main Kustomization and should be applied manually if needed.

**Important**: These manifests use the Feast Operator's remote client ConfigMaps (`feast-salesforecasting-client`, `feast-salesforecasting-client-ca`), which are created when you deploy the FeatureStore CR. This matches the notebook approach.

```bash
# Alternative to 01a-local.ipynb (manual Feast setup - only if not using Feast Operator)
oc apply -f manifests/feast/dataprep-scripts-configmap.yaml
oc apply -f manifests/feast/dataprep-job.yaml

# Alternative to 02-training.ipynb
oc apply -f manifests/training/training-script-configmap.yaml
oc apply -f manifests/training/trainjob.yaml

# Alternative to 03-inference.ipynb
oc apply -f manifests/serving/serve-configmap.yaml
oc apply -f manifests/serving/kserve-inference.yaml
```

**Recommended**: Use the notebooks for interactive development and experimentation. The manifests are useful for CI/CD pipelines or production deployments.

## Namespace

All resources deploy to `feast-trainer-demo` namespace by default. To change:

```bash
# Create new namespace
oc new-project my-namespace

# Apply with namespace override
oc apply -k manifests/ -n my-namespace
```

## Secrets Configuration

**Important**: Before deploying, you must configure secrets with your own credentials.

### Setup Steps

1. **Generate secure passwords**:
   ```bash
   # PostgreSQL password
   openssl rand -base64 16
   
   # MLflow session secret
   openssl rand -hex 16
   ```

2. **Update secret files** with generated values:
   
   | File | Placeholder | Description |
   |------|-------------|-------------|
   | `databases/postgres.yaml` | `<CHANGE_ME>` | PostgreSQL password |
   | `mlflow/mlflow.yaml` | `<GENERATE_WITH_openssl_rand_-hex_16>` | OAuth session secret |
   | `feast/dataprep-scripts-configmap.yaml` | `<CHANGE_ME>` | Must match PostgreSQL password |

3. **Example templates** are provided for reference:
   - `databases/postgres-secret.example.yaml`
   - `mlflow/mlflow-secrets.example.yaml`
   - `feast/feast-secrets.example.yaml`

### Security Notes

- Never commit actual credentials to git
- The `.gitignore` excludes `*-secret.yaml` and `*-secrets.yaml` files
- Use External Secrets Operator or Sealed Secrets for production

## Prerequisites

| Operator | Purpose |
|----------|---------|
| Feast Operator | FeatureStore CR support |
| Kubeflow Training Operator | TrainJob CR support |
| KServe | InferenceService CR support |
| MLflow Operator (RHOAI) | MLflow workspace support |

## Verification

Check deployment status:

```bash
# All pods running
oc get pods -n feast-trainer-demo

# Feast services
oc get featurestore -n feast-trainer-demo

# MLflow
oc get mlflow -n feast-trainer-demo

# Ray cluster
oc get raycluster -n feast-trainer-demo
```
