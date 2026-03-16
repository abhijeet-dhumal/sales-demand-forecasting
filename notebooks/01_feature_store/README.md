# Feast Feature Store Notebooks

## Workflow

![Feature Store Workflow](../../docs/diagrams/01-features-workflow.png)

## Overview

Two notebooks for different use cases:

| Notebook | Mode | Use Case |
|----------|------|----------|
| `01a-local.ipynb` | Direct Connection | Full control: register features, materialize, manage data |
| `01b-remote.ipynb` | Remote Client | Read-only: retrieve features for training/inference |

---

## Which Notebook to Use?

| I need to... | Use |
|--------------|-----|
| Register new features (`feast apply`) | `01a-local` |
| Materialize to online store | `01a-local` |
| Generate/ingest new data | `01a-local` |
| Get features for training | Either (01b is simpler) |
| Get features for inference | Either (01b is simpler) |

---

## 01a-local.ipynb (Direct Connection)

### Prerequisites

| Requirement | Details |
|-------------|---------|
| Environment | OpenShift AI Workbench with shared PVC at `/opt/app-root/src/shared` |
| Infrastructure | `oc apply -k manifests/` deployed |
| Services | PostgreSQL, Redis, RayCluster `feast-ray` |
| Python | `feast[postgres,ray,redis]`, `codeflare-sdk` |

### Connection

Generates local `feature_store.yaml` with direct DB connections:

| Component | Type | Connection |
|-----------|------|------------|
| Registry | `sql` | `postgresql://postgres:5432` |
| Offline Store | `ray` | `feast-ray` cluster |
| Online Store | `redis` | `redis:6379` |

### Steps

1. Install dependencies & configure CodeFlare auth
2. Generate synthetic sales data (65K rows)
3. Engineer features (lags, rolling stats)
4. Create `feature_store.yaml`
5. Save data to parquet
6. `feast apply` - register features
7. `feast materialize` - populate online store
8. Verify retrieval

---

## 01b-remote.ipynb (Remote Client)

### Prerequisites

| Requirement | Details |
|-------------|---------|
| Environment | OpenShift AI Workbench with Feature Store connection |
| Config Path | `/opt/app-root/src/feast-config/salesforecasting` (auto-mounted) |
| Infrastructure | Feast Operator deployed |
| Features | Already registered and materialized |
| Python | `feast` (minimal, no extras) |

### Connection

Uses operator-provided config with gRPC:

| Component | Type | Connection |
|-----------|------|------------|
| Registry | `remote` | `feast-*-registry:443` |
| Offline Store | `remote` | `feast-*-offline:443` |
| Online Store | `remote` | `feast-*-online:443` |

### Steps

1. Install dependencies
2. Connect using operator config
3. List registered features
4. Get online features
5. Get historical features

---

## Capabilities Comparison

| Operation | 01a (Local) | 01b (Remote) |
|-----------|-------------|--------------|
| `feast apply` | ✅ | ❌ |
| `feast materialize` | ✅ | ❌ |
| `get_online_features()` | ✅ | ✅ |
| `get_historical_features()` | ✅ | ✅ |
| `list_entities()` | ✅ | ✅ |

---

## Feature Store Architecture

### Lineage View

![Feature Store Lineage](../../docs/images/FeatureStoreOverview.png)

### Feature Services

![Feature Services](../../docs/images/FeatureServices.png)

### Data Sources

| File | Description |
|------|-------------|
| `sales_features.parquet` | Weekly sales with temporal features |
| `store_features.parquet` | Store metadata (type, size, region) |

### Entities

| Entity | Range | Description |
|--------|-------|-------------|
| `store_id` | 1-45 | Retail store identifier |
| `dept_id` | 1-14 | Department identifier |

### Feature Views

| Feature View | Count | Features |
|--------------|-------|----------|
| `sales_features` | 19 | `weekly_sales`, `lag_1-8`, `rolling_mean_4w`, `rolling_std_4w`, temporal, economic |
| `store_features` | 3 | `store_type`, `store_size`, `region` |

### Feature Services

| Service | Features | Use Case |
|---------|----------|----------|
| `training_features` | 22 (includes target) | Model training |
| `inference_features` | 21 (excludes target) | Real-time predictions |

---

## Online vs Offline Features

| | Online | Offline |
|---|---|---|
| Purpose | Real-time inference | Model training |
| Latency | Milliseconds | Seconds |
| Data | Latest values | Historical point-in-time |
| Store | Redis | Parquet via Ray |
| Method | `get_online_features()` | `get_historical_features()` |
