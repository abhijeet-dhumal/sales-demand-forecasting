# Getting Started with OpenShift AI

This guide walks you through setting up the infrastructure and creating a workbench for the Sales Demand Forecasting demo.

## Prerequisites

Before starting, ensure you have:

- Access to an OpenShift cluster with **OpenShift AI** installed
- `oc` CLI configured with cluster access
- A storage class with **ReadWriteMany (RWX)** support

> **Note**: If your cluster doesn't have RWX support, Red Hat recommends [OpenShift Data Foundation](https://www.redhat.com/en/technologies/cloud-computing/openshift-data-foundation). Alternatively, install an [NFS dynamic provisioner](https://github.com/opendatahub-io/distributed-workloads/tree/main/workshops/llm-fine-tuning#nfs-provisioner-optional).

---

## Part 1: Deploy Infrastructure

### Step 1: Clone the Repository

```bash
git clone https://github.com/abhijeet-dhumal/sales-demand-forecasting.git
cd sales-demand-forecasting
```

### Step 2: Deploy Components

```bash
oc apply -k manifests/
```

This deploys:

| Component | Purpose |
|-----------|---------|
| PostgreSQL | Feast registry (feature metadata) |
| Redis | Feast online store (low-latency serving) |
| RayCluster | Distributed compute for feature processing |
| FeatureStore | Feast Operator managed services |
| MLflow | Experiment tracking and model registry |
| Shared PVC | Persistent storage for data and models |

### Step 3: Verify Deployment

```bash
oc get pods -n feast-trainer-demo -w
```

Wait until all pods are `Running`:

```
NAME                                    READY   STATUS    RESTARTS   AGE
feast-ray-head-xxxxx                    1/1     Running   0          2m
feast-ray-worker-xxxxx                  1/1     Running   0          2m
feast-salesforecasting-server-xxxxx     4/4     Running   0          2m
postgres-xxxxx                          1/1     Running   0          2m
redis-xxxxx                             1/1     Running   0          2m
```

---

## Part 2: Create a Workbench

A workbench is a Jupyter notebook environment hosted on OpenShift AI.

### Step 1: Access OpenShift AI Dashboard

From the OpenShift web console, click the application launcher (grid icon) and select **Red Hat OpenShift AI**.

![Access OpenShift AI](docs/images/access-openshift-ai.png)

### Step 2: Create a Data Science Project

Navigate to **Projects** and click **Create project**.

![Create Project](docs/images/create-project.png)

Enter:
- **Name**: `feast-trainer-demo`
- **Description**: Sales demand forecasting ML pipeline

### Step 3: Open the Project

After creation, you'll see the project overview with options to create workbenches and pipelines.

![Project Overview](docs/images/project-overview.png)

Click **Create a workbench**.

### Step 4: Configure Workbench Image

Configure the workbench:

- **Name**: `notebook`
- **Image**: `Jupyter | Data Science | CPU | Python 3.12`
- **Version**: `3.4` (Latest)

![Workbench Image](docs/images/workbench-image.png)

### Step 5: Attach Shared Storage

Scroll down to **Cluster storage** and configure:

1. Click **Attach existing storage** or create new
2. **Name**: `notebook-storage`
3. **Storage class**: Select your RWX-capable class (e.g., `nfs-csi`)
4. **Access mode**: **ReadWriteMany (RWX)** - Important for shared access
5. **Size**: `50 GiB`
6. **Mount path**: `/opt/app-root/src/shared`

![Attach Storage](docs/images/attach-storage.png)

### Step 6: Add Feature Store Connection

Scroll to **Feature stores** section:

1. Click the dropdown
2. Select `salesforecasting`

This mounts the Feast client configuration automatically.

![Feature Store Connection](docs/images/feature-store-connection.png)

### Step 7: Create the Workbench

Click **Create workbench** at the bottom of the form.

### Step 8: Wait for Workbench to Start

The workbench will show as "Starting" then transition to "Running".

![Workbench Running](docs/images/workbench-running.png)

Click **Open** to launch JupyterLab.

---

## Part 3: Access MLflow

MLflow is deployed for experiment tracking and model registry.

### Step 1: Access MLflow UI

From the OpenShift console application launcher, select **MLflow**.

### Step 2: Select Workspace

Select the `feast-trainer-demo` workspace from the dropdown.

![MLflow Workspace](docs/images/mlflow-workspace.png)

---

## Part 4: Explore Feature Store

The Feast Feature Store UI shows registered features and lineage.

### Feature Store Lineage

Navigate to **Develop & train** → **Feature store** → **Overview** → **Lineage** tab.

![Feature Store Lineage](docs/images/FeatureStoreOverview.png)

### Feature Services

View the registered feature services:

![Feature Services](docs/images/FeatureServices.png)

| Service | Use Case |
|---------|----------|
| `training_features` | Model training (includes target variable) |
| `inference_features` | Real-time predictions (excludes target) |

---

## Part 5: Clone Repository in Workbench

Once JupyterLab opens:

1. Open a terminal (File → New → Terminal)
2. Clone the repository:

```bash
cd /opt/app-root/src
git clone https://github.com/abhijeet-dhumal/sales-demand-forecasting.git
```

3. Navigate to notebooks and start with your chosen workflow.

---

## Next Steps

| Workflow | Notebook | Description |
|----------|----------|-------------|
| **Option A** (Admin) | `notebooks/01_feature_store/01a-local.ipynb` | Register features, materialize data |
| **Option B** (User) | `notebooks/01_feature_store/01b-remote.ipynb` | Use pre-registered features |
| **Training** | `notebooks/02_training/02-training.ipynb` | Train model with Kubeflow |
| **Inference** | `notebooks/03_inferencing/03-inference.ipynb` | Deploy with KServe |

See the README in each notebook directory for detailed instructions.

---

## Troubleshooting

### Feature Store Connection Not Found

If `/opt/app-root/src/feast-config/salesforecasting` doesn't exist:
1. Edit workbench settings
2. Add Feature Store connection
3. Restart workbench

### Storage Not Mounted

Verify the PVC is bound:
```bash
oc get pvc -n feast-trainer-demo
```

### Pods Not Starting

Check pod events:
```bash
oc describe pod <pod-name> -n feast-trainer-demo
```
