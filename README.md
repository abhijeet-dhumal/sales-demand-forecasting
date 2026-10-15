# Feast + Ray + OpenShift AI: Walmart Sales Forecasting

A three-notebook, CPU-first demo that uses Walmart's real Store Sales Forecasting competition data to exercise Feast feature management, Ray-backed historical retrieval, Kubeflow Trainer, MLflow, and KServe.

The notebooks are designed for a data scientist working in an OpenShift AI Workbench after a platform engineer has prepared the namespace, storage, services, credentials, and cluster permissions. They do not create the platform stack themselves.

## Data and forecast definition

The input is the Kaggle [Walmart Sales Forecast dataset](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast), the Walmart Store Sales Forecast data family. The notebook reads the downloaded CSV files from `WALMART_DATA_DIR`; it does not fetch them itself. Use the dataset page's download terms and place `train.csv`, `test.csv`, `features.csv`, and `stores.csv` there. Keep those files out of Git. The first `test.csv` week is the demo forecast target. Its sales labels are not available and are never added to training.

Notebook 01 validates and joins the files, removes incomplete Store/Dept time series, and builds a one-week-ahead feature table. Its inputs are limited to prior sales lags, rolling statistics, calendar/holiday fields, and store size. The model does not consume future weather, markdown, or economic observations. Negative net weekly sales are retained and evaluated as negative values where present.

Feast provides feature definitions, point-in-time historical retrieval, online materialization, and the feature service shared by training and inference. Notebook 01 uses pandas to prepare the source Parquet files; Feast does not automatically delegate this preparation to Ray. Notebook 02 configures Feast's Ray offline store and Ray batch engine for historical retrieval. It currently collects that bounded retrieval into pandas on Trainer rank 0 for the rolling backtest and PyTorch input preparation; this is not a Ray Data training pipeline.

The backtest compares last-week naive, seasonal naive, and a pooled tree across eight rolling validation weeks and an eight-week final holdout drawn from the latest 104 weekly targets. The MLP run exercises the RHOAI `TransformersTrainer`/MLflow/KServe path using Hugging Face `Trainer`; the MLP architecture stays unchanged. RHOAI Trainer publishes live progress and manages periodic/JIT checkpoints. The artifact remains explicitly marked as an integration candidate, not a selected or promoted model.

## Platform engineer: prepare the demo namespace

These steps are done once per cluster/namespace. The examples use `sales-demand-forecasting`.

1. Confirm OpenShift AI has the required Feature Store, Ray, Trainer v2, MLflow, and KServe components; confirm an RWX shared PVC and a supported Workbench image. The Workbench and Ray/Trainer/KServe pods must mount the same shared PVC. Notebook 02 installs the RHOAI SDK fork `v0.4.1+rhaiv.2`, which carries Kubeflow Trainer API 2.3 `RuntimePatch` support plus RHOAI `TransformersTrainer` checkpointing. Confirm the namespace LocalQueue and permissions if Kueue is enabled. The `04c-workbench-rbac.yaml` manifest grants the Workbench service account access to submit TrainJobs, read the selected runtime, and manage only the demo ConfigMap and InferenceService. Its ClusterRoleBinding is limited to `get` on the configured ClusterTrainingRuntime.
2. Copy `.env.example` to a protected platform-engineer `.env` and set cluster-specific endpoints, PostgreSQL app/admin credentials, storage class, and image. `MLFLOW_TRACKING_URI` should point to the RHOAI-managed endpoint (this cluster: `https://rh-ai.apps.oai-kft-ibm.ibm.rh-ods.com/mlflow`). `RAY_IMAGE` defaults to the image digest already built and pushed for this demo. Keep this credential-bearing file out of Git and out of the Workbench. Render the config:

   ```bash
   cp .env.example .env
   chmod 600 .env
   # Edit .env with cluster-specific values
   python3 scripts/render_config.py
   ```

   The renderer writes credential-bearing files to the ignored `.generated/` directory. Do not commit that directory.
3. For a new namespace, apply `.generated/manifests/00-prereqs.yaml` to create the namespace and shared PVC. The configured Ray/Feast image is public, so it does not need a namespace Quay pull secret. If you replace it with a private image, create a read-only pull secret in this namespace.

   ```bash
   kubectl apply -f .generated/manifests/00-prereqs.yaml
   ```

   For an existing namespace, preserve its current PVC and size.
4. Confirm the namespace's existing PVC names, storage classes, and requested sizes. Then validate the rendered resources locally and apply the infrastructure manifests:

   ```bash
   kubectl apply --dry-run=client -k .generated/manifests
   kubectl apply -k .generated/manifests
   ```

   Do not print or diff `.generated/manifests`; it contains rendered Secret values. For an existing namespace, set `.env` to the existing claim names, storage classes, and sizes; do not reduce a claim's requested size. The current manifests include PostgreSQL, KubeRay, and a legacy manually managed Feast server/UI. They do not deploy an MLflow server, database, or artifact store: tracking uses the RHOAI-managed MLflow instance. The Feast operator is installed but is not used by this legacy manifest path; the current security and compatibility limits are in the [cluster compatibility record](docs/cluster-compatibility-matrix.md).
5. Wait for PostgreSQL, Ray, and Feast to be ready, then confirm the RHOAI-managed MLflow endpoint is available. The existing Workbench is already annotated with `opendatahub.io/mlflow-instance=mlflow`, which enables the platform MLflow integration. For a new Workbench, select the `mlflow` instance so RHOAI injects its tracking URI and Kubernetes-backed authentication. Configure the Workbench to mount the shared PVC and receive `POSTGRES_URL`, `POSTGRES_PY_URL`, and `POSTGRES_PASSWORD` from Kubernetes Secrets rather than putting database credentials in notebook source. For an existing Workbench, add those Secret-backed environment variables through the OpenShift AI UI or Notebook resource; this repository does not patch the Workbench automatically. Updating the Notebook spec can restart Jupyter; home and shared PVC files persist. The runtime patch mounts the same shared PVC and Secret into the Trainer worker. The training notebook logs to MLflow from the Workbench after the Trainer job completes; it does not forward the Workbench OAuth token to the Trainer pod.
6. Create/start the Workbench in `sales-demand-forecasting`, mount the shared PVC at `/opt/app-root/src/shared`, and clone this repository. Set `.env` `WORKBENCH_SERVICE_ACCOUNT` to the Workbench's actual service account (the inspected `demo-notebook` Workbench uses `demo-notebook`), and set `WORKBENCH_TRAINING_RUNTIME` to the same ClusterTrainingRuntime name used in notebook 02. Re-render and apply the manifests if either value changes. The notebooks do not read `.env` or `project_config.py`; each notebook has an editable settings cell. Inject `POSTGRES_URL`, `POSTGRES_PY_URL`, and `POSTGRES_PASSWORD` from `postgres-secret` as Workbench environment variables. Do not copy `POSTGRES_ADMIN_PASSWORD` into the Workbench.

The installed MinIO AIStor Operator is not used by this demo. Without a valid AIStor license, current AIStor releases start in offline mode with S3 operations blocked, so the demo keeps using the already-running standalone MinIO S3 service. MinIO stores bucket data on its own RWO volume; the sync Job copies the dataset and provenance sidecar onto the shared RWX PVC used by Workbench, Ray, and Trainer. See the [AIStor OpenShift deployment and license requirements](https://docs.min.io/aistor/installation/kubernetes/install/deploy-aistor-on-openshift/).

The checked-in compatibility record says the Feast 0.67.0 + Ray 2.58.0 image built successfully, but the Feast-Ray client/server/KubeRay runtime integration has not passed an end-to-end cluster smoke test. Treat that compatibility check as a platform gate, not as a successful run.

The RHOAI Data Science Gateway exposes the Ray dashboard over an OAuth-protected path. For this cluster, open [Ray Dashboard](https://rh-ai.apps.oai-kft-ibm.ibm.rh-ods.com/ray/sales-demand-forecasting/feast-ray/). The path is `/ray/<namespace>/<RayCluster>/`; for another cluster, get the hostname with `oc get gatewayconfig default-gateway -o jsonpath='{.status.domain}'`. Confirm the generated HTTPRoute is accepted before relying on the link: `oc get httproute <namespace>-<raycluster> -n redhat-ods-applications`.

The checked-in Feast Server and UI Services are ClusterIP-only. New applies no longer create public Routes. For a local UI session, use `oc port-forward -n sales-demand-forecasting svc/feast-ui 8888:8888` and open `http://localhost:8888`. If the namespace already has old Feast Routes, applying these files does not delete them; inspect and remove those old Routes separately after confirming ownership. Use the Feature Store operator's OIDC configuration before exposing a customer-facing Feature Store endpoint.

### Seed the Walmart dataset

The input is the public Kaggle [Walmart Sales Forecast dataset](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast). The MinIO bucket is already bootstrapped. To download through the cluster, first accept the dataset's Kaggle terms if prompted. KaggleHub only needs authentication for resources requiring consent or private resources; if required, create the optional token Secret without putting the token in Git or notebook code:

```zsh
umask 077
read -r -s "KAGGLE_API_TOKEN?Kaggle API token (hidden): "
printf '%s' "$KAGGLE_API_TOKEN" > /tmp/kaggle-api-token
unset KAGGLE_API_TOKEN
oc create secret generic kaggle-api-token -n sales-demand-forecasting \
  --from-file=KAGGLE_API_TOKEN=/tmp/kaggle-api-token --dry-run=client -o yaml \
  | oc apply -f -
rm -f /tmp/kaggle-api-token
```

Render the manifest, then run the optional importer Job. It downloads the dataset, selects `train.csv`, `test.csv`, `features.csv`, and `stores.csv`, makes a ZIP with those files at its root, verifies the uploaded checksum, and stores a JSON provenance sidecar in MinIO. The Job installs its pinned helper packages at startup, so it needs outbound access to Kaggle and PyPI; it does not use a GPU or require a new Ray image build.

```bash
python3 scripts/render_config.py
oc delete job kaggle-minio-import -n sales-demand-forecasting --ignore-not-found
oc apply -f .generated/manifests/12-kaggle-minio-import.yaml
oc wait --for=condition=complete --timeout=30m \
  job/kaggle-minio-import -n sales-demand-forecasting
oc logs job/kaggle-minio-import -n sales-demand-forecasting
oc delete job minio-dataset-sync -n sales-demand-forecasting --ignore-not-found
oc apply -f .generated/manifests/11-minio-dataset-sync.yaml
oc wait --for=condition=complete --timeout=10m \
  job/minio-dataset-sync -n sales-demand-forecasting
```

The sync Job copies the ZIP and available provenance sidecar from MinIO to the common RWX PVC at `/shared/walmart-competition-data`. From the Workbench, that directory is `/opt/app-root/src/shared/walmart-competition-data`. To use the Console instead, upload the Kaggle ZIP to bucket `walmart-sales` at `raw/walmart-sales-forecast.zip`, then run only the sync Job above. KaggleHub documents the optional token-based authentication and dataset download flow in its [official repository](https://github.com/Kaggle/kagglehub).

## Data scientist: run the notebooks

1. Confirm `/opt/app-root/src/shared/walmart-competition-data/walmart-sales-forecast.zip` is present. Notebook 01 validates the ZIP and extracts the four expected CSVs at its root. The loader also accepts the source files as individual `.csv.zip` archives.
2. Open `notebooks/01-feast-features.ipynb` and run every cell from top to bottom. It validates the source files, writes `sales_features.parquet` and `store_features.parquet` to the shared PVC, registers definitions with `FeatureStore.apply()`, materializes data with `FeatureStore.materialize()`, and verifies a real entity lookup. Missing input, permissions, or Feast errors stop the notebook at the failing cell.
3. Open `notebooks/02-training.ipynb` and run every cell. It retrieves Walmart rows through the configured Feast/Ray path, runs the rolling-origin baselines/tree comparison, then submits a uniquely named CPU-first TrainJob with the RHOAI `TransformersTrainer`. While the job is running, view live progress in **OpenShift AI → TrainingJobs**. Periodic and SIGTERM-triggered checkpoints are written under `pvc://<shared-pvc>/models/<run-id>/checkpoints`; a Trainer `RuntimePatch` mounts the shared PVC and Feast Secret into the worker. The Workbench logs the report and MLP artifacts to MLflow using its RHOAI-managed authentication after training completes. The MLP is explicitly an unselected integration candidate. The latest 8 weeks remain outside model fitting. Package installation and backtesting can take time on the first run.
4. Review `rolling_origin_report.json`, `training_metrics.json`, and `model_metadata.json` in the run directory referenced by `WORKBENCH_MODEL_DIR/latest_candidate.json`. The report does not select a production model; this artifact only exercises the Trainer, checkpoint, MLflow, and serving integrations.
5. Before notebook 03, build and push the digest-pinned `Dockerfile.ray` image described below and paste its digest into the `PREDICTOR_IMAGE` setting. Open `notebooks/03-inference.ipynb` and run every cell. Its configuration explicitly allows deployment of the marked unselected integration candidate; set `DEPLOY_UNSELECTED_CANDIDATE` to `False` to block that demo deployment. The notebook creates/updates a KServe InferenceService labelled with candidate status, waits for readiness, scores valid Store/Dept IDs from the loaded data, and saves predictions under `PREDICTIONS_DIR` on the shared PVC. API, readiness, and batch-response errors stop execution instead of printing a success message.

Notebooks 01 and 02 keep runtime settings inline in a `%%yaml parameters` cell. Workbench paths use `/opt/app-root/src/shared`; Trainer and predictor paths use `/shared`, both backed by the same PVC. Notebook 01 generates both Feast client configurations from its YAML settings and uses the Feast Python SDK, Notebook 02 defines the complete serialized training worker and reads the shared Ray-backed Feast config, and Notebook 03 embeds the serving script in a ConfigMap. Run notebooks in order because each produces artifacts consumed by the next. Platform engineers use `.env` to render Kubernetes manifests.

## Configuration and image

Platform deployment settings live in `.env`, with defaults documented in `.env.example`. The config loader uses only Python's standard library. Kubernetes does not read `.env` directly; `scripts/render_config.py` substitutes platform settings into ignored `.generated/` manifests. Notebook settings are configured separately in the notebooks.

`Dockerfile.ray` pins the Ray base image by digest and bakes Feast, CodeFlare, gRPC, Feast UI, and predictor dependencies into one image. `manifests/09-ray-image-buildconfig.yaml` emits a unique `v2` tag; after building it, capture the resulting image digest and set `RAY_IMAGE` in `.env` and `PREDICTOR_IMAGE` in notebook 03 to that digest before deployment. The currently recorded digest is the earlier image and is not suitable as a predictor. Notebook 03 no longer installs packages at pod startup. Notebook 02 installs its pinned Feast/Ray/CodeFlare packages into the Trainer job; it relies on the selected RHOAI runtime for its supported PyTorch, Transformers, and Accelerate versions instead of replacing those framework packages. The Workbench installs the RHOAI Kubeflow SDK fork from the pinned Git tag shown in Notebook 02.

The training notebook uses the platform-native Trainer features that fit this workload: the preinstalled `ClusterTrainingRuntime`, `TrainerClient`, progress metrics in the dashboard, periodic epoch checkpoints, and just-in-time checkpoints on SIGTERM. It uses the SDK's `RuntimePatch` option to attach the shared PVC and Secret to Trainer pods; it does not construct TrainJobs through the Kubernetes API. The selected ODH SDK tag contains Trainer API 2.3 models and is compatible with the inspected cluster's `spec.runtimePatches` API. Checkpoints use the existing RWX PVC; S3 checkpoint storage is unnecessary for this small demo unless portability or larger capacity becomes a requirement. Training Hub's LLM fine-tuning algorithms and the distributed data cache target different workloads and are intentionally not part of this tabular forecasting example. See the [Kubeflow SDK Training Options](https://sdk.kubeflow.org/en/latest/train/options.html), the [RHOAI Kubeflow Trainer v2 guide](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/working_with_distributed_workloads/running-kubeflow-trainerv2_distributed-workloads), and [RHOAI 3.6 EA release notes](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/3.6_early_access_release_notes/release-notes-3-6-ea_release-notes-3-6-ea).

## Other platform-native capabilities to adopt

| Capability | Current use | Recommended next step |
|---|---|---|
| Kubeflow Trainer v2 | Used now for the MLP training job, live progress, and PVC-backed checkpoints. | Keep this as the single model-training orchestrator. Use the supported RHOAI runtime's preinstalled Transformers/Accelerate packages; do not install an LLM fine-tuning algorithm for this tabular workload. |
| RHOAI Feature Store operator | The cluster serves its CRD, but this demo still runs manually managed Feast Server/UI Deployments. | Migrate to an operator-managed `FeatureStore` after recording the installed RHOAI/server versions and validating Feast 0.67 client compatibility. Configure its supported authentication before exposing any endpoint; remove the old public Routes as part of that migration. |
| AI Pipelines | Available on the platform; the demo is currently a guided notebook sequence. | Package the validated notebook steps as pipeline components once the dataset and artifacts have a durable object-store URI. This makes ingestion, feature materialization, training, evaluation, and promotion reproducible without hiding the interactive data-science flow. |
| Model Registry | Platform component is enabled, but the notebooks only log candidate artifacts to managed MLflow. | Add registry registration and approval only after the rolling-origin evaluation selects a candidate. Keep the current MLP explicitly unpromoted. |
| Ray Data | Feast's Ray offline store and batch engine execute distributed historical feature retrieval; the resulting bounded frame is collected on Trainer rank 0. | Keep that path for the current Walmart-sized dataset. Add an explicit Ray Data preparation stage only when profiling shows the current pandas step is a bottleneck or the demo is expanded to data large enough to justify distributed transforms. |

The inspected cluster runs RHOAI 3.6.0 EA.2 and serves Trainer 2.3's `runtimePatches` API. Notebook 02 now targets that API using the RHOAI SDK fork `v0.4.1+rhaiv.2`. The inspected cluster is Early Access and has not completed an end-to-end notebook run. See [the Kubeflow SDK RuntimePatch guide](https://sdk.kubeflow.org/en/latest/train/options.html), [RHOAI Trainer v2 guide](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/working_with_distributed_workloads/running-kubeflow-trainerv2_distributed-workloads), [RHOAI 3.6 EA release notes](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/3.6_early_access_release_notes/release-notes-3-6-ea_release-notes-3-6-ea), and [RHOAI supported configurations](https://access.redhat.com/articles/rhoai-supported-configs-3.x).

To rebuild the custom image, first confirm `quay-push-secret` exists in `sales-demand-forecasting` and grants push access to `quay.io/abdhumal`. The binary build uploads only the Dockerfile, not `.env`, the dataset, or the rest of the repository:

The Docker build strategy requires a privileged pod. A cluster admin must grant the OpenShift `privileged` SCC to the namespace's `builder` service account once:

```bash
oc adm policy add-scc-to-user privileged -z builder -n sales-demand-forecasting
```

The BuildConfig explicitly uses that service account. Do not grant the SCC to the Workbench or default service account.

```bash
oc project sales-demand-forecasting
oc apply -f .generated/manifests/09-ray-image-buildconfig.yaml
oc start-build ray-image-builder --from-file=Dockerfile.ray --follow
skopeo inspect --override-arch amd64 --override-os linux \
  --format '{{.Digest}}' \
  docker://quay.io/abdhumal/sales-demand-forecasting-ray:feast-0.67.0-ray-2.58.0-th081-v2-20261015
```

Set `RAY_IMAGE` in `.env` to `quay.io/abdhumal/sales-demand-forecasting-ray@<digest>` from that output, then rerun `python3 scripts/render_config.py`. Keep the BuildConfig's output tag and the `.env` digest in sync. The build will fail if the Quay push secret is absent or invalid.

The manifests under `manifests/05-dataprep-job.yaml`, `06-trainjob.yaml`, and `08-kserve-inference.yaml` are not used by the three-notebook flow. `05` creates synthetic data. `06` is deprecated and unsupported because it tries to use the Workbench-only MLflow authentication from a standalone TrainJob. `08` is an optional internal Service template; KServe already creates a predictor Service for the notebook path. The default kustomization does not apply these files. `09-ray-image-buildconfig.yaml` is applied separately only when rebuilding the custom image.

The default profile is CPU-first. Measure and profile the representative workload before asking for GPUs or more Ray workers. This dataset is intentionally used to teach the end-to-end interfaces; it does not prove that distributed data processing or accelerators improve forecast quality or run time.

## Troubleshooting

| Issue | Check |
|-------|-------|
| Notebook cannot find config | Confirm the protected `.env` is in the repository root in the Workbench. |
| Notebook 01 cannot find data | Confirm all four dataset CSVs are directly under `WALMART_DATA_DIR`. |
| Feast cannot connect to PostgreSQL | Confirm the Workbench and Trainer receive the PostgreSQL Secret values and that PostgreSQL is ready. URL-reserved characters in database credentials are percent-encoded by the config loader. |
| Ray/Feast pod is `ImagePullBackOff` | Confirm the configured image digest is reachable from the cluster; add a read-only pull secret only if using a private image. |
| Trainer cannot read files | Confirm the Trainer runtime mounts the same shared PVC at `/shared`. |
| MLflow cell fails authentication | Confirm the Workbench is connected to the RHOAI-managed `mlflow` instance and restart the Workbench so its Kubernetes authentication variables are injected. |
| Trainer or predictor remains pending | Check Kueue admission, namespace quota, PVC mount, and configured CPU/memory requests. |
| Feast-Ray retrieval fails | Check the exact client/server/Ray versions against `docs/cluster-compatibility-matrix.md`; the combination is not yet runtime-validated. |

## Cleanup

When the demo is finished, remove only demo resources after confirming that the namespace does not contain other workloads. Keep the shared Workbench and PVC if they are used by other work.
