# Cluster compatibility record

Observed on 2026-10-15 with read-only cluster queries. The demo target is `sales-demand-forecasting`; this records visible state and does not establish support for untested client/server combinations. Recheck before a demo run because quotas and allocatable resources change. No cluster resources were changed during this inspection.

## Platform inventory

| Area | Observed state | Demo implication |
|---|---|---|
| OpenShift | 4.21.32; local `oc` client 4.19.0 | Platform and client versions differ; use the server version when checking APIs. |
| OpenShift AI | `default-dsc` reports `Ready`, `ComponentsReady`, and `ModulesReady` true. The installed `rhods-operator` CSV is `3.6.0-ea.2`. | This is an Early Access platform. Keep the compatibility gate explicit before using the demo in a customer-facing or production-supported setting. |
| Feature Store | `featurestores.feast.dev/v1` API is served; no `FeatureStore` objects are present in the demo namespace. The legacy `feast-server` and `feast-ui` Deployments are running. Their public edge Routes still exist and are not protected by the Feature Store operator's OIDC configuration. | The current namespace uses the legacy manual server/UI path. Remove or secure the old Routes before a customer demonstration. The checked-in manifests no longer create Routes, but `kubectl apply` does not delete existing Routes. |
| Ray | Ray component is Managed and ready. `RayCluster/feast-ray` is ready with one worker; it reports 8 CPUs, 32 GiB, and 0 GPUs across the cluster. | The namespace RayCluster is now present; Feast-Ray client/offline-store execution remains unverified. |
| Kubeflow Trainer | Trainer component is Managed and ready. The available ClusterTrainingRuntimes include `torch-with-storage`, `torch-distributed`, and `torch-distributed-cpu`. The TrainJob CRD serves `trainer.kubeflow.org/v1alpha1` with `spec.runtimePatches`; it does not expose `spec.podTemplateOverrides`. A server-side dry run of the runtime patch schema succeeded; no demo TrainJobs were present. | Notebook 02 now uses the ODH SDK fork `v0.4.1+rhaiv.2`, which pins Kubeflow Trainer API 2.3.0rc3 and provides RuntimePatch plus the RHOAI TransformersTrainer. The CRD accepts `runtimePatches`, but no demo TrainJob has been submitted, so SDK-to-cluster behavior remains unverified. |
| AI Pipelines | A `DataSciencePipelinesApplication` named `dspa` was visible in namespace `test`; the DSC component reports ready. | Pipeline API is present, but demo namespace configuration and artifact access remain to be checked. |
| MLflow and Model Registry | Both DSC components report ready. | Use the platform-managed services after verifying their namespace configuration and client authentication path. |
| KServe | DSC component reports ready. | The serving API is present; this does not verify the demo predictor image or its Feast access. |
| Kueue | Kueue component reports ready; `training-cluster-queue` is Active and `nvidia-gpu-flavor` exists. The demo namespace `LocalQueue/default` is Active and maps to `ClusterQueue/default`; the installed Red Hat build of Kueue is 1.4.3. | The `.env.example` selects `default`. Admission still depends on quota and flavor availability. RHOAI 3.6 EA notes require RHBoK 1.4 or later for Trainer/Kueue integration. |
| Demo namespace | `sales-demand-forecasting` exists and is Active. The `demo-notebook` Workbench is running. PVCs are `demo-notebook-storage` (20 Gi, RWO, `ibmc-vpc-block-10iops-tier`), `shared` (100 Gi, RWX, `nfs-csi`), and `postgres-data` (10 Gi, RWO, `ibmc-vpc-block-10iops-tier`). `feast-ray`, `feast-server`, `feast-ui`, and `postgres` pods were Running. | Reuse `shared` at its existing 100 Gi size. Preserve the Workbench and all existing claims; do not reduce their requested sizes or replace their storage. |
| Workbench image and SDKs | `demo-notebook` uses `training-hub-universal-cuda:3.6`. Its Python environment has scikit-learn 1.9.0, PyTorch 2.13.0, and MLflow 3.15.1; Feast, Ray, CodeFlare SDK, and Kubeflow SDK were not installed at inspection. | The Workbench package install is still a network-dependent bootstrap. A curated, digest-pinned Workbench image is preferable for a fully repeatable customer demo. |

## Feature Store operator path

The installed CRD serves stable `feast.dev/v1`. Its schema supports Git-backed feature repositories, online/offline store configuration, registry configuration, and PVC mounts. The RHOAI 3.5 guide confirms that the operator owns the FeatureStore server/UI lifecycle, while backing databases and object storage remain separate prerequisites. The minimal local-provider configuration uses local SQLite/Parquet stores and needs persistent volumes if state must survive pod replacement. See [RHOAI 3.5 Feature Store setup](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/working_with_machine_learning_features/configuring_feature_store).

There is no demo `FeatureStore` instance yet. The current feature repository contains local `.env` render tokens and file-source paths on the shared PVC, so it cannot yet be consumed as a Git-backed operator repository without publishing a rendered, immutable revision and verifying that the operator pod can mount the same dataset. Keep the existing Feast server resources until those checks pass; do not treat the CRD's presence alone as a working migration.

## GPU profile

The demo namespace RayCluster currently requests CPU and memory only; the Workbench and Feast/Ray pods reported no GPU requests in this inspection. Node-level GPU availability was not rechecked on 2026-10-15. The example remains CPU-first. Check allocatable resources, Kueue quota, node taints/labels, and namespace admission immediately before requesting GPUs; a free device is not a scheduling guarantee.

## Feast and Ray compatibility gates

| Check | State |
|---|---|
| Custom Ray image build | BuildConfig completed and pushed `quay.io/abdhumal/sales-demand-forecasting-ray@sha256:e52027bb99e21c3993ffc576db059de2adc9334cf035052e083cc7b5588ff500` with Feast 0.67.0 and CodeFlare SDK 0.40.1 on the `quay.io/modh/ray:2.58.0-py312-cu129-th081` base. The running Ray and Feast pods use this digest. This verifies the image build/push only, not runtime compatibility. The updated Dockerfile pins the base digest and adds server/predictor dependencies, but that v2 image has not been built. |
| Example Feast client version | `.env.example` selects `0.67.0` for the demo client. |
| Feast 0.67.0 Python client | Published on PyPI on 2026-10-13 and requires Python >=3.10. Selected for the demo, but not smoke-tested against this cluster. Do not claim cluster support yet. |
| Operator-managed FeatureStore | Stable `feast.dev/v1` CRD is served; no demo FeatureStore instance or server image version is recorded. |
| Operator-managed Feature Store server image/version | Not recorded; no demo `FeatureStore` object exists to inspect. |
| Feast core checks (`apply`, online read, materialize, historical retrieval, TLS) | Not run as a complete notebook flow. The server/UI pods are Running, but pod readiness does not verify feature registration, retrieval, or TLS behavior. |
| Native Feast-Ray retrieval and `to_ray_dataset()` | Not run. Treat as optional and decide independently from core Feast support. |
| Ray Data fallback | Not exercised by this demo. |
| Storage | The shared RWX claim is Bound on `nfs-csi`; PostgreSQL uses its separate RWO claim. The Kaggle dataset and complete notebook workflow have not been verified on the shared PVC. |

## Next compatibility work

1. Record the installed OpenShift AI operator bundle versions, supported runtime images, and namespace queue/storage settings.
2. In a disposable demo namespace, create the operator-managed `FeatureStore` and validate the platform's supported Feast client first.
3. Run the pinned Python 3.10+ Feast 0.67.0 client through the core Feast smoke checks without replacing the operator-managed server image.
4. Check native Feast-Ray integration separately. If only that plugin fails, keep Feast core on the supported path and export bounded historical retrieval to Parquet for Ray Data.
5. Update this record with exact client/server/Ray versions, image digests, commands, and pass/fail results before claiming this client/server/Ray combination is supported.
