# GitHub Issue Draft for feast-dev/feast

---

## Title

`get_historical_features()` hangs with "infeasible resource requests" in KubeRay mode

---

## Labels

`bug`, `ray`, `kuberay`

---

## Description

### Summary

When using the Ray offline store with `use_kuberay=True` and CodeFlare SDK authentication, `get_historical_features()` hangs indefinitely. The Ray cluster logs show repeated "infeasible resource requests" warnings, but no error is surfaced to the caller.

### Environment

- **Feast version**: 0.59.0
- **Ray version**: 2.52.1
- **Python version**: 3.12
- **Platform**: OpenShift AI / Kubernetes with KubeRay operator
- **CodeFlare SDK version**: latest

### Configuration

```yaml
# feature_store.yaml
project: sales_forecasting
provider: local

offline_store:
  type: ray
  use_kuberay: true
  kuberay_conf:
    cluster_name: my-ray-cluster
    namespace: my-namespace
    skip_tls: true
  enable_ray_logging: true
```

### Steps to Reproduce

1. Deploy a RayCluster via KubeRay operator
2. Configure Feast with `use_kuberay=True` pointing to the cluster
3. Run `get_historical_features()`:

```python
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="./feature_repo")

entity_df = pd.DataFrame({
    "entity_id": [1, 2, 3],
    "event_timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
})

# This hangs indefinitely
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["my_feature_view:feature1", "my_feature_view:feature2"]
).to_df()
```

### Expected Behavior

Feature retrieval completes and returns a DataFrame.

### Actual Behavior

The call hangs indefinitely. Ray cluster logs show:

```
(raylet) There are tasks with infeasible resource requests that cannot be scheduled.
See https://docs.ray.io/en/latest/ray-core/scheduling/index.html#ray-scheduling-resources
```

This message repeats every few seconds but no error is raised to the Python caller.

### Root Cause Analysis

The issue is in `feast/infra/ray_initializer.py` in the `CodeFlareRayWrapper` class. The wrapper uses nested `@ray.remote` functions:

```python
# Current implementation (problematic)
def from_pandas(self, df: Any) -> Any:
    from feast.infra.ray_shared_utils import RemoteDatasetProxy

    @ray.remote  # Requests 1 CPU by default
    def _remote_from_pandas(dataframe):
        import ray
        return ray.data.from_pandas(dataframe)  # Creates MORE internal Ray tasks

    return RemoteDatasetProxy(_remote_from_pandas.remote(df))
```

**Why this causes scheduling failures:**

1. The outer `@ray.remote` task requests 1 CPU (default)
2. Ray Client schedules this task on a cluster worker
3. Inside, `ray.data.from_pandas()` creates additional internal tasks for parallel processing
4. These nested tasks compete for resources on the same node
5. If the cluster has limited resources or specific resource labels (e.g., `accelerator_type`), the nested tasks become infeasible
6. Ray's default behavior is to wait indefinitely for infeasible tasks rather than failing

The same pattern affects `read_parquet()`, `read_csv()`, `from_arrow()`, and other methods in `CodeFlareRayWrapper`.

### Suggested Fix

**Option 1: Minimal change - use `num_cpus=0` on wrapper functions**

```python
def from_pandas(self, df: Any) -> Any:
    from feast.infra.ray_shared_utils import RemoteDatasetProxy

    @ray.remote(num_cpus=0, num_gpus=0)  # Don't consume resources for the wrapper
    def _remote_from_pandas(dataframe):
        import ray
        return ray.data.from_pandas(dataframe)

    return RemoteDatasetProxy(_remote_from_pandas.remote(df))
```

**Option 2: Remove wrapper pattern entirely (recommended)**

Since `ray.init()` already connects to the KubeRay cluster, Ray Data operations will automatically execute on cluster workers. The `@ray.remote` wrapper is unnecessary:

```python
class CodeFlareRayWrapper:
    def __init__(self, ...):
        # ... authentication and ray.init() ...
        pass
    
    # Use Ray Data directly - it already runs on the connected cluster
    def read_parquet(self, path, **kwargs):
        return ray.data.read_parquet(path, **kwargs)
    
    def from_pandas(self, df):
        return ray.data.from_pandas(df)
```

### Additional Context

- The `RemoteDatasetProxy` class in `ray_shared_utils.py` has the same issue - all its methods use `@ray.remote` wrappers without `num_cpus=0`
- Setting `RAY_enable_infeasible_task_early_exit=true` on the Ray cluster makes the tasks fail fast instead of hanging, which helps with debugging but doesn't fix the underlying issue
- The issue does NOT occur with `FEAST_RAY_EXECUTION_MODE=local` or `FEAST_RAY_EXECUTION_MODE=remote` (direct Ray address) - only with the KubeRay/CodeFlare path

### Workaround

Until fixed, users can:
1. Set `FEAST_RAY_EXECUTION_MODE=local` to run Ray Data locally (loses distributed processing benefit)
2. Pre-compute features using the dataprep job and load cached data during training

---

## Related

- Ray documentation on resource requirements: https://docs.ray.io/en/latest/ray-core/scheduling/index.html
- Ray Data architecture: https://docs.ray.io/en/latest/data/data-internals.html

---

**I'm happy to submit a PR with the fix if this approach is acceptable.**
