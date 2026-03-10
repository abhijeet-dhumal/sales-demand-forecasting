# GitHub Issue Drafts for feast-dev/feast

---

# Issue 1: "Put failed" error in Ray client mode with KubeRay

## Title

`[Bug] ray.put() fails with "Put failed" error in Ray client mode (KubeRay)`

## Labels

`bug`, `ray`, `kuberay`, `offline-store`

---

## Expected Behavior

When using the Ray offline store with `FEAST_RAY_EXECUTION_MODE=remote` connecting to a KubeRay cluster via Ray client protocol (`ray://`), `get_historical_features()` should successfully retrieve features using distributed Ray processing and return a DataFrame.

## Current Behavior

The `ray.put()` call in the broadcast join logic fails silently with a "Put failed" error message. No stack trace is provided. The feature retrieval fails and falls back to cached data (if available) or fails entirely.

Logs show:
```
INFO - Ray initialized successfully in remote mode
Put failed:
```

The error originates from `ray.put()` in:
- `feast/infra/offline_stores/contrib/ray_offline_store/ray.py` line 606
- `feast/infra/ray_shared_utils.py` line 403
- `feast/infra/compute_engines/ray/nodes.py` line 159

## Steps to Reproduce

1. Deploy a RayCluster via KubeRay operator with TLS enabled
2. Configure Feast to connect via Ray client mode:

```yaml
# feature_store.yaml
project: sales_forecasting
provider: local

offline_store:
  type: ray
  ray_address: "ray://ray-head-svc.namespace.svc.cluster.local:10001"
  enable_ray_logging: true
```

Or via environment:
```bash
export FEAST_RAY_EXECUTION_MODE=remote
export RAY_ADDRESS=ray://ray-head-svc.namespace.svc.cluster.local:10001
```

3. Run feature retrieval:

```python
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="./feature_repo")

entity_df = pd.DataFrame({
    "store_id": [1, 2, 3],
    "dept_id": [1, 1, 1],
    "event_timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
})

# This fails with "Put failed"
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["store_features:temperature", "store_features:fuel_price"]
).to_df()
```

### Specifications

- **Version**: Feast 0.59.0, 0.60.0 (confirmed on both)
- **Platform**: OpenShift AI / Kubernetes with KubeRay operator
- **Subsystem**: Ray offline store, CodeFlare SDK integration

Additional versions:
- Ray: 2.52.1
- Python: 3.12
- CodeFlare SDK: 0.35.0

## Possible Solution

**Option 1: Use Ray Data instead of ray.put() for broadcasting**

```python
def _broadcast_join_feature_view(...):
    # Instead of ray.put(), use Ray Data which handles client mode properly
    feature_ds = ray.data.from_pandas(feature_df)
    
    def join_batch_with_features(batch: pd.DataFrame) -> pd.DataFrame:
        features = feature_ds.to_pandas()
        # ... rest of join logic
```

**Option 2: Detect client mode and use alternative serialization**

```python
def _broadcast_join_feature_view(...):
    is_client_mode = ray.util.client.ray.is_connected()
    
    if is_client_mode:
        import pickle
        feature_bytes = pickle.dumps(feature_df)
        
        def join_batch_with_features(batch: pd.DataFrame) -> pd.DataFrame:
            features = pickle.loads(feature_bytes)
            # ... join logic
    else:
        feature_ref = ray.put(feature_df)
        # ... existing logic
```

**Option 3: Add error handling with informative message**

```python
def _broadcast_join_feature_view(...):
    try:
        feature_ref = ray.put(feature_df)
    except Exception as e:
        if ray.util.client.ray.is_connected():
            raise RuntimeError(
                f"ray.put() failed in client mode. This is a known limitation. "
                f"Consider using FEAST_RAY_EXECUTION_MODE=local. Original error: {e}"
            )
        raise
```

**Workaround**: Use `FEAST_RAY_EXECUTION_MODE=local` or pre-compute features as cached parquet files.

---

# Issue 2: `get_historical_features()` hangs with infeasible resource requests

## Title

`[Bug] get_historical_features() hangs indefinitely with KubeRay - infeasible resource requests`

## Labels

`bug`, `ray`, `kuberay`, `offline-store`

---

## Expected Behavior

When using the Ray offline store with `use_kuberay=True` and CodeFlare SDK authentication, `get_historical_features()` should either:
1. Complete successfully and return features
2. Fail with a clear error message if resources are unavailable

## Current Behavior

The call hangs indefinitely with no error surfaced to the Python caller. Ray cluster logs show repeated warnings:

```
(raylet) There are tasks with infeasible resource requests that cannot be scheduled.
See https://docs.ray.io/en/latest/ray-core/scheduling/index.html#ray-scheduling-resources
```

The root cause is nested `@ray.remote` functions in `CodeFlareRayWrapper` that create resource scheduling deadlocks:

```python
# feast/infra/ray_initializer.py - problematic pattern
def from_pandas(self, df: Any) -> Any:
    @ray.remote  # Requests 1 CPU by default
    def _remote_from_pandas(dataframe):
        return ray.data.from_pandas(dataframe)  # Creates MORE internal Ray tasks
    return RemoteDatasetProxy(_remote_from_pandas.remote(df))
```

The outer task holds resources while inner tasks can't be scheduled, causing deadlock.

## Steps to Reproduce

1. Deploy a RayCluster via KubeRay operator
2. Configure Feast with KubeRay mode:

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

3. Run feature retrieval:

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

### Specifications

- **Version**: Feast 0.59.0, 0.60.0
- **Platform**: OpenShift AI / Kubernetes with KubeRay operator
- **Subsystem**: Ray offline store, CodeFlareRayWrapper, RemoteDatasetProxy

Additional versions:
- Ray: 2.52.1
- Python: 3.12
- CodeFlare SDK: 0.35.0

## Possible Solution

**Option 1: Use `num_cpus=0` on wrapper functions (minimal change)**

```python
def from_pandas(self, df: Any) -> Any:
    from feast.infra.ray_shared_utils import RemoteDatasetProxy

    @ray.remote(num_cpus=0, num_gpus=0)  # Don't consume resources for wrapper
    def _remote_from_pandas(dataframe):
        import ray
        return ray.data.from_pandas(dataframe)

    return RemoteDatasetProxy(_remote_from_pandas.remote(df))
```

**Option 2: Remove wrapper pattern entirely (recommended)**

Since `ray.init()` already connects to KubeRay, Ray Data operations automatically execute on cluster workers. The `@ray.remote` wrapper is unnecessary:

```python
class CodeFlareRayWrapper:
    def __init__(self, ...):
        # ... authentication and ray.init() ...
        pass
    
    def read_parquet(self, path, **kwargs):
        return ray.data.read_parquet(path, **kwargs)
    
    def from_pandas(self, df):
        return ray.data.from_pandas(df)
```

**Option 3: Add timeout with helpful error message**

```python
from ray.exceptions import GetTimeoutError

@ray.remote(num_cpus=0)
def _remote_from_pandas(dataframe):
    return ray.data.from_pandas(dataframe)

try:
    result = ray.get(_remote_from_pandas.remote(df), timeout=300)
except GetTimeoutError:
    raise RuntimeError(
        "Ray task timed out. Check Ray dashboard for 'infeasible resource requests'. "
        "This may indicate resource scheduling issues with nested Ray tasks."
    )
```

**Workaround**: Set `RAY_enable_infeasible_task_early_exit=true` on Ray cluster to fail fast instead of hanging indefinitely.

---

## Related

- Ray resource scheduling: https://docs.ray.io/en/latest/ray-core/scheduling/index.html
- Ray client limitations: https://docs.ray.io/en/latest/cluster/running-applications/job-submission/ray-client.html
- Ray Data internals: https://docs.ray.io/en/latest/data/data-internals.html

---

**I'm happy to submit PRs with fixes for either or both issues if the approaches are acceptable.**
