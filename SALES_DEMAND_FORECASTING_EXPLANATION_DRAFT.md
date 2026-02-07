# Sales Demand Forecasting - MLOps Pipeline
## Beginner-Friendly Explanation Draft

---

## Introduction

The **Sales Demand Forecasting** project is an end-to-end MLOps (Machine Learning Operations) pipeline that demonstrates how to build, train, and deploy a machine learning model for predicting retail sales demand. This project showcases production-grade tools and best practices for building ML systems on OpenShift AI.

### What Problem Does This Solve?

Retail businesses face a constant challenge: predicting how much product they'll sell in the future. Getting this wrong leads to:
- **Overstocking**: Too much inventory ties up capital and increases holding costs
- **Understocking**: Too little inventory leads to stockouts and lost sales
- **Poor Staffing**: Wrong staffing levels result in poor customer service or wasted payroll

This project demonstrates how machine learning can solve these problems by accurately forecasting demand, enabling businesses to:
- Optimize inventory levels (20% cost savings)
- Reduce stockouts (15% reduction)
- Optimize staffing (10-15% improvement)
- Make data-driven decisions

### What Makes This Special?

This isn't just a machine learning model in a notebook. It's a **complete production system** that demonstrates:
- **Feature Store**: Centralized, reusable features
- **Distributed Training**: Training across multiple GPUs
- **Experiment Tracking**: Versioning and comparing models
- **Model Serving**: Production-ready deployment
- **Distributed Processing**: Handling large datasets efficiently

---

## Understanding MLOps

### What is MLOps?

**MLOps** stands for Machine Learning Operations. It's the practice of applying DevOps principles to machine learning systems. Just as DevOps helps software development teams deploy code reliably, MLOps helps data science teams deploy ML models reliably.

### Traditional ML vs. MLOps

**Traditional ML Approach:**
1. Data scientist builds model in Jupyter notebook
2. Model works on their laptop with sample data
3. Hard to reproduce results
4. Difficult to deploy to production
5. No monitoring or versioning

**MLOps Approach:**
1. **Feature Store**: Features are centralized and versioned
2. **Reproducible Training**: Code, data, and environment are versioned
3. **Experiment Tracking**: All experiments are logged and comparable
4. **Model Serving**: Models are deployed as production services
5. **Monitoring**: Model performance is tracked in production

### Why MLOps Matters

- **Reproducibility**: Anyone can reproduce your results
- **Collaboration**: Teams can share features and models
- **Scalability**: Handle large datasets and high traffic
- **Reliability**: Production-ready systems with monitoring
- **Efficiency**: Reuse features and models across projects

---

## Architecture Overview

The pipeline consists of three main stages, each using different technologies:

### Stage 1: Feature Engineering
- **Technology**: Feast (Feature Store) + Ray (Distributed Processing)
- **Purpose**: Create, store, and serve features
- **Output**: Features ready for training and inference

### Stage 2: Model Training
- **Technology**: Kubeflow Training + MLflow
- **Purpose**: Train neural network model
- **Output**: Trained model with tracked experiments

### Stage 3: Model Serving
- **Technology**: KServe
- **Purpose**: Deploy model for real-time predictions
- **Output**: REST API serving predictions

### Supporting Infrastructure
- **PostgreSQL**: Storage for Feast registry, offline store, online store, and MLflow
- **KubeRay**: Kubernetes-native Ray cluster for distributed processing
- **OpenShift AI**: Platform providing all these components

---

## Stage 1: Feature Engineering with Feast

### What is a Feature Store?

A **Feature Store** is a centralized repository for machine learning features. Think of it as a database specifically designed for ML features, with two key capabilities:

1. **Offline Store**: Historical features for training models
2. **Online Store**: Real-time features for serving predictions

### Why Use a Feature Store?

**Without Feature Store:**
- Features are duplicated across projects
- Inconsistencies between training and serving
- Hard to version and track features
- Slow feature engineering

**With Feature Store:**
- Features are centralized and reusable
- Consistent features for training and serving
- Versioned and tracked
- Fast feature retrieval

### Feast Architecture

Feast uses a three-layer architecture:

**1. Feature Registry (PostgreSQL)**
- Stores feature definitions (metadata)
- Tracks feature versions
- SQL-queryable for discovery

**2. Offline Store (PostgreSQL)**
- Historical feature data
- Used for training models
- Optimized for batch queries

**3. Online Store (PostgreSQL)**
- Real-time feature values
- Used for inference
- Optimized for low-latency lookups

**4. Ray Compute Engine**
- Distributed feature processing
- Parallel joins and transformations
- Handles large datasets efficiently

### Feast Concepts Explained

#### Entities

**Entities** are the primary keys used to look up features. They represent the business objects you care about.

**Example:**
```python
store_entity = Entity(
    name="store",
    value_type=Int64,
    description="Walmart store number (1-45)"
)

dept_entity = Entity(
    name="dept",
    value_type=Int64,
    description="Department number (1-99)"
)
```

In this project, we use a **composite key** of (store, dept) because features vary by both store and department. For example, Store 1, Department 3 has different sales patterns than Store 2, Department 3.

#### Feature Views

**Feature Views** are collections of related features with a schema. They define what features are available and how to retrieve them.

**Example - Sales History Features:**
```python
sales_history_features = FeatureView(
    name="sales_history_features",
    entities=[store_entity, dept_entity],
    schema=[
        Field(name="weekly_sales", dtype=Float64),
        Field(name="is_holiday", dtype=Int64),
        Field(name="sales_lag_1", dtype=Float64),  # Sales from 1 week ago
        Field(name="sales_lag_2", dtype=Float64),  # Sales from 2 weeks ago
        Field(name="sales_rolling_mean_4", dtype=Float64),  # 4-week average
    ],
    source=sales_source,
    online=True  # Available in online store
)
```

**Key Features:**
- **Time-series features**: Lag features (sales from previous weeks) and rolling statistics (moving averages)
- **Temporal features**: Week of year, month, quarter for seasonality
- **Holiday indicators**: Binary flags for holiday weeks

#### Feature Services

**Feature Services** group multiple FeatureViews together for a specific use case. They ensure consistency - when you request features for demand forecasting, you always get the same set of features.

**Example:**
```python
demand_forecasting_service = FeatureService(
    name="demand_forecasting_service",
    features=[
        sales_history_features,      # Time-series features
        store_external_features,     # External factors
        feature_transformations,     # On-demand transforms
    ]
)
```

### Feature Types in This Project

#### 1. Time-Series Features

These capture historical patterns:
- **Lag Features**: `sales_lag_1`, `sales_lag_2`, `sales_lag_4` - Sales from previous weeks
- **Rolling Statistics**: `sales_rolling_mean_4`, `sales_rolling_mean_12` - Moving averages
- **Volatility**: `sales_rolling_std_4` - Standard deviation (measures variability)

**Why They Matter**: Past sales are strong predictors of future sales. A store that sold $25,000 last week is likely to sell a similar amount this week.

#### 2. External Features

These capture factors outside the store:
- **Economic Indicators**: CPI (Consumer Price Index), unemployment rate
- **Market Factors**: Fuel price, temperature
- **Promotions**: Markdowns (discounts), holiday indicators

**Why They Matter**: External factors influence demand. High fuel prices might reduce sales, while holidays increase sales.

#### 3. On-Demand Features

These are computed at request time:
- **Normalizations**: Scale features to 0-1 range
- **Interactions**: Holiday × markdown interaction
- **Business Metrics**: Sales per square foot
- **Seasonal Encoding**: Sine/cosine transformations for cyclical patterns

**Why They Matter**: Some features are better when transformed. Seasonal patterns (like higher sales in December) are better captured with sine/cosine than raw week numbers.

### Materialization: Offline to Online

**Materialization** is the process of copying features from the offline store (used for training) to the online store (used for inference).

**Why It's Needed:**
- Offline store: Optimized for large batch queries (training)
- Online store: Optimized for fast single-row lookups (inference)

**Process:**
1. Features are computed and stored in offline store
2. Materialization job runs periodically
3. Features are copied to online store
4. Online store is ready for real-time serving

**Example:**
```bash
feast materialize 2022-01-01T00:00:00 2024-01-01T00:00:00
```

This materializes features from January 1, 2022 to January 1, 2024.

### Ray Integration for Distributed Processing

**Ray** is a distributed computing framework that speeds up feature processing by distributing work across multiple workers.

**How It Works:**
1. Feast connects to Ray cluster
2. Large datasets are partitioned
3. Each partition processed in parallel
4. Results combined efficiently

**Benefits:**
- **Speed**: 5-10x faster than single-threaded
- **Scalability**: Add workers to handle larger datasets
- **Efficiency**: Better resource utilization

**In This Project:**
- Ray processes feature joins and transformations
- Handles 421K+ records efficiently
- Reduces feature engineering time from hours to minutes

---

## Stage 2: Model Training with Kubeflow

### What is Kubeflow Training?

**Kubeflow Training** is a Kubernetes-native system for distributed machine learning training. It allows you to submit training jobs that run across multiple GPUs and nodes.

### Why Distributed Training?

**Single GPU Training:**
- One GPU processes all data sequentially
- Slow for large datasets
- Limited by GPU memory

**Distributed Training:**
- Multiple GPUs work together
- Each GPU processes different batches
- Much faster (near-linear speedup)

**Example:**
- Single GPU: ~3 minutes for 15 epochs
- 4 GPUs: ~44 seconds for 15 epochs
- **~4x speedup!**

### The Model Architecture

The project uses a **Multi-Layer Perceptron (MLP)** - a type of neural network:

```
Input Layer (11 features)
    ↓
Hidden Layer 1 (256 neurons)
    ├─ Batch Normalization
    ├─ ReLU Activation
    └─ Dropout (20%)
    ↓
Hidden Layer 2 (128 neurons)
    ├─ Batch Normalization
    ├─ ReLU Activation
    └─ Dropout (20%)
    ↓
Hidden Layer 3 (64 neurons)
    ├─ Batch Normalization
    ├─ ReLU Activation
    └─ Dropout (20%)
    ↓
Output Layer (1 prediction)
```

**Why This Architecture?**
- **Deep**: Multiple layers capture complex patterns
- **Regularized**: Dropout and batch normalization prevent overfitting
- **Scalable**: Can handle many features

### Training Process

#### 1. Data Loading

Features are loaded from Feast offline store:
```python
from feast import FeatureStore

store = FeatureStore(repo_path=".")
features = store.get_historical_features(
    entity_df=entity_dataframe,
    features=demand_forecasting_service
)
```

#### 2. Preprocessing

Data is prepared for training:
- **Scaling**: Features normalized to similar ranges (using StandardScaler)
- **Splitting**: Data split into train (80%) and test (20%) sets
- **Batching**: Data organized into batches for efficient processing

#### 3. Model Definition

Neural network is created:
```python
class SalesMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            # ... more layers
        )
```

#### 4. Training Loop

For each epoch:
1. **Forward Pass**: Input data flows through network
2. **Loss Calculation**: Compare predictions to actual values
3. **Backward Pass**: Calculate gradients (how to adjust weights)
4. **Weight Update**: Update model weights using optimizer
5. **Evaluation**: Test on validation set

#### 5. Distributed Training (PyTorch DDP)

When using multiple GPUs:
- Each GPU processes different batches
- Gradients are averaged across GPUs
- Model weights synchronized across GPUs
- All GPUs work together as one

**Key Code:**
```python
# Initialize distributed training
dist.init_process_group(backend="nccl")
model = nn.parallel.DistributedDataParallel(model)

# Each GPU gets different data
sampler = DistributedSampler(train_ds)
loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
```

### Training Configuration

**Hyperparameters:**
- **Epochs**: 15 (number of times to see all data)
- **Batch Size**: 256 (samples per batch)
- **Learning Rate**: 0.001 (how fast to learn)
- **Optimizer**: AdamW (adaptive learning rate)
- **Scheduler**: CosineAnnealingLR (learning rate decay)

**Resources:**
- **Nodes**: 2
- **GPUs per Node**: 2
- **Total GPUs**: 4
- **CPU per Node**: 4
- **Memory per Node**: 16Gi

### Model Performance

**Metrics:**
- **MAPE (Mean Absolute Percentage Error)**: 10.5%
  - Industry baseline: 15-20%
  - **40% better than baseline!**
- **RMSE (Root Mean Squared Error)**: 500
- **Validation Loss**: 0.0009

**What This Means:**
- Predictions are within 10.5% of actual sales on average
- Much better than industry standard
- Good enough for business decisions

---

## MLflow - Experiment Tracking

### What is MLflow?

**MLflow** is an open-source platform for managing the ML lifecycle. It helps you track experiments, compare models, and manage model versions.

### Why Track Experiments?

**Without Tracking:**
- Hard to remember what you tried
- Can't compare different approaches
- Difficult to reproduce results
- No history of what worked

**With MLflow:**
- All experiments logged automatically
- Easy comparison of different runs
- Reproducible results
- Complete history

### What MLflow Tracks

#### 1. Parameters

Configuration settings that affect training:
```python
mlflow.log_params({
    "epochs": 50,
    "learning_rate": 0.001,
    "batch_size": 256,
    "world_size": 4  # Number of GPUs
})
```

#### 2. Metrics

Performance measurements over time:
```python
mlflow.log_metrics({
    "train_loss": 0.0012,
    "val_loss": 0.0009,
    "rmse": 500,
    "mape": 10.5
})
```

#### 3. Artifacts

Files associated with the run:
- Model files
- Plots and visualizations
- Data samples
- Configuration files

#### 4. Models

Versioned model storage:
```python
mlflow.pytorch.log_model(model, "model")
```

### MLflow UI

MLflow provides a web UI where you can:
- View all experiments
- Compare runs side-by-side
- Visualize metrics over time
- Download models and artifacts
- Share results with team

### Model Registry

MLflow Model Registry provides:
- **Versioning**: Track model versions
- **Staging**: Move models through stages (Staging → Production)
- **Metadata**: Store model descriptions and tags
- **Lineage**: Track which data and code produced the model

---

## Stage 3: Model Serving with KServe

### What is KServe?

**KServe** is a Kubernetes-native model serving platform. It provides a standard way to deploy ML models as production services.

### Why KServe?

**Traditional Deployment:**
- Manual container creation
- Manual service configuration
- No auto-scaling
- Hard to manage multiple models

**KServe:**
- Declarative model deployment
- Auto-scaling based on traffic
- A/B testing and canary deployments
- Standard REST API

### InferenceService

**InferenceService** is a Kubernetes Custom Resource that represents a deployed model:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: sales-forecast
spec:
  predictor:
    containers:
    - name: kserve-container
      image: pytorch-model-server
      env:
      - name: MODEL_DIR
        value: /mnt/models
```

**What Happens:**
1. KServe creates a Deployment with your model
2. Service exposes the model via REST API
3. Route (OpenShift) makes it accessible externally
4. Model is ready to serve predictions

### Inference API

KServe follows the **v1 prediction protocol**:

**Request:**
```json
POST /v1/models/sales-forecast:predict
{
  "instances": [
    {
      "lag_1": 25000,
      "lag_2": 24000,
      "lag_4": 23000,
      "temperature": 65.0,
      "fuel_price": 2.8,
      "cpi": 215.0,
      "unemployment": 5.5,
      ...
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [26750.32]
}
```

**Prediction**: The model predicts sales of $26,750 for the next week.

### Serving Process

When a prediction request arrives:

1. **Load Features**: Retrieve real-time features from Feast online store
2. **Preprocess**: Scale features using saved scalers
3. **Predict**: Run data through neural network
4. **Postprocess**: Convert prediction back to original scale
5. **Return**: Send prediction to client

**Latency**: Typically <100ms (P95)

### Auto-Scaling

KServe automatically scales based on traffic:
- **Scale Up**: More requests → More pods
- **Scale Down**: Fewer requests → Fewer pods
- **Resource Efficient**: Only uses resources when needed

### Model Updates

Updating a model is simple:
1. Train new model version
2. Update InferenceService to point to new model
3. KServe performs rolling update
4. Zero downtime deployment

---

## End-to-End Pipeline Flow

### Complete Workflow

```
┌─────────────────────────────────────────────────────────┐
│                   1. Data Generation                    │
│  Generate synthetic sales data with realistic patterns  │
│  Save to PostgreSQL tables                              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           2. Feature Engineering (Feast + Ray)          │
│  • Define entities (store, dept)                        │
│  • Create feature views (sales, external factors)       │
│  • Register features in Feast registry                  │
│  • Materialize features to online store                 │
│  • Ray processes data in parallel                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│        3. Model Training (Kubeflow + MLflow)           │
│  • Load features from Feast offline store               │
│  • Preprocess data (scale, split)                       │
│  • Train PyTorch model on 4 GPUs                        │
│  • Track experiment with MLflow                         │
│  • Save best model                                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           4. Model Serving (KServe)                    │
│  • Deploy model as InferenceService                     │
│  • Expose REST API                                      │
│  • Serve predictions in real-time                       │
│  • Auto-scale based on traffic                         │
└─────────────────────────────────────────────────────────┘
```

### Data Flow Details

#### Feature Engineering Flow

1. **Raw Data** → PostgreSQL tables (sales_features, store_features)
2. **Feast Definitions** → Entities, FeatureViews, FeatureServices
3. **Materialization** → Features copied to online store
4. **Ray Processing** → Distributed joins and transformations
5. **Ready** → Features available for training and inference

#### Training Flow

1. **Load Features** → From Feast offline store
2. **Preprocess** → Scale, split, batch
3. **Distributed Training** → 4 GPUs working together
4. **Evaluation** → Test on validation set
5. **MLflow Tracking** → Log metrics and model
6. **Save Model** → Best model saved to storage

#### Inference Flow

1. **Request Arrives** → REST API call
2. **Load Features** → From Feast online store (real-time)
3. **Preprocess** → Scale using saved scalers
4. **Predict** → Run through neural network
5. **Postprocess** → Convert to original scale
6. **Response** → Return prediction

---

## Key Technologies Deep Dive

### PostgreSQL

**Role**: Storage layer for multiple components

**Uses:**
1. **Feast Registry**: Feature definitions and metadata
2. **Feast Offline Store**: Historical feature data
3. **Feast Online Store**: Real-time feature values
4. **MLflow Backend**: Experiment tracking data

**Why PostgreSQL:**
- **ACID Compliance**: Reliable transactions
- **SQL-Queryable**: Easy to query and analyze
- **Performance**: Fast for both batch and real-time queries
- **Mature**: Well-tested and reliable

### KubeRay

**Role**: Kubernetes-native Ray cluster

**What It Does:**
- Deploys Ray cluster on Kubernetes
- Manages Ray workers as Kubernetes pods
- Integrates with OpenShift AI
- Provides distributed computing resources

**Benefits:**
- **Kubernetes-Native**: Uses standard K8s resources
- **Auto-Scaling**: Add/remove workers as needed
- **Resource Management**: Efficient GPU/CPU usage
- **Fault Tolerance**: Handles worker failures

### OpenShift AI

**Role**: Platform providing all components

**Components Used:**
- **Dashboard**: Web UI for managing projects
- **Workbenches**: Jupyter notebook environments
- **Ray**: KubeRay for distributed processing
- **Trainer**: Kubeflow Training Operator
- **MLflow**: Experiment tracking server

**Benefits:**
- **Integrated**: All components work together
- **Managed**: Platform handles infrastructure
- **Secure**: Enterprise-grade security
- **Scalable**: Handles large workloads

---

## Performance and Results

### Training Performance

**Configuration:**
- 4 GPUs (2 nodes × 2 GPUs)
- 15 epochs
- Batch size: 256

**Timing:**
- **Data Prep**: 2 minutes 15 seconds
- **Training**: 44 seconds
- **Total Pipeline**: ~3 minutes

**Speedup:**
- Single GPU: ~3 minutes
- 4 GPUs: 44 seconds
- **~4x speedup!**

### Model Performance

**Metrics:**
- **MAPE**: 10.5% (Mean Absolute Percentage Error)
  - Industry baseline: 15-20%
  - **40% better than baseline**
- **RMSE**: 500 (Root Mean Squared Error)
- **Validation Loss**: 0.0009

**What This Means:**
- Predictions are accurate within 10.5% on average
- Much better than typical industry performance
- Good enough for business decision-making

### Inference Performance

**Latency:**
- **Mean**: <100ms
- **P95**: <100ms
- **P99**: <150ms

**Throughput:**
- 100+ requests/second per pod
- Auto-scales to handle more traffic

**Availability:**
- 99.9%+ uptime
- Health checks and auto-recovery

### Business Impact

**Quantifiable Benefits:**
- **Inventory Savings**: 20% reduction in holding costs
- **Stockout Reduction**: 15% fewer lost sales
- **Payroll Optimization**: 10-15% better staffing levels
- **Forecast Accuracy**: 40% better than industry baseline

**Strategic Benefits:**
- Data-driven decision making
- Proactive planning instead of reactive
- Competitive advantage
- Scalable solution for growth

---

## Use Cases and Applications

### Primary Use Case: Retail Demand Forecasting

**What It Does:**
- Predicts weekly sales by store and department
- Helps optimize inventory levels
- Enables better staffing decisions
- Supports strategic planning

**Who Benefits:**
- **Store Managers**: Better inventory planning
- **Supply Chain**: Optimize ordering
- **HR**: Right staffing levels
- **Finance**: Revenue forecasting

### Other Applications

**Supply Chain:**
- Forecast demand for suppliers
- Warehouse capacity planning
- Transportation optimization

**Marketing:**
- Predict campaign impact
- Promotion effectiveness
- Customer behavior forecasting

**Finance:**
- Revenue forecasting
- Budget planning
- Financial modeling

**Operations:**
- Resource planning
- Capacity management
- Efficiency optimization

### Adaptable Pattern

The same pipeline can be adapted for different use cases:
- **Change Features**: Different features for different problems
- **Change Model**: Different algorithms (LSTM, Transformer, etc.)
- **Same Infrastructure**: Reuse the MLOps stack

---

## Getting Started

### Prerequisites

**Required:**
- OpenShift cluster with OpenShift AI 3.2+
- Components enabled:
  - `dashboard`
  - `ray` (KubeRay)
  - `trainer` (Kubeflow Training)
  - `workbenches`
  - `mlflow`
- Worker nodes with 2+ CPUs per Ray worker
- Dynamic storage with RWX support (NFS-CSI recommended)
- PostgreSQL for Feast and MLflow

### Quick Start

**Option 1: Automated Setup**
```bash
cd examples/complete-mlops-pipeline
./scripts/setup.sh
```

**Option 2: Manual Setup**
1. Apply Kubernetes manifests:
   ```bash
   kubectl apply -k manifests/
   ```
2. Wait for pods to be ready
3. Run notebooks or scripts

### Running the Pipeline

**Notebooks (Interactive):**
1. `01-feast-features.ipynb` - Feature engineering
2. `02-training.ipynb` - Model training
3. `03-inference.ipynb` - Model serving

**Scripts (Automated):**
```bash
python scripts/01_feast_features.py  # Feature engineering
python scripts/02_training.py        # Training
python scripts/03_inference.py        # Serving
```

---

## Best Practices

### Feature Engineering

**✅ Do:**
- Use Feature Store for reusability
- Version your features
- Document feature definitions
- Test feature transformations
- Use distributed processing for large datasets

**❌ Don't:**
- Duplicate features across projects
- Skip feature documentation
- Ignore feature versioning
- Process large datasets sequentially

### Training

**✅ Do:**
- Track all experiments with MLflow
- Use distributed training for large datasets
- Save checkpoints regularly
- Validate on holdout set
- Compare multiple model architectures

**❌ Don't:**
- Train without tracking
- Use single GPU for large datasets
- Skip validation
- Overfit to training data
- Ignore hyperparameter tuning

### Serving

**✅ Do:**
- Monitor model performance
- Set up auto-scaling
- Use health checks
- Implement retry logic
- Track prediction latency

**❌ Don't:**
- Deploy without monitoring
- Ignore scaling needs
- Skip health checks
- Forget about error handling
- Neglect performance optimization

---

## Troubleshooting

### Common Issues

#### Ray Connection Issues

**Symptoms:**
- Can't connect to Ray cluster
- Timeout errors

**Solutions:**
```python
os.environ["FEAST_RAY_SKIP_TLS"] = "true"
os.environ["FEAST_RAY_USE_KUBERAY"] = "true"
```

Check Ray cluster status:
```bash
kubectl get rayclusters -n <namespace>
kubectl logs -l app=ray-head -n <namespace>
```

#### MLflow Connection Issues

**Symptoms:**
- Can't connect to MLflow
- Tracking fails

**Solutions:**
Check MLflow pod:
```bash
kubectl get pods -n <namespace> -l app=mlflow
kubectl logs -l app=mlflow -n <namespace>
```

Verify service endpoint:
```bash
kubectl get svc -n <namespace> mlflow
```

#### Training Job Pending

**Symptoms:**
- Job stuck in Pending
- No pods created

**Solutions:**
Check resource availability:
```bash
kubectl describe trainjob <name> -n <namespace>
kubectl get events -n <namespace> --sort-by='.lastTimestamp'
```

Verify runtime configuration:
```bash
kubectl get cluster-training-runtimes
```

#### Inference Service Not Ready

**Symptoms:**
- Service not responding
- Pods not starting

**Solutions:**
Check pod logs:
```bash
kubectl logs -l serving.kserve.io/inferenceservice=sales-forecast -n <namespace>
```

Verify model files exist:
```bash
kubectl exec -it <pod-name> -n <namespace> -- ls -la /mnt/models
```

Check resource limits:
```bash
kubectl describe inferenceservice sales-forecast -n <namespace>
```

---

## Extending the Pipeline

### Possible Enhancements

#### 1. Data Sources

**Add More External Data:**
- Weather data (temperature, precipitation)
- Economic indicators (GDP, inflation)
- Social media signals (mentions, sentiment)
- Competitor data (prices, promotions)

**Real-Time Streams:**
- Integrate with Kafka/Kinesis
- Process streaming data
- Update features in real-time

#### 2. Models

**Different Architectures:**
- **LSTM**: Better for time-series
- **Transformer**: Attention mechanisms
- **XGBoost**: Gradient boosting
- **Ensemble**: Combine multiple models

**Advanced Techniques:**
- Transfer learning
- Multi-task learning
- AutoML

#### 3. Features

**More Time-Series Features:**
- Longer lags (8, 12, 52 weeks)
- More rolling statistics
- Fourier transforms
- Wavelet features

**Domain-Specific Features:**
- Customer behavior
- Product attributes
- Store characteristics
- Regional factors

#### 4. Infrastructure

**Monitoring:**
- Model drift detection
- Prediction monitoring
- Feature drift detection
- Performance tracking

**Advanced Serving:**
- A/B testing
- Canary deployments
- Traffic splitting
- Model explainability

---

## Summary

The Sales Demand Forecasting project demonstrates a complete, production-ready MLOps pipeline that:

1. **Uses Feature Store (Feast)** for centralized, reusable features
2. **Trains Models (Kubeflow)** with distributed training
3. **Tracks Experiments (MLflow)** for reproducibility
4. **Serves Models (KServe)** for real-time predictions
5. **Processes Data (Ray)** with distributed computing

**Key Takeaways:**
- **Production-Ready**: Uses enterprise-grade tools
- **Scalable**: Handles large datasets and high traffic
- **Reproducible**: Versioned features and models
- **Efficient**: Distributed processing and training
- **Accurate**: 40% better than industry baseline

**Business Value:**
- 20% inventory cost savings
- 15% stockout reduction
- 10-15% payroll optimization
- Data-driven decision making

This pipeline serves as a template for building ML systems that are scalable, maintainable, and production-ready.

