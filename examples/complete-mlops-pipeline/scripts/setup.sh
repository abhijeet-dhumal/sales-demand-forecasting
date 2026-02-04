#!/bin/bash
# =============================================================================
# Feast + Ray + Kubeflow + MLflow + Model Registry - Automated Setup
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFESTS_DIR="${SCRIPT_DIR}/../manifests"
NAMESPACE="feast-mlops-demo"

echo "============================================================"
echo "  Complete MLOps Pipeline Setup"
echo "  Feast + Ray + Kubeflow + MLflow + Model Registry"
echo "============================================================"
echo ""

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl not found. Please install kubectl."
    exit 1
fi

if ! kubectl auth can-i create namespace &> /dev/null; then
    echo "❌ Insufficient permissions. Please login to your cluster."
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Apply manifests in order
echo "📦 Applying manifests..."
echo ""

echo "Step 1/6: Creating namespace and ClusterTrainingRuntime..."
kubectl apply -f "${MANIFESTS_DIR}/00-prereqs.yaml"
sleep 3

echo "Step 2/6: Deploying PostgreSQL..."
kubectl apply -f "${MANIFESTS_DIR}/01-postgres.yaml"
echo "   Waiting for PostgreSQL to be ready..."
kubectl wait --for=condition=available deployment/feast-postgres -n ${NAMESPACE} --timeout=180s

echo "Step 3/6: Deploying MLflow..."
kubectl apply -f "${MANIFESTS_DIR}/02-mlflow.yaml"
echo "   Waiting for MLflow to be ready..."
sleep 10
kubectl wait --for=condition=available deployment/mlflow -n ${NAMESPACE} --timeout=180s || true

echo "Step 4/6: Deploying KubeRay cluster..."
kubectl apply -f "${MANIFESTS_DIR}/03-kuberay.yaml"
echo "   Waiting for Ray cluster to be ready..."
kubectl wait --for=condition=ready pod -l ray.io/cluster=feast-ray,ray.io/node-type=head -n ${NAMESPACE} --timeout=300s

echo "Step 5/6: Creating RBAC and shared storage..."
kubectl apply -f "${MANIFESTS_DIR}/04-feast-prereqs.yaml"

echo "Step 6/7: Submitting dataprep job (Feast + KubeRay)..."
kubectl apply -f "${MANIFESTS_DIR}/05-dataprep-job.yaml"
echo "   Waiting for dataprep to complete (this may take 2-3 minutes)..."
kubectl wait --for=condition=complete job/feast-dataprep-ray -n ${NAMESPACE} --timeout=600s

echo "Step 7/7: Deploying Model Registry..."
kubectl apply -f "${MANIFESTS_DIR}/07-model-registry.yaml"
echo "   Waiting for Model Registry to be ready..."
sleep 10
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=sales-model-registry -n ${NAMESPACE} --timeout=180s || true

echo ""
echo "============================================================"
echo "  ✅ Setup Complete!"
echo "============================================================"
echo ""
echo "Resources deployed in namespace: ${NAMESPACE}"
echo ""
echo "📊 Check status:"
echo "   kubectl get pods -n ${NAMESPACE}"
echo ""
echo "🔗 Access URLs (OpenShift routes):"
MLFLOW_URL=$(kubectl get route mlflow -n ${NAMESPACE} -o jsonpath='{.spec.host}' 2>/dev/null || echo 'Not available')
RAY_URL=$(kubectl get route feast-ray-dashboard -n ${NAMESPACE} -o jsonpath='{.spec.host}' 2>/dev/null || echo 'Not available')
REGISTRY_URL=$(kubectl get route sales-model-registry -n ${NAMESPACE} -o jsonpath='{.spec.host}' 2>/dev/null || echo 'Not available')
echo "   MLflow:         https://${MLFLOW_URL}"
echo "   Ray Dashboard:  https://${RAY_URL}"
echo "   Model Registry: https://${REGISTRY_URL}"
echo ""
echo "📓 Next steps:"
echo "   1. Create a workbench in OpenShift AI Dashboard"
echo "   2. Attach the 'feast-pvc' storage to the workbench"
echo "   3. Clone this repo and navigate to examples/feast-ray-mlops/notebooks/"
echo "   4. Run notebooks: 01-feast-features → 02-training → 03-inference"
echo ""
echo "🚀 Or submit training job directly:"
echo "   kubectl apply -f ${MANIFESTS_DIR}/06-trainjob.yaml"
echo ""
