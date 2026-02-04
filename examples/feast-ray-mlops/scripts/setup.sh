#!/bin/bash
# =============================================================================
# Feast + Ray + Kubeflow + MLflow - Automated Setup
# Based on working manifests from v2/manifests/
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFESTS_DIR="${SCRIPT_DIR}/../manifests"
NAMESPACE="feast-mlops-demo"

echo "============================================================"
echo "  MLOps Pipeline Setup"
echo "  Feast + Ray + Kubeflow + MLflow on OpenShift AI"
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

echo "Step 1/4: Creating namespace and prerequisites..."
kubectl apply -f "${MANIFESTS_DIR}/00-prereqs.yaml"
echo "   Waiting for namespace to be ready..."
sleep 3

echo "Step 2/4: Deploying PostgreSQL..."
kubectl apply -f "${MANIFESTS_DIR}/01-postgres.yaml"
echo "   Waiting for PostgreSQL to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n ${NAMESPACE} --timeout=180s || {
    echo "   ⚠️ PostgreSQL pods not ready yet, continuing..."
}

echo "Step 3/4: Deploying MLflow..."
kubectl apply -f "${MANIFESTS_DIR}/02-mlflow.yaml"
echo "   Waiting for MLflow to be ready..."
sleep 10  # Give time for init container and deployment
kubectl wait --for=condition=ready pod -l app=mlflow -n ${NAMESPACE} --timeout=180s || {
    echo "   ⚠️ MLflow not ready yet, continuing..."
}

echo "Step 4/4: Deploying KubeRay cluster..."
kubectl apply -f "${MANIFESTS_DIR}/03-kuberay.yaml"
echo "   Waiting for Ray cluster to be ready..."
sleep 15  # Ray cluster takes time to initialize
kubectl wait --for=condition=ready pod -l ray.io/cluster=feast-ray -n ${NAMESPACE} --timeout=300s || {
    echo "   ⚠️ Ray cluster not fully ready, continuing..."
}

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
echo "   MLflow:        https://${MLFLOW_URL}"
echo "   Ray Dashboard: https://${RAY_URL}"
echo ""
echo "📓 Next steps:"
echo "   1. Create a workbench in OpenShift AI Dashboard"
echo "   2. Clone this repo and navigate to examples/feast-ray-mlops/notebooks/"
echo "   3. Run notebooks in order: 01-setup → 02-feast-features → 03-training → 04-inference"
echo ""
echo "💡 Troubleshooting:"
echo "   kubectl get pods -n ${NAMESPACE}"
echo "   kubectl describe pod <pod-name> -n ${NAMESPACE}"
echo "   kubectl logs <pod-name> -n ${NAMESPACE}"
echo ""
