#!/bin/bash
# =============================================================================
# Feast + Ray + Kubeflow + MLflow - Resource Cleanup
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFESTS_DIR="${SCRIPT_DIR}/../manifests"
NAMESPACE="feast-mlops-demo"

echo "============================================================"
echo "  MLOps Pipeline Cleanup"
echo "============================================================"
echo ""
echo "⚠️  This will delete ALL resources in namespace: ${NAMESPACE}"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🗑️  Cleaning up resources..."
echo ""

# Delete TrainJobs first
echo "Deleting TrainJobs..."
kubectl delete trainjob --all -n ${NAMESPACE} --ignore-not-found=true 2>/dev/null || true

# Delete RayJobs
echo "Deleting RayJobs..."
kubectl delete rayjob --all -n ${NAMESPACE} --ignore-not-found=true 2>/dev/null || true

# Delete manifests in reverse order
echo "Deleting KubeRay cluster..."
kubectl delete -f "${MANIFESTS_DIR}/03-kuberay.yaml" --ignore-not-found=true 2>/dev/null || true

echo "Deleting MLflow..."
kubectl delete -f "${MANIFESTS_DIR}/02-mlflow.yaml" --ignore-not-found=true 2>/dev/null || true

echo "Deleting PostgreSQL..."
kubectl delete -f "${MANIFESTS_DIR}/01-postgres.yaml" --ignore-not-found=true 2>/dev/null || true

# Delete init jobs
echo "Deleting init jobs..."
kubectl delete job postgres-init -n ${NAMESPACE} --ignore-not-found=true 2>/dev/null || true

echo "Deleting prerequisites..."
kubectl delete -f "${MANIFESTS_DIR}/00-prereqs.yaml" --ignore-not-found=true 2>/dev/null || true

# Option to delete the entire namespace
echo ""
read -p "Delete the entire namespace ${NAMESPACE}? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Deleting namespace..."
    kubectl delete namespace ${NAMESPACE} --ignore-not-found=true 2>/dev/null || true
    echo "   ⏳ Namespace deletion may take a few minutes..."
fi

echo ""
echo "============================================================"
echo "  ✅ Cleanup Complete!"
echo "============================================================"
