# Sales Forecasting MLOps Pipeline - Makefile
#
# Usage:
#   make deploy        - Deploy all infrastructure
#   make dataprep      - Run data preparation job
#   make train         - Run training job
#   make logs          - Tail training logs
#   make status        - Show all pod status
#   make clean         - Delete namespace (destructive!)
#

NAMESPACE := feast-trainer-demo
KUBECTL := kubectl
KUSTOMIZE := kubectl kustomize

.PHONY: help deploy dataprep train logs status clean build-image push-image

help:
	@echo "Sales Forecasting MLOps Pipeline"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make deploy       - Deploy all infrastructure (Postgres, MLflow, Ray, Feast)"
	@echo "  make status       - Show all pod status"
	@echo "  make clean        - Delete namespace (WARNING: destructive!)"
	@echo ""
	@echo "Jobs:"
	@echo "  make dataprep     - Run Feast data preparation job"
	@echo "  make train        - Run distributed training job"
	@echo "  make logs         - Tail training job logs"
	@echo ""
	@echo "Development:"
	@echo "  make build-image  - Build custom training image"
	@echo "  make push-image   - Push training image to registry"
	@echo "  make lint         - Validate YAML manifests"
	@echo ""
	@echo "URLs (after deploy):"
	@echo "  make urls         - Show service URLs"

# Infrastructure deployment
deploy:
	@echo "Deploying infrastructure..."
	$(KUBECTL) apply -k manifests/
	@echo ""
	@echo "Waiting for pods to be ready..."
	$(KUBECTL) wait --for=condition=available deployment/postgres -n $(NAMESPACE) --timeout=120s || true
	$(KUBECTL) wait --for=condition=available deployment/mlflow -n $(NAMESPACE) --timeout=120s || true
	@echo ""
	@echo "Infrastructure deployed! Run 'make status' to check pods."

# Data preparation
dataprep:
	@echo "Starting Feast data preparation..."
	$(KUBECTL) delete job feast-dataprep -n $(NAMESPACE) --ignore-not-found
	$(KUBECTL) apply -f manifests/05-dataprep-job.yaml
	@echo ""
	@echo "Following logs..."
	sleep 5
	$(KUBECTL) logs -f job/feast-dataprep -n $(NAMESPACE) || true

# Training job
train:
	@echo "Starting distributed training..."
	$(KUBECTL) delete trainjob sales-training -n $(NAMESPACE) --ignore-not-found
	$(KUBECTL) apply -f manifests/06-trainjob.yaml
	@echo ""
	@echo "Training job submitted. Run 'make logs' to follow."

# Tail logs
logs:
	@echo "Tailing training logs (Ctrl+C to exit)..."
	$(KUBECTL) logs -f -l trainer.kubeflow.org/trainjob-name=sales-training -n $(NAMESPACE) --all-containers=true 2>/dev/null || \
	$(KUBECTL) logs -f -l app=sales-training -n $(NAMESPACE) --all-containers=true

# Status
status:
	@echo "=== Pod Status ==="
	$(KUBECTL) get pods -n $(NAMESPACE) -o wide
	@echo ""
	@echo "=== Training Jobs ==="
	$(KUBECTL) get trainjob -n $(NAMESPACE) 2>/dev/null || echo "No TrainJobs"
	@echo ""
	@echo "=== Ray Cluster ==="
	$(KUBECTL) get raycluster -n $(NAMESPACE) 2>/dev/null || echo "No RayClusters"

# URLs
urls:
	@echo "=== Service URLs ==="
	@echo ""
	@echo "MLflow UI:"
	@$(KUBECTL) get route mlflow -n $(NAMESPACE) -o jsonpath='  https://{.spec.host}{"\n"}' 2>/dev/null || echo "  Not available"
	@echo ""
	@echo "Feast UI:"
	@$(KUBECTL) get route feast-ui -n $(NAMESPACE) -o jsonpath='  https://{.spec.host}{"\n"}' 2>/dev/null || echo "  Not available"
	@echo ""
	@echo "Ray Dashboard:"
	@$(KUBECTL) get route feast-ray-dashboard -n $(NAMESPACE) -o jsonpath='  https://{.spec.host}{"\n"}' 2>/dev/null || echo "  Not available"

# Build custom training image
IMAGE_REPO ?= quay.io/your-org
IMAGE_TAG ?= v1

build-image:
	@echo "Building training image..."
	podman build -t $(IMAGE_REPO)/sales-training:$(IMAGE_TAG) images/training/
	@echo ""
	@echo "Built: $(IMAGE_REPO)/sales-training:$(IMAGE_TAG)"

push-image:
	@echo "Pushing training image..."
	podman push $(IMAGE_REPO)/sales-training:$(IMAGE_TAG)

# Lint manifests
lint:
	@echo "Validating YAML manifests..."
	@for f in manifests/*.yaml; do \
		echo "Checking $$f..."; \
		$(KUBECTL) apply --dry-run=client -f $$f > /dev/null || exit 1; \
	done
	@echo "All manifests valid!"

# Clean up (destructive!)
clean:
	@echo "WARNING: This will delete the entire namespace and all data!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	$(KUBECTL) delete namespace $(NAMESPACE) --ignore-not-found
	@echo "Namespace deleted."

# Port forwards for local development
port-forward-mlflow:
	$(KUBECTL) port-forward svc/mlflow 5000:5000 -n $(NAMESPACE)

port-forward-feast:
	$(KUBECTL) port-forward svc/feast-server 6566:6566 -n $(NAMESPACE)

# Watch training progress
watch:
	watch -n 5 "kubectl get pods -n $(NAMESPACE) && echo && kubectl get trainjob -n $(NAMESPACE)"
