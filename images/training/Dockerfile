# Custom Training Image for Sales Forecasting
#
# Pre-bakes all dependencies to eliminate runtime pip install (~60s savings per run)
#
# Build:
#   podman build -t quay.io/<org>/sales-training:v1 images/training/
#
# Push:
#   podman push quay.io/<org>/sales-training:v1
#
# Then update 06-trainjob.yaml:
#   trainer.image: quay.io/<org>/sales-training:v1
#
FROM quay.io/modh/training:py312-cuda128-torch280

# Install ML dependencies
RUN pip install --no-cache-dir \
    "feast[postgres,ray]==0.59.0" \
    codeflare-sdk \
    psycopg2-binary \
    scikit-learn \
    joblib \
    mlflow \
    model-registry \
    pandas \
    pyarrow \
    numpy

# Create directories
RUN mkdir -p /app /tmp/artifacts

# Copy training script (optional - can still use inline script)
COPY train_ddp.py /app/train_ddp.py

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command (overridden by TrainJob)
CMD ["python", "/app/train_ddp.py"]
