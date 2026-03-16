# Feast Feature Server with Ray + PostgreSQL + Redis support
#
# This image extends the default Feast feature server with:
# - Ray for distributed offline store operations
# - PostgreSQL for registry and offline store
# - Redis for online store
# - gRPC for registry server
#
# Build:
#   podman build -t quay.io/abdhumal/feast-ray-server:latest images/feast-server/
#
# Push:
#   podman push quay.io/abdhumal/feast-ray-server:latest

FROM registry.access.redhat.com/ubi9/python-311:1

USER 0

# Install system dependencies for psycopg2 and grpcio
RUN dnf install -y --nodocs \
    libpq-devel \
    gcc \
    python3.11-devel \
    && dnf clean all

# Copy uv for fast package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_CACHE_DIR=/tmp/uv-cache

# Install Feast with all required extras (as root for permissions)
# - postgres: PostgreSQL registry and offline store
# - ray: Distributed offline store and batch engine
# - redis: Redis online store
# - grpcio: gRPC server support
RUN uv pip install --system --python /usr/bin/python3.11 --no-cache \
    "feast[postgres,ray,redis,grpcio]==0.61.0" \
    "psycopg2-binary" \
    "codeflare-sdk==0.35.0" \
    "pyarrow>=14.0.0"

# Switch to non-root user
# Note: UI write access handled via volumeMount in FeatureStore CR (not chmod)
USER 1001

# Set environment variables
ENV FEAST_USAGE=false
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden by operator)
CMD ["feast", "serve", "--help"]
