# Feast Feature Server with Ray support
# Base: OpenShift AI Ray image (has Ray pre-installed)
FROM quay.io/modh/ray:2.52.1-py312-cu128

USER root

# Install Feast with Ray and PostgreSQL support
RUN pip install --no-cache-dir \
    'feast[postgres,ray]==0.59.0' \
    psycopg2-binary \
    uvicorn \
    codeflare-sdk>=0.31.1 \
    && pip cache purge

# Create feast user and directories
RUN mkdir -p /feast-data /feast && \
    chmod -R 777 /feast-data /feast

# Set working directory
WORKDIR /feast

# Default command for feature server
CMD ["feast", "serve", "-h", "0.0.0.0", "-p", "6566"]

# Expose ports
# 6566 - Feature server HTTP
# 6567 - Online server
# 8816 - Offline server gRPC
EXPOSE 6566 6567 8816
