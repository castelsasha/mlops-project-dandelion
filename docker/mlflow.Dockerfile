# -----------------------------------------------------------------------------
# MLflow server image:
# - Artifacts pushed to MinIO (S3-compatible)
# - Tracking backend stored locally in a mounted volume (SQLite)
# -----------------------------------------------------------------------------
FROM python:3.11-slim

# Useful base tools (curl for health/debug if needed)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# MLflow + MinIO/S3 client
RUN pip install --no-cache-dir \
      "mlflow==3.5.1" \
      "boto3==1.40.62"

# Persist MLflow state (SQLite DB + local files)
WORKDIR /mlflow
VOLUME /mlflow

# Start MLflow server:
# - backend-store-uri -> local SQLite file
# - artifacts-destination -> MinIO bucket path
CMD ["mlflow","server", \
     "--backend-store-uri","sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination","s3://plant-models/mlflow/", \
     "--serve-artifacts", \
     "--host","0.0.0.0","--port","5000"]