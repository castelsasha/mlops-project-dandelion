FROM python:3.11-slim

# Outils utiles
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dépendances serveur MLflow + S3
RUN pip install --no-cache-dir \
      "mlflow==3.5.1" \
      "boto3==1.40.62"

# Dossier persistant (SQLite + state)
WORKDIR /mlflow
VOLUME /mlflow

# MLflow server (backend local sqlite + artifacts sur MinIO)
CMD ["mlflow","server", \
     "--backend-store-uri","sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination","s3://plant-models/mlflow/", \
     "--serve-artifacts", \
     "--host","0.0.0.0","--port","5000"]
