import os
from dotenv import load_dotenv

"""
Centralise toutes les variables d'environnement utilisées par le projet
(dev: Docker Compose / prod: K8s ConfigMap & Secret).
"""

# Charge un éventuel .env en local (inoffensif si déjà fourni par Docker/K8s)
load_dotenv()

# ----------------------------
# Dossiers locaux (montés dans les containers)
# ----------------------------
LOCAL_DATA_DIR = os.getenv("LOCAL_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "models/")

# ----------------------------
# Stockage objet (MinIO / S3)
# ----------------------------
S3_DATA_BUCKET = os.getenv("S3_DATA_BUCKET", "plant-images")
S3_MODEL_BUCKET = os.getenv("S3_MODEL_BUCKET", "plant-models")

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")

# ----------------------------
# MLflow (tracking + artifacts)
# ----------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "dandelion_vs_grass")

# Utilisés quand MLflow stocke des artifacts sur MinIO
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
MLFLOW_ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", "s3://plant-models/mlflow/")

# ----------------------------
# Model registry / serving
# ----------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "dandelion-grass")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")  # ou "Staging"

# ----------------------------
# Base de données (PostgreSQL)
# ----------------------------
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB = os.getenv("PG_DB", "plants")
PG_USER = os.getenv("PG_USER", "mlops")
PG_PASSWORD = os.getenv("PG_PASSWORD", "mlops")

# ----------------------------
# Options d'entraînement
# ----------------------------
PRETRAINED = os.getenv("PRETRAINED", "false").lower() == "true"
