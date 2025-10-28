import os
from dotenv import load_dotenv

load_dotenv()

LOCAL_DATA_DIR = os.getenv("LOCAL_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "models/")

S3_DATA_BUCKET = os.getenv("S3_DATA_BUCKET", "plant-images")
S3_MODEL_BUCKET = os.getenv("S3_MODEL_BUCKET", "plant-models")

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "dandelion_vs_grass")
