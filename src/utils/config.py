"""
src/utils/config.py

Single source of truth for configuration variables.

- In dev, values typically come from a local `.env` file or docker-compose `env_file`.
- In prod, values should be injected via Kubernetes ConfigMaps/Secrets.
- Import from this module instead of calling os.getenv() all over the codebase.

Tip:
    Keep sensible defaults so local runs “just work”, but never hardcode secrets
    in source control for real production projects.
"""

from __future__ import annotations

import os
from typing import Optional
from dotenv import load_dotenv

# Load a local .env if present (harmless if everything is already provided by Docker/K8s)
load_dotenv()


# ---------------------------------------------------------------------------
# Small helpers to parse environment variables safely
# ---------------------------------------------------------------------------

def env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean env var with common spellings: true/1/yes/on."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    """Parse an int env var, falling back to a default on error/absence."""
    val = os.getenv(name)
    try:
        return int(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def env_str(name: str, default: str) -> str:
    """Get a string env var with a default."""
    return os.getenv(name, default)


# ---------------------------------------------------------------------------
# Local filesystem paths (mounted into containers when running with Docker)
# ---------------------------------------------------------------------------

LOCAL_DATA_DIR: str = env_str("LOCAL_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR: str = env_str("PROCESSED_DATA_DIR", "data/processed")
LOCAL_MODEL_DIR: str = env_str("LOCAL_MODEL_DIR", "models/")  # API reads models from here


# ---------------------------------------------------------------------------
# Object storage (MinIO / S3-compatible)
# ---------------------------------------------------------------------------

# Buckets
S3_DATA_BUCKET: str = env_str("S3_DATA_BUCKET", "plant-images")   # raw/processed images
S3_MODEL_BUCKET: str = env_str("S3_MODEL_BUCKET", "plant-models") # trained models

# Endpoint + credentials (MinIO by default in docker-compose)
S3_ENDPOINT_URL: str = env_str("S3_ENDPOINT_URL", "http://localhost:9000")
S3_ACCESS_KEY: str = env_str("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY: str = env_str("S3_SECRET_KEY", "minioadmin")


# ---------------------------------------------------------------------------
# MLflow tracking / artifacts
# ---------------------------------------------------------------------------

# Tracking server URL (exposed as 5001 on host by docker-compose)
MLFLOW_TRACKING_URI: str = env_str("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
MLFLOW_EXPERIMENT_NAME: str = env_str("MLFLOW_EXPERIMENT_NAME", "dandelion_vs_grass")

# When MLflow stores artifacts on MinIO, it needs the S3 endpoint + keys
MLFLOW_S3_ENDPOINT_URL: str = env_str("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
AWS_ACCESS_KEY_ID: str = env_str("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY: str = env_str("AWS_SECRET_ACCESS_KEY", "minioadmin")
MLFLOW_ARTIFACT_ROOT: str = env_str("MLFLOW_ARTIFACT_ROOT", "s3://plant-models/mlflow/")


# ---------------------------------------------------------------------------
# Model registry / serving (generic knobs you may use in the future)
# ---------------------------------------------------------------------------

MODEL_NAME: str = env_str("MODEL_NAME", "dandelion-grass")
MODEL_STAGE: str = env_str("MODEL_STAGE", "Production")  # or "Staging"


# ---------------------------------------------------------------------------
# Relational databases
# Choose one in practice — we expose both for flexibility.
# ---------------------------------------------------------------------------

# PostgreSQL (used by some dev setups; currently optional in this project)
PG_HOST: str = env_str("PG_HOST", "localhost")
PG_PORT: int = env_int("PG_PORT", 5432)
PG_DB: str = env_str("PG_DB", "plants")
PG_USER: str = env_str("PG_USER", "mlops")
PG_PASSWORD: str = env_str("PG_PASSWORD", "mlops")

# MySQL (recommended for the optional plants_data table in the assignment)
MYSQL_HOST: str = env_str("MYSQL_HOST", "localhost")
MYSQL_PORT: int = env_int("MYSQL_PORT", 3306)
MYSQL_DB: str = env_str("MYSQL_DB", "mlops")
MYSQL_USER: str = env_str("MYSQL_USER", "mlops")
MYSQL_PASSWORD: str = env_str("MYSQL_PASSWORD", "mlops")

# Optional DSN helpers (handy for libs that accept a URL)
def mysql_dsn() -> str:
    """Return a MySQL connection URL (mysql-connector-python flavor)."""
    return f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"

def postgres_dsn() -> str:
    """Return a PostgreSQL connection URL (psycopg / SQLAlchemy flavor)."""
    return f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"


# ---------------------------------------------------------------------------
# Training switches
# ---------------------------------------------------------------------------

# Whether to start from a pretrained backbone or train from scratch
PRETRAINED: bool = env_bool("PRETRAINED", False)