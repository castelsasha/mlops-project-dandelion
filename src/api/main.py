"""
FastAPI inference service for the dandelion vs grass classifier.

Key responsibilities:
- On startup: fetch the latest model from MinIO/S3 (if available) and load it.
- Expose /health to report readiness.
- Expose /predict to classify one uploaded image.

Notes:
- Configuration values (S3 endpoint, bucket names, local model dir) are loaded
  from src.utils.config. This keeps environment-specific details out of the code.
- Downloading the latest model is best-effort: if MinIO/S3 is unavailable,
  the API falls back to any model already present in LOCAL_MODEL_DIR.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Tuple

import boto3
from botocore.config import Config as BotoConfig
from fastapi import FastAPI, UploadFile, File, HTTPException

from src.model.inference import predict_image, load_latest_model
from src.utils.config import (
    S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY,
    S3_MODEL_BUCKET, LOCAL_MODEL_DIR
)

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
app = FastAPI(title="Dandelion vs Grass Classifier API")
logger = logging.getLogger("uvicorn.error")

# The in-memory model instance. Loaded once at startup.
model = None  # type: ignore[var-annotated]


def _download_latest_from_s3() -> Optional[str]:
    """
    Download the most recently modified model file from
    s3://{S3_MODEL_BUCKET}/models/ into LOCAL_MODEL_DIR.

    Returns:
        The local file path of the downloaded model, or None if nothing was downloaded.

    Behavior:
        - If MinIO/S3 is unreachable or empty, we log a warning and return None.
        - We intentionally do not crash on network issues to allow the API to
          start with any existing local model.
    """
    # Configure a short network timeout & path-style addressing for MinIO
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=BotoConfig(s3={"addressing_style": "path"}, retries={"max_attempts": 3}),
    )
    prefix = "models/"

    try:
        resp = s3.list_objects_v2(Bucket=S3_MODEL_BUCKET, Prefix=prefix)
    except Exception as e:
        logger.warning("[API] MinIO unavailable or misconfigured: %s", e)
        return None

    contents = resp.get("Contents") or []
    if not contents:
        logger.info("[API] No objects under s3://%s/%s (fallback to local).", S3_MODEL_BUCKET, prefix)
        return None

    # Pick most recently modified object
    latest_obj = max(contents, key=lambda x: x["LastModified"])
    latest_key = latest_obj["Key"]

    # Ensure local dir exists and download
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    local_path = os.path.join(LOCAL_MODEL_DIR, os.path.basename(latest_key))
    try:
        s3.download_file(S3_MODEL_BUCKET, latest_key, local_path)
        logger.info("[API] Downloaded model: %s -> %s", latest_key, local_path)
        return local_path
    except Exception as e:
        logger.warning("[API] Failed to download %s: %s", latest_key, e)
        return None


@app.on_event("startup")
def _startup() -> None:
    """
    FastAPI startup hook:
    - Try to fetch the freshest model from MinIO/S3
    - Load the latest available model from LOCAL_MODEL_DIR
    """
    global model

    logger.info("[API] Model initialization…")
    try:
        _download_latest_from_s3()  # best-effort; ignore return value here
    except Exception as e:
        logger.warning("[API] S3 download step failed: %s", e)

    try:
        # load_latest_model() should pick the newest file present in LOCAL_MODEL_DIR
        model_local = load_latest_model()
        model = model_local
        logger.info("[API] Model loaded ✅")
    except Exception as e:
        logger.exception("[API] Could not load a local model: %s", e)
        model = None


@app.get("/health")
def health() -> dict:
    """
    Liveness/readiness probe.
    Returns {"status": "ok", "model_loaded": true/false}
    """
    return {"status": "ok", "model_loaded": bool(model)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """
    Predict the class of the uploaded image.

    Request:
      multipart/form-data with a single 'file' part.

    Response (200):
      {
        "prediction": "dandelion" | "grass",
        "confidence": float   # e.g. 0.97
      }

    Errors:
      400 if the image cannot be decoded.
      500 if the model is not loaded.
    """
    if model is None:
        # Model was not loaded at startup; surface a clear 500 to the caller
        raise HTTPException(status_code=500, detail="Model is not loaded")

    try:
        image_bytes: bytes = await file.read()
        label, confidence = predict_image(model, image_bytes)  # type: Tuple[str, float]
        return {"prediction": label, "confidence": confidence}
    except HTTPException:
        raise
    except Exception as e:
        # Convert any parsing/decoding error to a user-facing 400
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}") from e