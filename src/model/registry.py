# src/model/registry.py
"""
Minimal S3/MinIO registry helper for model files (.pt/.pth).
- upload_model_to_s3(): push a local checkpoint under the 'models/' prefix and
  return a s3://... URI. The API container knows how to fetch the 'latest' file.
"""

from __future__ import annotations

import os
import boto3
from src.utils.config import (
    S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY, S3_MODEL_BUCKET
)


def upload_model_to_s3(local_path: str) -> str:
    """
    Upload a local checkpoint to S3/MinIO under 'models/<filename>'.
    Returns:
        s3_uri (str): e.g. s3://plant-models/models/model_1234567890.pt
    """
    if not os.path.isfile(local_path):
        raise FileNotFoundError(local_path)

    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )
    key = f"models/{os.path.basename(local_path)}"
    s3.upload_file(
        local_path,
        S3_MODEL_BUCKET,
        key,
        ExtraArgs={"ContentType": "application/octet-stream"},
    )
    uri = f"s3://{S3_MODEL_BUCKET}/{key}"
    print(f"[REGISTRY] Uploaded to {uri}")
    return uri