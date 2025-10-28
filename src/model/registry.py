import os
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError, NoCredentialsError
from src.utils.config import (
    S3_ENDPOINT_URL,
    S3_ACCESS_KEY,
    S3_SECRET_KEY,
    S3_MODEL_BUCKET,
)

def _get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        endpoint_url=S3_ENDPOINT_URL,
        config=Config(signature_version="s3v4"),
    )

def ensure_bucket_exists(bucket_name: str):
    s3 = _get_s3_client()
    existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if bucket_name not in existing:
        s3.create_bucket(Bucket=bucket_name)
        print(f"[REGISTRY] Created bucket '{bucket_name}'")

def upload_model_to_s3(local_model_path: str) -> str | None:
    if not os.path.exists(local_model_path):
        print(f"[REGISTRY] ❌ Model file does not exist: {local_model_path}")
        return None

    ensure_bucket_exists(S3_MODEL_BUCKET)

    filename = os.path.basename(local_model_path)
    object_key = f"models/{filename}"  # ex: models/model_1761675398.pt
    s3_uri = f"s3://{S3_MODEL_BUCKET}/{object_key}"

    s3 = _get_s3_client()

    print(f"[REGISTRY] Uploading {local_model_path} -> {s3_uri}")
    try:
        s3.upload_file(local_model_path, S3_MODEL_BUCKET, object_key)
        print(f"[REGISTRY] ✅ Upload success: {s3_uri}")
        return s3_uri
    except NoCredentialsError:
        print("[REGISTRY] ❌ No S3 credentials found.")
    except ClientError as e:
        print(f"[REGISTRY] ❌ Client error during upload: {e}")

    return None