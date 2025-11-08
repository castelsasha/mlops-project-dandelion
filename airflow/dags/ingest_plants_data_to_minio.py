# airflow/dags/ingest_plants_data_to_minio.py
from __future__ import annotations

import io
import os
import time
import certifi
import requests
import boto3
import pendulum
from datetime import timedelta
from urllib.parse import urlparse
from pathlib import Path
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException

import mysql.connector
from mysql.connector import Error as MySQLError
from botocore.exceptions import ClientError

# Environment-driven configuration (provided by docker-compose.airflow env_file)
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "mlops")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "mlops")
MYSQL_DB = os.getenv("MYSQL_DB", "mlops")

# Use your *data* bucket here (not the model bucket):
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
S3_BUCKET = os.getenv("S3_DATA_BUCKET", "plant-images")
S3_PREFIX = os.getenv("S3_PREFIX", "plants")  # e.g. plants/{label}/...

# Friendly UA + referer for CDNs like Pexels/Unsplash
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/129.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": UA,
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.pexels.com/",
}

def _mysql_conn():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
    )

def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

def _ensure_bucket(s3, bucket: str):
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        s3.create_bucket(Bucket=bucket)

def _choose_key(row: dict, default_ext: str = ".jpg") -> str:
    """
    Build a stable S3 key: plants/{label}/{id}_{basename}
    """
    src = row["url_source"]
    label = row["label"] or "unknown"
    rid = row["id"]
    base = Path(urlparse(src).path).name or f"{rid}{default_ext}"
    if not Path(base).suffix:
        base = f"{base}{default_ext}"
    return f"{S3_PREFIX}/{label}/{rid}_{base}"

def _download_bytes(url: str, timeout=20, retries=3) -> bytes | None:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout, verify=certifi.where(), headers=HEADERS)
            r.raise_for_status()
            return r.content
        except Exception as e:
            time.sleep(0.5 * attempt)
    return None

DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

@dag(
    dag_id="ingest_plants_data_to_minio",
    default_args=DEFAULT_ARGS,
    schedule=None,  # manual trigger
    start_date=pendulum.datetime(2024, 1, 1, tz="Europe/Paris"),
    catchup=False,
    tags=["ingestion", "minio", "sql"],
    max_active_runs=1,
)
def ingest_plants_data_to_minio():

    @task
    def fetch_rows(limit: int = 200, label: str | None = None) -> list[dict]:
        """
        Read rows where url_s3 IS NULL (to process only missing images).
        Optionally filter by label.
        """
        q = ["SELECT id, url_source, label FROM plants_data WHERE url_s3 IS NULL"]
        args = []
        if label:
            q.append("AND label = %s")
            args.append(label)
        q.append("ORDER BY id ASC LIMIT %s")
        args.append(limit)

        cnx = _mysql_conn()
        try:
            cur = cnx.cursor(dictionary=True)
            cur.execute(" ".join(q), args)
            rows = cur.fetchall()
            return rows
        finally:
            cnx.close()

    @task
    def push_and_update(rows: list[dict]) -> dict:
        """
        For each row: download -> upload to S3 -> set url_s3.
        """
        if not rows:
            return {"processed": 0, "updated": 0, "skipped": 0}

        s3 = _s3_client()
        _ensure_bucket(s3, S3_BUCKET)

        updated = 0
        skipped = 0

        cnx = _mysql_conn()
        try:
            cur = cnx.cursor()
            for row in rows:
                src = row["url_source"]
                data = _download_bytes(src)
                if not data:
                    skipped += 1
                    continue

                key = _choose_key(row)
                s3.put_object(Bucket=S3_BUCKET, Key=key, Body=io.BytesIO(data), ContentType="image/jpeg")
                s3_uri = f"s3://{S3_BUCKET}/{key}"

                cur.execute("UPDATE plants_data SET url_s3 = %s WHERE id = %s", (s3_uri, row["id"]))
                updated += 1

            cnx.commit()
        finally:
            cnx.close()

        return {"processed": len(rows), "updated": updated, "skipped": skipped}

    summary = push_and_update(fetch_rows())
    # You can add a tiny assertion task if you want to fail on 0 updates.

dag = ingest_plants_data_to_minio()