from __future__ import annotations

import os
import requests
from datetime import datetime, timedelta

import psycopg2
from airflow.decorators import dag, task

# ---------- Config via ENV (compat Docker Compose) ----------
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
S3_ACCESS_KEY   = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY   = os.getenv("S3_SECRET_KEY", "minioadmin")
S3_DATA_BUCKET  = os.getenv("S3_DATA_BUCKET", "plant-images")

PG_HOST = os.getenv("PG_HOST", "postgres")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB   = os.getenv("PG_DB", "plants")
PG_USER = os.getenv("PG_USER", "mlops")
PG_PASS = os.getenv("PG_PASSWORD", "mlops")

TIMEOUT = 20
RETRIES = 3


def _db_conn():
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB, user=PG_USER, password=PG_PASS
    )


@dag(
    dag_id="etl_plants_to_s3",
    description="ETL: read plants_data (url_source) -> validate -> upload to MinIO -> set url_s3",
    start_date=datetime(2025, 10, 28),
    schedule="0 1 * * *",
    catchup=False,
    default_args={
        "owner": "mlops-team",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["etl", "minio", "postgres"],
)
def etl_plants_to_s3():

    @task()
    def extract_rows() -> list[tuple[int, str, str]]:
        """Retourne (id, url_source, label) où url_s3 est NULL."""
        with _db_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, url_source, label
                FROM plants_data
                WHERE url_source IS NOT NULL AND url_source <> ''
                  AND url_s3 IS NULL
                  AND label IN ('dandelion','grass')
                LIMIT 500
                """
            )
            rows = cur.fetchall()
        return rows

    @task()
    def download_and_upload(rows: list[tuple[int, str, str]]) -> list[tuple[int, str]]:
        """Télécharge, valide, envoie sur S3. Retourne [(id, s3_uri), ...]."""
        if not rows:
            return []

        from src.data.upload_s3 import (
            make_s3_client,
            validate_image_bytes,
            img_bytes_and_key,
            put_s3_image,
        )

        s3 = make_s3_client(S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY)

        updated: list[tuple[int, str]] = []
        for (rid, url, label) in rows:
            content = None
            for _ in range(RETRIES):
                try:
                    r = requests.get(url, timeout=TIMEOUT)
                    if r.ok:
                        content = r.content
                        break
                except Exception:
                    pass
            if not content:
                continue

            img = validate_image_bytes(content)
            if not img:
                continue

            data, key = img_bytes_and_key(img, label)
            s3_uri = put_s3_image(s3, S3_DATA_BUCKET, data, key)
            updated.append((rid, s3_uri))

        return updated

    @task()
    def update_db(updated: list[tuple[int, str]]) -> None:
        if not updated:
            return
        with _db_conn() as conn, conn.cursor() as cur:
            for rid, s3_uri in updated:
                cur.execute("UPDATE plants_data SET url_s3=%s WHERE id=%s", (s3_uri, rid))
            conn.commit()

    rows = extract_rows()
    uploaded = download_and_upload(rows)
    update_db(uploaded)


etl_plants_to_s3()
