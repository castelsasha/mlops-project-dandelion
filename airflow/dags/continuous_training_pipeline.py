# airflow/dags/continuous_training_pipeline.py
from __future__ import annotations

import os
from datetime import timedelta
import pendulum

from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException

# This DAG fetches the dataset (local CSV/images) and then trains the model.
# It relies on your repo code:
#   - python -m src.data.fetch_image
#   - python -m src.model.train
#
# Notes:
# - We keep tasks as Python callables that shell out to your modules.
# - Retries & timeouts make failures visible but resilient.
# - Use env inherited from `env_file: .env` in docker-compose.airflow.yml.

DEFAULT_ARGS = {
    "owner": "mlops-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

@dag(
    dag_id="continuous_training_pipeline",
    description="CT: fetch dataset -> train model -> log to MLflow (+ upload .pt to MinIO)",
    default_args=DEFAULT_ARGS,
    start_date=pendulum.datetime(2025, 10, 28, tz="Europe/Paris"),
    schedule="0 2 * * *",  # every day at 02:00
    catchup=False,
    tags=["ct", "training", "mlflow"],
    max_active_runs=1,
)
def continuous_training_pipeline():

    @task.execution_timeout(timedelta(minutes=15))
    def fetch_latest_data():
        """
        Run the local fetch script to download/refresh images + metadata.csv.
        We shell out to keep the exact same codepath as your CLI usage.
        """
        import subprocess, sys
        cmd = [sys.executable, "-m", "src.data.fetch_image"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print(res.stderr)
            raise AirflowFailException("fetch_image failed")

    @task.execution_timeout(timedelta(minutes=60))
    def train_and_register_model():
        """
        Run the training script. It logs to MLflow and uploads the best .pt to MinIO.
        """
        import subprocess, sys
        cmd = [sys.executable, "-m", "src.model.train"]
        env = os.environ.copy()
        # Example: override planning epochs for CT if you want
        # env["EPOCHS"] = os.getenv("EPOCHS", "10")
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            print(res.stderr)
            raise AirflowFailException("training failed")

    fetch_latest_data() >> train_and_register_model()

dag = continuous_training_pipeline()