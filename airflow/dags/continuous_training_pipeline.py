from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="continuous_training_pipeline",
    description="CT: fetch dataset -> train model -> log to MLflow (+ upload .pt if configured)",
    default_args=default_args,
    start_date=datetime(2025, 10, 28),
    schedule_interval="0 2 * * *",  # tous les jours à 02:00
    catchup=False,
    tags=["ct", "training", "mlflow"],
) as dag:

    fetch_data = BashOperator(
        task_id="fetch_latest_data",
        bash_command="python -m src.data.fetch_image",
    )

    train_model = BashOperator(
        task_id="train_and_register_model",
        bash_command="python -m src.model.train",
    )

    fetch_data >> train_model
