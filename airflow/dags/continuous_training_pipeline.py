from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

"""
DAG de Continuous Training :
1. Télécharge / met à jour les données
2. Réentraîne le modèle
3. (Implicitement) pousse le modèle dans la registry S3/MinIO via train.py

Hypothèse:
- Airflow tourne dans un conteneur Docker qui a accès au repo (monté en volume)
- Le venv / deps sont disponibles dans l'image Airflow
"""

default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="continuous_training_pipeline",
    description="Pipeline quotidien de réentraînement du classifieur dandelion_vs_grass",
    default_args=default_args,
    start_date=datetime(2025, 10, 28),
    schedule_interval="0 2 * * *",  # tous les jours à 02h00
    catchup=False,
) as dag:

    fetch_data = BashOperator(
        task_id="fetch_latest_data",
        bash_command="python -m src.data.fetch_images",
    )

    train_model = BashOperator(
        task_id="train_and_register_model",
        bash_command="python -m src.model.train",
    )

    fetch_data >> train_model