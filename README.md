# MLOps Project — Dandelion vs Grass Classification

Goal: classify images and detect if the plant is **dandelion** or **grass**.

This repository is not “just training a model”.  
It implements a COMPLETE modern MLOps stack:

- reproducible dataset creation
- training with MLflow tracking
- artifact versioning in S3 (MinIO)
- API deployment (FastAPI)
- user interface (Streamlit)
- docker-compose for local orchestration
- (optional) Airflow + Kubernetes + CI/CD GitHub Actions

Author: **Arnaud Fernandes**

---

## What the system does end-to-end

1) download raw images
2) train a ResNet18 (transfer learning)
3) log metrics + artifacts into MLflow
4) push best `.pt` model → MinIO (S3)
5) FastAPI loads the latest model at startup
6) Streamlit allows a user to drag & drop an image and see prediction

result: fully automated ML product, not just code

---

## Architecture (text version diagram)

User   → Streamlit → FastAPI → Model → Response
                                ↑
                               S3 (model versions)
                              MLflow (metrics)

dataset can come from local CSV OR SQL + S3 ingestion

---

## Repository structure

.
├─ src/data/             → dataset ingestion, CSV, SQL, S3 push/pull
├─ src/model/            → training / inference / registry → MLflow
├─ src/api/              → FastAPI inference server
├─ src/utils/            → config helpers (ENV based)
├─ webapp/               → Streamlit UI
├─ docker/               → Dockerfiles
├─ data/raw/             → local images + metadata.csv
├─ tests/                → tests
└─ docker-compose*.yml   → launch dev stack (minio/mlflow/api/ui/db)

---

## How to run locally (development)

### 1) Launch infra (MinIO + MLflow locally)

docker compose up -d minio mlflow

### 2) download small dataset (if needed)

python -m src.data.fetch_image

### 3) train model

python -m src.model.train

the best `.pt` is saved locally and pushed into S3 (plant-models/models/*)

### 4) launch API locally

uvicorn src.api.main:app --reload

### 5) launch Streamlit UI

streamlit run webapp/app.py

---

## API smoke test

curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict -F "file=@data/raw/example.jpg"

---

## Where models are stored

MinIO bucket: `plant-models/models/`  
each training run produces a timestamped `.pt`

best model (by val F1) overwrites API target

---

## CI/CD & Kubernetes (optional)

- artifact container images pushed to GHCR
- Deployment updated automatically
- API pods reload model on startup
- Airflow DAGs can trigger continuous training

---

## Why this project matters

this repo demonstrates a real production pipeline:  
**not just training → but deploying + monitoring + versioning**

This is the core of modern ML engineer work.