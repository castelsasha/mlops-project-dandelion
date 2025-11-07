# MLOps Project — Image Classification (Dandelion vs Grass)

This repository implements a full end-to-end MLOps pipeline for a binary image classifier that distinguishes **dandelions** from **grass**.

This project follows the assignment requirements:

- Modular code (preprocessing, training, inference, web UI)
- Model tracking + versioning with MLflow
- Model artifacts stored in S3 (MinIO)
- Deployable inference API (FastAPI)
- Interactive WebApp (Streamlit)
- Dockerized services
- Ready for CI/CD & Kubernetes deployment

> **Main student:** **Arnaud Fernandes**

---

## 1) Repository structure

.
├─ src/
│  ├─ data/
│  ├─ model/           ← training, registry, inference
│  ├─ api/             ← FastAPI app
│  └─ utils/
├─ webapp/             ← Streamlit UI
├─ docker/             ← Dockerfiles
├─ tests/              ← unit & integration tests
├─ docker-compose.yml
├─ docker-compose.app.yml
├─ docker-compose.db.yml
└─ README.md

---

## 2) Functional Components

| Component | Technology |
|----------|------------|
| Model | PyTorch — ResNet18 |
| Tracking + Registry | MLflow |
| Artifact Store | MinIO S3 |
| API | FastAPI / Uvicorn |
| Web UI | Streamlit |
| Containerization | Docker |
| Local Orchestration | docker-compose |
| CI/CD | GitHub Actions (to be added) |
| Datastore (optional) | SQL table `plants_data` |

---

## 3) Pipeline Summary

1) Data is downloaded from URLs (greenr-airflow dataset)  
2) (Optional) insert metadata into SQL  
3) (Optional) Airflow DAG downloads images + pushes to MinIO  
4) Training script trains ResNet18  
5) MLflow logs metrics & model version  
6) Model is saved + pushed to S3 under `plant-models/models/`  
7) API loads latest model from S3 at startup  
8) WebApp sends images to API to get predictions

---

## 4) MLflow / S3 Proof of Work

Artifacts stored under:
s3://plant-models/models/
Example file after training:
model_1762541755.pt
---

## 5) How to run locally (dev)

```bash
docker compose up -d minio mlflow
python -m src.model.train
uvicorn src.api.main:app --reload
streamlit run webapp/app.py

curl -s http://127.0.0.1:8000/health

curl -s -X POST http://127.0.0.1:8000/predict \
  -F "file=@data/raw/dandelion_00000010.jpg"

curl -s -X POST http://127.0.0.1:8000/predict \
  -F "file=@data/raw/grass_00000010.jpg"

  ghcr.io/castelsasha/mlops-project-dandelion-api:latest

  --------------------------------------------------------------------------------------------

ça c’est 100% propre  
→ tu le colles directement dans ton README.md

---

si validé → réponds juste :  

**“go SQL”**  
et j’enchaîne instant avec `plants_data.sql` (le fichier optionnel de l’énoncé).