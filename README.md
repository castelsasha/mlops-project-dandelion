# 🌼 MLOps Project — Dandelion vs Grass Classification

Goal: classify images and detect if the plant is **dandelion** or **grass**.

This repository implements a **complete modern MLOps stack**, not just a model:
- reproducible dataset creation  
- training with MLflow tracking  
- model artifact versioning in S3 (MinIO)  
- API serving with FastAPI  
- WebApp with Streamlit  
- Docker Compose for local orchestration  
- optional Kubernetes deployment + GitHub Actions CI/CD  

Author: **Arnaud Fernandes - Sasha Castel - Saber Dhib - Noa Sebag - Camil Nitel**

---

## ⚙️ End-to-End Pipeline Overview

1. Download and clean raw images  
2. Train a **ResNet-18** (transfer learning, PyTorch)  
3. Track experiments in **MLflow**  
4. Push best `.pt` checkpoint to **MinIO (S3)**  
5. Serve predictions via **FastAPI**  
6. Interact through **Streamlit** (drag-and-drop interface)

Result: a fully automated ML product — **train → version → serve → interact**

---

## 🧩 Architecture

User → Streamlit → FastAPI → Model → Prediction  
           ↑  
        S3 (models)  
        MLflow (metrics)

---

## 🗂 Repository structure

.  
├─ src/data/    → data ingestion & preprocessing  
├─ src/model/   → training, evaluation, S3 registry  
├─ src/api/    → FastAPI inference service  
├─ src/utils/   → config helpers (env-based)  
├─ webapp/    → Streamlit interface  
├─ docker/    → Dockerfiles  
├─ k8s/      → Kubernetes manifests  
├─ scripts/    → deployment automation  
├─ data/raw/   → local dataset  
├─ .github/workflows/ → CI/CD workflows  
└─ docker-compose*.yml → local orchestration  

---

## 🧪 Run locally (development mode)

# Setup environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start infra (MinIO + MLflow)
docker compose up -d minio mlflow

# Train model
python -m src.model.train

# Run API (FastAPI)
uvicorn src.api.main:app --reload

# Healthcheck & prediction
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -F "file=@data/raw/dandelion_00000010.jpg"

# Run Streamlit UI
streamlit run webapp/app.py
# → http://localhost:8501

---

## 🐳 Run with Docker Compose (full stack)

docker compose -f docker-compose.yml -f docker-compose.app.yml up -d minio mlflow api webapp

# Access:
# MinIO:  http://127.0.0.1:9001
# MLflow: http://127.0.0.1:5001
# API:    http://127.0.0.1:8000/docs
# WebApp: http://localhost:8501

curl http://127.0.0.1:8000/health

---

## ☸️ Run with Kubernetes (Docker Desktop)

# Apply manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check deployment
kubectl -n dandelion get pods,svc

# Forward API to localhost
kubectl -n dandelion port-forward svc/dandelion-api 8000:8000
curl http://127.0.0.1:8000/health

# Or use the helper script
chmod +x scripts/deploy_local.sh
./scripts/deploy_local.sh

---

## 🔁 Local CI/CD (self-hosted runner)

A local GitHub Actions workflow (`.github/workflows/dev-local.yaml`) automates:
- Docker image build (`dandelion-api:latest`)
- Kubernetes deployment & restart
- Healthcheck smoke test  

To run it, configure a **self-hosted runner** with Docker + kubectl enabled.

---

## 🧱 Model Storage

Models are saved automatically after training:  
- local path → `models/model_<timestamp>_best.pt`  
- remote copy → `MinIO / plant-models/models/`  

The API automatically loads the latest model from MinIO at startup.

---

## 💡 Why this project matters

This project demonstrates a **real MLOps pipeline**:
- automated data → training → deployment  
- reproducible experiment tracking with MLflow  
- containerized, production-ready API  
- live monitoring via health & prediction endpoints  

A true **end-to-end ML product**, not just a notebook.