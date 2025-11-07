import os
import boto3
from fastapi import FastAPI, UploadFile, File, HTTPException

from src.model.inference import predict_image, load_latest_model
from src.utils.config import (
    S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY,
    S3_MODEL_BUCKET, LOCAL_MODEL_DIR
)

app = FastAPI(title="Dandelion vs Grass Classifier API")

# --------------------------------------------------
# téléchargement auto du dernier .pt depuis S3
# --------------------------------------------------
def _download_latest_from_s3():
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )
    try:
        resp = s3.list_objects_v2(Bucket=S3_MODEL_BUCKET, Prefix="models/")
    except Exception as e:
        print(f"[API] MinIO indisponible / mauvaise conf: {e}")
        return None

    if "Contents" not in resp or not resp["Contents"]:
        print(f"[API] Aucun .pt sous s3://{S3_MODEL_BUCKET}/models/ -- fallback local")
        return None

    files = sorted(resp["Contents"], key=lambda x: x["LastModified"], reverse=True)
    latest_key = files[0]["Key"]

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    local_path = os.path.join(LOCAL_MODEL_DIR, os.path.basename(latest_key))
    s3.download_file(S3_MODEL_BUCKET, latest_key, local_path)
    print(f"[API] Modèle téléchargé: {latest_key} -> {local_path}")
    return local_path


print("[API] Initialisation du modèle...")
try:
    _download_latest_from_s3()
except Exception as e:
    print(f"[API] ⚠️ Téléchargement depuis MinIO impossible: {e}")

try:
    model = load_latest_model()
    print("[API] Modèle chargé ✅")
except Exception as e:
    print(f"[API] ❌ Impossible de charger le modèle local: {e}")
    model = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(model)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    try:
        image_bytes = await file.read()
        label, confidence = predict_image(model, image_bytes)
        return {"prediction": label, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
