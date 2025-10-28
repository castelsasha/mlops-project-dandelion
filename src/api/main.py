import io
import torch
import boto3
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from src.model.inference import predict_image, load_latest_model
from src.utils.config import (
    S3_ENDPOINT_URL,
    S3_ACCESS_KEY,
    S3_SECRET_KEY,
    S3_MODEL_BUCKET,
)

app = FastAPI(title="Dandelion vs Grass Classifier API")

# Chargement du modèle (local ou S3)
print("[API] Initialisation du modèle...")

try:
    # Connexion à MinIO (S3)
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )

    # Liste des objets dans le bucket
    response = s3.list_objects_v2(Bucket=S3_MODEL_BUCKET, Prefix="models/")
    if "Contents" not in response:
        print("[API] Aucun modèle trouvé sur MinIO, fallback en local.")
        model = load_latest_model()
    else:
        # Télécharge le plus récent modèle
        files = sorted(response["Contents"], key=lambda x: x["LastModified"], reverse=True)
        latest_key = files[0]["Key"]

        local_path = f"/tmp/{latest_key.split('/')[-1]}"
        s3.download_file(S3_MODEL_BUCKET, latest_key, local_path)
        print(f"[API] Modèle téléchargé depuis MinIO: {latest_key}")

        # Charge le modèle
        model = load_latest_model()
        print("[API] Modèle chargé avec succès ✅")

except Exception as e:
    print(f"[API] ⚠️ Erreur MinIO, chargement local: {e}")
    model = load_latest_model()

# Healthcheck
@app.get("/health")
def health():
    return {"status": "ok"}

# Endpoint principal
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Envoie une image -> renvoie la classe prédite + confiance
    """
    try:
        image_bytes = await file.read()
        label, confidence = predict_image(model, image_bytes)
        return {"prediction": label, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))