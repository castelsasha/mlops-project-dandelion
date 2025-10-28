import os
from io import BytesIO

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from src.utils.config import (
    LOCAL_MODEL_DIR,
    S3_ENDPOINT_URL,
    S3_ACCESS_KEY,
    S3_SECRET_KEY,
    S3_MODEL_BUCKET,
)

# Mapping index -> label (même ordre qu'au training)
IDX_TO_LABEL = {0: "dandelion", 1: "grass"}

# Préprocessing identique à l'entraînement (très important)
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

device = "cuda" if torch.cuda.is_available() else "cpu"


def _build_empty_model(num_classes: int = 2):
    """
    Reconstruit la même archi que celle utilisée lors du training :
    ResNet18 + dernière couche remplacée par une Linear(num_classes).
    Pas de poids ImageNet ici car on va charger nos poids entraînés.
    """
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


def _load_state_dict_into_model(model_path: str):
    """
    Charge les poids depuis un .pt sauvegardé et renvoie un modèle prêt pour l'inférence.
    """
    model = _build_empty_model(num_classes=2)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    return model


def load_latest_model():
    """
    Charge automatiquement le modèle le plus récent dans LOCAL_MODEL_DIR.
    On suppose que les fichiers suivent le format model_<timestamp>.pt
    """
    if not os.path.exists(LOCAL_MODEL_DIR):
        raise RuntimeError(
            f"Le dossier modèle '{LOCAL_MODEL_DIR}' n'existe pas. "
            "Lance d'abord l'entraînement pour générer un modèle."
        )

    model_files = [
        f for f in os.listdir(LOCAL_MODEL_DIR)
        if f.startswith("model_") and f.endswith(".pt")
    ]
    if not model_files:
        raise RuntimeError(
            "Aucun modèle trouvé dans le dossier models/. "
            "Lance d'abord l'entraînement."
        )

    # tri décroissant => le plus récent en premier
    model_files.sort(reverse=True)
    latest_path = os.path.join(LOCAL_MODEL_DIR, model_files[0])

    print(f"[INFERENCE] Loading latest local model: {latest_path}")
    return _load_state_dict_into_model(latest_path)


def predict_image(model, image_bytes: bytes):
    """
    Prend une image brute (bytes), applique le preprocess
    et renvoie (classe_str, confiance_float)
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = PREPROCESS(image).unsqueeze(0).to(device)  # shape [1,3,224,224]

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    conf, pred_idx = torch.max(probs, dim=0)
    pred_label = IDX_TO_LABEL[pred_idx.item()]
    confidence = float(conf.item())

    return pred_label, confidence


def _get_s3_client():
    """
    Client S3-compatible (MinIO en dev, AWS S3 en prod).
    """
    return boto3.client(
        "s3",
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        endpoint_url=S3_ENDPOINT_URL,
        config=Config(signature_version="s3v4"),
    )


def download_latest_model_from_s3(tmp_dir: str = "/tmp"):
    """
    Va chercher le dernier modèle présent dans le bucket S3_MODEL_BUCKET / 'models/'.
    Le télécharge localement dans /tmp, charge les poids dans un ResNet18,
    et renvoie le modèle prêt pour l'inférence.

    -> Utilisable si on veut que l'API charge le modèle depuis le registry S3
       au lieu du disque local.
    """
    s3 = _get_s3_client()

    try:
        objects = s3.list_objects_v2(
            Bucket=S3_MODEL_BUCKET,
            Prefix="models/",
        )
    except ClientError as e:
        raise RuntimeError(f"[S3] Unable to list models in bucket {S3_MODEL_BUCKET}: {e}")

    if "Contents" not in objects or len(objects["Contents"]) == 0:
        raise RuntimeError("[S3] No models found in registry (bucket empty).")

    # Trier par LastModified décroissant => dernier modèle en premier
    files = sorted(objects["Contents"], key=lambda x: x["LastModified"], reverse=True)
    latest_key = files[0]["Key"]  # ex: models/model_1761675398.pt

    local_path = os.path.join(tmp_dir, os.path.basename(latest_key))

    print(f"[S3] Downloading latest model {latest_key} -> {local_path}")
    s3.download_file(S3_MODEL_BUCKET, latest_key, local_path)

    # Charger ce modèle en mémoire
    model = _load_state_dict_into_model(local_path)
    print(f"[S3] ✅ Model loaded from S3: {latest_key}")
    return model