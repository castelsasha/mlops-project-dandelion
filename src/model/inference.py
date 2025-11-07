import io, os, torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from torchvision import models
from src.utils.config import LOCAL_MODEL_DIR

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_model(num_classes=2, pretrained=False):
    # Aucun téléchargement de poids dans le container par défaut
    if pretrained:
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m.to(DEVICE)

def _load_any_torch_object(path: str):
    """
    Tente d'abord un chargement 'sécurisé' (weights_only=True).
    Si ça échoue (modèle picklé complet), retente avec weights_only=False.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # ⚠️ weights_only=False peut exécuter du code arbitraire si la source n’est pas fiable.
        # Ici on charge un modèle que TU as produit: ok.
        return torch.load(path, map_location="cpu", weights_only=False)

def load_latest_model():
    if not os.path.isdir(LOCAL_MODEL_DIR):
        raise RuntimeError(f"Le dossier modèle '{LOCAL_MODEL_DIR}' n'existe pas.")
    # accepte .pt et .pth
    candidates = sorted([p for p in os.listdir(LOCAL_MODEL_DIR) if p.endswith((".pt", ".pth"))])
    if not candidates:
        raise RuntimeError("Aucun modèle trouvé dans le dossier models/. Lance l'entraînement.")
    latest = os.path.join(LOCAL_MODEL_DIR, candidates[-1])
    print(f"[INFERENCE] Loading latest local model: {latest}")

    obj = _load_any_torch_object(latest)

    # Cas 1 : un state_dict -> on instancie l’archi et on charge
    if isinstance(obj, dict):
        model = build_model(pretrained=False)  # pas de download
        model.load_state_dict(obj)
        model.eval()
        return model

    # Cas 2 : un module picklé complet
    if isinstance(obj, nn.Module):
        model = obj.to(DEVICE)
        model.eval()
        return model

    # Sinon, format inconnu
    raise RuntimeError(f"Fichier modèle non supporté: {type(obj)}")

_pre = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

@torch.inference_mode()
def predict_image(model, image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = _pre(img).unsqueeze(0).to(DEVICE)
    out = model(x)
    prob = torch.softmax(out, dim=1)[0].detach().cpu()
    idx = int(prob.argmax().item())
    label = "dandelion" if idx == 1 else "grass"
    return label, float(prob[idx].item())