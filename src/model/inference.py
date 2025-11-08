# src/model/inference.py
"""
Lightweight inference utilities:
- build_model(): construct a ResNet-18 with the correct output head.
- load_latest_model(): load last saved .pt/.pth checkpoint from LOCAL_MODEL_DIR.
- predict_image(): run preprocessing + model forward and return (label, confidence).

Notes:
- We purposely keep preprocessing simple at inference time (Resize+Normalize).
- The label mapping must match your training dataset indices. Here we assume:
    0 -> "grass", 1 -> "dandelion".
"""

from __future__ import annotations

import io
import os
from typing import Tuple

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from torchvision import models

from src.utils.config import LOCAL_MODEL_DIR

# ImageNet normalization expected by torchvision ResNet backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["grass", "dandelion"]  # must match training label order!


def build_model(num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    """
    Build a ResNet-18 backbone with a custom classification head.
    If pretrained=True, ImageNet weights are used (requires internet unless cached).
    """
    if pretrained:
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m.to(DEVICE)


def _load_any_torch_object(path: str):
    """
    Try a safe load (weights_only=True). If that fails (pickled module),
    fallback to a full load (weights_only=False).
    WARNING: full pickle load can execute arbitrary code; only load trusted files.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def load_latest_model() -> nn.Module:
    """
    Load the newest .pt/.pth file from LOCAL_MODEL_DIR.
    Supports:
      - state_dict checkpoints (dict)
      - full pickled nn.Module
    """
    if not os.path.isdir(LOCAL_MODEL_DIR):
        raise RuntimeError(f"Model folder '{LOCAL_MODEL_DIR}' does not exist.")

    candidates = []
    for fname in os.listdir(LOCAL_MODEL_DIR):
        if fname.endswith((".pt", ".pth")):
            full = os.path.join(LOCAL_MODEL_DIR, fname)
            candidates.append((full, os.path.getmtime(full)))

    if not candidates:
        raise RuntimeError("No model file found in models/. Run training first.")

    latest_path = sorted(candidates, key=lambda x: x[1])[-1][0]
    print(f"[INFERENCE] Loading latest local model: {latest_path}")

    obj = _load_any_torch_object(latest_path)

    if isinstance(obj, dict):  # state_dict
        model = build_model(pretrained=False)
        model.load_state_dict(obj, strict=True)
        model.eval()
        return model

    if isinstance(obj, nn.Module):
        model = obj.to(DEVICE)
        model.eval()
        return model

    raise RuntimeError(f"Unsupported model payload type: {type(obj)}")


_preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


@torch.inference_mode()
def predict_image(model: nn.Module, image_bytes: bytes) -> Tuple[str, float]:
    """
    Convert bytes -> RGB PIL -> tensor -> forward pass.
    Returns:
        label (str), confidence (float in [0, 1]).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = _preprocess(img).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu()
    idx = int(probs.argmax().item())
    label = CLASS_NAMES[idx]
    return label, float(probs[idx].item())