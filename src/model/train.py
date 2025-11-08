# src/model/train.py
"""
Training script for the Dandelion vs Grass classifier (ResNet-18).

This version focuses on better generalization and reproducibility:
- Data augmentation (RandomResizedCrop, HorizontalFlip, ColorJitter).
- Label smoothing + weight decay.
- CosineAnnealingLR scheduler.
- Early stopping on val_f1 with patience.
- AMP (mixed precision) when CUDA is available.
- Best-model checkpointing (by val_f1).
- MLflow logging (metrics/params/artifacts) + S3 upload of the best .pt.

Expected dataset interface:
- If USE_DB=False (default): src.data.dataset.PlantDataset(csv_file="data/raw/metadata.csv")
- If USE_DB=True: src.data.dataset_db.PlantDatasetDB()  (reads from MySQL/S3)

Label order should be consistent with inference:
    0 -> "grass", 1 -> "dandelion"
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms as T
from sklearn.metrics import f1_score, accuracy_score, recall_score
from tqdm import tqdm
import mlflow

from src.utils.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    LOCAL_MODEL_DIR,
)

# =========================
# Hyperparameters & knobs
# =========================

USE_DB = False            # switch to DB-backed dataset if you need it
EPOCHS = int(os.getenv("EPOCHS", "10"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LR = float(os.getenv("LR", "3e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))
VAL_RATIO = float(os.getenv("VAL_RATIO", "0.2"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
LABEL_SMOOTHING = float(os.getenv("LABEL_SMOOTHING", "0.05"))
PATIENCE = int(os.getenv("PATIENCE", "3"))           # early stopping patience (epochs)
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))     # macOS: keep 0 to avoid spawn issues
PIN_MEMORY = bool(os.getenv("PIN_MEMORY", "0") == "1")

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class History:
    best_f1: float = -1.0
    best_path: Optional[str] = None
    epochs_no_improve: int = 0


# ================
# Data pipeline
# ================

# Train-time augmentations for robustness
train_tfms = T.Compose([
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Validation-time transforms (deterministic)
val_tfms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_dataloaders(metadata_csv: str | None = "data/raw/metadata.csv") -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val dataloaders. When USE_DB=True, we use a DB-backed dataset that
    fetches image bytes from S3; otherwise a CSV-based dataset reads local files.
    """
    print("[DATA] Preparing datasets and dataloaders...")
    if USE_DB:
        from src.data.dataset_db import PlantDatasetDB
        full_dataset = PlantDatasetDB(train_transforms=train_tfms, val_transforms=val_tfms)
    else:
        from src.data.dataset import PlantDataset
        full_dataset = PlantDataset(csv_file=metadata_csv, train_transforms=train_tfms, val_transforms=val_tfms)

    val_size = max(int(len(full_dataset) * VAL_RATIO), 1)
    train_size = max(len(full_dataset) - val_size, 1)

    torch.manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Tell split subsets which transforms to use (if your dataset supports it)
    if hasattr(train_dataset.dataset, "set_split_transforms"):
        train_dataset.dataset.set_split_transforms(train_tfms, val_tfms)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    print(f"[DATA] Train size: {train_size}, Val size: {val_size}")
    return train_loader, val_loader


# ==================
# Model & training
# ==================

def build_model(num_classes: int = 2) -> nn.Module:
    """
    Start from ImageNet weights for a stronger baseline.
    """
    print("[MODEL] ResNet18 (ImageNet pretrained)…")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
    """
    Run a full validation pass and compute loss/acc/f1/recall.
    """
    model.eval()
    all_preds, all_targets = [], []
    running = 0.0

    for x, y in tqdm(loader, leave=False, desc="Val"):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        running += loss.item() * x.size(0)
        preds = out.argmax(1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y.detach().cpu().numpy())

    n = max(len(loader.dataset), 1)
    return {
        "loss": running / n,
        "acc": accuracy_score(all_targets, all_preds),
        "f1": f1_score(all_targets, all_preds, average="macro"),
        "recall": recall_score(all_targets, all_preds, average="macro"),
    }


def train_one_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Dict[str, float]:
    """
    Standard train loop with optional AMP. We compute running loss and simple metrics.
    """
    model.train()
    all_preds, all_targets = [], []
    running = 0.0

    for x, y in tqdm(loader, leave=False, desc="Train"):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and device == "cuda":
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        running += loss.item() * x.size(0)
        preds = out.argmax(1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y.detach().cpu().numpy())

    n = max(len(loader.dataset), 1)
    return {
        "loss": running / n,
        "acc": accuracy_score(all_targets, all_preds),
        "f1": f1_score(all_targets, all_preds, average="macro"),
        "recall": recall_score(all_targets, all_preds, average="macro"),
    }


def save_model(model: nn.Module, output_dir: str = LOCAL_MODEL_DIR, tag: str | None = None) -> str:
    """
    Save a state_dict checkpoint with an optional tag (e.g., 'best').
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    name = f"model_{timestamp}{('_' + tag) if tag else ''}.pt"
    model_path = os.path.join(output_dir, name)
    torch.save(model.state_dict(), model_path)
    print(f"[MODEL] Saved at {model_path}")
    return model_path


def main():
    # ==============
    # MLflow setup
    # ==============
    print("[MLFLOW] Setting up tracking...")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", MLFLOW_EXPERIMENT_NAME)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    registered_model_name = os.getenv("MODEL_NAME", "").strip()  # optional MLflow registry name
    target_stage = os.getenv("MODEL_STAGE", "None").strip()      # e.g. "Production"

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"[MLFLOW] Run id: {run_id}")

        # -------- Data --------
        train_loader, val_loader = get_dataloaders("data/raw/metadata.csv")

        # -------- Model / Loss / Optim / Sched --------
        model = build_model(num_classes=2)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(EPOCHS - 1, 1))
        scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

        # -------- Training loop with Early Stopping --------
        hist = History()
        for epoch in range(1, EPOCHS + 1):
            tr = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
            va = evaluate(model, val_loader, criterion)
            scheduler.step()

            # Log to MLflow
            mlflow.log_metrics({
                "train_loss": tr["loss"], "train_acc": tr["acc"], "train_f1": tr["f1"],
                "val_loss": va["loss"],   "val_acc": va["acc"],   "val_f1": va["f1"],
                "val_recall": va["recall"], "lr": scheduler.get_last_lr()[0],
            }, step=epoch)

            print(f"[EPOCH {epoch}/{EPOCHS}] "
                  f"train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
                  f"val_acc={va['acc']:.3f} val_f1={va['f1']:.3f} lr={scheduler.get_last_lr()[0]:.2e}")

            # Checkpoint best model by val_f1
            if va["f1"] > hist.best_f1:
                hist.best_f1 = va["f1"]
                hist.epochs_no_improve = 0
                best_path = save_model(model, tag="best")
                hist.best_path = best_path
                mlflow.log_artifact(best_path, artifact_path="weights")
            else:
                hist.epochs_no_improve += 1

            # Early stopping
            if hist.epochs_no_improve >= PATIENCE:
                print(f"[EARLY STOP] No improvement in val_f1 for {PATIENCE} epoch(s).")
                break

        # Final save (last epoch)
        last_path = save_model(model, tag="last")
        mlflow.log_artifact(last_path, artifact_path="weights")

        # Params for reproducibility
        mlflow.log_params({
            "epochs_planned": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "val_ratio": VAL_RATIO,
            "label_smoothing": LABEL_SMOOTHING,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "device": device,
            "use_db": USE_DB,
        })

        # Log an MLflow model (flavor PyTorch) using the current model weights
        example = (np.random.rand(1, 3, 224, 224).astype("float32"))
        log_kwargs = {
            "pytorch_model": model,
            "artifact_path": "model",
            "input_example": example,
        }
        if registered_model_name:
            log_kwargs["registered_model_name"] = registered_model_name
        mlflow.pytorch.log_model(**log_kwargs)

        print("\n[TRAINING] ✅ Done.")
        print(f"🏃 View run at: {tracking_uri}/#/experiments/"
              f"{mlflow.get_experiment(run.info.experiment_id).experiment_id}/runs/{run_id}")

        # Upload the BEST checkpoint to S3 (if any), so the API pulls the newest good model
        try:
            from src.model.registry import upload_model_to_s3
            target_to_push = hist.best_path if hist.best_path else last_path
            s3_uri = upload_model_to_s3(target_to_push)
            mlflow.log_text(s3_uri, artifact_file="deployed_model_s3_uri.txt")
            print(f"[REGISTRY] pushed to {s3_uri}")
        except Exception as e:
            print(f"[REGISTRY] ⚠️ upload to S3 failed: {e}")

        # Optional: promote to a registry stage (if using MLflow Model Registry)
        if registered_model_name and target_stage and target_stage.lower() != "none":
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            vers = client.search_model_versions(f"name='{registered_model_name}'")
            linked = [v for v in vers if getattr(v, "run_id", "") == run_id]
            if linked:
                v = sorted(linked, key=lambda x: int(x.version))[-1]
                client.transition_model_version_stage(
                    name=registered_model_name,
                    version=v.version,
                    stage=target_stage,
                    archive_existing_versions=True
                )
                print(f"[REGISTRY] {registered_model_name} v{v.version} -> {target_stage}")


if __name__ == "__main__":
    main()