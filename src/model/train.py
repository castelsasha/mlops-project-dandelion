# src/model/train.py
import os
import time
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from sklearn.metrics import f1_score, accuracy_score, recall_score
from tqdm import tqdm
import mlflow

from src.utils.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    LOCAL_MODEL_DIR,
)

# ----------------------------
# Config entraînement
# ----------------------------
USE_DB = False           # bascule sur Postgres+S3 si besoin (True)
EPOCHS = 3
BATCH_SIZE = 32
LR = 1e-3
VAL_RATIO = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 0          # macOS: laissez 0 pour éviter les soucis
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_dataloaders(metadata_csv: str | None = "data/raw/metadata.csv") -> Tuple[DataLoader, DataLoader]:
    """
    Si USE_DB=True, utilise PlantDatasetDB (Postgres + S3).
    Sinon, fallback CSV local via PlantDataset.
    """
    print("[DATA] Preparing datasets and dataloaders...")
    if USE_DB:
        from src.data.dataset_db import PlantDatasetDB
        full_dataset = PlantDatasetDB()
    else:
        from src.data.dataset import PlantDataset
        full_dataset = PlantDataset(csv_file=metadata_csv)

    val_size = max(int(len(full_dataset) * VAL_RATIO), 1)
    train_size = max(len(full_dataset) - val_size, 1)

    torch.manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    print(f"[DATA] Train size: {train_size}, Val size: {val_size}")
    return train_loader, val_loader


def build_model(num_classes: int = 2) -> nn.Module:
    print("[MODEL] ResNet18 from scratch…")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


@torch.no_grad()
def eval_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
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


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    model.train()
    all_preds, all_targets = [], []
    running = 0.0

    for x, y in tqdm(loader, leave=False, desc="Train"):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad(set_to_none=True)
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


def save_model(model: nn.Module, output_dir: str = LOCAL_MODEL_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    model_path = os.path.join(output_dir, f"model_{timestamp}.pt")
    # on sauvegarde un state_dict, compatible avec notre inference loader
    torch.save(model.state_dict(), model_path)
    print(f"[MODEL] Saved at {model_path}")
    return model_path


def main():
    # ----------------------------
    # MLflow setup
    # ----------------------------
    print("[MLFLOW] Setting up tracking...")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", MLFLOW_EXPERIMENT_NAME)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Optionnel : nom & stage pour le registry MLflow
    registered_model_name = os.getenv("MODEL_NAME", "").strip()  # e.g. "dandelion-grass"
    target_stage = os.getenv("MODEL_STAGE", "None").strip()      # e.g. "Production"

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"[MLFLOW] Run id: {run_id}")

        # ----------------------------
        # Data + Model
        # ----------------------------
        train_loader, val_loader = get_dataloaders("data/raw/metadata.csv")
        model = build_model(num_classes=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # ----------------------------
        # Train loop
        # ----------------------------
        for epoch in range(1, EPOCHS + 1):
            tr = train_one_epoch(model, train_loader, criterion, optimizer)
            va = eval_one_epoch(model, val_loader, criterion)

            mlflow.log_metrics({
                "train_loss": tr["loss"], "train_acc": tr["acc"], "train_f1": tr["f1"],
                "val_loss": va["loss"],   "val_acc": va["acc"],   "val_f1": va["f1"],
                "val_recall": va["recall"]
            }, step=epoch)

            print(f"[EPOCH {epoch}/{EPOCHS}] "
                  f"train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} "
                  f"val_acc={va['acc']:.3f} val_f1={va['f1']:.3f}")

        # ----------------------------
        # Params pour reproductibilité
        # ----------------------------
        mlflow.log_params({
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "val_ratio": VAL_RATIO,
            "arch": "resnet18",
            "use_db": USE_DB,
            "device": device,
        })

        # ----------------------------
        # Sauvegarde locale + artifact brut
        # ----------------------------
        path_pt = save_model(model)
        mlflow.log_artifact(path_pt, artifact_path="weights")

        # ----------------------------
        # Log d’un modèle MLflow (flavor PyTorch)
        # + input_example au bon format (numpy.ndarray)
        # ----------------------------
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

        # ----------------------------
        # NEW: push automatique du .pt vers S3/MinIO (pour l’API)
        # ----------------------------
        try:
            from src.model.registry import upload_model_to_s3
            s3_uri = upload_model_to_s3(path_pt)
            mlflow.log_text(s3_uri, artifact_file="deployed_model_s3_uri.txt")
            print(f"[REGISTRY] pushed to {s3_uri}")
        except Exception as e:
            print(f"[REGISTRY] ⚠️ upload to S3 failed: {e}")

        # ----------------------------
        # Promotion auto (facultatif) si MODEL_STAGE est fourni
        # ----------------------------
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