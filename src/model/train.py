import os
import time
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import models
from sklearn.metrics import f1_score, accuracy_score, recall_score
from tqdm import tqdm  # ✅ pour les barres de progression
from src.data.dataset import PlantDataset
from src.utils.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    LOCAL_MODEL_DIR,
)

# -------------------------
# Hyperparamètres
# -------------------------
EPOCHS = 5
BATCH_SIZE = 32
LR = 1e-3
VAL_RATIO = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 2  # si problème sur Mac, mets 0

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_dataloaders(metadata_csv):
    print("[DATA] Preparing datasets and dataloaders...")
    full_dataset = PlantDataset(csv_file=metadata_csv)

    val_size = int(len(full_dataset) * VAL_RATIO)
    train_size = len(full_dataset) - val_size

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


def build_model(num_classes=2):
    print("[MODEL] Loading ResNet18 pre-trained on ImageNet...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    print("[MODEL] Model ready ✅")
    return model.to(device)


def train_one_epoch(model, optimizer, criterion, train_loader):
    model.train()
    all_preds, all_targets = [], []
    running_loss = 0.0

    for batch_imgs, batch_labels in tqdm(train_loader, desc="Training", leave=False):
        batch_imgs = batch_imgs.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_imgs)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_imgs.size(0)

        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        targets = batch_labels.detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro")
    epoch_recall = recall_score(all_targets, all_preds, average="macro")

    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
        "f1": epoch_f1,
        "recall": epoch_recall,
    }


def eval_one_epoch(model, criterion, val_loader):
    model.eval()
    all_preds, all_targets = [], []
    running_loss = 0.0

    with torch.no_grad():
        for batch_imgs, batch_labels in tqdm(val_loader, desc="Validation", leave=False):
            batch_imgs = batch_imgs.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_labels)
            running_loss += loss.item() * batch_imgs.size(0)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = batch_labels.cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro")
    epoch_recall = recall_score(all_targets, all_preds, average="macro")

    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
        "f1": epoch_f1,
        "recall": epoch_recall,
    }


def save_model(model, output_dir=LOCAL_MODEL_DIR):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    model_path = os.path.join(output_dir, f"model_{timestamp}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[MODEL] Saved at {model_path}")
    return model_path


def run_training(metadata_csv="data/raw/metadata.csv"):
    print("[TRAINING] Starting training process 🚀")
    start_time = time.time()

    train_loader, val_loader = get_dataloaders(metadata_csv)
    model = build_model(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = []
    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{EPOCHS} =====")
        train_metrics = train_one_epoch(model, optimizer, criterion, train_loader)
        val_metrics = eval_one_epoch(model, criterion, val_loader)

        print(
            f"[EPOCH {epoch+1}/{EPOCHS}] "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            **train_metrics,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

    model_path = save_model(model)
    total_time = (time.time() - start_time) / 60
    print(f"[TRAINING] Completed in {total_time:.2f} min ⏱️")

    final_metrics = history[-1]
    return final_metrics, model_path


def main():
    print("[MLFLOW] Setting up tracking...")
    tracking_uri = MLFLOW_TRACKING_URI
    if tracking_uri.startswith("http://localhost"):
        tracking_uri = "mlruns"
        os.makedirs(tracking_uri, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        final_metrics, model_path = run_training(metadata_csv="data/raw/metadata.csv")

        mlflow.log_metric("val_accuracy", final_metrics["val_acc"])
        mlflow.log_metric("val_f1", final_metrics["val_f1"])
        mlflow.log_metric("val_recall", final_metrics["val_recall"])
        mlflow.log_metric("val_loss", final_metrics["val_loss"])

        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("lr", LR)
        mlflow.log_param("val_ratio", VAL_RATIO)
        mlflow.log_param("arch", "resnet18")

        mlflow.log_artifact(model_path, artifact_path="model")

        # upload vers MinIO
        try:
            from src.model.registry import upload_model_to_s3
            s3_uri = upload_model_to_s3(model_path)
            if s3_uri:
                mlflow.log_param("model_registry_uri", s3_uri)
        except Exception as e:
            print(f"[WARNING] Could not upload model to S3/MinIO: {e}")

        print("\n[TRAINING] ✅ Done.")
        print(f"[TRAINING] Model saved at {model_path}")
        print(f"[TRAINING] MLflow run stored in {tracking_uri}")


# ✅ Ligne indispensable pour exécuter le script directement
if __name__ == "__main__":
    main()