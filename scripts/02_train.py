"""
02_train.py
Trains MobileNetV2 / EfficientNetB0 transfer-learning models in PyTorch,
matching the original TF architecture: frozen base -> GAP -> Dropout(0.3)
-> Dense(128, relu) -> Dropout(0.2) -> Dense(num_classes).

Config (data paths, epochs, batch size, etc.) is centralized in config.py
and overridable via environment variables (e.g. OCT_EPOCHS, OCT_DATA_DIR).

Usage:
  python scripts/02_train.py --quick_test   # 1 model, 2 epochs, sanity check
  python scripts/02_train.py                # full run, both models
"""
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import config
import importlib
prep = importlib.import_module("01_prepare_data")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = config.MODELS_DIR
LOGS_DIR = config.LOGS_DIR
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


class TransferModel(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super().__init__()
        if backbone_name == "MobileNetV2":
            base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            self.features = base.features
            feat_dim = base.last_channel
        elif backbone_name == "EfficientNetB0":
            base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.features = base.features
            feat_dim = 1280
        else:
            raise ValueError("Choose 'MobileNetV2' or 'EfficientNetB0'.")

        for param in self.features.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def compute_weights(train_loader, num_classes):
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist())
    weights = compute_class_weight(class_weight="balanced",
                                    classes=np.arange(num_classes),
                                    y=all_labels)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def train_model(backbone_name, train_loader, val_loader, classes, epochs, patience):
    print(f"\nTraining {backbone_name} on {DEVICE} for {epochs} epoch(s)...\n")
    model = TransferModel(backbone_name, len(classes)).to(DEVICE)

    class_weights = compute_weights(train_loader, len(classes))
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=1e-4)

    best_val_acc = 0.0
    epochs_no_improve = 0
    checkpoint_path = os.path.join(MODELS_DIR, f"best_{backbone_name}.pt")
    log_path = os.path.join(LOGS_DIR, f"train_{backbone_name}.log")

    with open(log_path, "w") as log_file:
        for epoch in range(epochs):
            start = time.time()
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            train_loss = running_loss / len(train_loader.dataset)

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            val_acc = correct / total
            elapsed = time.time() - start

            line = (f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} "
                    f"| val_acc={val_acc:.4f} | {elapsed:.1f}s")
            print(line)
            log_file.write(line + "\n")
            log_file.flush()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    log_file.write(f"Early stopping at epoch {epoch+1}\n")
                    break

    print(f"Best val_acc: {best_val_acc:.4f} | saved to {checkpoint_path}")
    return checkpoint_path, best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick_test", action="store_true",
                        help="Run 1 model, 2 epochs, to sanity-check the pipeline")
    args = parser.parse_args()

    train_loader, val_loader, test_loader, classes = prep.get_dataloaders()

    if args.quick_test:
        train_model("MobileNetV2", train_loader, val_loader, classes, epochs=2, patience=config.PATIENCE)
    else:
        for backbone in ["MobileNetV2", "EfficientNetB0"]:
            train_model(backbone, train_loader, val_loader, classes, epochs=config.EPOCHS, patience=config.PATIENCE)
