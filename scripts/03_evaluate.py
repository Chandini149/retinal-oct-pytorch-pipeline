"""
03_evaluate.py
Evaluates both trained models on the test set (classification report,
confusion matrix, ROC-AUC) and generates Grad-CAM visualizations.
Uses centralized config.py for all paths.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import config
import importlib
prep = importlib.import_module("01_prepare_data")
train_mod = importlib.import_module("02_train")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS_DIR = config.MODELS_DIR
OUTPUTS_DIR = config.OUTPUTS_DIR
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def load_model(backbone_name, num_classes):
    model = train_mod.TransferModel(backbone_name, num_classes)
    checkpoint_path = os.path.join(MODELS_DIR, f"best_{backbone_name}.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def evaluate_model(model, test_loader, classes, model_name):
    y_true, y_pred, y_probs = [], [], []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = softmax(outputs).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            y_probs.extend(probs)
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    y_true, y_pred, y_probs = np.array(y_true), np.array(y_pred), np.array(y_probs)

    print(f"\nClassification Report: {model_name}")
    report = classification_report(y_true, y_pred, target_names=classes)
    print(report)

    with open(os.path.join(OUTPUTS_DIR, f"{model_name}_classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    save_path = os.path.join(OUTPUTS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {save_path}")

    y_true_onehot = np.eye(len(classes))[y_true]
    auc_score = roc_auc_score(y_true_onehot, y_probs, multi_class="ovr")
    accuracy = float(np.mean(y_true == y_pred))

    print(f"Test Accuracy: {accuracy:.4f} | ROC-AUC: {auc_score:.4f}")
    return {"model": model_name, "accuracy": round(accuracy, 4), "roc_auc": round(auc_score, 4)}


def get_last_conv_layer(model, backbone_name):
    return model.features[-1]


def generate_gradcam(model, backbone_name, image_tensor, class_idx):
    activations = {}
    gradients = {}
    layer = get_last_conv_layer(model, backbone_name)

    def forward_hook(module, inp, out):
        activations["value"] = out

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    fh = layer.register_forward_hook(forward_hook)
    bh = layer.register_full_backward_hook(backward_hook)

    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    image_tensor.requires_grad_(True)

    output = model(image_tensor)
    model.zero_grad()
    output[0, class_idx].backward()

    fh.remove()
    bh.remove()

    acts = activations["value"][0].detach().cpu().numpy()
    grads = gradients["value"][0].detach().cpu().numpy()

    weights = grads.mean(axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    return cam


def save_gradcam_examples(model, backbone_name, classes, model_name):
    test_dir = os.path.join(config.DATA_DIR, "test")
    sample_images = []
    for cls in classes:
        class_folder = os.path.join(test_dir, cls)
        first_img = sorted(os.listdir(class_folder))[0]
        sample_images.append((cls, os.path.join(class_folder, first_img)))

    fig, axes = plt.subplots(len(sample_images), 2, figsize=(8, 14))

    for i, (true_class, img_path) in enumerate(sample_images):
        img = Image.open(img_path).convert("RGB")
        tensor = prep.eval_transform(img)

        with torch.no_grad():
            pred_probs = nn.Softmax(dim=1)(model(tensor.unsqueeze(0).to(DEVICE)))
        pred_idx = pred_probs.argmax(dim=1).item()
        confidence = pred_probs[0, pred_idx].item()
        pred_class = classes[pred_idx]

        cam = generate_gradcam(model, backbone_name, tensor, pred_idx)
        cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(img.size))

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original\nTrue: {true_class}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img)
        axes[i, 1].imshow(cam_resized, cmap="jet", alpha=0.4)
        axes[i, 1].set_title(f"Grad-CAM\nPred: {pred_class} ({confidence:.2f})")
        axes[i, 1].axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUTPUTS_DIR, f"{model_name}_gradcam_examples.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Grad-CAM figure to {save_path}")


if __name__ == "__main__":
    _, _, test_loader, classes = prep.get_dataloaders()

    results = []
    for backbone in ["MobileNetV2", "EfficientNetB0"]:
        model = load_model(backbone, len(classes))
        results.append(evaluate_model(model, test_loader, classes, backbone))
        save_gradcam_examples(model, backbone, classes, backbone)

    print("\n=== Summary ===")
    for r in results:
        print(r)