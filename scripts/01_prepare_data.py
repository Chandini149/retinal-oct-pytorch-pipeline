"""
01_prepare_data.py
Loads OCT train/val/test datasets and prepares PyTorch DataLoaders.
Data location is configurable via config.py / OCT_DATA_DIR env var.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import config
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_dataloaders():
    if not os.path.isdir(config.DATA_DIR):
        raise FileNotFoundError(
            f"Data directory not found: {config.DATA_DIR}\n"
            f"Set OCT_DATA_DIR to the correct path, e.g.:\n"
            f"  export OCT_DATA_DIR=/path/to/OCT_subset"
        )

    train_ds = datasets.ImageFolder(os.path.join(config.DATA_DIR, "train"), transform=train_transform)
    val_ds   = datasets.ImageFolder(os.path.join(config.DATA_DIR, "val"),   transform=eval_transform)
    test_ds  = datasets.ImageFolder(os.path.join(config.DATA_DIR, "test"), transform=eval_transform)

    print(f"Data source: {config.DATA_DIR}")
    print(f"Classes found: {train_ds.classes}")
    print(f"Train samples: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    return train_loader, val_loader, test_loader, train_ds.classes


if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels: {labels[:5]}")
