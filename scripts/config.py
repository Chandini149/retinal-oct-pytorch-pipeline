"""
config.py
Centralized, environment-driven configuration.
Override any value by setting the corresponding environment variable
before running the pipeline, e.g.:
    export OCT_DATA_DIR=/path/to/your/OCT_subset
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = os.environ.get("OCT_DATA_DIR", str(PROJECT_ROOT / "data" / "OCT_subset"))
MODELS_DIR = os.environ.get("OCT_MODELS_DIR", str(PROJECT_ROOT / "models"))
LOGS_DIR = os.environ.get("OCT_LOGS_DIR", str(PROJECT_ROOT / "logs"))
OUTPUTS_DIR = os.environ.get("OCT_OUTPUTS_DIR", str(PROJECT_ROOT / "outputs"))

IMG_SIZE = int(os.environ.get("OCT_IMG_SIZE", 160))
BATCH_SIZE = int(os.environ.get("OCT_BATCH_SIZE", 32))
NUM_WORKERS = int(os.environ.get("OCT_NUM_WORKERS", 2))
EPOCHS = int(os.environ.get("OCT_EPOCHS", 10))
PATIENCE = int(os.environ.get("OCT_PATIENCE", 2))
