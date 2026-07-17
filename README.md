# Retinal OCT Disease Classification — PyTorch Pipeline

Automated, config-driven image classification pipeline for retinal OCT
(optical coherence tomography) scans, classifying four disease categories:
**CNV, DME, DRUSEN, NORMAL**. Built as a PyTorch + Bash rework of an earlier
TensorFlow prototype, focused on reproducibility and automation rather than
one-off notebook execution.

## Results

| Model | Test Accuracy | ROC-AUC |
|---|---|---|
| **MobileNetV2** | **84.6%** | **96.9%** |
| EfficientNetB0 | 79.4% | 94.6% |

Full pipeline (data prep → training both models → evaluation) runs in
**~22 minutes on CPU only** (no GPU required).

Per-class performance (MobileNetV2) and Grad-CAM interpretability figures
are in [`outputs/`](outputs/).

## Pipeline
Both models use the same head architecture: frozen pretrained backbone →
global average pooling → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.2) →
Dense(4). Class-balanced loss weighting handles class imbalance.

## Setup

```bash
git clone https://github.com/Chandini149/retinal-oct-pytorch-pipeline.git
cd retinal-oct-pytorch-pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

No paths are hardcoded. Set your data directory via environment variable
before running (expects `train/`, `val/`, `test/` subfolders, each containing
one folder per class):

```bash
export OCT_DATA_DIR=/path/to/your/OCT_subset
```

Optional overrides (defaults shown):
```bash
export OCT_EPOCHS=10
export OCT_BATCH_SIZE=32
export OCT_PATIENCE=2
```

## Run

```bash
./run_pipeline.sh
```

Runs all three stages in order, stops immediately on failure (`set -e`),
and logs each run with a timestamp to `logs/`.

To sanity-check the setup quickly before a full run (1 model, 2 epochs,
~2-3 min):
```bash
python scripts/02_train.py --quick_test
```

## Dataset

Retinal OCT B-scan images (Kermany et al., Mendeley/Kaggle OCT2017 dataset),
four-class subset: CNV, DME, DRUSEN, NORMAL.

## Notes

- CPU-only training; no CUDA dependency.
- Original TensorFlow/Keras prototype (full 88K-image dataset,
  EfficientNetB0 baseline) available at
  [retinal-oct-disease-classification](https://github.com/Chandini149/retinal-oct-disease-classification).
