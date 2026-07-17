#!/bin/bash
set -e

LOG_FILE="logs/pipeline_run_$(date +%Y%m%d_%H%M%S).log"
echo "Pipeline started: $(date)" | tee -a "$LOG_FILE"

source venv/bin/activate

echo "[1/3] Preparing data..." | tee -a "$LOG_FILE"
python scripts/01_prepare_data.py 2>&1 | tee -a "$LOG_FILE"

echo "[2/3] Training models..." | tee -a "$LOG_FILE"
python scripts/02_train.py 2>&1 | tee -a "$LOG_FILE"

echo "[3/3] Evaluating models..." | tee -a "$LOG_FILE"
python scripts/03_evaluate.py 2>&1 | tee -a "$LOG_FILE"

echo "Pipeline completed: $(date)" | tee -a "$LOG_FILE"
