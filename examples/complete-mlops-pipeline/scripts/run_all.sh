#!/bin/bash
# MLOps Pipeline Runner
# Usage: ./run_all.sh [--skip-features] [--inference-only]

set -euo pipefail
cd "$(dirname "$0")"

echo "=============================================="
echo "MLOps Pipeline"
echo "=============================================="

case "${1:-}" in
    --inference-only) python3.12 03_inference.py; exit 0 ;;
    --skip-features) shift ;;
    *) echo ">>> Step 1: Feature Engineering"; python3 01_feast_features.py ;;
esac

echo -e "\n>>> Step 2: Model Training"
python3 02_training.py

echo -e "\n>>> Step 3: Deploy + Test Inference"
python3.12 03_inference.py

echo -e "\n✅ Pipeline Complete!"
