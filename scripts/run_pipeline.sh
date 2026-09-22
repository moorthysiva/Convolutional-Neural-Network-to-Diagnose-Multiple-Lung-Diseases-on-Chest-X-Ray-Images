#!/usr/bin/env bash
# End-to-end: train -> evaluate -> (optionally) serve.
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
EXP_DIR="artifacts/$(python -c "import yaml;print(yaml.safe_load(open('${CONFIG}'))['experiment_name'])")"

echo ">> Training with ${CONFIG}"
python -m chestxray.training.train --config "${CONFIG}"

echo ">> Evaluating on held-out test set"
python -m chestxray.evaluate --checkpoint "${EXP_DIR}/best_model.pt" --config "${CONFIG}"

echo ">> Done. Report at ${EXP_DIR}/test_report.json"
echo ">> To serve:  MODEL_CHECKPOINT=${EXP_DIR}/best_model.pt uvicorn api.main:app --port 8000"
