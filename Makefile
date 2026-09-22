.PHONY: install install-dev lint format test test-light train evaluate serve docker

install:
	pip install -r requirements.txt && pip install -e .

install-dev:
	pip install -r requirements-dev.txt && pip install -e .

lint:
	ruff check src api monitoring tests
	black --check src api monitoring tests

format:
	ruff check --fix src api monitoring tests
	black src api monitoring tests

# Full suite (needs torch/timm/fastapi).
test:
	pytest --cov=src --cov=monitoring --cov-report=term-missing

# Lightweight suite used by the fast CI job (no torch needed).
test-light:
	pytest tests/test_data.py tests/test_metrics.py tests/test_drift.py

train:
	python -m chestxray.training.train --config configs/default.yaml

evaluate:
	python -m chestxray.evaluate --checkpoint artifacts/densenet121_baseline/best_model.pt

serve:
	MODEL_CHECKPOINT=artifacts/densenet121_baseline/best_model.pt \
		uvicorn api.main:app --host 0.0.0.0 --port 8000

docker:
	docker build -t chestxray-cad:latest .

hpo:
	python -m chestxray.training.hpo --config configs/default.yaml --trials 20

eda:
	python scripts/run_eda_baselines.py --csv data/Data_Entry_2017.csv --out artifacts/eda
