# ChestX-ray CAD — Multi-Label Thoracic Disease Classification

A deep-learning system that detects **14 thoracic pathologies** from frontal chest
X-rays (NIH **ChestX-ray14**), built as a complete, reproducible and deployable
application: a data pipeline that runs from raw images through training,
validation and testing to a served inference API — with class-imbalance
handling, bias/fairness auditing, robustness testing, drift monitoring and
Grad-CAM explainability.

> **Intended use:** research and decision-support / triage. This is **not** an
> autonomous diagnostic device; see [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

![python](https://img.shields.io/badge/python-3.10%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green) ![tests](https://img.shields.io/badge/tests-pytest-informational)

---

## Why it's built the way it is

Chest radiography is the most common thoracic examination, but reading it is
hard and a single film can show several findings at once. That shapes three core
design decisions:

- **Multi-label, not single-label.** Each of the 14 pathologies is an
  independent `sigmoid` output trained with binary cross-entropy, so findings
  don't compete.
- **Patient-disjoint evaluation.** Because one patient contributes many images,
  all splits are made by *patient* (official NIH lists when available, otherwise
  a grouped split), and disjointness is asserted before training — no patient
  leaks between train, validation and test.
- **Honest metrics.** Classes are highly imbalanced, so the headline metric is
  ROC-AUC (plus average precision, sensitivity/specificity and calibration),
  never raw accuracy.

## Features

- Multi-label classification of 14 pathologies (DenseNet-121 by default; any
  `timm` backbone).
- Leakage-free, patient-grouped data splitting with an automated guard.
- Class-imbalance handling: weighted BCE, focal loss, or an optional
  `WeightedRandomSampler`.
- Feature engineering: CLAHE contrast enhancement and engineered patient-metadata
  features.
- Model selection & hyperparameter optimisation with Optuna.
- Rigorous evaluation: ROC-AUC, average precision, sensitivity/specificity, F1
  at validation-tuned thresholds, and expected calibration error.
- Exploratory analysis, sound baselines, patient-grouped cross-validation and
  bootstrap confidence intervals.
- Bias/fairness auditing by sex, view position and age band.
- Robustness testing under image corruptions; data-drift monitoring for
  production.
- Grad-CAM explainability.
- Reproducible (typed config, global seeding, persisted artifacts), tested
  (pytest + CI) and deployable (FastAPI, Docker).

## Results

Measured on a patient-disjoint held-out test set:

| Model | Macro ROC-AUC | Notes |
|---|---|---|
| Metadata-only baseline (logistic regression) | 0.647 | patient-grouped 5-fold CV; a sanity floor |
| DenseNet-121 (this project) | **0.718** | leakage-free test set; per-class 0.60–0.84 |
| DenseNet-121, full-data target | ~0.84 | published CheXNet benchmark (Rajpurkar et al., 2017) |

The reported 0.718 comes from a run on the public **sample subset** (~5,600
images, 8 epochs) for tractability; training on the full 112,120-image dataset
is expected to approach the ~0.84 benchmark. The pipeline is robust to most
image corruptions (worst case: −0.07 AUC under Gaussian noise) and the fairness
audit surfaces a modest age-band performance gap to address before any
deployment.

## Dataset

NIH **ChestX-ray14** — 112,120 frontal images from 30,805 patients, 14
pathology labels mined from radiology reports. It is ~42 GB and not included
here; see [`scripts/download_data.md`](scripts/download_data.md) for Kaggle / NIH
download and the expected `data/` layout.

## Quickstart

```bash
# 1) install
python -m venv .venv && source .venv/bin/activate
make install-dev

# 2) run the tests (the light subset needs no GPU or data)
make test-light

# 3) download the data (see scripts/download_data.md), then run the pipeline
bash scripts/run_pipeline.sh configs/default.yaml       # train + evaluate
python scripts/make_results.py --exp artifacts/densenet121_baseline   # figures + tables

# 4) explore the data and baselines
python scripts/run_eda_baselines.py --csv data/Data_Entry_2017.csv --out artifacts/eda

# 5) serve the trained model
make serve
curl -F "file=@some_xray.png" http://localhost:8000/predict
```

## Run on Colab (no local GPU)

Open [`notebooks/ChestXray_Colab.ipynb`](notebooks/ChestXray_Colab.ipynb) in
Google Colab. It installs the package, pulls the NIH data from Kaggle, trains,
and produces the ROC, training, fairness, robustness and Grad-CAM figures, then
bundles them for download.

## How the pipeline works

```
prepare ─► split ─► train ─► validate ─► test ─► inference
```

- **prepare** — build multi-hot labels, filter ultra-rare classes.
- **split** — patient-disjoint train / val / test.
- **train** — full-set epochs, class-weighted loss, AMP, early stopping, LR
  scheduling; persists config, labels and per-epoch history.
- **test** — thresholds tuned on validation, then frozen; report ROC-AUC, AP,
  sensitivity/specificity, calibration.
- **inference** — a `Predictor` class used by the CLI and the API.

Full component-by-component map: [`docs/PIPELINE.md`](docs/PIPELINE.md).

## Inference API

```bash
MODEL_CHECKPOINT=artifacts/densenet121_baseline/best_model.pt \
  uvicorn api.main:app --host 0.0.0.0 --port 8000
# POST an image:
curl -F "file=@some_xray.png" http://localhost:8000/predict
```

Returns per-pathology probabilities, the model version and inference latency.
Containerised via `Dockerfile` / `docker-compose.yml`.

## Project structure

```
src/chestxray/        # installable package
  config.py           #   typed YAML configuration
  data/               #   prepare (splits), dataset, transforms, features
  models/factory.py   #   backbone + sigmoid head
  training/           #   train loop, losses, metrics, HPO
  evaluation/         #   fairness, robustness, overfitting diagnostics
  eda.py, baselines.py, validation.py
  evaluate.py         #   held-out test evaluation + report
  explain.py          #   Grad-CAM
  inference.py        #   Predictor used by the API
api/                  # FastAPI service
monitoring/           # data-drift detection (PSI)
tests/                # pytest suite
configs/              # default / smoke configs
notebooks/            # Colab pipeline
docs/                 # model card, pipeline map
Dockerfile · docker-compose.yml · Makefile · .github/workflows/ci.yml
```

## Development

```bash
make install-dev     # install with dev tools
make test            # full test suite (needs torch)
make test-light      # fast subset (no torch)
make lint            # ruff + black
make hpo             # hyperparameter search / model selection
```

## Responsible use

The model is trained on weak, NLP-derived labels from a small number of
institutions; expect performance to drop under distribution shift, audit
subgroup performance, and keep a qualified clinician in the loop. See
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## License

MIT — see [`LICENSE`](LICENSE). The NIH ChestX-ray14 dataset is subject to its
own terms.
