# Pipeline & Competency Map

How the reproducible pipeline is structured, and where each ML-engineering
competency is implemented. Every item below is real, runnable code with tests
where the logic is testable without a GPU.

## The reproducible pipeline

```
prepare ─► split ─► train ─► validate ─► test ─► inference
  │          │        │         │          │         │
  │          │        │         │          │         └─ inference.py / api/main.py
  │          │        │         │          └─ evaluate.py (held-out test, frozen thresholds)
  │          │        │         └─ train.py (val loss early-stop, val-AUC checkpoint)
  │          │        └─ train.py (whole-set epochs, AMP, LR schedule, seeded)
  │          └─ prepare.make_splits (patient-disjoint; official NIH lists)
  └─ prepare.build_label_frame (multi-hot labels, min-case filtering)
```

Reproducibility: `utils.seed.seed_everything` seeds Python/NumPy/PyTorch/cuDNN;
`config.py` persists the exact config, `train.py` writes `labels.json`,
`history.json` and `best_model.pt`. A run is reproducible from its config + code
commit.

## Competency → component

| Competency | Where | Notes |
|---|---|---|
| **Reproducible data-prep / train / val / test / inference** | `data/prepare.py`, `training/train.py`, `evaluate.py`, `inference.py`, `api/` | config-driven, seeded, artifacts persisted |
| **Feature engineering** | `data/features.py`, `data/transforms.py` | CLAHE contrast feature (`data.clahe: true`); engineered patient-metadata features (age/sex/view/follow-up) for fusion & fairness |
| **Model selection** | `training/hpo.py` (`BACKBONES`) | DenseNet121 / ResNet50 / MobileNetV3 / EfficientNet compared on val AUC |
| **Hyperparameter optimisation** | `training/hpo.py` | Optuna search over backbone, lr, weight decay, dropout, loss (random-search fallback) |
| **Data leakage** | `data/prepare.py` `assert_no_patient_leakage` | patient-disjoint splits, asserted before training |
| **Overfitting** | `evaluation/diagnostics.py` + dropout, weight decay, augmentation, early stopping | quantifies train/val gap and flags the overfitting signature |
| **Concept drift** | `monitoring/drift.py` | PSI on inputs and predictions vs. a training baseline |
| **Robustness** | `evaluation/robustness.py` | AUC degradation under noise/blur/brightness/rotation corruptions |
| **Bias / fairness** | `evaluation/fairness.py` | macro AUC by sex / age band / view position; flags disparities |
| **Explainability** | `explain.py` | Grad-CAM heat-maps; audits for shortcut learning |

## Data exploration, baselines & validation (second competency block)

| Competency | Where | Notes |
|---|---|---|
| **Explore complex datasets** | `eda.py`, `scripts/run_eda_baselines.py` | label prevalence & co-occurrence, demographics, images-per-patient |
| **Establish sound baselines** | `baselines.py` | prevalence floor + metadata logistic-regression / random-forest |
| **Experimental design & validation** | `validation.py` | patient-grouped K-fold; bootstrap AUC confidence intervals |
| **Missing data** | `eda._missing`, `data/features.py` | drops empty phantom column; median-imputes missing age |
| **Outliers** | `eda.py`, `data/features.py` | flags & clips implausible ages (>100, up to 414) |
| **Sampling bias** | `eda.py`, `evaluation/fairness.py` | sex/view skew, multiple-images-per-patient |
| **Class imbalance** | `data.positive_weights`, `data.sample_weights` | weighted loss + optional resampling |
| **Leakage** | `data.assert_no_patient_leakage` | patient-disjoint splits, asserted |
| **Compare a few methods** | `baselines.compare_baselines` | prevalence vs logistic vs random forest, grouped CV, macro AUC ± std |

Real result on the full 112,120-row metadata (patient-grouped 5-fold CV):
prevalence 0.500, logistic regression 0.647 ± 0.004, random forest 0.637 ± 0.006.

## Commands

```bash
# reproducible train + test + results
bash scripts/run_pipeline.sh configs/default.yaml
python scripts/make_results.py --exp artifacts/densenet121_baseline

# hyperparameter optimisation / model selection
python -m chestxray.training.hpo --config configs/default.yaml --trials 20

# feature engineering: enable CLAHE in config
#   data: { clahe: true }

# imbalance resampling: train.sampler: weighted   (with loss: bce)
```

Fairness and robustness reports are produced from saved test predictions
(`test_predictions.npz` + the metadata frame), so they can be run post-hoc on
any checkpoint without retraining.
