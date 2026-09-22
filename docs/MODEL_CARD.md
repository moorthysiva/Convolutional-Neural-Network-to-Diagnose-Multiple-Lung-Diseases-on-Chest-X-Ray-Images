# Model Card — ChestX-ray CAD

## Overview
Multi-label classifier that estimates the probability of up to 14 thoracic
pathologies from a single frontal chest X-ray. Intended as a **decision-support
/ triage aid**, not an autonomous diagnostic device.

## Intended use
- **In scope:** research, education, prioritising worklists, as a "second reader"
  that a qualified clinician reviews.
- **Out of scope:** standalone diagnosis, use on paediatric or lateral films,
  or any use without clinician oversight.

## Data
- **Source:** NIH ChestX-ray14 (112,120 frontal images, 30,805 patients).
- **Labels:** mined from radiology reports with NLP by the dataset authors, so
  they are *weak* labels with known noise.
- **Splits:** patient-disjoint (official NIH lists or a grouped split). No
  patient appears in more than one split.

## Architecture
- timm backbone (default DenseNet121, ImageNet-pretrained) + dropout + linear
  head with **one sigmoid logit per pathology**.
- Trained with `BCEWithLogitsLoss` (optionally class-weighted or focal).

## Evaluation
- Primary: macro & per-class **ROC-AUC**. Secondary: average precision,
  precision/recall/F1 at F1-tuned thresholds, sensitivity/specificity, and
  **expected calibration error**.
- Thresholds tuned on validation and frozen before touching test.

## Limitations & risks
- Label noise caps achievable performance and can bias metrics.
- Trained on one dataset from a small number of institutions → expect
  performance drop under distribution shift; monitor drift in production.
- Risk of **shortcut learning** (attending to text markers/borders); Grad-CAM is
  provided to audit this.
- Known dataset biases across sex/age/view position; audit subgroup AUC before
  any deployment.

## Ethical & regulatory
- De-identified research data. Any clinical use would require regulatory
  clearance (e.g. FDA/CE/UKCA), prospective validation, and a human in the loop.
