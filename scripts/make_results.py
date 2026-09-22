"""Turn a completed evaluation run into paper-ready assets.

After ``python -m chestxray.evaluate`` has written ``test_report.json`` and
``test_predictions.npz`` into an experiment folder, this script produces:

  * ``results_perclass.csv``  – per-class AUC/AP/precision/recall/F1/sens/spec
  * ``results_summary.json``  – macro AUC, macro AP, calibration, n_test
  * ``roc_curves.png``        – per-class ROC curves (like the original Fig. 10)
  * ``per_class_auc.png``     – per-class AUC bar chart
  * ``training_curves.png``   – loss and val-AUC over epochs (if history present)
  * ``results_block.md``      – a ready-to-paste prose summary of the numbers

These are the exact figures/tables to drop into the report — with real numbers
from your own training run, not estimates.

Usage:
    python scripts/make_results.py --exp artifacts/densenet121_baseline
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    _HAVE_PLOT = True
except Exception:  # matplotlib/sklearn optional for the CSV/JSON outputs
    _HAVE_PLOT = False


def _write_perclass_csv(report: dict, out: Path) -> None:
    tf = report["threshold_free"]["per_class"]
    op = report.get("operating_point", {})
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Pathology", "AUC", "AP", "Precision", "Recall",
                    "F1", "Sensitivity", "Specificity", "N_pos"])
        for name, m in tf.items():
            o = op.get(name, {})
            w.writerow([
                name, f"{m['auc']:.3f}", f"{m['ap']:.3f}",
                f"{o.get('precision', float('nan')):.3f}",
                f"{o.get('recall', float('nan')):.3f}",
                f"{o.get('f1', float('nan')):.3f}",
                f"{o.get('sensitivity', float('nan')):.3f}",
                f"{o.get('specificity', float('nan')):.3f}",
                m.get("n_pos", ""),
            ])


def _plot_roc(npz_path: Path, out: Path) -> None:
    data = np.load(npz_path, allow_pickle=True)
    scores, targets, labels = data["scores"], data["targets"], data["labels"]
    plt.figure(figsize=(6, 6))
    for i, name in enumerate(labels):
        if targets[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(targets[:, i].astype(int), scores[:, i])
        plt.plot(fpr, tpr, label=f"{name} (AUC {auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves (test set)")
    plt.legend(fontsize=6, loc="lower right")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def _plot_auc_bar(report: dict, out: Path) -> None:
    tf = report["threshold_free"]["per_class"]
    names = list(tf.keys())
    aucs = [tf[n]["auc"] for n in names]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(names)), aucs)
    plt.axhline(0.5, color="k", ls="--", lw=0.8)
    plt.xticks(range(len(names)), names, rotation=90)
    plt.ylabel("AUC")
    plt.title(f"Per-class AUC (macro = {report['threshold_free']['auc_macro']:.3f})")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def _plot_history(history_path: Path, out: Path) -> None:
    history = json.loads(history_path.read_text())
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("Loss")
    ax[1].plot(ep, [h["val_auc_macro"] for h in history])
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("macro AUC"); ax[1].set_title("Validation AUC")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def _write_block(report: dict, out: Path) -> None:
    tf = report["threshold_free"]
    lines = [
        f"On the held-out test set of {report.get('n_test_images', 'N')} patient-disjoint "
        f"images, the model achieved a macro-averaged AUC of {tf['auc_macro']:.3f} "
        f"and a macro average precision of {tf['ap_macro']:.3f}. "
        f"The expected calibration error was {report.get('calibration_ece', float('nan')):.3f}. "
        "Per-class AUC, average precision, and operating-point sensitivity/specificity "
        "are reported in Table [X]. Thresholds were tuned on the validation set and frozen "
        "before evaluation, so no operating-point information leaked from the test set.",
    ]
    out.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description="Build paper-ready results assets.")
    p.add_argument("--exp", required=True, help="experiment folder with test_report.json")
    a = p.parse_args()
    exp = Path(a.exp)

    report = json.loads((exp / "test_report.json").read_text())
    _write_perclass_csv(report, exp / "results_perclass.csv")
    (exp / "results_summary.json").write_text(json.dumps({
        "auc_macro": report["threshold_free"]["auc_macro"],
        "ap_macro": report["threshold_free"]["ap_macro"],
        "calibration_ece": report.get("calibration_ece"),
        "n_test_images": report.get("n_test_images"),
    }, indent=2))
    _write_block(report, exp / "results_block.md")

    if _HAVE_PLOT:
        _plot_auc_bar(report, exp / "per_class_auc.png")
        if (exp / "test_predictions.npz").exists():
            _plot_roc(exp / "test_predictions.npz", exp / "roc_curves.png")
        if (exp / "history.json").exists():
            _plot_history(exp / "history.json", exp / "training_curves.png")

    print(f"Wrote results assets to {exp}/")


if __name__ == "__main__":
    main()
