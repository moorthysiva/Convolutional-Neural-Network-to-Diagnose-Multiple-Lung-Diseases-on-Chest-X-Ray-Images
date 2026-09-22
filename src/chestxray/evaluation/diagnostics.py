"""Overfitting diagnostics.

Reads the ``history.json`` a training run writes and quantifies the
generalisation gap (train vs. validation loss) and whether validation loss has
started to rise while training loss keeps falling — the classic overfitting
signature. Complements the structural defences already in place (dropout,
weight decay, augmentation, early stopping).
"""
from __future__ import annotations

import json
from pathlib import Path


def overfitting_report(history: list[dict] | str | Path) -> dict:
    """Summarise the generalisation gap from a training history.

    Accepts the history list or a path to ``history.json``.
    """
    if isinstance(history, (str, Path)):
        history = json.loads(Path(history).read_text())
    if not history:
        return {"epochs": 0}

    train = [h["train_loss"] for h in history]
    val = [h["val_loss"] for h in history]
    final_gap = val[-1] - train[-1]
    best_val = min(val)
    best_epoch = int(min(range(len(val)), key=lambda i: val[i]))

    # Overfitting signal: val loss rose after its minimum while train loss fell.
    rose_after_best = val[-1] > best_val + 1e-4
    train_still_falling = train[-1] < train[best_epoch] - 1e-4
    overfitting = rose_after_best and train_still_falling

    return {
        "epochs": len(history),
        "final_train_loss": round(train[-1], 4),
        "final_val_loss": round(val[-1], 4),
        "generalisation_gap": round(final_gap, 4),
        "best_val_epoch": best_epoch,
        "overfitting_suspected": bool(overfitting),
        "note": (
            "Validation loss rose after its minimum while training loss fell — "
            "consider more regularisation or earlier stopping."
            if overfitting else
            "No strong overfitting signature; early stopping selected the best epoch."
        ),
    }
