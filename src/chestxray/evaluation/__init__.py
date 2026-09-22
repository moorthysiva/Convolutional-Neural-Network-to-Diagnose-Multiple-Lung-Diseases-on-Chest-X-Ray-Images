"""Assessment tools beyond headline metrics: bias/fairness, robustness and
overfitting diagnostics. Kept separate from the training metrics so they can be
run post-hoc on any set of saved predictions."""
from .fairness import subgroup_auc, fairness_report
from .diagnostics import overfitting_report
from .robustness import corrupt, robustness_report, CORRUPTIONS

__all__ = [
    "subgroup_auc", "fairness_report",
    "overfitting_report",
    "corrupt", "robustness_report", "CORRUPTIONS",
]
