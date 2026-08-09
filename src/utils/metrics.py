"""Shared metric definitions for dementia classification experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)

SUPPORTED_METRICS: frozenset[str] = frozenset(
    {
        "balanced_accuracy",
        "roc_auc",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
    }
)


def build_classification_scoring(metric_names: Sequence[str]) -> dict[str, Any]:
    """Return validated sklearn scorers without ordinary accuracy."""
    if not metric_names:
        raise ValueError("At least one evaluation metric is required")
    duplicates = sorted(
        {name for name in metric_names if metric_names.count(name) > 1}
    )
    if duplicates:
        raise ValueError(f"Metric names must be unique; duplicates: {duplicates}")
    unsupported = sorted(set(metric_names).difference(SUPPORTED_METRICS))
    if unsupported:
        raise ValueError(
            f"Unsupported metrics {unsupported}; ordinary accuracy is not permitted"
        )

    scorers: dict[str, Any] = {}
    for name in metric_names:
        if name == "sensitivity":
            scorers[name] = make_scorer(
                recall_score, pos_label=1, zero_division=0
            )
        elif name == "specificity":
            scorers[name] = make_scorer(
                recall_score, pos_label=0, zero_division=0
            )
        elif name == "precision":
            scorers[name] = make_scorer(precision_score, zero_division=0)
        elif name == "f1":
            scorers[name] = make_scorer(f1_score, zero_division=0)
        else:
            scorers[name] = name
    return scorers


def calculate_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_score: Any,
) -> dict[str, float | int]:
    """Calculate the approved held-out binary classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
