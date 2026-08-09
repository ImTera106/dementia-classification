"""Shared prediction-score helpers for fitted binary classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def get_positive_class_score(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return finite positive-class probabilities or decision scores."""
    classes = np.asarray(getattr(model, "classes_", []))
    if set(classes.tolist()) != {0, 1}:
        raise ValueError(f"Saved model classes must be [0, 1], got {classes}")
    if callable(getattr(model, "predict_proba", None)):
        probabilities = np.asarray(model.predict_proba(features))
        positive_index = int(np.flatnonzero(classes == 1)[0])
        scores = probabilities[:, positive_index]
    elif callable(getattr(model, "decision_function", None)):
        scores = np.asarray(model.decision_function(features))
        if scores.ndim != 1:
            raise ValueError("Binary decision_function must return one score")
    else:
        raise ValueError(
            "Saved model must provide predict_proba or decision_function"
        )
    if len(scores) != len(features) or not np.isfinite(scores).all():
        raise ValueError("Saved model produced invalid continuous scores")
    return scores.astype(float)
