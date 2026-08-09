"""Tests for Phase 8 orchestration helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from src.phase8 import _comparison_table


def test_comparison_uses_matching_real_only_experiment() -> None:
    real = pd.DataFrame(
        {
            "algorithm": ["svm", "svm"],
            "feature_set": ["clinical", "clinical_imaging"],
            "training_condition": ["real_only", "real_only"],
            "balanced_accuracy": [0.70, 0.80],
        }
    )
    augmented = real.copy()
    augmented["training_condition"] = "real_plus_synthetic"
    augmented["balanced_accuracy"] = [0.75, 0.78]
    comparison = _comparison_table(real, augmented)
    deltas = comparison.loc[
        comparison["training_condition"] == "real_plus_synthetic",
        "balanced_accuracy_delta_vs_real_only",
    ].tolist()
    assert deltas == pytest.approx([0.05, -0.02])
