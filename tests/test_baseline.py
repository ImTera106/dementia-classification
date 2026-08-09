"""Tests for training-only Phase 2 baseline evaluation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.baseline import evaluate_baselines, run_baseline_pipeline
from src.utils.metrics import build_classification_scoring


def baseline_config() -> dict:
    """Return a small configuration suitable for targeted unit tests."""
    return {
        "random_state": 9,
        "split": {
            "subject_id_column": "subject_id",
            "target_column": "dementia",
        },
        "cross_validation": {
            "method": "repeated_stratified_kfold",
            "n_splits": 2,
            "n_repeats": 2,
            "shuffle": True,
        },
        "preprocessing": {
            "numeric_imputer": {"strategy": "median"},
            "categorical_imputer": {"strategy": "most_frequent"},
            "one_hot_encoder": {"handle_unknown": "ignore"},
            "scale_numeric": True,
        },
        "baseline": {
            "training_condition": "real_only",
            "algorithms": {
                "dummy": {"strategy": "most_frequent"},
                "logistic_regression": {
                    "regularization": "l2",
                    "solver": "lbfgs",
                    "C": 1.0,
                    "max_iter": 500,
                },
            },
            "metrics": {
                "primary": "balanced_accuracy",
                "secondary": [
                    "roc_auc",
                    "sensitivity",
                    "specificity",
                    "precision",
                    "f1",
                ],
            },
        },
        "feature_sets": {
            "clinical": {
                "numeric": ["age", "education_years", "ses", "mmse"],
                "categorical": ["sex"],
            },
            "clinical_imaging": {
                "numeric": [
                    "age",
                    "education_years",
                    "ses",
                    "mmse",
                    "etiv",
                    "nwbv",
                    "asf",
                ],
                "categorical": ["sex"],
            },
        },
    }


def training_frame() -> pd.DataFrame:
    """Return a balanced one-row-per-subject real-style training table."""
    rows = 20
    return pd.DataFrame(
        {
            "subject_id": [f"S{i:02d}" for i in range(rows)],
            "sex": ["female", "male"] * (rows // 2),
            "age": list(range(65, 65 + rows)),
            "education_years": [10, 12, 14, 16] * (rows // 4),
            "ses": [1, 2, 3, pd.NA, 5] * (rows // 5),
            "mmse": [29, 20, 28, 19, 27] * (rows // 5),
            "etiv": list(range(1300, 1300 + rows)),
            "nwbv": [0.60 + i / 100 for i in range(rows)],
            "asf": [1.0 + i / 100 for i in range(rows)],
            "dementia": [0, 1] * (rows // 2),
        }
    )


class BaselineTests(unittest.TestCase):
    """Verify experiment dimensions, metrics, and saved outputs."""

    def test_ordinary_accuracy_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ordinary accuracy"):
            build_classification_scoring(["balanced_accuracy", "accuracy"])

    def test_all_experiments_share_expected_fold_count(self) -> None:
        folds, summary = evaluate_baselines(training_frame(), baseline_config())
        self.assertEqual(len(folds), 16)
        self.assertEqual(len(summary), 4)
        self.assertNotIn("accuracy", folds.columns)
        self.assertEqual(set(folds["training_condition"]), {"real_only"})
        counts = folds.groupby(["algorithm", "feature_set"]).size()
        self.assertTrue((counts == 4).all())

    def test_runner_reads_only_supplied_training_path_and_saves_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train_real.csv"
            fold_path = root / "outputs" / "folds.csv"
            summary_path = root / "outputs" / "summary.csv"
            training_frame().to_csv(train_path, index=False)
            paths = run_baseline_pipeline(
                train_path,
                fold_path,
                summary_path,
                model_config=baseline_config(),
            )
            self.assertEqual(set(paths), {"fold_metrics", "summary"})
            self.assertTrue(all(path.is_file() for path in paths.values()))
            self.assertNotIn("accuracy", pd.read_csv(fold_path).columns)
