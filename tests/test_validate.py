"""Tests for fixed-prediction Phase 5 robustness validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.validate import (
    ValidationError,
    calculate_bootstrap_validation,
    run_validation_pipeline,
)

ALGORITHMS = (
    "logistic_regression",
    "svm",
    "decision_tree",
    "random_forest",
    "xgboost",
)


def evaluation_config() -> dict:
    """Return the approved metrics without ordinary accuracy."""
    return {
        "metrics": {
            "primary": "balanced_accuracy",
            "secondary": [
                "roc_auc",
                "sensitivity",
                "specificity",
                "precision",
                "f1",
            ],
        }
    }


def validation_config() -> dict:
    """Return a lightweight deterministic bootstrap configuration."""
    return {
        "training_condition": "real_only",
        "expected_subject_count": 6,
        "bootstrap": {
            "n_resamples": 100,
            "confidence_level": 0.95,
            "random_state": 7,
            "stratified": True,
        },
        "comparison": {
            "reference_feature_set": "clinical",
            "comparison_feature_set": "clinical_imaging",
            "metric": "balanced_accuracy",
        },
        "figures": {"dpi": 72},
    }


def prediction_frame() -> pd.DataFrame:
    """Return complete paired predictions for five algorithms and two feature sets."""
    subjects = [f"T{i}" for i in range(6)]
    targets = np.array([0, 0, 0, 1, 1, 1])
    records: list[dict] = []
    for algorithm in ALGORITHMS:
        for feature_set in ("clinical", "clinical_imaging"):
            predicted = (
                targets.copy()
                if feature_set == "clinical_imaging"
                else np.array([0, 0, 1, 1, 1, 0])
            )
            scores = np.where(predicted == 1, 0.8, 0.2)
            for subject, target, prediction, score in zip(
                subjects, targets, predicted, scores, strict=True
            ):
                records.append(
                    {
                        "algorithm": algorithm,
                        "feature_set": feature_set,
                        "training_condition": "real_only",
                        "subject_id": subject,
                        "target": target,
                        "prediction": prediction,
                        "score": score,
                    }
                )
    return pd.DataFrame.from_records(records)


class ValidateTests(unittest.TestCase):
    """Verify complete paired subject-level bootstrap behavior."""

    def test_intervals_and_paired_differences_are_reproducible(self) -> None:
        first_intervals, first_differences = calculate_bootstrap_validation(
            prediction_frame(), validation_config(), evaluation_config()
        )
        second_intervals, second_differences = calculate_bootstrap_validation(
            prediction_frame(), validation_config(), evaluation_config()
        )
        pd.testing.assert_frame_equal(first_intervals, second_intervals)
        pd.testing.assert_frame_equal(first_differences, second_differences)
        self.assertEqual(len(first_intervals), 60)
        self.assertEqual(len(first_differences), 5)
        self.assertTrue((first_differences["estimate"] > 0).all())
        self.assertNotIn("accuracy", set(first_intervals["metric"]))

    def test_duplicate_prediction_is_rejected(self) -> None:
        predictions = prediction_frame()
        predictions = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValidationError, "Duplicate"):
            calculate_bootstrap_validation(
                predictions, validation_config(), evaluation_config()
            )

    def test_runner_saves_two_tables_and_two_figures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions_path = root / "predictions.csv"
            prediction_frame().to_csv(predictions_path, index=False)
            output_paths = {
                "test_metric_bootstrap_intervals": root / "intervals.csv",
                "feature_set_balanced_accuracy_differences": root / "differences.csv",
                "test_balanced_accuracy_intervals_figure": root / "intervals.png",
                "feature_set_balanced_accuracy_differences_figure": root / "differences.png",
            }
            paths = run_validation_pipeline(
                predictions_path,
                validation_config=validation_config(),
                evaluation_config=evaluation_config(),
                output_paths=output_paths,
            )
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(path.is_file() for path in paths.values()))
