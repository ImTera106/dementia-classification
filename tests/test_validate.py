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
    calculate_training_condition_bootstrap_validation,
    run_validation_pipeline,
    run_training_condition_validation_pipeline,
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
        "training_condition_comparison": {
            "reference_training_condition": "real_only",
            "comparison_training_condition": "real_plus_synthetic",
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


def augmented_prediction_frame() -> pd.DataFrame:
    """Return paired augmented predictions with a clinical-only improvement."""
    frame = prediction_frame()
    frame["training_condition"] = "real_plus_synthetic"
    clinical = frame["feature_set"] == "clinical"
    frame.loc[clinical, "prediction"] = frame.loc[clinical, "target"]
    frame.loc[clinical, "score"] = np.where(
        frame.loc[clinical, "target"] == 1, 0.9, 0.1
    )
    return frame


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

    def test_training_condition_differences_are_paired_and_reproducible(self) -> None:
        first_intervals, first_differences = (
            calculate_training_condition_bootstrap_validation(
                prediction_frame(),
                augmented_prediction_frame(),
                validation_config(),
                evaluation_config(),
            )
        )
        second_intervals, second_differences = (
            calculate_training_condition_bootstrap_validation(
                prediction_frame(),
                augmented_prediction_frame(),
                validation_config(),
                evaluation_config(),
            )
        )
        pd.testing.assert_frame_equal(first_intervals, second_intervals)
        pd.testing.assert_frame_equal(first_differences, second_differences)
        self.assertEqual(len(first_intervals), 60)
        self.assertEqual(len(first_differences), 10)
        self.assertEqual(
            set(first_intervals["training_condition"]),
            {"real_plus_synthetic"},
        )
        clinical = first_differences["feature_set"] == "clinical"
        self.assertTrue((first_differences.loc[clinical, "estimate"] > 0).all())
        self.assertTrue(
            (first_differences.loc[~clinical, "estimate"] == 0).all()
        )
        self.assertIn("interval_includes_zero", first_differences.columns)
        self.assertNotIn("accuracy", set(first_intervals["metric"]))

    def test_training_condition_subject_or_target_mismatch_is_rejected(self) -> None:
        for mismatch in ("subject", "target"):
            augmented = augmented_prediction_frame()
            if mismatch == "subject":
                augmented.loc[augmented["subject_id"] == "T0", "subject_id"] = "TX"
            else:
                augmented.loc[augmented["subject_id"] == "T0", "target"] = 1
            with self.subTest(mismatch=mismatch), self.assertRaisesRegex(
                ValidationError, "identical subjects and targets"
            ):
                calculate_training_condition_bootstrap_validation(
                    prediction_frame(),
                    augmented,
                    validation_config(),
                    evaluation_config(),
                )

    def test_training_condition_runner_saves_two_tables_and_figures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "reference.csv"
            comparison_path = root / "comparison.csv"
            prediction_frame().to_csv(reference_path, index=False)
            augmented_prediction_frame().to_csv(comparison_path, index=False)
            output_paths = {
                "synthetic_test_metric_bootstrap_intervals": root / "intervals.csv",
                "training_condition_balanced_accuracy_differences": root
                / "differences.csv",
                "synthetic_test_balanced_accuracy_intervals_figure": root
                / "intervals.png",
                "training_condition_balanced_accuracy_differences_figure": root
                / "differences.png",
            }
            paths = run_training_condition_validation_pipeline(
                reference_path,
                comparison_path,
                validation_config=validation_config(),
                evaluation_config=evaluation_config(),
                output_paths=output_paths,
            )
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(path.is_file() for path in paths.values()))
