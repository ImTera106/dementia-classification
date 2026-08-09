"""Tests for evaluation-only scoring of saved fitted models."""

from __future__ import annotations

import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from src.evaluate import EvaluationError, evaluate_saved_models, run_evaluation_pipeline
from src.utils.io import save_joblib


ALGORITHMS = (
    "logistic_regression",
    "svm",
    "decision_tree",
    "random_forest",
    "xgboost",
)


class ProbabilityOnlyModel(ClassifierMixin, BaseEstimator):
    """Small fitted test double that must never be refitted during evaluation."""

    def __init__(self) -> None:
        self.classes_ = np.array([0, 1])
        self.fitted_ = True

    def fit(self, *_: object, **__: object) -> None:
        raise AssertionError("Evaluation must not call fit")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.arange(len(features)) % 2

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive = np.where(np.arange(len(features)) % 2 == 1, 0.9, 0.1)
        return np.column_stack([1 - positive, positive])


class DecisionOnlyModel(ClassifierMixin, BaseEstimator):
    """Small fitted SVM-like test double exposing only decision scores."""

    def __init__(self) -> None:
        self.classes_ = np.array([0, 1])
        self.fitted_ = True

    def fit(self, *_: object, **__: object) -> None:
        raise AssertionError("Evaluation must not call fit")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.arange(len(features)) % 2

    def decision_function(self, features: pd.DataFrame) -> np.ndarray:
        return np.where(np.arange(len(features)) % 2 == 1, 1.0, -1.0)


def model_config() -> dict:
    """Return the two configured feature settings."""
    return {
        "split": {
            "subject_id_column": "subject_id",
            "target_column": "dementia",
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


def evaluation_config() -> dict:
    """Return approved held-out metrics and figure settings."""
    return {
        "training_condition": "real_only",
        "positive_label": 1,
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
        "figures": {"dpi": 72},
    }


def held_out_frame() -> pd.DataFrame:
    """Return four held-out-style subjects with both binary classes."""
    return pd.DataFrame(
        {
            "subject_id": ["T1", "T2", "T3", "T4"],
            "sex": ["female", "male", "female", "male"],
            "age": [70, 71, 72, 73],
            "education_years": [12, 14, 16, 18],
            "ses": [1, 2, 3, 4],
            "mmse": [29, 20, 28, 19],
            "etiv": [1300, 1310, 1320, 1330],
            "nwbv": [0.7, 0.6, 0.72, 0.61],
            "asf": [1.1, 1.2, 1.09, 1.19],
            "dementia": [0, 1, 0, 1],
        }
    )


def manifest(root: Path) -> dict:
    """Save evaluation-only doubles and return a complete model manifest."""
    probability_path = save_joblib(ProbabilityOnlyModel(), root / "prob.joblib")
    decision_path = save_joblib(DecisionOnlyModel(), root / "decision.joblib")
    records = []
    for feature_set, columns in {
        "clinical": ["age", "education_years", "ses", "mmse", "sex"],
        "clinical_imaging": [
            "age",
            "education_years",
            "ses",
            "mmse",
            "etiv",
            "nwbv",
            "asf",
            "sex",
        ],
    }.items():
        for algorithm in ALGORITHMS:
            records.append(
                {
                    "algorithm": algorithm,
                    "feature_set": feature_set,
                    "training_condition": "real_only",
                    "path": str(decision_path if algorithm == "svm" else probability_path),
                    "feature_columns": columns,
                    "selected_parameters": {},
                }
            )
    return {
        "training_condition": "real_only",
        "versions": {
            "scikit_learn": version("scikit-learn"),
            "xgboost": "3.4.0",
        },
        "models": records,
    }


class EvaluateTests(unittest.TestCase):
    """Verify identical held-out subjects, score interfaces, and saved outputs."""

    def test_saved_models_are_scored_without_refitting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics, predictions, confusion = evaluate_saved_models(
                held_out_frame(),
                manifest(Path(directory)),
                model_config(),
                evaluation_config(),
            )
            self.assertEqual(len(metrics), 10)
            self.assertEqual(len(predictions), 40)
            self.assertEqual(len(confusion), 10)
            self.assertTrue((metrics["balanced_accuracy"] == 1.0).all())
            self.assertTrue((metrics["roc_auc"] == 1.0).all())

    def test_evaluation_runner_saves_tables_and_figures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_path = root / "test_real.csv"
            held_out_frame().to_csv(test_path, index=False)
            output_paths = {
                "test_metrics": root / "outputs" / "metrics.csv",
                "test_predictions": root / "outputs" / "predictions.csv",
                "test_confusion_matrices": root / "outputs" / "confusion.csv",
                "test_balanced_accuracy_figure": root / "outputs" / "balanced.png",
                "test_roc_figure": root / "outputs" / "roc.png",
            }
            paths = run_evaluation_pipeline(
                test_path,
                manifest=manifest(root),
                model_config=model_config(),
                evaluation_config=evaluation_config(),
                output_paths=output_paths,
            )
            self.assertEqual(len(paths), 5)
            self.assertTrue(all(path.is_file() for path in paths.values()))

    def test_xgboost_runtime_must_match_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            saved_manifest = manifest(Path(directory))
            saved_manifest["versions"]["xgboost"] = "0.0.0"
            with self.assertRaisesRegex(ValueError, "runtime has"):
                evaluate_saved_models(
                    held_out_frame(),
                    saved_manifest,
                    model_config(),
                    evaluation_config(),
                )

    def test_scikit_learn_runtime_must_match_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            saved_manifest = manifest(Path(directory))
            saved_manifest["versions"]["scikit_learn"] = "0.0.0"
            with self.assertRaisesRegex(ValueError, "runtime has"):
                evaluate_saved_models(
                    held_out_frame(),
                    saved_manifest,
                    model_config(),
                    evaluation_config(),
                )

    def test_scikit_learn_version_must_be_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            saved_manifest = manifest(Path(directory))
            del saved_manifest["versions"]["scikit_learn"]
            with self.assertRaisesRegex(EvaluationError, "scikit-learn version"):
                evaluate_saved_models(
                    held_out_frame(),
                    saved_manifest,
                    model_config(),
                    evaluation_config(),
                )

    def test_augmented_condition_is_preserved_in_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            saved_manifest = manifest(Path(directory))
            saved_manifest["training_condition"] = "real_plus_synthetic"
            for record in saved_manifest["models"]:
                record["training_condition"] = "real_plus_synthetic"
            config = evaluation_config()
            config["training_condition"] = "real_plus_synthetic"
            metrics, predictions, confusion = evaluate_saved_models(
                held_out_frame(), saved_manifest, model_config(), config
            )
            assert set(metrics["training_condition"]) == {"real_plus_synthetic"}
            assert set(predictions["training_condition"]) == {"real_plus_synthetic"}
            assert set(confusion["training_condition"]) == {"real_plus_synthetic"}
