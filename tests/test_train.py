"""Tests for final training without held-out test access."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.train import TrainingConfigurationError, train_final_models


ALGORITHMS = (
    "logistic_regression",
    "svm",
    "decision_tree",
    "random_forest",
    "xgboost",
)


def model_config() -> dict:
    """Return feature and preprocessing settings for final-training tests."""
    return {
        "split": {
            "subject_id_column": "subject_id",
            "target_column": "dementia",
        },
        "preprocessing": {
            "numeric_imputer": {"strategy": "median"},
            "categorical_imputer": {"strategy": "most_frequent"},
            "one_hot_encoder": {"handle_unknown": "ignore"},
            "scale_numeric": True,
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


def tuning_config() -> dict:
    """Return lightweight fixed estimators covering all model families."""
    return {
        "random_state": 5,
        "training_condition": "real_only",
        "algorithms": {
            "logistic_regression": {
                "scale_numeric": True,
                "estimator": {"solver": "lbfgs", "max_iter": 200},
            },
            "svm": {
                "scale_numeric": True,
                "estimator": {"kernel": "rbf"},
            },
            "decision_tree": {"scale_numeric": False, "estimator": {}},
            "random_forest": {
                "scale_numeric": False,
                "estimator": {"n_jobs": 1},
            },
            "xgboost": {
                "required_version": "3.4.0",
                "scale_numeric": False,
                "estimator": {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "tree_method": "hist",
                    "n_jobs": 1,
                    "verbosity": 0,
                },
            },
        },
    }


def best_parameters() -> dict:
    """Return one selected parameter record per required experiment."""
    parameters = {
        "logistic_regression": {"model__C": 1.0},
        "svm": {"model__C": 1.0, "model__gamma": "scale"},
        "decision_tree": {"model__max_depth": 3},
        "random_forest": {"model__n_estimators": 5},
        "xgboost": {"model__n_estimators": 5},
    }
    return {
        "final_models_fitted": False,
        "experiments": [
            {
                "algorithm": algorithm,
                "feature_set": feature_set,
                "training_condition": "real_only",
                "parameters": parameters[algorithm],
            }
            for feature_set in ("clinical", "clinical_imaging")
            for algorithm in ALGORITHMS
        ],
    }


def training_frame() -> pd.DataFrame:
    """Return balanced real-style subject-level training data."""
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


class TrainTests(unittest.TestCase):
    """Verify exact experiment coverage and fitted model persistence."""

    def test_all_ten_final_pipelines_are_fitted_and_saved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = train_final_models(
                training_frame(),
                model_config(),
                tuning_config(),
                best_parameters(),
                models_dir=directory,
            )
            self.assertEqual(manifest["trained_subjects"], 20)
            self.assertEqual(len(manifest["models"]), 10)
            self.assertTrue(
                all(Path(record["path"]).is_file() for record in manifest["models"])
            )
            self.assertEqual(
                {(record["algorithm"], record["feature_set"]) for record in manifest["models"]},
                {
                    (algorithm, feature_set)
                    for feature_set in ("clinical", "clinical_imaging")
                    for algorithm in ALGORITHMS
                },
            )

    def test_missing_selected_experiment_is_rejected(self) -> None:
        selected = best_parameters()
        selected["experiments"].pop()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TrainingConfigurationError, "missing"):
                train_final_models(
                    training_frame(),
                    model_config(),
                    tuning_config(),
                    selected,
                    models_dir=directory,
                )
