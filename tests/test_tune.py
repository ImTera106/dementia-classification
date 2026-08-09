"""Tests for configuration-driven, training-only Phase 3 tuning."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.tune import (
    TuningConfigurationError,
    resolve_tuning_settings,
    run_tuning_pipeline,
    tune_candidates,
)


def model_config() -> dict:
    """Return the model settings shared with the Phase 3 runner."""
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
    """Return minimal searches covering every required algorithm."""
    return {
        "random_state": 11,
        "training_condition": "real_only",
        "cross_validation": {
            "method": "repeated_stratified_kfold",
            "n_splits": 2,
            "n_repeats": 1,
            "shuffle": True,
        },
        "search": {
            "method": "randomized_search",
            "primary_metric": "balanced_accuracy",
            "secondary_metrics": [
                "roc_auc",
                "sensitivity",
                "specificity",
                "precision",
                "f1",
            ],
            "n_jobs": 1,
            "return_train_score": False,
        },
        "algorithms": {
            "logistic_regression": {
                "n_iter": 1,
                "scale_numeric": True,
                "estimator": {"solver": "lbfgs", "max_iter": 200},
                "parameters": {"model__C": [1.0]},
            },
            "svm": {
                "n_iter": 1,
                "scale_numeric": True,
                "estimator": {"kernel": "rbf"},
                "parameters": {"model__C": [1.0]},
            },
            "decision_tree": {
                "n_iter": 1,
                "scale_numeric": False,
                "estimator": {},
                "parameters": {"model__max_depth": [3]},
            },
            "random_forest": {
                "n_iter": 1,
                "scale_numeric": False,
                "estimator": {"n_jobs": 1},
                "parameters": {"model__n_estimators": [5]},
            },
            "xgboost": {
                "required_version": "3.4.0",
                "n_iter": 1,
                "scale_numeric": False,
                "estimator": {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "tree_method": "hist",
                    "n_jobs": 1,
                    "verbosity": 0,
                },
                "parameters": {"model__n_estimators": [5]},
            },
        },
    }


def training_frame() -> pd.DataFrame:
    """Return balanced subject-level data with deliberate numeric missingness."""
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


class TuningTests(unittest.TestCase):
    """Verify search coverage, non-refitting selection, and persistence."""

    def test_all_ten_experiments_run_without_final_refit(self) -> None:
        results, summary, best = tune_candidates(
            training_frame(), model_config(), tuning_config()
        )
        self.assertEqual(len(results), 10)
        self.assertEqual(len(summary), 10)
        self.assertEqual(len(best["experiments"]), 10)
        self.assertFalse(best["final_models_fitted"])
        self.assertEqual(set(summary["training_condition"]), {"real_only"})
        self.assertEqual(set(summary["feature_set"]), {"clinical", "clinical_imaging"})

    def test_missing_required_algorithm_is_rejected(self) -> None:
        config = tuning_config()
        del config["algorithms"]["xgboost"]
        with self.assertRaisesRegex(TuningConfigurationError, "missing"):
            resolve_tuning_settings(model_config(), config)

    def test_runner_saves_selection_artifacts_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train_real.csv"
            training_frame().to_csv(train_path, index=False)
            paths = run_tuning_pipeline(
                train_path,
                model_config=model_config(),
                tuning_config=tuning_config(),
                cv_results_path=root / "outputs" / "cv.csv",
                summary_path=root / "outputs" / "summary.csv",
                best_parameters_path=root / "outputs" / "best.json",
            )
            self.assertTrue(all(path.is_file() for path in paths.values()))
            saved = json.loads(
                paths["best_parameters"].read_text(encoding="utf-8")
            )
            self.assertFalse(saved["final_models_fitted"])
            self.assertEqual(len(pd.read_csv(paths["summary"])), 10)
            self.assertFalse(any(path.suffix in {".pkl", ".joblib"} for path in paths.values()))
