"""Tests for fixed-model Phase 6 explainability."""

from __future__ import annotations

import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from src.explain import (
    ExplainabilityError,
    calculate_logistic_coefficients,
    calculate_permutation_importance,
    calculate_tree_shap,
    resolve_explainability_settings,
    validate_explanation_contract,
)
from src.features import split_features_target
from src.models import build_model_pipeline
from src.utils.io import save_joblib

ALGORITHMS = (
    "logistic_regression",
    "svm",
    "decision_tree",
    "random_forest",
    "xgboost",
)


class PredictionOnlyEstimator(ClassifierMixin, BaseEstimator):
    """Fitted explanation double that fails if refitting is attempted."""

    def __init__(self) -> None:
        self.classes_ = np.array([0, 1])
        self.fitted_ = True

    def fit(self, *_: object, **__: object) -> None:
        raise AssertionError("Explainability must not call fit")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return (features["mmse"].to_numpy() < 25).astype(int)


def model_config() -> dict:
    """Return the two approved feature sets and preprocessing settings."""
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


def explanation_config() -> dict:
    """Return lightweight approved explanation settings."""
    return {
        "training_condition": "real_only",
        "expected_subject_count": 10,
        "random_state": 7,
        "permutation_importance": {
            "scoring": "balanced_accuracy",
            "n_repeats": 3,
            "n_jobs": 1,
        },
        "logistic_coefficients": {
            "algorithm": "logistic_regression",
            "numeric_unit": "one_training_standard_deviation",
            "categorical_contrast": "male_minus_female",
        },
        "tree_shap": {
            "algorithms": ["decision_tree", "random_forest", "xgboost"],
            "class_label": 1,
            "aggregate_encoded_features": True,
            "check_additivity": True,
        },
        "figures": {"dpi": 72},
    }


def subject_frame() -> pd.DataFrame:
    """Return a small balanced held-out-style subject table."""
    rows = 10
    return pd.DataFrame(
        {
            "subject_id": [f"T{i:02d}" for i in range(rows)],
            "sex": ["female", "male"] * 5,
            "age": np.arange(65, 75),
            "education_years": [10, 12, 14, 16, 18] * 2,
            "ses": [1, 2, 3, 4, 5] * 2,
            "mmse": [29, 20, 28, 19, 27, 18, 26, 17, 25, 16],
            "etiv": np.arange(1300, 1310),
            "nwbv": np.linspace(0.60, 0.78, rows),
            "asf": np.linspace(1.0, 1.18, rows),
            "dementia": [0, 1] * 5,
        }
    )


def fitted_records(root: Path, algorithms: tuple[str, ...]) -> list[dict]:
    """Fit lightweight pipelines before explanation and save their records."""
    frame = subject_frame()
    config = model_config()
    estimator_configs = {
        "logistic_regression": {"solver": "lbfgs", "max_iter": 200},
        "decision_tree": {"max_depth": 2},
        "random_forest": {"n_estimators": 3, "n_jobs": 1},
        "xgboost": {
            "n_estimators": 3,
            "max_depth": 2,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": 1,
            "verbosity": 0,
        },
    }
    records = []
    for feature_set in ("clinical", "clinical_imaging"):
        features, target = split_features_target(
            frame, feature_set, config["feature_sets"]
        )
        for algorithm in algorithms:
            pipeline = build_model_pipeline(
                algorithm,
                feature_set,
                preprocessing_config=config["preprocessing"],
                feature_sets_config=config["feature_sets"],
                estimator_config=estimator_configs[algorithm],
                scale_numeric=algorithm == "logistic_regression",
                random_state=7,
            )
            pipeline.fit(features, target)
            path = save_joblib(pipeline, root / f"{algorithm}_{feature_set}.joblib")
            records.append(
                {
                    "algorithm": algorithm,
                    "feature_set": feature_set,
                    "path": str(path),
                    "feature_columns": list(features.columns),
                }
            )
    return records


def explanation_manifest() -> dict:
    """Return the exact saved-model grid needed for contract-only tests."""
    records = [
        {"algorithm": algorithm, "feature_set": feature_set}
        for feature_set in ("clinical", "clinical_imaging")
        for algorithm in ALGORITHMS
    ]
    return {
        "training_condition": "real_only",
        "versions": {
            "scikit_learn": version("scikit-learn"),
            "xgboost": version("xgboost"),
        },
        "models": records,
    }


class ExplainTests(unittest.TestCase):
    """Verify model-appropriate explanations and no-refit behavior."""

    def test_permutation_covers_all_models_without_refitting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = save_joblib(
                PredictionOnlyEstimator(), Path(directory) / "predict_only.joblib"
            )
            records = []
            for feature_set in ("clinical", "clinical_imaging"):
                columns = (
                    model_config()["feature_sets"][feature_set]["numeric"]
                    + model_config()["feature_sets"][feature_set]["categorical"]
                )
                for algorithm in ALGORITHMS:
                    records.append(
                        {
                            "algorithm": algorithm,
                            "feature_set": feature_set,
                            "path": str(path),
                            "feature_columns": columns,
                        }
                    )
            importance = calculate_permutation_importance(
                subject_frame(),
                records,
                model_config(),
                resolve_explainability_settings(explanation_config()),
            )
            self.assertEqual(
                set(importance[["algorithm", "feature_set"]].itertuples(index=False, name=None)),
                {
                    (algorithm, feature_set)
                    for algorithm in ALGORITHMS
                    for feature_set in ("clinical", "clinical_imaging")
                },
            )
            self.assertEqual(set(importance["scoring"]), {"balanced_accuracy"})

    def test_logistic_coefficients_use_one_sex_contrast(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = fitted_records(Path(directory), ("logistic_regression",))
            coefficients = calculate_logistic_coefficients(records, model_config())
            sex = coefficients.loc[coefficients["feature"] == "sex"]
            self.assertEqual(len(sex), 2)
            self.assertEqual(set(sex["term"]), {"sex_male_vs_female"})
            self.assertTrue((coefficients["odds_ratio"] > 0).all())

    def test_tree_shap_covers_six_models_and_aggregates_sex(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = fitted_records(
                Path(directory), ("decision_tree", "random_forest", "xgboost")
            )
            importance, values = calculate_tree_shap(
                subject_frame(),
                records,
                model_config(),
                resolve_explainability_settings(explanation_config()),
            )
            experiments = set(
                importance[["algorithm", "feature_set"]].itertuples(
                    index=False, name=None
                )
            )
            self.assertEqual(len(experiments), 6)
            self.assertIn("sex", set(values["feature"]))
            self.assertFalse(values["feature"].str.startswith("categorical__").any())
            self.assertNotIn("dementia", values.columns)

    def test_ordinary_accuracy_is_rejected(self) -> None:
        config = explanation_config()
        config["permutation_importance"]["scoring"] = "accuracy"
        with self.assertRaisesRegex(ExplainabilityError, "balanced_accuracy"):
            resolve_explainability_settings(config)

    def test_scikit_learn_runtime_must_match_manifest(self) -> None:
        saved_manifest = explanation_manifest()
        saved_manifest["versions"]["scikit_learn"] = "0.0.0"
        with self.assertRaisesRegex(ValueError, "runtime has"):
            validate_explanation_contract(
                saved_manifest,
                model_config(),
                resolve_explainability_settings(explanation_config()),
                subject_frame(),
            )

    def test_scikit_learn_version_must_be_recorded(self) -> None:
        saved_manifest = explanation_manifest()
        del saved_manifest["versions"]["scikit_learn"]
        with self.assertRaisesRegex(ExplainabilityError, "scikit-learn version"):
            validate_explanation_contract(
                saved_manifest,
                model_config(),
                resolve_explainability_settings(explanation_config()),
                subject_frame(),
            )
