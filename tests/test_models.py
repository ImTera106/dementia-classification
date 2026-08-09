"""Tests for configuration-driven, leakage-safe baseline pipelines."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models import (
    ModelConfigurationError,
    build_baseline_pipeline,
    build_preprocessor,
    build_tuning_pipeline,
)


def preprocessing_config() -> dict:
    """Return the supported Phase 2 preprocessing settings."""
    return {
        "numeric_imputer": {"strategy": "median"},
        "categorical_imputer": {"strategy": "most_frequent"},
        "one_hot_encoder": {"handle_unknown": "ignore"},
        "scale_numeric": True,
    }


def feature_sets_config() -> dict:
    """Return YAML-style feature definitions for pipeline construction."""
    return {
        "clinical": {
            "numeric": ["age", "education_years", "ses", "mmse"],
            "categorical": ["sex"],
        }
    }


class ModelTests(unittest.TestCase):
    """Verify pipeline structure, configuration use, and fold-only fitting."""

    def test_pipeline_contains_preprocessing_before_estimator(self) -> None:
        pipeline = build_baseline_pipeline(
            "logistic_regression",
            "clinical",
            preprocessing_config=preprocessing_config(),
            feature_sets_config=feature_sets_config(),
            algorithm_config={
                "regularization": "l2",
                "solver": "lbfgs",
                "C": 0.25,
                "max_iter": 321,
            },
            random_state=17,
        )
        self.assertEqual(list(pipeline.named_steps), ["preprocess", "model"])
        self.assertEqual(pipeline.named_steps["model"].C, 0.25)
        self.assertEqual(pipeline.named_steps["model"].max_iter, 321)
        self.assertEqual(pipeline.named_steps["model"].random_state, 17)

    def test_preprocessor_learns_statistics_from_fit_fold_only(self) -> None:
        fold_train = pd.DataFrame(
            {
                "age": [60.0, 62.0, 64.0, 66.0],
                "education_years": [10, 12, 14, 16],
                "ses": [1.0, np.nan, 3.0, 5.0],
                "mmse": [20.0, 22.0, 24.0, 26.0],
                "sex": ["female", "male", "female", "male"],
            }
        )
        held_out_fold = pd.DataFrame(
            {
                "age": [1000.0],
                "education_years": [20],
                "ses": [999.0],
                "mmse": [1.0],
                "sex": ["unknown_to_training_fold"],
            }
        )
        preprocessor = build_preprocessor(
            "clinical",
            preprocessing_config(),
            feature_sets_config(),
            scale_numeric=True,
        )
        preprocessor.fit(fold_train)
        numeric_imputer = (
            preprocessor.named_transformers_["numeric"].named_steps["imputer"]
        )
        self.assertEqual(float(numeric_imputer.statistics_[2]), 3.0)
        preprocessor.transform(held_out_fold)
        self.assertNotIn(999.0, numeric_imputer.statistics_)

    def test_unknown_algorithm_is_rejected(self) -> None:
        with self.assertRaisesRegex(ModelConfigurationError, "Unsupported"):
            build_baseline_pipeline(
                "random_forest",
                "clinical",
                preprocessing_config=preprocessing_config(),
                feature_sets_config=feature_sets_config(),
                algorithm_config={},
                random_state=1,
            )

    def test_tuning_pipelines_apply_model_specific_scaling(self) -> None:
        estimator_configs = {
            "logistic_regression": {"solver": "lbfgs", "max_iter": 100},
            "svm": {"kernel": "rbf"},
            "decision_tree": {},
            "random_forest": {"n_estimators": 5, "n_jobs": 1},
            "xgboost": {
                "n_estimators": 5,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "n_jobs": 1,
                "verbosity": 0,
            },
        }
        for algorithm, estimator_config in estimator_configs.items():
            should_scale = algorithm in {"logistic_regression", "svm"}
            pipeline = build_tuning_pipeline(
                algorithm,
                "clinical",
                preprocessing_config=preprocessing_config(),
                feature_sets_config=feature_sets_config(),
                estimator_config=estimator_config,
                scale_numeric=should_scale,
                random_state=7,
            )
            numeric_steps = pipeline.named_steps[
                "preprocess"
            ].transformers[0][1].named_steps
            self.assertEqual("scaler" in numeric_steps, should_scale)
            self.assertEqual(
                pipeline.named_steps["model"].random_state,
                7,
            )
