"""Tests for explicit, no-refit saved-model inference."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from src.predict import (
    InferenceError,
    predict_subjects,
    run_prediction_pipeline,
    select_model_record,
)
from src.utils.io import save_joblib, save_json


class FittedProbabilityModel(ClassifierMixin, BaseEstimator):
    """Inference test double that raises if code attempts to refit it."""

    def __init__(self) -> None:
        self.classes_ = np.array([0, 1])
        self.fitted_ = True

    def fit(self, *_: object, **__: object) -> None:
        raise AssertionError("Inference must never fit a model")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return (features["mmse"].fillna(30) < 25).astype(int).to_numpy()

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        positive = np.where(self.predict(features) == 1, 0.8, 0.2)
        return np.column_stack([1 - positive, positive])


def model_config() -> dict:
    return {
        "split": {"subject_id_column": "subject_id", "target_column": "dementia"},
        "feature_sets": {
            "clinical": {
                "numeric": ["age", "education_years", "ses", "mmse"],
                "categorical": ["sex"],
            }
        },
    }


def inference_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": ["NEW_1", "NEW_2"],
            "age": [70, 75],
            "education_years": [12, 16],
            "ses": [2, pd.NA],
            "mmse": [29, 20],
            "sex": ["female", "male"],
        }
    )


def manifest(model_path: Path) -> dict:
    return {
        "training_condition": "real_only",
        "versions": {
            "scikit_learn": version("scikit-learn"),
            "xgboost": version("xgboost"),
        },
        "models": [
            {
                "algorithm": "logistic_regression",
                "feature_set": "clinical",
                "training_condition": "real_only",
                "path": str(model_path),
                "feature_columns": [
                    "age", "education_years", "ses", "mmse", "sex"
                ],
            }
        ],
    }


def test_predict_subjects_uses_named_fitted_model_without_target(tmp_path: Path) -> None:
    path = save_joblib(FittedProbabilityModel(), tmp_path / "model.joblib")
    result = predict_subjects(
        inference_frame(),
        manifest(path),
        model_config(),
        algorithm="logistic_regression",
        feature_set="clinical",
        training_condition="real_only",
    )
    assert list(result["subject_id"]) == ["NEW_1", "NEW_2"]
    assert list(result["prediction"]) == [0, 1]
    assert list(result["score"]) == pytest.approx([0.2, 0.8])
    assert "target" not in result.columns


def test_missing_predictor_and_duplicate_identifier_are_rejected() -> None:
    record = manifest(Path("unused.joblib"))["models"][0]
    missing = inference_frame().drop(columns="education_years")
    with pytest.raises(InferenceError, match="missing required columns"):
        from src.predict import validate_inference_frame

        validate_inference_frame(missing, record, model_config())
    duplicate = inference_frame()
    duplicate["subject_id"] = "same"
    with pytest.raises(InferenceError, match="must be unique"):
        validate_inference_frame(duplicate, record, model_config())


def test_model_selection_has_no_fallback() -> None:
    saved = manifest(Path("unused.joblib"))
    with pytest.raises(InferenceError, match="Expected one model"):
        select_model_record(
            saved,
            algorithm="random_forest",
            feature_set="clinical",
            training_condition="real_only",
        )


def test_runner_refuses_to_overwrite_output_without_flag(tmp_path: Path) -> None:
    model_path = save_joblib(FittedProbabilityModel(), tmp_path / "model.joblib")
    manifest_path = save_json(manifest(model_path), tmp_path / "manifest.json")
    source = tmp_path / "subjects.csv"
    destination = tmp_path / "predictions.csv"
    inference_frame().to_csv(source, index=False)
    destination.write_text("existing", encoding="utf-8")
    paths = {"model_manifest": str(manifest_path)}
    with pytest.raises(FileExistsError, match="--overwrite"):
        run_prediction_pipeline(
            source,
            destination,
            paths_config=paths,
            model_config=model_config(),
            algorithm="logistic_regression",
            feature_set="clinical",
            training_condition="real_only",
        )
    run_prediction_pipeline(
        source,
        destination,
        paths_config=paths,
        model_config=model_config(),
        algorithm="logistic_regression",
        feature_set="clinical",
        training_condition="real_only",
        overwrite=True,
    )
    assert len(pd.read_csv(destination)) == 2
