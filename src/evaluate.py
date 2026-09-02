"""Pure evaluation helpers for fitted pipelines and held-out observations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from src.features import split_features_target
from src.utils.io import (
    load_joblib,
    require_package_version,
)
from src.utils.metrics import (
    build_classification_scoring,
    calculate_classification_metrics,
)
from src.utils.plotting import plot_balanced_accuracy, plot_roc_curves
from src.utils.prediction import get_positive_class_score

LOGGER = logging.getLogger(__name__)

class EvaluationError(ValueError):
    """Raised when saved models or held-out evaluation inputs are invalid."""


def _validate_evaluation_contract(
    manifest: dict[str, Any],
    model_config: dict[str, Any],
    evaluation_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate the exact final experiment set and approved metrics."""
    versions = manifest.get("versions")
    if not isinstance(versions, dict) or not isinstance(
        versions.get("scikit_learn"), str
    ):
        raise EvaluationError("Model manifest must record the scikit-learn version")
    if not isinstance(versions.get("xgboost"), str):
        raise EvaluationError("Model manifest must record the XGBoost version")
    require_package_version(
        "scikit-learn",
        versions["scikit_learn"],
        context="Saved-model evaluation",
    )
    require_package_version(
        "xgboost",
        versions["xgboost"],
        context="Saved-model evaluation",
    )
    training_condition = evaluation_config.get("training_condition")
    if training_condition not in {"real_only", "real_plus_synthetic"}:
        raise EvaluationError("Unsupported evaluation training condition")
    if manifest.get("training_condition") != training_condition:
        raise EvaluationError(
            "Model manifest and evaluation training conditions must match"
        )
    if evaluation_config.get("positive_label") != 1:
        raise EvaluationError("Positive label must remain 1")
    metrics_config = evaluation_config.get("metrics")
    feature_sets = model_config.get("feature_sets")
    if not isinstance(metrics_config, dict) or not isinstance(feature_sets, dict):
        raise EvaluationError("Evaluation metrics and feature sets must be mappings")
    primary = metrics_config.get("primary")
    secondary = metrics_config.get("secondary")
    if primary != "balanced_accuracy" or not isinstance(secondary, list):
        raise EvaluationError("Balanced accuracy must remain the primary metric")
    build_classification_scoring([primary, *secondary])

    records = manifest.get("models")
    if not isinstance(records, list):
        raise EvaluationError("Model manifest models must be a list")
    expected_algorithms = model_config.get("experiment_algorithms")
    if not isinstance(expected_algorithms, list) or not expected_algorithms:
        raise EvaluationError("model_config.experiment_algorithms must be a list")
    expected = {
        (algorithm, feature_set)
        for feature_set in feature_sets
        for algorithm in expected_algorithms
    }
    actual: set[tuple[str, str]] = set()
    for record in records:
        if not isinstance(record, dict):
            raise EvaluationError("Each model manifest record must be a mapping")
        key = (str(record.get("algorithm")), str(record.get("feature_set")))
        if record.get("training_condition") != training_condition:
            raise EvaluationError(f"Model record training condition mismatch: {key}")
        if key in actual:
            raise EvaluationError(f"Duplicate model manifest record: {key}")
        actual.add(key)
    missing = sorted(expected.difference(actual))
    extra = sorted(actual.difference(expected))
    if missing or extra:
        raise EvaluationError(
            f"Model manifest experiment mismatch; missing={missing}, extra={extra}"
        )
    return records


def generate_saved_predictions(
    test_frame: pd.DataFrame,
    manifest: dict[str, Any],
    model_config: dict[str, Any],
    evaluation_config: dict[str, Any],
) -> pd.DataFrame:
    """Generate predictions from fitted pipelines without calculating metrics."""
    records = _validate_evaluation_contract(
        manifest, model_config, evaluation_config
    )
    training_condition = str(evaluation_config["training_condition"])
    feature_sets = model_config["feature_sets"]
    split_config = model_config["split"]
    subject_column = str(split_config["subject_id_column"])
    prediction_frames: list[pd.DataFrame] = []

    for record in records:
        algorithm = str(record["algorithm"])
        feature_set_name = str(record["feature_set"])
        features, target = split_features_target(
            test_frame,
            feature_set_name,
            feature_sets,
            target_column=str(split_config["target_column"]),
            subject_id_column=subject_column,
        )
        if list(features.columns) != record.get("feature_columns"):
            raise EvaluationError(
                f"Feature columns do not match manifest for "
                f"{(algorithm, feature_set_name)}"
            )
        model = load_joblib(record["path"])
        check_is_fitted(model)
        predictions = np.asarray(model.predict(features))
        if predictions.shape != target.shape or not set(predictions).issubset({0, 1}):
            raise EvaluationError(
                f"Invalid predictions from {(algorithm, feature_set_name)}"
            )
        scores = get_positive_class_score(model, features)
        identifiers = {
            "algorithm": algorithm,
            "feature_set": feature_set_name,
            "training_condition": training_condition,
        }
        prediction_frames.append(
            pd.DataFrame(
                {
                    **identifiers,
                    "subject_id": test_frame[subject_column].astype(str),
                    "target": target.astype(int),
                    "prediction": predictions.astype(int),
                    "score": scores,
                }
            )
        )

    return pd.concat(prediction_frames, ignore_index=True)


def evaluate_saved_models(
    test_frame: pd.DataFrame,
    manifest: dict[str, Any],
    model_config: dict[str, Any],
    evaluation_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compatibility helper deriving tables from generated predictions."""
    predictions = generate_saved_predictions(
        test_frame, manifest, model_config, evaluation_config
    )
    metrics, confusion = calculate_evaluation_tables(predictions)
    return metrics, predictions, confusion


def calculate_evaluation_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive model-level metrics and confusion counts from frozen predictions."""
    required = {
        "algorithm", "feature_set", "training_condition", "target", "prediction", "score"
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise EvaluationError(f"Predictions missing evaluation columns: {missing}")
    metrics_records: list[dict[str, Any]] = []
    confusion_records: list[dict[str, Any]] = []
    identifiers = ["algorithm", "feature_set", "training_condition"]
    for keys, group in predictions.groupby(identifiers, sort=True):
        metrics = calculate_classification_metrics(
            group["target"].to_numpy(dtype=int),
            group["prediction"].to_numpy(dtype=int),
            group["score"].to_numpy(dtype=float),
        )
        identity = dict(zip(identifiers, keys, strict=True))
        metrics_records.append({**identity, **metrics})
        confusion_records.append(
            {**identity, **{name: metrics[name] for name in ("tn", "fp", "fn", "tp")}}
        )
    return (
        pd.DataFrame.from_records(metrics_records),
        pd.DataFrame.from_records(confusion_records),
    )


def save_evaluation_outputs(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    confusion_matrices: pd.DataFrame,
    *,
    metrics_path: str | Path,
    predictions_path: str | Path | None,
    confusion_path: str | Path,
    balanced_accuracy_figure_path: str | Path,
    roc_figure_path: str | Path,
    dpi: int,
) -> dict[str, Path]:
    """Save held-out tables and figures without modifying fitted models."""
    paths = {
        "metrics": Path(metrics_path),
        "confusion_matrices": Path(confusion_path),
        "balanced_accuracy_figure": Path(balanced_accuracy_figure_path),
        "roc_figure": Path(roc_figure_path),
    }
    if predictions_path is not None:
        paths["predictions"] = Path(predictions_path)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        metrics.to_csv(paths["metrics"], index=False)
        if "predictions" in paths:
            predictions.to_csv(paths["predictions"], index=False)
        confusion_matrices.to_csv(paths["confusion_matrices"], index=False)
    except OSError as exc:
        raise OSError(f"Could not save held-out evaluation tables: {exc}") from exc
    plot_balanced_accuracy(
        metrics, paths["balanced_accuracy_figure"], dpi=dpi
    )
    plot_roc_curves(predictions, paths["roc_figure"], dpi=dpi)
    return paths
