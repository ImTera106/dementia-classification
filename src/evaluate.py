"""Evaluate saved Phase 4 pipelines once on the persistent real test set."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from src.features import split_features_target
from src.tune import REQUIRED_ALGORITHMS
from src.utils.io import (
    load_joblib,
    load_json,
    load_yaml_config,
    require_package_version,
)
from src.utils.metrics import (
    build_classification_scoring,
    calculate_classification_metrics,
)
from src.utils.plotting import plot_balanced_accuracy, plot_roc_curves
from src.utils.prediction import get_positive_class_score

LOGGER = logging.getLogger(__name__)

DEFAULT_PATHS_CONFIG: Final[Path] = Path("config/paths.yaml")
DEFAULT_MODEL_CONFIG: Final[Path] = Path("config/model_config.yaml")
DEFAULT_EVALUATION_CONFIG: Final[Path] = Path("config/evaluation_config.yaml")


class EvaluationError(ValueError):
    """Raised when saved models or held-out evaluation inputs are invalid."""


def load_real_test_data(path: str | Path) -> pd.DataFrame:
    """Load the persistent real held-out partition for final evaluation only."""
    test_path = Path(path)
    if not test_path.is_file():
        raise FileNotFoundError(f"Real test partition not found: {test_path}")
    frame = pd.read_csv(test_path)
    LOGGER.info("Loaded %d real held-out test subjects", len(frame))
    return frame


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
    expected = {
        (algorithm, feature_set)
        for feature_set in feature_sets
        for algorithm in REQUIRED_ALGORITHMS
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


def evaluate_saved_models(
    test_frame: pd.DataFrame,
    manifest: dict[str, Any],
    model_config: dict[str, Any],
    evaluation_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score every saved pipeline on the same held-out real subjects."""
    records = _validate_evaluation_contract(
        manifest, model_config, evaluation_config
    )
    training_condition = str(evaluation_config["training_condition"])
    feature_sets = model_config["feature_sets"]
    split_config = model_config["split"]
    subject_column = str(split_config["subject_id_column"])
    metric_records: list[dict[str, Any]] = []
    confusion_records: list[dict[str, Any]] = []
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
        metrics = calculate_classification_metrics(target, predictions, scores)
        identifiers = {
            "algorithm": algorithm,
            "feature_set": feature_set_name,
            "training_condition": training_condition,
        }
        metric_records.append({**identifiers, **metrics})
        confusion_records.append(
            {
                **identifiers,
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
            }
        )
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

    return (
        pd.DataFrame.from_records(metric_records),
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame.from_records(confusion_records),
    )


def save_evaluation_outputs(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    confusion_matrices: pd.DataFrame,
    *,
    metrics_path: str | Path,
    predictions_path: str | Path,
    confusion_path: str | Path,
    balanced_accuracy_figure_path: str | Path,
    roc_figure_path: str | Path,
    dpi: int,
) -> dict[str, Path]:
    """Save held-out tables and figures without modifying fitted models."""
    paths = {
        "metrics": Path(metrics_path),
        "predictions": Path(predictions_path),
        "confusion_matrices": Path(confusion_path),
        "balanced_accuracy_figure": Path(balanced_accuracy_figure_path),
        "roc_figure": Path(roc_figure_path),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        metrics.to_csv(paths["metrics"], index=False)
        predictions.to_csv(paths["predictions"], index=False)
        confusion_matrices.to_csv(paths["confusion_matrices"], index=False)
    except OSError as exc:
        raise OSError(f"Could not save held-out evaluation tables: {exc}") from exc
    plot_balanced_accuracy(
        metrics, paths["balanced_accuracy_figure"], dpi=dpi
    )
    plot_roc_curves(predictions, paths["roc_figure"], dpi=dpi)
    return paths


def run_evaluation_pipeline(
    test_path: str | Path,
    *,
    manifest: dict[str, Any],
    model_config: dict[str, Any],
    evaluation_config: dict[str, Any],
    output_paths: dict[str, Any],
) -> dict[str, Path]:
    """Load the held-out partition once, score saved models, and save outputs."""
    test_frame = load_real_test_data(test_path)
    metrics, predictions, confusion_matrices = evaluate_saved_models(
        test_frame, manifest, model_config, evaluation_config
    )
    figure_config = evaluation_config.get("figures")
    if not isinstance(figure_config, dict):
        raise EvaluationError("evaluation figures must be a mapping")
    paths = save_evaluation_outputs(
        metrics,
        predictions,
        confusion_matrices,
        metrics_path=output_paths["test_metrics"],
        predictions_path=output_paths["test_predictions"],
        confusion_path=output_paths["test_confusion_matrices"],
        balanced_accuracy_figure_path=output_paths[
            "test_balanced_accuracy_figure"
        ],
        roc_figure_path=output_paths["test_roc_figure"],
        dpi=int(figure_config["dpi"]),
    )
    LOGGER.info("Saved held-out results for %d models", len(metrics))
    return paths


def parse_args() -> argparse.Namespace:
    """Parse YAML configuration paths for held-out evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument(
        "--evaluation-config", type=Path, default=DEFAULT_EVALUATION_CONFIG
    )
    return parser.parse_args()


def main() -> int:
    """Evaluate final saved pipelines once on the configured real test set."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    try:
        paths_config = load_yaml_config(args.paths_config)
        run_evaluation_pipeline(
            paths_config["data"]["test"],
            manifest=load_json(paths_config["model_manifest"]),
            model_config=load_yaml_config(args.model_config),
            evaluation_config=load_yaml_config(args.evaluation_config),
            output_paths=paths_config["outputs"],
        )
    except (
        EvaluationError,
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        LOGGER.error("Held-out evaluation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
