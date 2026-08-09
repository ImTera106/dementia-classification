"""Quantify uncertainty in fixed Phase 4 held-out predictions without refitting."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from src.tune import REQUIRED_ALGORITHMS
from src.utils.io import load_yaml_config
from src.utils.metrics import SUPPORTED_METRICS, calculate_classification_metrics
from src.utils.plotting import (
    plot_balanced_accuracy_intervals,
    plot_feature_set_differences,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_PATHS_CONFIG: Final[Path] = Path("config/paths.yaml")
DEFAULT_EVALUATION_CONFIG: Final[Path] = Path("config/evaluation_config.yaml")
DEFAULT_VALIDATION_CONFIG: Final[Path] = Path("config/validation_config.yaml")
PREDICTION_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "algorithm",
        "feature_set",
        "training_condition",
        "subject_id",
        "target",
        "prediction",
        "score",
    }
)


class ValidationError(ValueError):
    """Raised when robustness configuration or predictions are invalid."""


def resolve_validation_settings(
    validation_config: dict[str, Any], evaluation_config: dict[str, Any]
) -> dict[str, Any]:
    """Validate and normalize the approved Phase 5 settings."""
    bootstrap = validation_config.get("bootstrap")
    comparison = validation_config.get("comparison")
    figures = validation_config.get("figures")
    metrics = evaluation_config.get("metrics")
    if not all(
        isinstance(value, dict)
        for value in (bootstrap, comparison, figures, metrics)
    ):
        raise ValidationError(
            "bootstrap, comparison, figures, and evaluation metrics must be mappings"
        )
    if validation_config.get("training_condition") != "real_only":
        raise ValidationError("Phase 5 training_condition must be real_only")
    if bootstrap.get("stratified") is not True:
        raise ValidationError("Phase 5 bootstrap must remain stratified")
    n_resamples = int(bootstrap.get("n_resamples", 0))
    confidence_level = float(bootstrap.get("confidence_level", 0))
    expected_subject_count = int(validation_config.get("expected_subject_count", 0))
    if n_resamples < 1 or not 0 < confidence_level < 1:
        raise ValidationError(
            "Bootstrap requires n_resamples >= 1 and 0 < confidence_level < 1"
        )
    if expected_subject_count < 2:
        raise ValidationError("expected_subject_count must be at least 2")
    metric_names = [metrics.get("primary"), *(metrics.get("secondary") or [])]
    if metric_names[0] != "balanced_accuracy" or any(
        not isinstance(metric, str) or metric not in SUPPORTED_METRICS
        for metric in metric_names
    ):
        raise ValidationError("Evaluation metrics must be approved project metrics")
    comparison_metric = comparison.get("metric")
    if comparison_metric != "balanced_accuracy":
        raise ValidationError("Feature-set comparison metric must be balanced_accuracy")
    reference = comparison.get("reference_feature_set")
    compared = comparison.get("comparison_feature_set")
    if {reference, compared} != {"clinical", "clinical_imaging"}:
        raise ValidationError(
            "Comparison must pair clinical with clinical_imaging"
        )
    return {
        "training_condition": "real_only",
        "expected_subject_count": expected_subject_count,
        "n_resamples": n_resamples,
        "confidence_level": confidence_level,
        "random_state": int(bootstrap["random_state"]),
        "metric_names": metric_names,
        "comparison_metric": comparison_metric,
        "reference_feature_set": str(reference),
        "comparison_feature_set": str(compared),
        "dpi": int(figures["dpi"]),
    }


def validate_prediction_frame(
    predictions: pd.DataFrame, settings: dict[str, Any]
) -> pd.DataFrame:
    """Validate one complete prediction per held-out subject and experiment."""
    missing_columns = sorted(PREDICTION_COLUMNS.difference(predictions.columns))
    if missing_columns:
        raise ValidationError(f"Prediction artifact missing columns: {missing_columns}")
    frame = predictions.loc[:, sorted(PREDICTION_COLUMNS)].copy()
    if frame.empty or frame[list(PREDICTION_COLUMNS)].isna().any().any():
        raise ValidationError("Prediction artifact must be nonempty and complete")
    if set(frame["training_condition"]) != {settings["training_condition"]}:
        raise ValidationError("Predictions must use training_condition real_only")
    if not set(frame["target"]).issubset({0, 1}) or set(frame["target"]) != {0, 1}:
        raise ValidationError("Prediction targets must contain both binary classes")
    if not set(frame["prediction"]).issubset({0, 1}):
        raise ValidationError("Predictions must be binary")
    if not np.isfinite(frame["score"].astype(float)).all():
        raise ValidationError("Prediction scores must be finite")
    keys = ["algorithm", "feature_set", "subject_id"]
    if frame.duplicated(keys).any():
        raise ValidationError("Duplicate algorithm/feature-set/subject predictions")

    subject_targets = frame.groupby("subject_id")["target"].nunique()
    if (subject_targets != 1).any():
        raise ValidationError("Targets must be consistent for every subject")
    subjects = set(frame["subject_id"].astype(str))
    if len(subjects) != settings["expected_subject_count"]:
        raise ValidationError(
            f"Expected {settings['expected_subject_count']} held-out subjects, "
            f"found {len(subjects)}"
        )
    expected_experiments = {
        (algorithm, feature_set)
        for algorithm in REQUIRED_ALGORITHMS
        for feature_set in ("clinical", "clinical_imaging")
    }
    actual_experiments = set(
        frame[["algorithm", "feature_set"]].itertuples(index=False, name=None)
    )
    if actual_experiments != expected_experiments:
        raise ValidationError(
            "Prediction artifact must contain the exact 10 final experiments"
        )
    for experiment, group in frame.groupby(["algorithm", "feature_set"]):
        if set(group["subject_id"].astype(str)) != subjects:
            raise ValidationError(f"Subject coverage mismatch for {experiment}")
    return frame


def _stratified_bootstrap_indices(
    targets: np.ndarray, *, n_resamples: int, random_state: int
) -> np.ndarray:
    """Draw shared within-class subject indices for every bootstrap replicate."""
    rng = np.random.default_rng(random_state)
    negative = np.flatnonzero(targets == 0)
    positive = np.flatnonzero(targets == 1)
    negative_draws = rng.choice(negative, size=(n_resamples, len(negative)), replace=True)
    positive_draws = rng.choice(positive, size=(n_resamples, len(positive)), replace=True)
    return np.concatenate([negative_draws, positive_draws], axis=1)


def _bootstrap_metric_values(
    target: np.ndarray,
    prediction: np.ndarray,
    score: np.ndarray,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Vectorize approved binary metrics across stratified bootstrap samples."""
    sampled_target = target[indices]
    sampled_prediction = prediction[indices]
    sampled_score = score[indices]
    positive = sampled_target == 1
    negative = ~positive
    tp = np.sum((sampled_prediction == 1) & positive, axis=1)
    fn = np.sum((sampled_prediction == 0) & positive, axis=1)
    tn = np.sum((sampled_prediction == 0) & negative, axis=1)
    fp = np.sum((sampled_prediction == 1) & negative, axis=1)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) != 0)

    positive_scores = sampled_score[positive].reshape(len(indices), -1)
    negative_scores = sampled_score[negative].reshape(len(indices), -1)
    comparisons = positive_scores[:, :, None] - negative_scores[:, None, :]
    roc_auc = np.mean((comparisons > 0) + 0.5 * (comparisons == 0), axis=(1, 2))
    return {
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "roc_auc": roc_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
    }


def calculate_bootstrap_validation(
    predictions: pd.DataFrame,
    validation_config: dict[str, Any],
    evaluation_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate fixed-model intervals and paired feature-set differences."""
    settings = resolve_validation_settings(validation_config, evaluation_config)
    frame = validate_prediction_frame(predictions, settings)
    subject_target = (
        frame[["subject_id", "target"]]
        .drop_duplicates()
        .sort_values("subject_id", kind="stable")
        .reset_index(drop=True)
    )
    target = subject_target["target"].to_numpy(dtype=int)
    indices = _stratified_bootstrap_indices(
        target,
        n_resamples=settings["n_resamples"],
        random_state=settings["random_state"],
    )
    alpha = 1 - settings["confidence_level"]
    quantiles = [alpha / 2, 1 - alpha / 2]
    interval_records: list[dict[str, Any]] = []
    bootstrap_values: dict[tuple[str, str, str], np.ndarray] = {}

    for (algorithm, feature_set), group in frame.groupby(
        ["algorithm", "feature_set"], sort=True
    ):
        aligned = subject_target[["subject_id"]].merge(
            group, on="subject_id", how="left", validate="one_to_one"
        )
        prediction = aligned["prediction"].to_numpy(dtype=int)
        score = aligned["score"].to_numpy(dtype=float)
        point_metrics = calculate_classification_metrics(target, prediction, score)
        values = _bootstrap_metric_values(target, prediction, score, indices)
        for metric in settings["metric_names"]:
            lower, upper = np.quantile(values[metric], quantiles)
            bootstrap_values[(str(algorithm), str(feature_set), metric)] = values[metric]
            interval_records.append(
                {
                    "algorithm": algorithm,
                    "feature_set": feature_set,
                    "training_condition": "real_only",
                    "metric": metric,
                    "estimate": point_metrics[metric],
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                    "confidence_level": settings["confidence_level"],
                    "n_resamples": settings["n_resamples"],
                }
            )

    difference_records: list[dict[str, Any]] = []
    metric = settings["comparison_metric"]
    reference = settings["reference_feature_set"]
    compared = settings["comparison_feature_set"]
    intervals = pd.DataFrame.from_records(interval_records)
    for algorithm in sorted(REQUIRED_ALGORITHMS):
        reference_values = bootstrap_values[(algorithm, reference, metric)]
        compared_values = bootstrap_values[(algorithm, compared, metric)]
        differences = compared_values - reference_values
        lower, upper = np.quantile(differences, quantiles)
        estimates = intervals.loc[
            (intervals["algorithm"] == algorithm)
            & (intervals["metric"] == metric)
        ].set_index("feature_set")["estimate"]
        difference_records.append(
            {
                "algorithm": algorithm,
                "training_condition": "real_only",
                "metric": metric,
                "comparison": f"{compared}_minus_{reference}",
                "estimate": float(estimates[compared] - estimates[reference]),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "confidence_level": settings["confidence_level"],
                "n_resamples": settings["n_resamples"],
            }
        )
    return intervals, pd.DataFrame.from_records(difference_records)


def run_validation_pipeline(
    predictions_path: str | Path,
    *,
    validation_config: dict[str, Any],
    evaluation_config: dict[str, Any],
    output_paths: dict[str, Any],
) -> dict[str, Path]:
    """Load fixed predictions, calculate intervals, and save Phase 5 outputs."""
    input_path = Path(predictions_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Held-out predictions not found: {input_path}")
    predictions = pd.read_csv(input_path, dtype={"subject_id": str})
    intervals, differences = calculate_bootstrap_validation(
        predictions, validation_config, evaluation_config
    )
    settings = resolve_validation_settings(validation_config, evaluation_config)
    paths = {
        "intervals": Path(output_paths["test_metric_bootstrap_intervals"]),
        "differences": Path(output_paths["feature_set_balanced_accuracy_differences"]),
        "interval_figure": Path(output_paths["test_balanced_accuracy_intervals_figure"]),
        "difference_figure": Path(
            output_paths["feature_set_balanced_accuracy_differences_figure"]
        ),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(paths["intervals"], index=False)
    differences.to_csv(paths["differences"], index=False)
    plot_balanced_accuracy_intervals(
        intervals, paths["interval_figure"], dpi=settings["dpi"]
    )
    plot_feature_set_differences(
        differences, paths["difference_figure"], dpi=settings["dpi"]
    )
    LOGGER.info(
        "Saved bootstrap intervals for %d fixed experiments", len(intervals) // 6
    )
    return paths


def parse_args() -> argparse.Namespace:
    """Parse paths for fixed-prediction robustness validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument(
        "--evaluation-config", type=Path, default=DEFAULT_EVALUATION_CONFIG
    )
    parser.add_argument(
        "--validation-config", type=Path, default=DEFAULT_VALIDATION_CONFIG
    )
    return parser.parse_args()


def main() -> int:
    """Run Phase 5 without loading training data or fitted model artifacts."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    try:
        paths = load_yaml_config(args.paths_config)
        run_validation_pipeline(
            paths["outputs"]["test_predictions"],
            validation_config=load_yaml_config(args.validation_config),
            evaluation_config=load_yaml_config(args.evaluation_config),
            output_paths=paths["outputs"],
        )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValidationError, ValueError) as exc:
        LOGGER.error("Robustness validation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
