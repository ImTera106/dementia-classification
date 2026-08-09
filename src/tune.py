"""Tune Phase 3 candidates using real training data and fold-safe pipelines."""

from __future__ import annotations

import argparse
import json
import logging
from time import perf_counter
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler

from src.baseline import create_cv_splits, load_real_training_data
from src.features import split_features_target
from src.models import ModelConfigurationError, build_tuning_pipeline
from src.synthesize import SynthesisError, generate_synthetic_subjects
from src.utils.io import load_yaml_config, require_package_version
from src.utils.metrics import build_classification_scoring, calculate_classification_metrics

LOGGER = logging.getLogger(__name__)

DEFAULT_PATHS_CONFIG: Final[Path] = Path("config/paths.yaml")
DEFAULT_MODEL_CONFIG: Final[Path] = Path("config/model_config.yaml")
DEFAULT_TUNING_CONFIG: Final[Path] = Path("config/tuning_config.yaml")
REQUIRED_ALGORITHMS: Final[frozenset[str]] = frozenset(
    {
        "logistic_regression",
        "svm",
        "decision_tree",
        "random_forest",
        "xgboost",
    }
)


class TuningConfigurationError(ValueError):
    """Raised when Phase 3 configuration is incomplete or unsupported."""


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise TuningConfigurationError(f"{key} must be a YAML mapping")
    return value


def resolve_tuning_settings(
    model_config: dict[str, Any], tuning_config: dict[str, Any]
) -> dict[str, Any]:
    """Validate and normalize all settings required for Phase 3."""
    preprocessing = _require_mapping(model_config, "preprocessing")
    feature_sets = _require_mapping(model_config, "feature_sets")
    split_config = _require_mapping(model_config, "split")
    cv_config = _require_mapping(tuning_config, "cross_validation")
    search_config = _require_mapping(tuning_config, "search")
    algorithms = _require_mapping(tuning_config, "algorithms")

    training_condition = tuning_config.get("training_condition")
    if training_condition not in {"real_only", "real_plus_synthetic"}:
        raise TuningConfigurationError(
            "training_condition must be real_only or real_plus_synthetic"
        )
    if cv_config.get("method") != "repeated_stratified_kfold":
        raise TuningConfigurationError(
            "Phase 3 requires repeated_stratified_kfold"
        )
    if cv_config.get("shuffle") is not True:
        raise TuningConfigurationError("Phase 3 folds must set shuffle: true")
    n_splits = int(cv_config.get("n_splits", 0))
    n_repeats = int(cv_config.get("n_repeats", 0))
    if n_splits < 2 or n_repeats < 1:
        raise TuningConfigurationError(
            "Tuning CV requires n_splits >= 2 and n_repeats >= 1"
        )
    if search_config.get("method") != "randomized_search":
        raise TuningConfigurationError("Phase 3 supports randomized_search only")
    if search_config.get("return_train_score") is not False:
        raise TuningConfigurationError("return_train_score must remain false")

    configured_algorithms = set(algorithms)
    missing = sorted(REQUIRED_ALGORITHMS.difference(configured_algorithms))
    extra = sorted(configured_algorithms.difference(REQUIRED_ALGORITHMS))
    if missing or extra:
        raise TuningConfigurationError(
            f"Tuning algorithms mismatch; missing={missing}, extra={extra}"
        )
    for algorithm, specification in algorithms.items():
        if not isinstance(specification, dict):
            raise TuningConfigurationError(
                f"algorithms.{algorithm} must be a YAML mapping"
            )
        estimator = specification.get("estimator")
        parameters = specification.get("parameters")
        if not isinstance(estimator, dict) or not isinstance(parameters, dict):
            raise TuningConfigurationError(
                f"algorithms.{algorithm} requires estimator and parameters mappings"
            )
        if int(specification.get("n_iter", 0)) < 1:
            raise TuningConfigurationError(
                f"algorithms.{algorithm}.n_iter must be positive"
            )
        if not isinstance(specification.get("scale_numeric"), bool):
            raise TuningConfigurationError(
                f"algorithms.{algorithm}.scale_numeric must be boolean"
            )
        if algorithm == "xgboost":
            required_version = specification.get("required_version")
            if not isinstance(required_version, str) or not required_version:
                raise TuningConfigurationError(
                    "algorithms.xgboost.required_version must be a version string"
                )
            require_package_version(
                "xgboost", required_version, context="Phase 3 tuning"
            )
        invalid_keys = sorted(
            key for key in parameters if not key.startswith("model__")
        )
        invalid_values = sorted(
            key
            for key, values in parameters.items()
            if not isinstance(values, list) or not values
        )
        if invalid_keys or invalid_values:
            raise TuningConfigurationError(
                f"algorithms.{algorithm} has invalid parameter keys={invalid_keys} "
                f"or values={invalid_values}"
            )

    primary_metric = search_config.get("primary_metric")
    secondary_metrics = search_config.get("secondary_metrics")
    if not isinstance(primary_metric, str) or not isinstance(
        secondary_metrics, list
    ):
        raise TuningConfigurationError(
            "search requires primary_metric and secondary_metrics"
        )
    metric_names = [primary_metric, *secondary_metrics]
    scoring = build_classification_scoring(metric_names)

    return {
        "random_state": int(tuning_config["random_state"]),
        "training_condition": training_condition,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "search_n_jobs": int(search_config["n_jobs"]),
        "primary_metric": primary_metric,
        "metric_names": metric_names,
        "scoring": scoring,
        "preprocessing": preprocessing,
        "feature_sets_config": feature_sets,
        "feature_set_names": list(feature_sets),
        "target_column": str(split_config["target_column"]),
        "subject_id_column": str(split_config["subject_id_column"]),
        "algorithms": algorithms,
    }


def build_randomized_search(
    pipeline: Any,
    algorithm_config: dict[str, Any],
    *,
    scoring: dict[str, Any],
    cv_splits: list[tuple[Any, Any]],
    random_state: int,
    n_jobs: int,
) -> RandomizedSearchCV:
    """Build a non-refitting search so final training remains in Phase 4."""
    parameters = algorithm_config["parameters"]
    unknown = sorted(set(parameters).difference(pipeline.get_params(deep=True)))
    if unknown:
        raise TuningConfigurationError(
            f"Search parameters are not valid pipeline parameters: {unknown}"
        )
    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameters,
        n_iter=int(algorithm_config["n_iter"]),
        scoring=scoring,
        refit=False,
        cv=cv_splits,
        random_state=random_state,
        n_jobs=n_jobs,
        return_train_score=False,
        error_score="raise",
    )


def _json_ready(value: Any) -> Any:
    """Convert numpy and nested search values to JSON-compatible objects."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _extract_search_results(
    search: RandomizedSearchCV,
    *,
    algorithm: str,
    feature_set: str,
    metric_names: list[str],
    primary_metric: str,
    cv_fold_count: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Normalize candidate results and select one configuration without refitting."""
    raw = pd.DataFrame(search.cv_results_)
    ranking = raw.sort_values(
        [f"mean_test_{primary_metric}", f"std_test_{primary_metric}"],
        ascending=[False, True],
        kind="stable",
    )
    best_index = int(ranking.index[0])
    best_row = raw.loc[best_index]
    best_params = _json_ready(best_row["params"])

    result_columns = [
        column
        for column in raw.columns
        if column.startswith(
            ("param_", "split", "mean_test_", "std_test_", "rank_test_")
        )
        or column in {"mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time"}
    ]
    results = raw.loc[:, result_columns].copy()
    results.insert(0, "candidate_index", range(len(results)))
    results.insert(0, "training_condition", "real_only")
    results.insert(0, "feature_set", feature_set)
    results.insert(0, "algorithm", algorithm)
    results["params_json"] = raw["params"].map(
        lambda value: json.dumps(_json_ready(value), sort_keys=True)
    )

    summary: dict[str, Any] = {
        "algorithm": algorithm,
        "feature_set": feature_set,
        "training_condition": "real_only",
        "search_candidates": int(len(raw)),
        "cv_fold_count": cv_fold_count,
        "best_candidate_index": best_index,
        "primary_metric": primary_metric,
    }
    for metric in metric_names:
        summary[f"{metric}_mean"] = float(best_row[f"mean_test_{metric}"])
        summary[f"{metric}_std"] = float(best_row[f"std_test_{metric}"])

    parameter_record = {
        "algorithm": algorithm,
        "feature_set": feature_set,
        "training_condition": "real_only",
        "best_candidate_index": best_index,
        "parameters": best_params,
        "selection": {
            "metric": primary_metric,
            "mean": summary[f"{primary_metric}_mean"],
            "std": summary[f"{primary_metric}_std"],
        },
    }
    return results, summary, parameter_record


def tune_candidates(
    train_frame: pd.DataFrame,
    model_config: dict[str, Any],
    tuning_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Tune all algorithms and feature sets using identical training-only folds."""
    settings = resolve_tuning_settings(model_config, tuning_config)
    if settings["training_condition"] != "real_only":
        raise TuningConfigurationError(
            "Use tune_augmented_candidates for real_plus_synthetic tuning"
        )
    reference_target: pd.Series | None = None
    shared_splits: list[tuple[Any, Any]] | None = None
    result_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    parameter_records: list[dict[str, Any]] = []

    for feature_set_name in settings["feature_set_names"]:
        features, target = split_features_target(
            train_frame,
            feature_set_name,
            settings["feature_sets_config"],
            target_column=settings["target_column"],
            subject_id_column=settings["subject_id_column"],
        )
        if reference_target is None:
            reference_target = target
            shared_splits = create_cv_splits(
                target,
                n_splits=settings["n_splits"],
                n_repeats=settings["n_repeats"],
                random_state=settings["random_state"],
            )
        elif not target.equals(reference_target):
            raise TuningConfigurationError(
                "Feature settings must use identical subjects and target order"
            )

        assert shared_splits is not None
        for algorithm, algorithm_config in settings["algorithms"].items():
            LOGGER.info("Tuning %s with %s", algorithm, feature_set_name)
            pipeline = build_tuning_pipeline(
                algorithm,
                feature_set_name,
                preprocessing_config=settings["preprocessing"],
                feature_sets_config=settings["feature_sets_config"],
                estimator_config=algorithm_config["estimator"],
                scale_numeric=algorithm_config["scale_numeric"],
                random_state=settings["random_state"],
            )
            search = build_randomized_search(
                pipeline,
                algorithm_config,
                scoring=settings["scoring"],
                cv_splits=shared_splits,
                random_state=settings["random_state"],
                n_jobs=settings["search_n_jobs"],
            )
            search.fit(features, target)
            results, summary, parameters = _extract_search_results(
                search,
                algorithm=algorithm,
                feature_set=feature_set_name,
                metric_names=settings["metric_names"],
                primary_metric=settings["primary_metric"],
                cv_fold_count=len(shared_splits),
            )
            result_frames.append(results)
            summaries.append(summary)
            parameter_records.append(parameters)

    all_results = pd.concat(result_frames, ignore_index=True)
    summary_frame = pd.DataFrame.from_records(summaries)
    best_parameters = {
        "training_condition": "real_only",
        "primary_metric": settings["primary_metric"],
        "random_state": settings["random_state"],
        "cross_validation": {
            "n_splits": settings["n_splits"],
            "n_repeats": settings["n_repeats"],
            "fold_count": settings["n_splits"] * settings["n_repeats"],
        },
        "final_models_fitted": False,
        "experiments": parameter_records,
    }
    return all_results, summary_frame, best_parameters


def _continuous_score(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return the positive-class score used for fold ROC AUC."""
    classes = np.asarray(model.classes_)
    if callable(getattr(model, "predict_proba", None)):
        probabilities = np.asarray(model.predict_proba(features))
        return probabilities[:, int(np.flatnonzero(classes == 1)[0])]
    return np.asarray(model.decision_function(features), dtype=float)


def _fold_augmented_data(
    train_frame: pd.DataFrame,
    *,
    feature_set_name: str,
    feature_sets_config: dict[str, Any],
    synthesis_config: dict[str, Any],
    target_column: str,
    subject_id_column: str,
    cv_splits: list[tuple[Any, Any]],
) -> list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    """Create fold-local augmented training data with entirely real validation rows."""
    columns = [
        *split_features_target(
            train_frame,
            feature_set_name,
            feature_sets_config,
            target_column=target_column,
            subject_id_column=subject_id_column,
        )[0].columns,
        target_column,
    ]
    folds = []
    for fold_index, (train_indices, validation_indices) in enumerate(cv_splits):
        real_fold_train = train_frame.iloc[train_indices].copy()
        real_fold_validation = train_frame.iloc[validation_indices].copy()
        synthetic, _ = generate_synthetic_subjects(
            real_fold_train,
            feature_set_name,
            feature_sets_config,
            synthesis_config,
            target_column=target_column,
            subject_id_column=subject_id_column,
            id_prefix=f"CV{fold_index + 1:02d}",
            evaluate_reports=False,
        )
        augmented = pd.concat(
            [real_fold_train.loc[:, columns], synthetic.loc[:, columns]],
            ignore_index=True,
        )
        folds.append(
            (
                augmented.drop(columns=target_column),
                augmented[target_column].astype(int),
                real_fold_validation.loc[:, columns].drop(columns=target_column),
                real_fold_validation[target_column].astype(int),
            )
        )
    return folds


def tune_augmented_candidates(
    train_frame: pd.DataFrame,
    model_config: dict[str, Any],
    tuning_config: dict[str, Any],
    synthesis_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Tune with fold-local synthesis and score exclusively on real validation rows."""
    settings = resolve_tuning_settings(model_config, tuning_config)
    if settings["training_condition"] != "real_plus_synthetic":
        raise TuningConfigurationError(
            "Augmented tuning requires training_condition real_plus_synthetic"
        )
    _, reference_target = split_features_target(
        train_frame,
        settings["feature_set_names"][0],
        settings["feature_sets_config"],
        target_column=settings["target_column"],
        subject_id_column=settings["subject_id_column"],
    )
    shared_splits = create_cv_splits(
        reference_target,
        n_splits=settings["n_splits"],
        n_repeats=settings["n_repeats"],
        random_state=settings["random_state"],
    )
    result_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    parameter_records: list[dict[str, Any]] = []

    for feature_set_name in settings["feature_set_names"]:
        LOGGER.info("Generating fold-local synthetic data for %s", feature_set_name)
        folds = _fold_augmented_data(
            train_frame,
            feature_set_name=feature_set_name,
            feature_sets_config=settings["feature_sets_config"],
            synthesis_config=synthesis_config,
            target_column=settings["target_column"],
            subject_id_column=settings["subject_id_column"],
            cv_splits=shared_splits,
        )
        for algorithm, algorithm_config in settings["algorithms"].items():
            LOGGER.info("Tuning augmented %s with %s", algorithm, feature_set_name)
            pipeline = build_tuning_pipeline(
                algorithm,
                feature_set_name,
                preprocessing_config=settings["preprocessing"],
                feature_sets_config=settings["feature_sets_config"],
                estimator_config=algorithm_config["estimator"],
                scale_numeric=algorithm_config["scale_numeric"],
                random_state=settings["random_state"],
            )
            candidates = list(
                ParameterSampler(
                    algorithm_config["parameters"],
                    n_iter=int(algorithm_config["n_iter"]),
                    random_state=settings["random_state"],
                )
            )
            algorithm_rows: list[dict[str, Any]] = []
            for candidate_index, parameters in enumerate(candidates):
                fold_metrics = {metric: [] for metric in settings["metric_names"]}
                fit_times: list[float] = []
                score_times: list[float] = []
                for augmented_x, augmented_y, validation_x, validation_y in folds:
                    candidate = clone(pipeline).set_params(**parameters)
                    started = perf_counter()
                    candidate.fit(augmented_x, augmented_y)
                    fit_times.append(perf_counter() - started)
                    started = perf_counter()
                    predictions = candidate.predict(validation_x)
                    scores = _continuous_score(candidate, validation_x)
                    metrics = calculate_classification_metrics(
                        validation_y, predictions, scores
                    )
                    score_times.append(perf_counter() - started)
                    for metric in settings["metric_names"]:
                        fold_metrics[metric].append(float(metrics[metric]))
                row: dict[str, Any] = {
                    "algorithm": algorithm,
                    "feature_set": feature_set_name,
                    "training_condition": "real_plus_synthetic",
                    "candidate_index": candidate_index,
                    "params_json": json.dumps(_json_ready(parameters), sort_keys=True),
                    "mean_fit_time": float(np.mean(fit_times)),
                    "std_fit_time": float(np.std(fit_times)),
                    "mean_score_time": float(np.mean(score_times)),
                    "std_score_time": float(np.std(score_times)),
                }
                for metric, values in fold_metrics.items():
                    row[f"mean_test_{metric}"] = float(np.mean(values))
                    row[f"std_test_{metric}"] = float(np.std(values))
                algorithm_rows.append(row)
                result_records.append(row)
            ranking = sorted(
                algorithm_rows,
                key=lambda row: (
                    -row[f"mean_test_{settings['primary_metric']}"],
                    row[f"std_test_{settings['primary_metric']}"],
                ),
            )
            best = ranking[0]
            for rank, row in enumerate(ranking, start=1):
                row[f"rank_test_{settings['primary_metric']}"] = rank
            summary = {
                "algorithm": algorithm,
                "feature_set": feature_set_name,
                "training_condition": "real_plus_synthetic",
                "search_candidates": len(candidates),
                "cv_fold_count": len(folds),
                "best_candidate_index": best["candidate_index"],
                "primary_metric": settings["primary_metric"],
            }
            for metric in settings["metric_names"]:
                summary[f"{metric}_mean"] = best[f"mean_test_{metric}"]
                summary[f"{metric}_std"] = best[f"std_test_{metric}"]
            summaries.append(summary)
            parameter_records.append(
                {
                    "algorithm": algorithm,
                    "feature_set": feature_set_name,
                    "training_condition": "real_plus_synthetic",
                    "best_candidate_index": best["candidate_index"],
                    "parameters": json.loads(best["params_json"]),
                    "selection": {
                        "metric": settings["primary_metric"],
                        "mean": best[f"mean_test_{settings['primary_metric']}"],
                        "std": best[f"std_test_{settings['primary_metric']}"],
                    },
                }
            )
    best_parameters = {
        "training_condition": "real_plus_synthetic",
        "primary_metric": settings["primary_metric"],
        "random_state": settings["random_state"],
        "cross_validation": {
            "n_splits": settings["n_splits"],
            "n_repeats": settings["n_repeats"],
            "fold_count": len(shared_splits),
            "validation_data": "real_only",
            "synthesis_fit_scope": "fold_training_only",
        },
        "final_models_fitted": False,
        "experiments": parameter_records,
    }
    return (
        pd.DataFrame.from_records(result_records),
        pd.DataFrame.from_records(summaries),
        best_parameters,
    )


def save_tuning_outputs(
    cv_results: pd.DataFrame,
    summary: pd.DataFrame,
    best_parameters: dict[str, Any],
    *,
    cv_results_path: str | Path,
    summary_path: str | Path,
    best_parameters_path: str | Path,
) -> dict[str, Path]:
    """Save Phase 3 results without serializing fitted search objects or models."""
    paths = {
        "cv_results": Path(cv_results_path),
        "summary": Path(summary_path),
        "best_parameters": Path(best_parameters_path),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cv_results.to_csv(paths["cv_results"], index=False)
        summary.to_csv(paths["summary"], index=False)
        paths["best_parameters"].write_text(
            json.dumps(best_parameters, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        raise OSError(f"Could not save Phase 3 tuning outputs: {exc}") from exc
    return paths


def run_tuning_pipeline(
    train_path: str | Path,
    *,
    model_config: dict[str, Any],
    tuning_config: dict[str, Any],
    cv_results_path: str | Path,
    summary_path: str | Path,
    best_parameters_path: str | Path,
) -> dict[str, Path]:
    """Load real training data, tune candidates, and save selection artifacts."""
    train_frame = load_real_training_data(train_path)
    cv_results, summary, best_parameters = tune_candidates(
        train_frame, model_config, tuning_config
    )
    paths = save_tuning_outputs(
        cv_results,
        summary,
        best_parameters,
        cv_results_path=cv_results_path,
        summary_path=summary_path,
        best_parameters_path=best_parameters_path,
    )
    LOGGER.info("Saved tuning results for %d experiments", len(summary))
    return paths


def parse_args() -> argparse.Namespace:
    """Parse YAML configuration paths for Phase 3."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--tuning-config", type=Path, default=DEFAULT_TUNING_CONFIG)
    return parser.parse_args()


def main() -> int:
    """Run Phase 3 tuning from YAML without accessing held-out test data."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    try:
        paths_config = load_yaml_config(args.paths_config)
        model_config = load_yaml_config(args.model_config)
        tuning_config = load_yaml_config(args.tuning_config)
        output_paths = paths_config["outputs"]
        run_tuning_pipeline(
            paths_config["data"]["train"],
            model_config=model_config,
            tuning_config=tuning_config,
            cv_results_path=output_paths["tuning_cv_results"],
            summary_path=output_paths["tuning_summary"],
            best_parameters_path=output_paths["tuning_best_parameters"],
        )
    except (
        FileNotFoundError,
        KeyError,
        ModelConfigurationError,
        OSError,
        TuningConfigurationError,
        TypeError,
        ValueError,
    ) as exc:
        LOGGER.error("Tuning failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
