"""Run configuration-driven Phase 2 baselines on the real training split only."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Final

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from src.features import split_features_target
from src.models import ModelConfigurationError, build_baseline_pipeline
from src.utils.io import load_yaml_config
from src.utils.metrics import build_classification_scoring

LOGGER = logging.getLogger(__name__)

DEFAULT_PATHS_CONFIG: Final[Path] = Path("config/paths.yaml")
DEFAULT_MODEL_CONFIG: Final[Path] = Path("config/model_config.yaml")


class BaselineConfigurationError(ValueError):
    """Raised when the Phase 2 experiment configuration is invalid."""


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise BaselineConfigurationError(f"{key} must be a YAML mapping")
    return value


def resolve_baseline_settings(model_config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize settings needed by the baseline runner."""
    cv_config = _require_mapping(model_config, "cross_validation")
    preprocessing_config = _require_mapping(model_config, "preprocessing")
    split_config = _require_mapping(model_config, "split")
    baseline_config = _require_mapping(model_config, "baseline")
    metrics_config = _require_mapping(baseline_config, "metrics")
    algorithms = _require_mapping(baseline_config, "algorithms")

    if cv_config.get("method") != "repeated_stratified_kfold":
        raise BaselineConfigurationError(
            "Phase 2 requires cross_validation.method=repeated_stratified_kfold"
        )
    if cv_config.get("shuffle") is not True:
        raise BaselineConfigurationError(
            "Repeated stratified folds must be shuffled reproducibly"
        )
    n_splits = int(cv_config.get("n_splits", 0))
    n_repeats = int(cv_config.get("n_repeats", 0))
    if n_splits < 2 or n_repeats < 1:
        raise BaselineConfigurationError(
            "cross_validation requires n_splits >= 2 and n_repeats >= 1"
        )
    if baseline_config.get("training_condition") != "real_only":
        raise BaselineConfigurationError(
            "Phase 2 baseline training_condition must be real_only"
        )

    feature_sets = _require_mapping(model_config, "feature_sets")
    if not feature_sets:
        raise BaselineConfigurationError("feature_sets must be a non-empty YAML mapping")
    primary_metric = metrics_config.get("primary")
    secondary_metrics = metrics_config.get("secondary")
    if not isinstance(primary_metric, str) or not isinstance(secondary_metrics, list):
        raise BaselineConfigurationError(
            "baseline.metrics requires one primary and a secondary metric list"
        )
    metric_names = [primary_metric, *secondary_metrics]
    scoring = build_classification_scoring(metric_names)

    return {
        "random_state": int(model_config["random_state"]),
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "feature_set_names": list(feature_sets.keys()),
        "feature_sets_config": feature_sets,
        "target_column": str(split_config["target_column"]),
        "subject_id_column": str(split_config["subject_id_column"]),
        "preprocessing": preprocessing_config,
        "algorithms": algorithms,
        "metric_names": metric_names,
        "primary_metric": primary_metric,
        "scoring": scoring,
    }


def load_real_training_data(path: str | Path) -> pd.DataFrame:
    """Load only the persistent real training partition."""
    train_path = Path(path)
    if not train_path.is_file():
        raise FileNotFoundError(f"Real training partition not found: {train_path}")
    frame = pd.read_csv(train_path)
    LOGGER.info("Loaded %d real training subjects", len(frame))
    return frame


def create_cv_splits(
    target: pd.Series,
    *,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> list[tuple[Any, Any]]:
    """Materialize one shared set of repeated stratified train/validation folds."""
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    placeholder = pd.DataFrame(index=target.index)
    return list(splitter.split(placeholder, target))


def evaluate_baselines(
    train_frame: pd.DataFrame, model_config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all configured baselines using identical training-only CV folds."""
    settings = resolve_baseline_settings(model_config)
    reference_target: pd.Series | None = None
    shared_splits: list[tuple[Any, Any]] | None = None
    records: list[dict[str, Any]] = []

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
            raise BaselineConfigurationError(
                "Feature settings must use identical subjects and target order"
            )

        assert shared_splits is not None
        for algorithm, algorithm_config in settings["algorithms"].items():
            if not isinstance(algorithm_config, dict):
                raise BaselineConfigurationError(
                    f"baseline.algorithms.{algorithm} must be a YAML mapping"
                )
            pipeline = build_baseline_pipeline(
                algorithm,
                feature_set_name,
                preprocessing_config=settings["preprocessing"],
                feature_sets_config=settings["feature_sets_config"],
                algorithm_config=algorithm_config,
                random_state=settings["random_state"],
            )
            scores = cross_validate(
                pipeline,
                features,
                target,
                cv=shared_splits,
                scoring=settings["scoring"],
                return_train_score=False,
                error_score="raise",
            )
            for split_index in range(len(shared_splits)):
                record: dict[str, Any] = {
                    "algorithm": algorithm,
                    "feature_set": feature_set_name,
                    "training_condition": "real_only",
                    "repeat": split_index // settings["n_splits"] + 1,
                    "fold": split_index % settings["n_splits"] + 1,
                }
                for metric in settings["metric_names"]:
                    record[metric] = float(scores[f"test_{metric}"][split_index])
                records.append(record)

    fold_metrics = pd.DataFrame.from_records(records)
    grouping = ["algorithm", "feature_set", "training_condition"]
    summary = (
        fold_metrics.groupby(grouping, sort=False)[settings["metric_names"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    summary["primary_metric"] = settings["primary_metric"]
    return fold_metrics, summary


def save_baseline_outputs(
    fold_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    fold_metrics_path: str | Path,
    summary_path: str | Path,
) -> dict[str, Path]:
    """Persist machine-readable Phase 2 results."""
    paths = {
        "fold_metrics": Path(fold_metrics_path),
        "summary": Path(summary_path),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fold_metrics.to_csv(paths["fold_metrics"], index=False)
        summary.to_csv(paths["summary"], index=False)
    except OSError as exc:
        raise OSError(f"Could not save Phase 2 baseline outputs: {exc}") from exc
    return paths


def run_baseline_pipeline(
    train_path: str | Path,
    fold_metrics_path: str | Path,
    summary_path: str | Path,
    *,
    model_config: dict[str, Any],
) -> dict[str, Path]:
    """Load real training data, cross-validate baselines, and save results."""
    train_frame = load_real_training_data(train_path)
    fold_metrics, summary = evaluate_baselines(train_frame, model_config)
    paths = save_baseline_outputs(
        fold_metrics,
        summary,
        fold_metrics_path=fold_metrics_path,
        summary_path=summary_path,
    )
    LOGGER.info("Saved %d fold-level baseline results", len(fold_metrics))
    return paths


def parse_args() -> argparse.Namespace:
    """Parse config locations for the Phase 2 command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    return parser.parse_args()


def main() -> int:
    """Run Phase 2 using paths and experiment settings from YAML."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    try:
        paths_config = load_yaml_config(args.paths_config)
        model_config = load_yaml_config(args.model_config)
        output_paths = paths_config["outputs"]
        run_baseline_pipeline(
            paths_config["data"]["train"],
            output_paths["baseline_fold_metrics"],
            output_paths["baseline_summary"],
            model_config=model_config,
        )
    except (
        BaselineConfigurationError,
        FileNotFoundError,
        KeyError,
        ModelConfigurationError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        LOGGER.error("Baseline modeling failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
