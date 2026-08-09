"""Fit and serialize final Phase 4 pipelines using real training data only."""

from __future__ import annotations

import argparse
import logging
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import pandas as pd

from src.baseline import load_real_training_data
from src.features import split_features_target
from src.models import ModelConfigurationError, build_model_pipeline
from src.synthesize import generate_synthetic_subjects, modeling_columns
from src.tune import REQUIRED_ALGORITHMS
from src.utils.io import (
    load_json,
    load_yaml_config,
    require_package_version,
    save_joblib,
    save_json,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_PATHS_CONFIG: Final[Path] = Path("config/paths.yaml")
DEFAULT_MODEL_CONFIG: Final[Path] = Path("config/model_config.yaml")
DEFAULT_TUNING_CONFIG: Final[Path] = Path("config/tuning_config.yaml")


class TrainingConfigurationError(ValueError):
    """Raised when selected parameters cannot define all final experiments."""


def _selected_parameter_map(
    best_parameters: dict[str, Any],
    *,
    feature_sets: dict[str, Any],
    training_condition: str = "real_only",
) -> dict[tuple[str, str], dict[str, Any]]:
    """Validate and index exactly one selected record per final experiment."""
    if best_parameters.get("final_models_fitted") is not False:
        raise TrainingConfigurationError(
            "Phase 3 parameter artifact must record final_models_fitted: false"
        )
    records = best_parameters.get("experiments")
    if not isinstance(records, list):
        raise TrainingConfigurationError("best_parameters.experiments must be a list")

    expected = {
        (algorithm, feature_set)
        for feature_set in feature_sets
        for algorithm in REQUIRED_ALGORITHMS
    }
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise TrainingConfigurationError(
                "Each selected-parameter record must be a mapping"
            )
        key = (str(record.get("algorithm")), str(record.get("feature_set")))
        if key in indexed:
            raise TrainingConfigurationError(f"Duplicate selected parameters: {key}")
        parameters = record.get("parameters")
        if not isinstance(parameters, dict):
            raise TrainingConfigurationError(f"Selected parameters missing for {key}")
        if record.get("training_condition") != training_condition:
            raise TrainingConfigurationError(f"Unsupported training condition for {key}")
        indexed[key] = parameters

    missing = sorted(expected.difference(indexed))
    extra = sorted(set(indexed).difference(expected))
    if missing or extra:
        raise TrainingConfigurationError(
            f"Selected experiment mismatch; missing={missing}, extra={extra}"
        )
    return indexed


def train_final_models(
    train_frame: pd.DataFrame,
    model_config: dict[str, Any],
    tuning_config: dict[str, Any],
    best_parameters: dict[str, Any],
    *,
    models_dir: str | Path,
) -> dict[str, Any]:
    """Fit all selected pipelines on real training subjects and save them."""
    if tuning_config.get("training_condition") != "real_only":
        raise TrainingConfigurationError("Final training condition must be real_only")
    feature_sets = model_config.get("feature_sets")
    preprocessing = model_config.get("preprocessing")
    split_config = model_config.get("split")
    algorithms = tuning_config.get("algorithms")
    if not all(
        isinstance(value, dict)
        for value in (feature_sets, preprocessing, split_config, algorithms)
    ):
        raise TrainingConfigurationError(
            "Model and tuning configs require feature, preprocessing, split, "
            "and algorithm mappings"
        )
    selected = _selected_parameter_map(
        best_parameters, feature_sets=feature_sets, training_condition="real_only"
    )
    if set(algorithms) != set(REQUIRED_ALGORITHMS):
        raise TrainingConfigurationError("Tuning config must define all algorithms")
    required_xgboost = algorithms["xgboost"].get("required_version")
    if not isinstance(required_xgboost, str) or not required_xgboost:
        raise TrainingConfigurationError(
            "algorithms.xgboost.required_version must be a version string"
        )
    require_package_version(
        "xgboost", required_xgboost, context="Final model training"
    )

    output_root = Path(models_dir) / "real_only"
    model_records: list[dict[str, Any]] = []
    for feature_set_name in feature_sets:
        features, target = split_features_target(
            train_frame,
            feature_set_name,
            feature_sets,
            target_column=str(split_config["target_column"]),
            subject_id_column=str(split_config["subject_id_column"]),
        )
        for algorithm in algorithms:
            algorithm_config = algorithms[algorithm]
            LOGGER.info("Fitting final %s with %s", algorithm, feature_set_name)
            pipeline = build_model_pipeline(
                algorithm,
                feature_set_name,
                preprocessing_config=preprocessing,
                feature_sets_config=feature_sets,
                estimator_config=algorithm_config["estimator"],
                scale_numeric=algorithm_config["scale_numeric"],
                random_state=int(tuning_config["random_state"]),
            )
            parameters = selected[(algorithm, feature_set_name)]
            unknown = sorted(
                set(parameters).difference(pipeline.get_params(deep=True))
            )
            if unknown:
                raise TrainingConfigurationError(
                    f"Invalid selected parameters for {(algorithm, feature_set_name)}: "
                    f"{unknown}"
                )
            pipeline.set_params(**parameters)
            pipeline.fit(features, target)
            model_path = output_root / feature_set_name / f"{algorithm}.joblib"
            save_joblib(pipeline, model_path)
            model_records.append(
                {
                    "algorithm": algorithm,
                    "feature_set": feature_set_name,
                    "training_condition": "real_only",
                    "path": str(model_path),
                    "feature_columns": list(features.columns),
                    "selected_parameters": parameters,
                }
            )

    return {
        "schema_version": 1,
        "training_condition": "real_only",
        "trained_subjects": int(len(train_frame)),
        "target_counts": {
            str(label): int(count)
            for label, count in train_frame[str(split_config["target_column"])]
            .value_counts()
            .sort_index()
            .items()
        },
        "versions": {
            "python": platform.python_version(),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
            "xgboost": version("xgboost"),
            "joblib": version("joblib"),
        },
        "models": model_records,
    }


def train_augmented_models(
    real_train_frame: pd.DataFrame,
    model_config: dict[str, Any],
    tuning_config: dict[str, Any],
    synthesis_config: dict[str, Any],
    best_parameters: dict[str, Any],
    *,
    models_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    """Generate full-training synthetic cohorts and fit all augmented pipelines."""
    if tuning_config.get("training_condition") != "real_plus_synthetic":
        raise TrainingConfigurationError(
            "Augmented final training requires real_plus_synthetic"
        )
    feature_sets = model_config.get("feature_sets")
    preprocessing = model_config.get("preprocessing")
    split_config = model_config.get("split")
    algorithms = tuning_config.get("algorithms")
    if not all(
        isinstance(value, dict)
        for value in (feature_sets, preprocessing, split_config, algorithms)
    ):
        raise TrainingConfigurationError("Augmented training configuration is incomplete")
    selected = _selected_parameter_map(
        best_parameters,
        feature_sets=feature_sets,
        training_condition="real_plus_synthetic",
    )
    required_xgboost = algorithms["xgboost"].get("required_version")
    require_package_version(
        "xgboost", required_xgboost, context="Augmented final model training"
    )
    target_column = str(split_config["target_column"])
    subject_id_column = str(split_config["subject_id_column"])
    output_root = Path(models_dir) / "real_plus_synthetic"
    model_records: list[dict[str, Any]] = []
    synthetic_frames: dict[str, pd.DataFrame] = {}
    synthesis_reports: dict[str, dict[str, Any]] = {}

    for feature_set_name in feature_sets:
        synthetic, report = generate_synthetic_subjects(
            real_train_frame,
            feature_set_name,
            feature_sets,
            synthesis_config,
            target_column=target_column,
            subject_id_column=subject_id_column,
        )
        synthetic_frames[feature_set_name] = synthetic
        synthesis_reports[feature_set_name] = report
        columns = modeling_columns(
            feature_set_name, feature_sets, target_column=target_column
        )
        combined = pd.concat(
            [real_train_frame.loc[:, columns], synthetic.loc[:, columns]],
            ignore_index=True,
        )
        features = combined.drop(columns=target_column)
        target = combined[target_column].astype(int)
        for algorithm, algorithm_config in algorithms.items():
            LOGGER.info("Fitting augmented %s with %s", algorithm, feature_set_name)
            pipeline = build_model_pipeline(
                algorithm,
                feature_set_name,
                preprocessing_config=preprocessing,
                feature_sets_config=feature_sets,
                estimator_config=algorithm_config["estimator"],
                scale_numeric=algorithm_config["scale_numeric"],
                random_state=int(tuning_config["random_state"]),
            )
            parameters = selected[(algorithm, feature_set_name)]
            pipeline.set_params(**parameters)
            pipeline.fit(features, target)
            model_path = output_root / feature_set_name / f"{algorithm}.joblib"
            save_joblib(pipeline, model_path)
            model_records.append(
                {
                    "algorithm": algorithm,
                    "feature_set": feature_set_name,
                    "training_condition": "real_plus_synthetic",
                    "path": str(model_path),
                    "feature_columns": list(features.columns),
                    "selected_parameters": parameters,
                }
            )
    manifest = {
        "schema_version": 1,
        "training_condition": "real_plus_synthetic",
        "real_trained_subjects": int(len(real_train_frame)),
        "synthetic_subjects_per_feature_set": {
            key: int(len(value)) for key, value in synthetic_frames.items()
        },
        "test_data_used_for_training_or_synthesis": False,
        "versions": {
            "python": platform.python_version(),
            "pandas": version("pandas"),
            "scikit_learn": version("scikit-learn"),
            "xgboost": version("xgboost"),
            "sdv": version("sdv"),
            "joblib": version("joblib"),
        },
        "models": model_records,
    }
    return manifest, synthetic_frames, synthesis_reports


def run_training_pipeline(
    train_path: str | Path,
    *,
    model_config: dict[str, Any],
    tuning_config: dict[str, Any],
    best_parameters: dict[str, Any],
    models_dir: str | Path,
    manifest_path: str | Path,
) -> Path:
    """Load the real training partition, fit final pipelines, and save a manifest."""
    train_frame = load_real_training_data(train_path)
    manifest = train_final_models(
        train_frame,
        model_config,
        tuning_config,
        best_parameters,
        models_dir=models_dir,
    )
    output_path = save_json(manifest, manifest_path)
    LOGGER.info("Saved %d final model pipelines", len(manifest["models"]))
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse YAML configuration paths for final training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--tuning-config", type=Path, default=DEFAULT_TUNING_CONFIG)
    return parser.parse_args()


def main() -> int:
    """Fit final models without accepting or loading a held-out test path."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    try:
        paths_config = load_yaml_config(args.paths_config)
        model_config = load_yaml_config(args.model_config)
        tuning_config = load_yaml_config(args.tuning_config)
        run_training_pipeline(
            paths_config["data"]["train"],
            model_config=model_config,
            tuning_config=tuning_config,
            best_parameters=load_json(
                paths_config["outputs"]["tuning_best_parameters"]
            ),
            models_dir=paths_config["models_dir"],
            manifest_path=paths_config["model_manifest"],
        )
    except (
        FileNotFoundError,
        KeyError,
        ModelConfigurationError,
        OSError,
        TrainingConfigurationError,
        TypeError,
        ValueError,
    ) as exc:
        LOGGER.error("Final training failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
