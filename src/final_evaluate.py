"""Evaluate the complete frozen experiment set on the held-out test data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Final

import pandas as pd

from src.evaluate import (
    EvaluationError,
    _validate_evaluation_contract,
    generate_saved_predictions,
)
from src.evaluation_release import (
    create_evaluation_release,
    release_id,
)
from src.freeze_experiment import git_state, manifest_sha256
from src.utils.io import load_json, load_yaml_config, sha256_file

LOGGER = logging.getLogger(__name__)
DEFAULT_PATHS_CONFIG: Final[Path] = Path("config/paths.yaml")
DEFAULT_MODEL_CONFIG: Final[Path] = Path("config/model_config.yaml")
DEFAULT_REAL_EVALUATION_CONFIG: Final[Path] = Path("config/evaluation_config.yaml")
DEFAULT_AUGMENTED_EVALUATION_CONFIG: Final[Path] = Path(
    "config/evaluation_synthetic_config.yaml"
)


def load_real_test_data(path: str | Path) -> pd.DataFrame:
    """Open the held-out partition only inside the canonical evaluator."""
    test_path = Path(path)
    if not test_path.is_file():
        raise FileNotFoundError(f"Real test partition not found: {test_path}")
    frame = pd.read_csv(test_path)
    LOGGER.info("Loaded %d real held-out test subjects", len(frame))
    return frame


def validate_frozen_experiment_set(
    frozen_manifest: dict[str, Any],
    model_config: dict[str, Any],
    evaluation_configs: dict[str, dict[str, Any]],
    *,
    project_root: str | Path,
) -> None:
    """Verify all development fingerprints before test-data access."""
    root = Path(project_root)
    if frozen_manifest.get("manifest_sha256") != manifest_sha256(frozen_manifest):
        raise EvaluationError("Frozen manifest fingerprint mismatch")
    experiment_set = frozen_manifest.get("experiment_set")
    if not isinstance(experiment_set, dict):
        raise EvaluationError("Frozen manifest is missing experiment_set")
    expected_conditions = model_config.get("training_conditions")
    if not isinstance(expected_conditions, list) or not expected_conditions:
        raise EvaluationError("model_config.training_conditions must be a list")
    expected_algorithms = model_config.get("experiment_algorithms")
    expected_features = model_config.get("feature_sets")
    if experiment_set != {
        "algorithms": expected_algorithms,
        "feature_sets": list(expected_features),
        "training_conditions": expected_conditions,
        "expected_experiment_count": (
            len(expected_conditions) * len(expected_features) * len(expected_algorithms)
        ),
    }:
        raise EvaluationError("Frozen experiment definition does not match configuration")
    manifests = frozen_manifest.get("condition_manifests")
    if not isinstance(manifests, dict) or set(manifests) != set(expected_conditions):
        raise EvaluationError("Frozen manifests do not cover all training conditions")
    if set(evaluation_configs) != set(expected_conditions):
        raise EvaluationError("Evaluation configs do not cover all training conditions")

    frozen_git = frozen_manifest.get("git")
    current_git = git_state(root)
    if not isinstance(frozen_git, dict) or frozen_git.get("dirty") is not False:
        raise EvaluationError("Final evaluation requires a clean frozen Git state")
    if current_git != frozen_git:
        raise EvaluationError("Current Git state does not match the frozen experiment")

    training_data = frozen_manifest.get("training_data")
    if not isinstance(training_data, dict):
        raise EvaluationError("Frozen manifest is missing training_data")
    training_path = Path(training_data.get("path", ""))
    if not training_path.is_absolute():
        training_path = root / training_path
    if sha256_file(training_path) != training_data.get("sha256"):
        raise EvaluationError("Training-data fingerprint mismatch")

    configurations = frozen_manifest.get("configurations")
    if not isinstance(configurations, dict):
        raise EvaluationError("Frozen manifest is missing configurations")
    current_configuration_values: dict[str, dict[str, Any]] = {}
    for name, record in configurations.items():
        if not isinstance(record, dict):
            raise EvaluationError(f"Invalid frozen configuration record: {name}")
        config_path = Path(record.get("path", ""))
        if not config_path.is_absolute():
            config_path = root / config_path
        if sha256_file(config_path) != record.get("sha256"):
            raise EvaluationError(f"Configuration fingerprint mismatch: {name}")
        current_configuration_values[name] = load_yaml_config(config_path)
    if current_configuration_values.get("model") != model_config:
        raise EvaluationError("Loaded model configuration does not match its fingerprint")
    if current_configuration_values.get("real_evaluation") != evaluation_configs.get(
        "real_only"
    ):
        raise EvaluationError("Loaded real-only evaluation configuration is not frozen")
    if current_configuration_values.get(
        "augmented_evaluation"
    ) != evaluation_configs.get("real_plus_synthetic"):
        raise EvaluationError("Loaded augmented evaluation configuration is not frozen")

    experiment_records = frozen_manifest.get("experiments")
    if not isinstance(experiment_records, list):
        raise EvaluationError("Frozen manifest is missing experiments")
    fingerprints = {
        (record.get("training_condition"), record.get("feature_set"), record.get("algorithm")): record
        for record in experiment_records
        if isinstance(record, dict)
    }
    if len(fingerprints) != experiment_set["expected_experiment_count"]:
        raise EvaluationError("Frozen experiment fingerprints are incomplete or duplicated")
    expected_paths: set[Path] = set()
    for condition in expected_conditions:
        records = _validate_evaluation_contract(
            manifests[condition], model_config, evaluation_configs[condition]
        )
        for record in records:
            key = (condition, record["feature_set"], record["algorithm"])
            fingerprint = fingerprints.get(key)
            if fingerprint is None:
                raise EvaluationError(f"Missing frozen experiment fingerprint: {key}")
            expected_id = f"{condition}__{record['feature_set']}__{record['algorithm']}"
            if fingerprint.get("experiment_id") != expected_id:
                raise EvaluationError(f"Invalid frozen experiment ID: {key}")
            model_path = Path(record["path"])
            if not model_path.is_absolute():
                model_path = root / model_path
            if not model_path.is_file():
                raise FileNotFoundError(f"Frozen model artifact not found: {model_path}")
            if sha256_file(model_path) != fingerprint.get("model_artifact_sha256"):
                raise EvaluationError(f"Frozen model fingerprint mismatch: {key}")
            expected_paths.add(model_path)
    expected_count = (
        len(expected_conditions)
        * len(model_config["feature_sets"])
        * len(model_config["experiment_algorithms"])
    )
    if len(expected_paths) != expected_count:
        raise EvaluationError("Frozen experiment records must reference unique models")


def run_final_evaluation(
    test_path: str | Path,
    *,
    frozen_manifest: dict[str, Any],
    model_config: dict[str, Any],
    evaluation_configs: dict[str, dict[str, Any]],
    output_paths: dict[str, Any],
    project_root: str | Path = ".",
) -> dict[str, Path]:
    """Validate frozen experiments, then open the test set once and score all models."""
    validate_frozen_experiment_set(
        frozen_manifest,
        model_config,
        evaluation_configs,
        project_root=project_root,
    )
    releases_root = Path(output_paths["final_evaluation_dir"])
    if not releases_root.is_absolute():
        releases_root = Path(project_root) / releases_root
    expected_release = releases_root / release_id(frozen_manifest)
    if expected_release.exists():
        raise FileExistsError(f"Evaluation release already exists: {expected_release}")
    resolved_test_path = Path(test_path)
    if not resolved_test_path.is_absolute():
        resolved_test_path = Path(project_root) / resolved_test_path
    test_frame = load_real_test_data(resolved_test_path)
    test_split = frozen_manifest.get("test_split")
    if (
        not isinstance(test_split, dict)
        or sha256_file(resolved_test_path) != test_split.get("sha256")
    ):
        raise EvaluationError("Held-out test fingerprint mismatch")
    manifests = frozen_manifest["condition_manifests"]
    prediction_frames: list[pd.DataFrame] = []
    for condition in model_config["training_conditions"]:
        predictions = generate_saved_predictions(
            test_frame,
            manifests[condition],
            model_config,
            evaluation_configs[condition],
        )
        predictions.insert(
            0,
            "experiment_id",
            predictions["training_condition"].astype(str)
            + "__"
            + predictions["feature_set"].astype(str)
            + "__"
            + predictions["algorithm"].astype(str),
        )
        prediction_frames.append(predictions)

    return create_evaluation_release(
        pd.concat(prediction_frames, ignore_index=True),
        frozen_manifest,
        releases_root=releases_root,
        test_sha256=str(test_split["sha256"]),
    )


def parse_args() -> argparse.Namespace:
    """Parse paths for the single supported held-out evaluation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument(
        "--real-evaluation-config", type=Path, default=DEFAULT_REAL_EVALUATION_CONFIG
    )
    parser.add_argument(
        "--augmented-evaluation-config",
        type=Path,
        default=DEFAULT_AUGMENTED_EVALUATION_CONFIG,
    )
    return parser.parse_args()


def main() -> int:
    """Evaluate all frozen conditions without exposing development operations."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    try:
        paths = load_yaml_config(args.paths_config)
        root = Path(paths["project_root"])
        run_final_evaluation(
            paths["data"]["test"],
            frozen_manifest=load_json(paths["frozen_experiment_manifest"]),
            model_config=load_yaml_config(args.model_config),
            evaluation_configs={
                "real_only": load_yaml_config(args.real_evaluation_config),
                "real_plus_synthetic": load_yaml_config(
                    args.augmented_evaluation_config
                ),
            },
            output_paths=paths["outputs"],
            project_root=root,
        )
    except (
        EvaluationError,
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        LOGGER.error("Final held-out evaluation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
