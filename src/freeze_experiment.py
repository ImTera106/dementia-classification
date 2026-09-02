"""Freeze the completed development experiment without opening held-out data."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Final

from src.utils.io import load_json, load_yaml_config, save_json, sha256_file

LOGGER = logging.getLogger(__name__)
DEFAULT_PATHS_CONFIG: Final[Path] = Path("config/paths.yaml")
DEFAULT_MODEL_CONFIG: Final[Path] = Path("config/model_config.yaml")
CONFIG_PATHS: Final[dict[str, Path]] = {
    "paths": Path("config/paths.yaml"),
    "model": Path("config/model_config.yaml"),
    "real_tuning": Path("config/tuning_config.yaml"),
    "augmented_tuning": Path("config/tuning_synthetic_config.yaml"),
    "synthesis": Path("config/synthetic_config.yaml"),
    "real_evaluation": Path("config/evaluation_config.yaml"),
    "augmented_evaluation": Path("config/evaluation_synthetic_config.yaml"),
    "validation": Path("config/validation_config.yaml"),
}


class FreezeError(ValueError):
    """Raised when development artifacts cannot be frozen safely."""


def manifest_sha256(manifest: dict[str, Any]) -> str:
    """Fingerprint a manifest using canonical JSON while excluding its digest."""
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def git_state(project_root: str | Path) -> dict[str, Any]:
    """Return the current commit and a fingerprint of any tracked-file changes."""
    root = Path(project_root)

    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments], cwd=root, check=False, capture_output=True, text=True
        )
        if result.returncode != 0:
            raise FreezeError(f"Git command failed: {' '.join(arguments)}")
        return result.stdout

    commit = run("rev-parse", "HEAD").strip()
    status = run("status", "--porcelain", "--untracked-files=all")
    tracked_diff = run("diff", "--binary", "HEAD")
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(
            tracked_diff.encode("utf-8")
        ).hexdigest(),
    }


def _expected_keys(model_config: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (condition, feature_set, algorithm)
        for condition in model_config["training_conditions"]
        for feature_set in model_config["feature_sets"]
        for algorithm in model_config["experiment_algorithms"]
    }


def build_frozen_manifest(
    *,
    condition_manifests: dict[str, dict[str, Any]],
    model_config: dict[str, Any],
    train_path: str | Path,
    split_summary: dict[str, Any],
    configuration_paths: dict[str, str | Path],
    project_root: str | Path,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    """Build a complete pre-evaluation manifest without receiving a test path."""
    state = git_state(project_root)
    if state["dirty"] and not allow_dirty:
        raise FreezeError("Refusing to freeze a dirty Git worktree")
    expected = _expected_keys(model_config)
    actual: dict[tuple[str, str, str], dict[str, Any]] = {}
    experiments: list[dict[str, Any]] = []
    configuration_values = {
        name: load_yaml_config(path) for name, path in configuration_paths.items()
    }
    for condition, manifest in condition_manifests.items():
        if manifest.get("training_condition") != condition:
            raise FreezeError(f"Manifest training condition mismatch: {condition}")
        records = manifest.get("models")
        if not isinstance(records, list):
            raise FreezeError(f"Manifest models must be a list: {condition}")
        for record in records:
            key = (condition, str(record.get("feature_set")), str(record.get("algorithm")))
            if key in actual:
                raise FreezeError(f"Duplicate experiment: {key}")
            actual[key] = record
    if set(actual) != expected:
        raise FreezeError(
            f"Frozen experiment mismatch; missing={sorted(expected - set(actual))}, "
            f"extra={sorted(set(actual) - expected)}"
        )
    for condition, feature_set, algorithm in sorted(expected):
        record = actual[(condition, feature_set, algorithm)]
        model_path = Path(record["path"])
        experiments.append(
            {
                "experiment_id": f"{condition}__{feature_set}__{algorithm}",
                "algorithm": algorithm,
                "feature_set": feature_set,
                "training_condition": condition,
                "feature_columns": record.get("feature_columns"),
                "hyperparameters": record.get("selected_parameters"),
                "preprocessing": model_config.get("preprocessing"),
                "cross_validation": configuration_values[
                    "augmented_tuning" if condition == "real_plus_synthetic" else "real_tuning"
                ].get("cross_validation"),
                "augmentation_settings": (
                    configuration_values["synthesis"]
                    if condition == "real_plus_synthetic"
                    else None
                ),
                "random_state": model_config.get("random_state"),
                "model_artifact_path": str(model_path),
                "model_artifact_sha256": sha256_file(model_path),
            }
        )
    file_hashes = split_summary.get("file_sha256")
    if not isinstance(file_hashes, dict) or not isinstance(file_hashes.get("test"), str):
        raise FreezeError("Split summary must contain the original test fingerprint")
    configs = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in sorted(configuration_paths.items())
    }
    frozen = {
        "schema_version": 1,
        "experiment_set": {
            "algorithms": list(model_config["experiment_algorithms"]),
            "feature_sets": list(model_config["feature_sets"]),
            "training_conditions": list(model_config["training_conditions"]),
            "expected_experiment_count": len(expected),
        },
        "git": state,
        "training_data": {
            "path": str(train_path),
            "sha256": sha256_file(train_path),
        },
        "test_split": {"sha256": file_hashes["test"]},
        "configurations": configs,
        "condition_manifests": condition_manifests,
        "experiments": experiments,
    }
    frozen["manifest_sha256"] = manifest_sha256(frozen)
    return frozen


def parse_args() -> argparse.Namespace:
    """Parse development-side freeze arguments; no test path is accepted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Create the frozen manifest after all development and fitting are complete."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    try:
        paths = load_yaml_config(args.paths_config)
        config_paths = {**CONFIG_PATHS, "paths": args.paths_config, "model": args.model_config}
        frozen = build_frozen_manifest(
            condition_manifests={
                "real_only": load_json(paths["model_manifest"]),
                "real_plus_synthetic": load_json(paths["synthetic_model_manifest"]),
            },
            model_config=load_yaml_config(args.model_config),
            train_path=paths["data"]["train"],
            split_summary=load_json(paths["outputs"]["split_summary"]),
            configuration_paths=config_paths,
            project_root=paths["project_root"],
            allow_dirty=args.allow_dirty,
        )
        save_json(frozen, paths["frozen_experiment_manifest"])
    except (FileNotFoundError, FreezeError, KeyError, OSError, TypeError, ValueError) as exc:
        LOGGER.error("Experiment freeze failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
