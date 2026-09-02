"""Regenerate formal evaluation artifacts from frozen predictions only."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Final

import pandas as pd

from src.evaluate import calculate_evaluation_tables, save_evaluation_outputs
from src.evaluation_release import load_evaluation_release, release_id
from src.freeze_experiment import manifest_sha256
from src.utils.io import load_json, load_yaml_config, sha256_file
from src.utils.plotting import plot_training_condition_comparison
from src.validate import (
    run_training_condition_validation_pipeline,
    run_validation_pipeline,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_PATHS_CONFIG: Final[Path] = Path("config/paths.yaml")
DEFAULT_MODEL_CONFIG: Final[Path] = Path("config/model_config.yaml")
DEFAULT_REAL_EVALUATION_CONFIG: Final[Path] = Path("config/evaluation_config.yaml")
DEFAULT_AUGMENTED_EVALUATION_CONFIG: Final[Path] = Path(
    "config/evaluation_synthetic_config.yaml"
)
DEFAULT_VALIDATION_CONFIG: Final[Path] = Path("config/validation_config.yaml")
ANALYSIS_MANIFEST_FILENAME: Final[str] = "analysis_manifest.json"


class ReleaseAnalysisError(ValueError):
    """Raised when a canonical release cannot produce a complete analysis."""


def verify_analysis_manifest(
    release_dir: str | Path,
    frozen_manifest: dict[str, Any],
    *,
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """Verify the release linkage and exact bytes of every formal artifact."""
    directory = Path(release_dir)
    _, receipt = load_evaluation_release(directory, frozen_manifest)
    path = directory / ANALYSIS_MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Release analysis manifest not found: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReleaseAnalysisError("Invalid release analysis manifest") from exc
    if manifest.get("manifest_sha256") != manifest_sha256(manifest):
        raise ReleaseAnalysisError("Release analysis manifest fingerprint mismatch")
    if manifest.get("frozen_manifest_sha256") != frozen_manifest.get(
        "manifest_sha256"
    ) or manifest.get("predictions_sha256") != receipt.get("predictions_sha256"):
        raise ReleaseAnalysisError("Release analysis input fingerprint mismatch")
    records = manifest.get("artifacts")
    if not isinstance(records, dict) or not records:
        raise ReleaseAnalysisError("Release analysis artifact records are missing")
    root = Path(project_root)
    for name, record in records.items():
        if not isinstance(record, dict):
            raise ReleaseAnalysisError(f"Invalid analysis artifact record: {name}")
        artifact_path = Path(record.get("path", ""))
        if not artifact_path.is_absolute():
            artifact_path = root / artifact_path
        if sha256_file(artifact_path) != record.get("sha256"):
            raise ReleaseAnalysisError(f"Analysis artifact fingerprint mismatch: {name}")
    return manifest


def _comparison_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate condition deltas for matched algorithm/feature experiments."""
    keys = ["algorithm", "feature_set"]
    real = metrics.loc[
        metrics["training_condition"] == "real_only",
        keys + ["balanced_accuracy"],
    ].rename(columns={"balanced_accuracy": "real_only_balanced_accuracy"})
    comparison = metrics.merge(real, on=keys, how="left", validate="many_to_one")
    comparison["balanced_accuracy_delta_vs_real_only"] = (
        comparison["balanced_accuracy"]
        - comparison["real_only_balanced_accuracy"]
    )
    return comparison


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def _write_analysis_manifest(
    release_dir: Path,
    frozen_manifest: dict[str, Any],
    receipt: dict[str, Any],
    artifacts: dict[str, Path],
    *,
    project_root: Path,
    configuration_fingerprints: dict[str, str] | None,
) -> Path:
    records = {
        name: {
            "path": _display_path(path, project_root),
            "sha256": sha256_file(path),
        }
        for name, path in sorted(artifacts.items())
    }
    manifest = {
        "schema_version": 1,
        "release_id": release_dir.name,
        "frozen_manifest_sha256": frozen_manifest["manifest_sha256"],
        "predictions_sha256": receipt["predictions_sha256"],
        "configuration_sha256": configuration_fingerprints or {},
        "artifacts": records,
    }
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    destination = release_dir / ANALYSIS_MANIFEST_FILENAME
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".analysis-manifest-", suffix=".json", dir=release_dir
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def run_release_analysis(
    release_dir: str | Path,
    *,
    frozen_manifest: dict[str, Any],
    evaluation_configs: dict[str, dict[str, Any]],
    validation_config: dict[str, Any],
    output_paths: dict[str, Any],
    project_root: str | Path = ".",
    configuration_fingerprints: dict[str, str] | None = None,
) -> dict[str, Path]:
    """Rebuild all formal aggregates without test data or fitted models."""
    directory = Path(release_dir)
    predictions, receipt = load_evaluation_release(directory, frozen_manifest)
    expected_conditions = {"real_only", "real_plus_synthetic"}
    if set(evaluation_configs) != expected_conditions:
        raise ReleaseAnalysisError("Evaluation configs must cover both conditions")

    artifacts: dict[str, Path] = {}
    condition_keys = {
        "real_only": (
            "test_metrics", "test_confusion_matrices",
            "test_balanced_accuracy_figure", "test_roc_figure",
        ),
        "real_plus_synthetic": (
            "synthetic_test_metrics", "synthetic_test_confusion_matrices",
            "synthetic_test_balanced_accuracy_figure", "synthetic_test_roc_figure",
        ),
    }
    metric_frames: list[pd.DataFrame] = []
    for condition, keys in condition_keys.items():
        frame = predictions.loc[
            predictions["training_condition"] == condition
        ].drop(columns="experiment_id")
        metrics, confusion = calculate_evaluation_tables(frame)
        metric_frames.append(metrics)
        saved = save_evaluation_outputs(
            metrics,
            frame,
            confusion,
            metrics_path=output_paths[keys[0]],
            predictions_path=None,
            confusion_path=output_paths[keys[1]],
            balanced_accuracy_figure_path=output_paths[keys[2]],
            roc_figure_path=output_paths[keys[3]],
            dpi=int(evaluation_configs[condition]["figures"]["dpi"]),
        )
        artifacts.update({f"{condition}_{name}": path for name, path in saved.items()})

    comparison = _comparison_table(pd.concat(metric_frames, ignore_index=True))
    comparison_path = Path(output_paths["training_condition_comparison"])
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)
    artifacts["training_condition_comparison"] = comparison_path
    artifacts["training_condition_comparison_figure"] = (
        plot_training_condition_comparison(
            comparison,
            output_paths["training_condition_comparison_figure"],
            dpi=int(evaluation_configs["real_only"]["figures"]["dpi"]),
        )
    )
    artifacts.update(
        {
            f"real_only_bootstrap_{name}": path
            for name, path in run_validation_pipeline(
                directory,
                frozen_manifest=frozen_manifest,
                validation_config=validation_config,
                evaluation_config=evaluation_configs["real_only"],
                output_paths=output_paths,
            ).items()
        }
    )
    artifacts.update(
        {
            f"training_condition_bootstrap_{name}": path
            for name, path in run_training_condition_validation_pipeline(
                directory,
                frozen_manifest=frozen_manifest,
                validation_config=validation_config,
                evaluation_config=evaluation_configs["real_only"],
                output_paths=output_paths,
            ).items()
        }
    )
    manifest_path = _write_analysis_manifest(
        directory,
        frozen_manifest,
        receipt,
        artifacts,
        project_root=Path(project_root),
        configuration_fingerprints=configuration_fingerprints,
    )
    LOGGER.info("Verified and recorded %d release-derived artifacts", len(artifacts))
    return {**artifacts, "analysis_manifest": manifest_path}


def parse_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--validation-config", type=Path, default=DEFAULT_VALIDATION_CONFIG
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    try:
        paths = load_yaml_config(args.paths_config)
        frozen = load_json(paths["frozen_experiment_manifest"])
        directory = Path(paths["outputs"]["final_evaluation_dir"]) / release_id(frozen)
        config_paths = {
            "paths": args.paths_config,
            "model": args.model_config,
            "real_evaluation": args.real_evaluation_config,
            "augmented_evaluation": args.augmented_evaluation_config,
            "validation": args.validation_config,
        }
        run_release_analysis(
            directory,
            frozen_manifest=frozen,
            evaluation_configs={
                "real_only": load_yaml_config(args.real_evaluation_config),
                "real_plus_synthetic": load_yaml_config(
                    args.augmented_evaluation_config
                ),
            },
            validation_config=load_yaml_config(args.validation_config),
            output_paths=paths["outputs"],
            project_root=paths["project_root"],
            configuration_fingerprints={
                name: sha256_file(path) for name, path in config_paths.items()
            },
        )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        LOGGER.error("Release analysis failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
