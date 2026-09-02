"""Publish a privacy-safe aggregate verification package from one release."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Final

import pandas as pd

from src.analyze_release import verify_analysis_manifest
from src.evaluation_release import load_evaluation_release, release_id
from src.freeze_experiment import manifest_sha256
from src.utils.io import load_json, load_yaml_config, sha256_file

LOGGER = logging.getLogger(__name__)
DEFAULT_PATHS_CONFIG: Final[Path] = Path("config/paths.yaml")
FORBIDDEN_COLUMNS: Final[frozenset[str]] = frozenset(
    {"subject_id", "prediction", "score", "synthetic_subject_id"}
)
SUBJECT_VALUE_PATTERN: Final[re.Pattern[str]] = re.compile(r"OAS2[_-]\d+", re.I)


class PublicationError(ValueError):
    """Raised when a candidate public package is unsafe or unverifiable."""


def verify_public_package(package_dir: str | Path) -> dict[str, Any]:
    """Verify package completeness, fingerprints, and disclosure restrictions."""
    directory = Path(package_dir)
    manifest_path = directory / "verification_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Verification manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PublicationError("Invalid public verification manifest") from exc
    if manifest.get("manifest_sha256") != manifest_sha256(manifest):
        raise PublicationError("Public verification manifest fingerprint mismatch")
    records = manifest.get("artifacts")
    if not isinstance(records, dict) or not records:
        raise PublicationError("Public verification artifact records are missing")
    expected = {Path("verification_manifest.json")}
    for name, record in records.items():
        if not isinstance(record, dict):
            raise PublicationError(f"Invalid public artifact record: {name}")
        relative = Path(record.get("path", ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise PublicationError(f"Unsafe public artifact path: {relative}")
        path = directory / relative
        _validate_safe_file(path)
        if sha256_file(path) != record.get("published_sha256"):
            raise PublicationError(f"Public artifact fingerprint mismatch: {name}")
        expected.add(relative)
    actual = {
        path.relative_to(directory)
        for path in directory.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise PublicationError("Public package contains unrecorded or missing files")
    return manifest


def _validate_safe_file(path: Path) -> None:
    """Reject subject-level columns, likely OASIS identifiers, and unsafe types."""
    if path.suffix.lower() not in {".csv", ".json", ".png", ".html"}:
        raise PublicationError(f"Unsupported public artifact type: {path}")
    if path.suffix.lower() == ".csv":
        columns = {str(value).strip().lower() for value in pd.read_csv(path, nrows=0).columns}
        forbidden = sorted(columns.intersection(FORBIDDEN_COLUMNS))
        if forbidden:
            raise PublicationError(
                f"Subject-level columns are forbidden in public artifacts: {forbidden}"
            )
    if path.suffix.lower() in {".csv", ".json", ".html"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if SUBJECT_VALUE_PATTERN.search(text):
            raise PublicationError(f"Possible OASIS subject identifier found: {path}")


def _copy_name(source: Path, project_root: Path) -> Path:
    """Preserve repository-relative provenance paths inside the package."""
    try:
        relative = source.resolve().relative_to(project_root.resolve())
    except ValueError as exc:
        raise PublicationError(f"Public source must be inside the project: {source}") from exc
    if relative.parts[0] not in {"outputs", "report"}:
        raise PublicationError(f"Public source is outside approved artifact roots: {source}")
    return relative


def publish_verification_package(
    release_dir: str | Path,
    frozen_manifest: dict[str, Any],
    *,
    project_root: str | Path,
    public_root: str | Path,
    additional_artifacts: dict[str, str | Path],
    split_summary: dict[str, Any],
) -> Path:
    """Atomically publish verified formal and approved aggregate artifacts."""
    root = Path(project_root)
    directory = Path(release_dir)
    _, receipt = load_evaluation_release(directory, frozen_manifest)
    analysis = verify_analysis_manifest(directory, frozen_manifest, project_root=root)
    sources = {
        f"formal_{name}": root / record["path"]
        for name, record in analysis["artifacts"].items()
    }
    for name, value in additional_artifacts.items():
        if name in sources:
            raise PublicationError(f"Duplicate public artifact name: {name}")
        sources[name] = Path(value)
    required_split_keys = {
        "train_subjects", "test_subjects", "subject_overlap_count",
        "train_target_counts", "test_target_counts", "random_state",
    }
    if not required_split_keys.issubset(split_summary):
        raise PublicationError("Split summary is missing public aggregate provenance")
    report_path = sources.get("rendered_report")
    if report_path is None:
        raise PublicationError("A rendered report is required for public verification")
    report_text = report_path.read_text(encoding="utf-8", errors="ignore")
    if directory.name not in report_text or analysis["manifest_sha256"] not in report_text:
        raise PublicationError("Rendered report does not identify this verified analysis release")

    destination_root = Path(public_root)
    destination = destination_root / directory.name
    if destination.exists():
        raise FileExistsError(f"Public verification package already exists: {destination}")
    destination_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".verification-", dir=destination_root))
    records: dict[str, dict[str, str]] = {}
    published_paths: set[Path] = set()
    try:
        for name, source in sorted(sources.items()):
            if not source.is_file():
                raise FileNotFoundError(f"Required public aggregate not found: {source}")
            _validate_safe_file(source)
            relative = _copy_name(source, root)
            if relative in published_paths:
                raise PublicationError(f"Duplicate public destination path: {relative}")
            published_paths.add(relative)
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            records[name] = {
                "path": str(relative),
                "source_sha256": sha256_file(source),
                "published_sha256": sha256_file(target),
            }
        public_split = {key: split_summary[key] for key in sorted(required_split_keys)}
        manifest = {
            "schema_version": 1,
            "release_id": directory.name,
            "historical_test_status": "previously_inspected_partition",
            "claim_scope": "aggregate computational verification; not renewed independence",
            "git": frozen_manifest["git"],
            "frozen_manifest_sha256": frozen_manifest["manifest_sha256"],
            "test_data_sha256": receipt["test_sha256"],
            "predictions_sha256": receipt["predictions_sha256"],
            "analysis_manifest_sha256": analysis["manifest_sha256"],
            "runtime_versions": receipt["runtime_versions"],
            "configuration_sha256": frozen_manifest["configurations"],
            "split_summary": public_split,
            "artifacts": records,
            "excluded_artifacts": [
                "raw_or_processed_oasis_data", "subject_ids",
                "subject_level_predictions", "train_test_membership",
                "synthetic_subject_rows", "fitted_models",
            ],
        }
        manifest["manifest_sha256"] = manifest_sha256(manifest)
        (temporary / "verification_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        verify_public_package(temporary)
        os.rename(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    try:
        paths = load_yaml_config(args.paths_config)
        root = Path(paths["project_root"])
        frozen = load_json(paths["frozen_experiment_manifest"])
        directory = root / paths["outputs"]["final_evaluation_dir"] / release_id(frozen)
        outputs = paths["outputs"]
        additional = {
            "data_assessment": root / outputs["data_assessment"],
            "split_summary": root / outputs["split_summary"],
            "baseline_cv_summary": root / outputs["baseline_summary"],
            "real_tuning_summary": root / outputs["tuning_summary"],
            "augmented_tuning_summary": root / outputs["synthetic_tuning_summary"],
            "synthetic_quality": root / outputs["synthetic_quality"],
            "permutation_importance": root / outputs["permutation_importance"],
            "logistic_coefficients": root / outputs["logistic_coefficients"],
            "tree_shap_importance": root / outputs["tree_shap_importance"],
            "permutation_importance_clinical_figure": root
            / outputs["permutation_importance_clinical_figure"],
            "permutation_importance_clinical_imaging_figure": root
            / outputs["permutation_importance_clinical_imaging_figure"],
            "logistic_coefficients_figure": root
            / outputs["logistic_coefficients_figure"],
            "rendered_report": root / "report/report.html",
        }
        for algorithm in ("decision_tree", "random_forest", "xgboost"):
            for feature_set in ("clinical", "clinical_imaging"):
                name = f"tree_shap_{algorithm}_{feature_set}_figure"
                additional[name] = (
                    root / outputs["tree_shap_figures_dir"]
                    / f"{algorithm}_{feature_set}.png"
                )
        publish_verification_package(
            directory,
            frozen,
            project_root=root,
            public_root=root / paths["public_verification_dir"],
            additional_artifacts=additional,
            split_summary=load_json(root / outputs["split_summary"]),
        )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        LOGGER.error("Public verification failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
