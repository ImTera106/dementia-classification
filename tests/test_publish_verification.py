"""Tests for privacy-safe public aggregate verification packages."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.analyze_release import run_release_analysis
from src.publish_verification import (
    PublicationError, publish_verification_package, verify_public_package,
)
from tests.test_analyze_release import (
    _analysis_outputs, _release, _validation_config,
)
from tests.test_final_evaluate import _evaluation_configs


def _split_summary() -> dict:
    return {
        "train_subjects": 120,
        "test_subjects": 30,
        "subject_overlap_count": 0,
        "train_target_counts": {"0": 58, "1": 62},
        "test_target_counts": {"0": 14, "1": 16},
        "random_state": 123,
    }


def _prepared_inputs(root: Path) -> tuple[dict, dict, dict[str, Path]]:
    frozen, release = _release(root)
    analysis = run_release_analysis(
        release["release_dir"],
        frozen_manifest=frozen,
        evaluation_configs=_evaluation_configs(),
        validation_config=_validation_config(),
        output_paths=_analysis_outputs(root),
        project_root=root,
    )
    aggregate = root / "outputs" / "safe_summary.csv"
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"algorithm": ["logistic_regression"], "mean": [0.8]}).to_csv(
        aggregate, index=False
    )
    report = root / "report" / "report.html"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"release {release['release_dir'].name} analysis "
        + json.loads(analysis["analysis_manifest"].read_text(encoding="utf-8"))[
            "manifest_sha256"
        ],
        encoding="utf-8",
    )
    return frozen, release, {"safe_summary": aggregate, "rendered_report": report}


def test_publication_contains_only_verified_aggregate_evidence() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        frozen, release, additional = _prepared_inputs(root)
        destination = publish_verification_package(
            release["release_dir"], frozen,
            project_root=root,
            public_root=root / "public_results",
            additional_artifacts=additional,
            split_summary=_split_summary(),
        )
        manifest = json.loads(
            (destination / "verification_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["historical_test_status"] == "previously_inspected_partition"
        assert manifest["release_id"] == release["release_dir"].name
        verify_public_package(destination)
        assert not list(destination.rglob("frozen_predictions.csv"))
        assert not list(destination.rglob("*.joblib"))
        first = next(
            path for path in destination.rglob("*.csv") if path.is_file()
        )
        first.write_text("changed", encoding="utf-8")
        with pytest.raises(PublicationError, match="fingerprint mismatch"):
            verify_public_package(destination)
        with pytest.raises(FileExistsError, match="already exists"):
            publish_verification_package(
                release["release_dir"], frozen,
                project_root=root,
                public_root=root / "public_results",
                additional_artifacts=additional,
                split_summary=_split_summary(),
            )


def test_subject_level_columns_are_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        frozen, release, additional = _prepared_inputs(root)
        unsafe = root / "outputs" / "unsafe.csv"
        pd.DataFrame({"subject_id": ["hidden"], "prediction": [1]}).to_csv(
            unsafe, index=False
        )
        additional["unsafe"] = unsafe
        with pytest.raises(PublicationError, match="Subject-level columns"):
            publish_verification_package(
                release["release_dir"], frozen,
                project_root=root,
                public_root=root / "public_results",
                additional_artifacts=additional,
                split_summary=_split_summary(),
            )
