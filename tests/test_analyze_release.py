"""Tests for recoverable, fingerprinted canonical-release analysis."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.analyze_release import ReleaseAnalysisError, run_release_analysis, verify_analysis_manifest
from src.final_evaluate import run_final_evaluation
from src.utils.io import sha256_file
from tests.test_final_evaluate import (
    CLEAN_GIT_STATE,
    _evaluation_configs,
    _frozen_fixture,
    _output_paths,
)
from tests.test_evaluate import held_out_frame, model_config
from tests.test_validate import validation_config


def _analysis_outputs(root: Path) -> dict:
    paths = _output_paths(root / "outputs")
    paths.update(
        {
            "test_metric_bootstrap_intervals": root / "outputs/real_intervals.csv",
            "feature_set_balanced_accuracy_differences": root / "outputs/feature_diff.csv",
            "test_balanced_accuracy_intervals_figure": root / "outputs/real_intervals.png",
            "feature_set_balanced_accuracy_differences_figure": root / "outputs/feature_diff.png",
            "synthetic_test_metric_bootstrap_intervals": root / "outputs/aug_intervals.csv",
            "training_condition_balanced_accuracy_differences": root / "outputs/condition_diff.csv",
            "synthetic_test_balanced_accuracy_intervals_figure": root / "outputs/aug_intervals.png",
            "training_condition_balanced_accuracy_differences_figure": root / "outputs/condition_diff.png",
        }
    )
    return paths


def _validation_config() -> dict:
    config = validation_config()
    config["expected_subject_count"] = 4
    return config


def _release(root: Path) -> tuple[dict, dict[str, Path]]:
    frozen, test_path, _ = _frozen_fixture(root)
    with patch("src.final_evaluate.git_state", return_value=CLEAN_GIT_STATE), patch(
        "src.final_evaluate.load_real_test_data", return_value=held_out_frame()
    ):
        release = run_final_evaluation(
            test_path,
            frozen_manifest=frozen,
            model_config=model_config(),
            evaluation_configs=_evaluation_configs(),
            output_paths=_analysis_outputs(root),
            project_root=root,
        )
    return frozen, release


def test_analysis_manifest_fingerprints_every_derived_artifact() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        frozen, release = _release(root)
        paths = run_release_analysis(
            release["release_dir"],
            frozen_manifest=frozen,
            evaluation_configs=_evaluation_configs(),
            validation_config=_validation_config(),
            output_paths=_analysis_outputs(root),
            project_root=root,
        )
        assert len(paths) == 19
        manifest = json.loads(paths["analysis_manifest"].read_text(encoding="utf-8"))
        assert manifest["predictions_sha256"] == sha256_file(release["predictions"])
        assert len(manifest["artifacts"]) == 18
        for record in manifest["artifacts"].values():
            assert sha256_file(root / record["path"]) == record["sha256"]
        verify_analysis_manifest(
            release["release_dir"], frozen, project_root=root
        )
        first_record = next(iter(manifest["artifacts"].values()))
        (root / first_record["path"]).write_bytes(b"changed")
        with pytest.raises(ReleaseAnalysisError, match="artifact fingerprint mismatch"):
            verify_analysis_manifest(
                release["release_dir"], frozen, project_root=root
            )


def test_failed_analysis_is_recoverable_without_regenerating_predictions() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        frozen, release = _release(root)
        original_digest = sha256_file(release["predictions"])
        with patch(
            "src.analyze_release.plot_training_condition_comparison",
            side_effect=RuntimeError("plot failed"),
        ), pytest.raises(RuntimeError, match="plot failed"):
            run_release_analysis(
                release["release_dir"],
                frozen_manifest=frozen,
                evaluation_configs=_evaluation_configs(),
                validation_config=_validation_config(),
                output_paths=_analysis_outputs(root),
                project_root=root,
            )
        assert not (release["release_dir"] / "analysis_manifest.json").exists()
        paths = run_release_analysis(
            release["release_dir"],
            frozen_manifest=frozen,
            evaluation_configs=_evaluation_configs(),
            validation_config=_validation_config(),
            output_paths=_analysis_outputs(root),
            project_root=root,
        )
        assert paths["analysis_manifest"].is_file()
        assert sha256_file(release["predictions"]) == original_digest
