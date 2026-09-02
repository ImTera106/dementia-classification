"""Tests for read-only legacy computational parity checks."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from src.check_legacy_parity import verify_legacy_parity
from src.evaluate import calculate_evaluation_tables
from src.final_evaluate import run_final_evaluation
from src.evaluation_release import load_evaluation_release
from tests.test_final_evaluate import (
    CLEAN_GIT_STATE, _evaluation_configs, _frozen_fixture, _output_paths,
)
from tests.test_evaluate import held_out_frame, model_config


def test_legacy_match_proves_computational_parity_only() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        frozen, test_path, _ = _frozen_fixture(root)
        with patch("src.final_evaluate.git_state", return_value=CLEAN_GIT_STATE), patch(
            "src.final_evaluate.load_real_test_data", return_value=held_out_frame()
        ):
            release = run_final_evaluation(
                test_path,
                frozen_manifest=frozen,
                model_config=model_config(),
                evaluation_configs=_evaluation_configs(),
                output_paths=_output_paths(root),
                project_root=root,
            )
        predictions, _ = load_evaluation_release(release["release_dir"], frozen)
        prediction_paths = {}
        metric_paths = {}
        metrics, _ = calculate_evaluation_tables(predictions.drop(columns="experiment_id"))
        for condition in ("real_only", "real_plus_synthetic"):
            prediction_path = root / f"{condition}_predictions.csv"
            predictions.loc[
                predictions["training_condition"] == condition
            ].drop(columns="experiment_id").to_csv(prediction_path, index=False)
            prediction_paths[condition] = prediction_path
            metric_path = root / f"{condition}_metrics.csv"
            metrics.loc[metrics["training_condition"] == condition].to_csv(
                metric_path, index=False
            )
            metric_paths[condition] = metric_path
        summary = verify_legacy_parity(
            release["release_dir"], frozen,
            legacy_prediction_paths=prediction_paths,
            legacy_metric_paths=metric_paths,
        )
        assert summary == {"prediction_rows": 80, "experiments": 20, "subjects": 4}
