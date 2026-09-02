"""Tests for immutable canonical held-out prediction releases."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation_release import (
    EvaluationReleaseError,
    create_evaluation_release,
    load_evaluation_release,
)
from src.freeze_experiment import manifest_sha256


def _manifest() -> dict:
    value = {
        "test_split": {"sha256": "test"},
        "experiments": [
            {
                "experiment_id": "real_only__clinical__logistic_regression",
                "algorithm": "logistic_regression",
                "feature_set": "clinical",
                "training_condition": "real_only",
            }
        ]
    }
    value["manifest_sha256"] = manifest_sha256(value)
    return value


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "experiment_id": ["real_only__clinical__logistic_regression"] * 2,
            "algorithm": ["logistic_regression"] * 2,
            "feature_set": ["clinical"] * 2,
            "training_condition": ["real_only"] * 2,
            "subject_id": ["T0", "T1"],
            "target": [0, 1],
            "prediction": [0, 1],
            "score": [0.1, 0.9],
        }
    )


def test_release_is_complete_verified_and_non_overwriting() -> None:
    with tempfile.TemporaryDirectory() as directory:
        paths = create_evaluation_release(
            _predictions(), _manifest(), releases_root=directory, test_sha256="test"
        )
        loaded, receipt = load_evaluation_release(paths["release_dir"], _manifest())
        assert len(loaded) == 2
        assert receipt["experiment_count"] == 1
        assert receipt["subject_count"] == 2
        with pytest.raises(FileExistsError, match="already exists"):
            create_evaluation_release(
                _predictions(), _manifest(), releases_root=directory, test_sha256="test"
            )


def test_changed_prediction_bytes_are_rejected() -> None:
    with tempfile.TemporaryDirectory() as directory:
        paths = create_evaluation_release(
            _predictions(), _manifest(), releases_root=directory, test_sha256="test"
        )
        paths["predictions"].write_text("changed", encoding="utf-8")
        with pytest.raises(EvaluationReleaseError, match="fingerprint mismatch"):
            load_evaluation_release(paths["release_dir"], _manifest())
