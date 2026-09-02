"""Create and verify one immutable held-out prediction release."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from src.freeze_experiment import manifest_sha256
from src.utils.io import sha256_file

PREDICTION_FILENAME = "frozen_predictions.csv"
RECEIPT_FILENAME = "evaluation_receipt.json"
PREDICTION_COLUMNS = [
    "experiment_id",
    "algorithm",
    "feature_set",
    "training_condition",
    "subject_id",
    "target",
    "prediction",
    "score",
]


class EvaluationReleaseError(ValueError):
    """Raised when a canonical prediction release is incomplete or changed."""


def release_id(frozen_manifest: dict[str, Any]) -> str:
    """Return the deterministic release identifier for one frozen experiment."""
    digest = frozen_manifest.get("manifest_sha256")
    if not isinstance(digest, str) or digest != manifest_sha256(frozen_manifest):
        raise EvaluationReleaseError("Frozen manifest fingerprint mismatch")
    return digest[:16]


def _runtime_versions() -> dict[str, str]:
    values = {"python": __import__("platform").python_version()}
    for package in ("pandas", "scikit-learn", "xgboost"):
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = "not-installed"
    return values


def validate_canonical_predictions(
    predictions: pd.DataFrame,
    frozen_manifest: dict[str, Any],
) -> pd.DataFrame:
    """Require one prediction for every frozen experiment and held-out subject."""
    missing = sorted(set(PREDICTION_COLUMNS).difference(predictions.columns))
    if missing:
        raise EvaluationReleaseError(f"Canonical predictions missing columns: {missing}")
    frame = predictions.loc[:, PREDICTION_COLUMNS].copy()
    if frame.empty or frame.isna().any().any():
        raise EvaluationReleaseError("Canonical predictions must be nonempty and complete")
    if frame.duplicated(["experiment_id", "subject_id"]).any():
        raise EvaluationReleaseError("Duplicate experiment/subject prediction")
    expected = {
        str(record["experiment_id"]): (
            str(record["algorithm"]),
            str(record["feature_set"]),
            str(record["training_condition"]),
        )
        for record in frozen_manifest.get("experiments", [])
    }
    actual = {
        str(row.experiment_id): (
            str(row.algorithm), str(row.feature_set), str(row.training_condition)
        )
        for row in frame[
            ["experiment_id", "algorithm", "feature_set", "training_condition"]
        ].drop_duplicates().itertuples(index=False)
    }
    if actual != expected:
        raise EvaluationReleaseError("Canonical predictions do not match frozen experiments")
    subjects = set(frame["subject_id"].astype(str))
    if not subjects:
        raise EvaluationReleaseError("Canonical predictions contain no subjects")
    for experiment_id, group in frame.groupby("experiment_id", sort=False):
        if set(group["subject_id"].astype(str)) != subjects:
            raise EvaluationReleaseError(
                f"Subject coverage mismatch for experiment {experiment_id}"
            )
    if len(frame) != len(expected) * len(subjects):
        raise EvaluationReleaseError("Canonical prediction row count is incomplete")
    if not set(frame["target"]).issubset({0, 1}) or set(frame["target"]) != {0, 1}:
        raise EvaluationReleaseError("Canonical targets must contain both binary classes")
    if not set(frame["prediction"]).issubset({0, 1}):
        raise EvaluationReleaseError("Canonical predictions must be binary")
    if not np.isfinite(frame["score"].astype(float)).all():
        raise EvaluationReleaseError("Canonical prediction scores must be finite")
    target_counts = frame.groupby("subject_id")["target"].nunique()
    if (target_counts != 1).any():
        raise EvaluationReleaseError("Targets must be consistent for each subject")
    return frame.sort_values(["experiment_id", "subject_id"], kind="stable").reset_index(
        drop=True
    )


def create_evaluation_release(
    predictions: pd.DataFrame,
    frozen_manifest: dict[str, Any],
    *,
    releases_root: str | Path,
    test_sha256: str,
) -> dict[str, Path]:
    """Atomically publish a new release and refuse to overwrite an existing one."""
    frame = validate_canonical_predictions(predictions, frozen_manifest)
    root = Path(releases_root)
    destination = root / release_id(frozen_manifest)
    if destination.exists():
        raise FileExistsError(f"Evaluation release already exists: {destination}")
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".evaluation-release-", dir=root))
    try:
        prediction_path = temporary / PREDICTION_FILENAME
        frame.to_csv(prediction_path, index=False)
        receipt = {
            "release_id": destination.name,
            "frozen_manifest_sha256": frozen_manifest["manifest_sha256"],
            "test_sha256": test_sha256,
            "predictions_sha256": sha256_file(prediction_path),
            "experiment_count": int(frame["experiment_id"].nunique()),
            "subject_count": int(frame["subject_id"].nunique()),
            "prediction_row_count": len(frame),
            "runtime_versions": _runtime_versions(),
        }
        (temporary / RECEIPT_FILENAME).write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )
        os.rename(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "release_dir": destination,
        "predictions": destination / PREDICTION_FILENAME,
        "receipt": destination / RECEIPT_FILENAME,
    }


def load_evaluation_release(
    release_dir: str | Path,
    frozen_manifest: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify receipt and bytes before returning canonical predictions."""
    directory = Path(release_dir)
    prediction_path = directory / PREDICTION_FILENAME
    receipt_path = directory / RECEIPT_FILENAME
    if not prediction_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"Incomplete evaluation release: {directory}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationReleaseError("Invalid evaluation receipt") from exc
    if receipt.get("release_id") != release_id(frozen_manifest):
        raise EvaluationReleaseError("Evaluation release ID does not match manifest")
    if receipt.get("frozen_manifest_sha256") != frozen_manifest.get("manifest_sha256"):
        raise EvaluationReleaseError("Evaluation receipt manifest mismatch")
    test_split = frozen_manifest.get("test_split")
    if not isinstance(test_split, dict) or receipt.get("test_sha256") != test_split.get(
        "sha256"
    ):
        raise EvaluationReleaseError("Evaluation receipt test fingerprint mismatch")
    if sha256_file(prediction_path) != receipt.get("predictions_sha256"):
        raise EvaluationReleaseError("Canonical prediction fingerprint mismatch")
    frame = pd.read_csv(prediction_path, dtype={"subject_id": str})
    frame = validate_canonical_predictions(frame, frozen_manifest)
    if len(frame) != receipt.get("prediction_row_count"):
        raise EvaluationReleaseError("Evaluation receipt row count mismatch")
    if frame["experiment_id"].nunique() != receipt.get("experiment_count"):
        raise EvaluationReleaseError("Evaluation receipt experiment count mismatch")
    if frame["subject_id"].nunique() != receipt.get("subject_count"):
        raise EvaluationReleaseError("Evaluation receipt subject count mismatch")
    return frame, receipt
