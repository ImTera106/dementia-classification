"""Run inference with one explicitly selected trusted local model pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from src.features import ALLOWED_SEX_VALUES, get_feature_set
from src.utils.io import (
    load_joblib,
    load_json,
    load_yaml_config,
    require_package_version,
)
from src.utils.prediction import get_positive_class_score

LOGGER = logging.getLogger(__name__)
DEFAULT_PATHS_CONFIG: Final[Path] = Path("config/paths.yaml")
DEFAULT_MODEL_CONFIG: Final[Path] = Path("config/model_config.yaml")
TRAINING_CONDITIONS: Final[frozenset[str]] = frozenset(
    {"real_only", "real_plus_synthetic"}
)


class InferenceError(ValueError):
    """Raised when an inference request violates the saved-model contract."""


def resolve_manifest_path(
    paths_config: dict[str, Any], training_condition: str
) -> Path:
    """Resolve a condition-specific manifest without selecting by test results."""
    if training_condition == "real_only":
        key = "model_manifest"
    elif training_condition == "real_plus_synthetic":
        key = "synthetic_model_manifest"
    else:
        valid = ", ".join(sorted(TRAINING_CONDITIONS))
        raise InferenceError( 
            f"Unknown training condition {training_condition!r}; choose: {valid}"
        )
    try:
        return Path(paths_config[key])
    except (KeyError, TypeError) as exc:
        raise InferenceError(f"paths configuration is missing {key}") from exc


def select_model_record(
    manifest: dict[str, Any],
    *,
    algorithm: str,
    feature_set: str,
    training_condition: str,
) -> dict[str, Any]:
    """Select exactly one named experiment from a matching manifest."""
    if manifest.get("training_condition") != training_condition:
        raise InferenceError("Manifest training condition does not match the request")
    records = manifest.get("models")
    if not isinstance(records, list):
        raise InferenceError("Model manifest models must be a list")
    matches = [
        record
        for record in records
        if isinstance(record, dict)
        and record.get("algorithm") == algorithm
        and record.get("feature_set") == feature_set
        and record.get("training_condition") == training_condition
    ]
    if len(matches) != 1:
        available = sorted(
            {
                (str(record.get("algorithm")), str(record.get("feature_set")))
                for record in records
                if isinstance(record, dict)
            }
        )
        raise InferenceError(
            f"Expected one model for {(algorithm, feature_set)}, found "
            f"{len(matches)}; available={available}"
        )
    return matches[0]


def validate_inference_frame(
    frame: pd.DataFrame,
    record: dict[str, Any],
    model_config: dict[str, Any],
) -> tuple[pd.Series, pd.DataFrame]:
    """Validate identifiers and raw predictors without learned preprocessing."""
    feature_sets = model_config.get("feature_sets")
    split = model_config.get("split")
    if not isinstance(feature_sets, dict) or not isinstance(split, dict):
        raise InferenceError("Model config requires feature_sets and split mappings")
    feature_set_name = str(record.get("feature_set"))
    definition = get_feature_set(feature_sets, feature_set_name)
    expected = list(definition.columns)
    if record.get("feature_columns") != expected:
        raise InferenceError(
            f"Manifest feature columns do not match YAML for {feature_set_name}"
        )
    subject_column = str(split["subject_id_column"])
    missing = sorted(set([subject_column, *expected]).difference(frame.columns))
    if missing:
        raise InferenceError(f"Inference input is missing required columns: {missing}")
    identifiers = frame[subject_column]
    if identifiers.isna().any() or identifiers.astype(str).str.strip().eq("").any():
        raise InferenceError(f"{subject_column} must be non-null and non-empty")
    if identifiers.duplicated().any():
        raise InferenceError(f"{subject_column} must be unique")

    features = frame.loc[:, expected].copy()
    for column in definition.numeric:
        converted = pd.to_numeric(features[column], errors="coerce")
        invalid = features[column].notna() & converted.isna()
        if invalid.any():
            raise InferenceError(f"{column} contains non-numeric values")
        finite = converted.dropna().map(np.isfinite)
        if not finite.all():
            raise InferenceError(f"{column} contains infinite values")
        features[column] = converted.astype("float64")
    if "sex" in definition.categorical:
        observed = set(features["sex"].dropna().astype(str))
        invalid_sex = sorted(observed.difference(ALLOWED_SEX_VALUES))
        if invalid_sex:
            raise InferenceError(
                f"sex contains unsupported values: {invalid_sex}; expected female/male"
            )
    return identifiers.astype(str), features


def predict_subjects(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    model_config: dict[str, Any],
    *,
    algorithm: str,
    feature_set: str,
    training_condition: str,
) -> pd.DataFrame:
    """Predict with one fitted pipeline and return identifiers plus scores."""
    record = select_model_record(
        manifest,
        algorithm=algorithm,
        feature_set=feature_set,
        training_condition=training_condition,
    )
    versions = manifest.get("versions")
    if not isinstance(versions, dict) or not isinstance(
        versions.get("scikit_learn"), str
    ):
        raise InferenceError("Manifest must record the scikit-learn version")
    require_package_version(
        "scikit-learn", versions["scikit_learn"], context="Saved-model inference"
    )
    if algorithm == "xgboost":
        required_xgboost = versions.get("xgboost")
        if not isinstance(required_xgboost, str):
            raise InferenceError("XGBoost manifest version is missing")
        require_package_version(
            "xgboost", required_xgboost, context="Saved-model inference"
        )
    identifiers, features = validate_inference_frame(frame, record, model_config)
    model = load_joblib(record["path"])
    check_is_fitted(model)
    predictions = np.asarray(model.predict(features))
    if predictions.shape != (len(features),) or not set(predictions).issubset({0, 1}):
        raise InferenceError("Saved model produced invalid binary predictions")
    scores = get_positive_class_score(model, features)
    return pd.DataFrame(
        {
            "algorithm": algorithm,
            "feature_set": feature_set,
            "training_condition": training_condition,
            str(model_config["split"]["subject_id_column"]): identifiers,
            "prediction": predictions.astype(int),
            "score": scores,
        }
    )


def run_prediction_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    *,
    paths_config: dict[str, Any],
    model_config: dict[str, Any],
    algorithm: str,
    feature_set: str,
    training_condition: str,
    overwrite: bool = False,
) -> Path:
    """Load new subjects, run one named model, and save predictions."""
    source = Path(input_path)
    destination = Path(output_path)
    if not source.is_file():
        raise FileNotFoundError(f"Inference input not found: {source}")
    if source.resolve() == destination.resolve():
        raise InferenceError("Inference output must not overwrite its input file")
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Inference output already exists: {destination}; pass --overwrite to replace it"
        )
    manifest_path = resolve_manifest_path(paths_config, training_condition)
    predictions = predict_subjects(
        pd.read_csv(source),
        load_json(manifest_path),
        model_config,
        algorithm=algorithm,
        feature_set=feature_set,
        training_condition=training_condition,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, index=False)
    LOGGER.info("Saved %d predictions to %s", len(predictions), destination)
    return destination


def parse_args() -> argparse.Namespace:
    """Parse an explicit inference request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--feature-set", required=True)
    parser.add_argument("--training-condition", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    return parser.parse_args()


def main() -> int:
    """Run inference without loading outcomes or fitting model state."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    try:
        run_prediction_pipeline(
            args.input,
            args.output,
            paths_config=load_yaml_config(args.paths_config),
            model_config=load_yaml_config(args.model_config),
            algorithm=args.algorithm,
            feature_set=args.feature_set,
            training_condition=args.training_condition,
            overwrite=args.overwrite,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        InferenceError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        LOGGER.error("Inference failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
