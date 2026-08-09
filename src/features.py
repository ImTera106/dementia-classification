"""Canonical feature-set definitions and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import pandas as pd


ALLOWED_SEX_VALUES: Final[frozenset[str]] = frozenset({"female", "male"})
TARGET_COLUMN: Final[str] = "dementia"
SUBJECT_ID_COLUMN: Final[str] = "subject_id"
LEAKAGE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "mri_id",
        "group",
        "visit",
        "mr_delay",
        "hand",
        "cdr",
    }
)


class FeatureValidationError(ValueError):
    """Raised when a cleaned table cannot supply a requested feature set."""


@dataclass(frozen=True)
class FeatureSet:
    """Column roles for one approved modeling setting."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        """Return all predictors in stable model-input order."""
        return self.numeric + self.categorical


def get_feature_set(
    config: dict[str, Any],
    name: str,
) -> FeatureSet:
    """Return one feature-set definition from the YAML config."""
    try:
        definition = config[name]
    except KeyError as exc:
        valid = ", ".join(config)
        raise FeatureValidationError(
            f"Unknown feature set {name!r}; choose: {valid}"
        ) from exc
    try:
        return FeatureSet(
            numeric=tuple(definition["numeric"]),
            categorical=tuple(definition["categorical"]),
        )
    except (KeyError, TypeError) as exc:
        raise FeatureValidationError(
            f"Feature set {name!r} must define numeric and categorical lists"
        ) from exc


def validate_modeling_frame(
    df: pd.DataFrame,
    feature_set: str,
    config: dict[str, Any],
    *,
    target_column: str = TARGET_COLUMN,
    subject_id_column: str = SUBJECT_ID_COLUMN,
) -> FeatureSet:
    """Validate required columns, binary target, and subject uniqueness."""
    feature_definition = get_feature_set(config, feature_set)

    required = set(feature_definition.columns) | {target_column, subject_id_column}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise FeatureValidationError(f"Missing required modeling columns: {missing}")
    if df[subject_id_column].isna().any():
        raise FeatureValidationError("subject_id contains missing values")
    if df[subject_id_column].duplicated().any():
        raise FeatureValidationError("subject_id must be unique before splitting")
    target_values = set(df[target_column].dropna().unique())
    if df[target_column].isna().any() or not target_values.issubset({0, 1}):
        raise FeatureValidationError(
            f"{target_column} must contain only non-null binary values 0/1"
        )
    sex_values = set(df["sex"].dropna().astype(str).unique())
    if df["sex"].isna().any() or not sex_values.issubset(ALLOWED_SEX_VALUES):
        raise FeatureValidationError("sex must contain only non-null 'female'/'male' values")
    for column in feature_definition.numeric:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if (df[column].notna() & numeric.isna()).any():
            raise FeatureValidationError(f"{column} contains non-numeric values")
    if set(feature_definition.columns).intersection(
        LEAKAGE_COLUMNS | {target_column, subject_id_column}
    ):
        raise FeatureValidationError(
            "Feature definition contains identifier or leakage columns"
        )
    return feature_definition


def split_features_target(
    df: pd.DataFrame,
    feature_set: str,
    config: dict[str, Any],
    *,
    target_column: str = TARGET_COLUMN,
    subject_id_column: str = SUBJECT_ID_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract predictors and target without fitting or transforming values."""
    feature_definition = validate_modeling_frame(
        df,
        feature_set,
        config,
        target_column=target_column,
        subject_id_column=subject_id_column,
    )
    features = df.loc[:, list(feature_definition.columns)].copy()
    for column in feature_definition.numeric:
        features[column] = pd.to_numeric(features[column], errors="raise").astype(
            "float64"
        )
    target = df.loc[:, target_column].copy()
    target.name = target_column
    return features, target
