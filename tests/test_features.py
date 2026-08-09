"""Tests for YAML-defined model feature settings."""

from __future__ import annotations

import unittest

import pandas as pd

from src.features import (
    FeatureValidationError,
    get_feature_set,
    split_features_target,
)


def model_config() -> dict:
    """Return config-defined feature sets and modeling column names."""
    return {
        "split": {
            "subject_id_column": "subject_id",
            "target_column": "dementia",
        },
        "feature_sets": {
            "clinical": {
                "numeric": ["age", "education_years", "ses", "mmse"],
                "categorical": ["sex"],
            },
            "clinical_imaging": {
                "numeric": [
                    "age",
                    "education_years",
                    "ses",
                    "mmse",
                    "etiv",
                    "nwbv",
                    "asf",
                ],
                "categorical": ["sex"],
            },
        },
    }


def modeling_frame() -> pd.DataFrame:
    """Return a valid one-row-per-subject cleaned table."""
    return pd.DataFrame(
        {
            "subject_id": ["S1", "S2"],
            "sex": ["female", "male"],
            "age": [70, 80],
            "education_years": [14, 16],
            "ses": [2, pd.NA],
            "mmse": [29, 22],
            "etiv": [1500, 1400],
            "nwbv": [0.75, 0.68],
            "asf": [1.1, 1.2],
            "dementia": [0, 1],
        }
    )


class FeatureTests(unittest.TestCase):
    """Verify configured feature boundaries and subject-level safeguards."""

    def test_clinical_excludes_all_mri_measurements(self) -> None:
        config = model_config()
        expected = get_feature_set(config["feature_sets"], "clinical")
        features, target = split_features_target(
            modeling_frame(), "clinical", config["feature_sets"]
        )
        self.assertEqual(tuple(features.columns), expected.columns)
        self.assertTrue({"etiv", "nwbv", "asf"}.isdisjoint(features.columns))
        self.assertEqual(target.tolist(), [0, 1])

    def test_clinical_imaging_adds_exactly_configured_mri_features(self) -> None:
        config = model_config()
        clinical = get_feature_set(config["feature_sets"], "clinical")
        clinical_imaging = get_feature_set(
            config["feature_sets"], "clinical_imaging"
        )
        features, _ = split_features_target(
            modeling_frame(), "clinical_imaging", config["feature_sets"]
        )
        self.assertEqual(tuple(features.columns), clinical_imaging.columns)
        self.assertEqual(
            set(clinical_imaging.columns) - set(clinical.columns),
            {"etiv", "nwbv", "asf"},
        )

    def test_duplicate_subjects_are_rejected(self) -> None:
        frame = modeling_frame()
        frame.loc[1, "subject_id"] = "S1"
        with self.assertRaisesRegex(FeatureValidationError, "unique"):
            split_features_target(
                frame, "clinical", model_config()["feature_sets"]
            )

    def test_unknown_feature_set_is_rejected(self) -> None:
        with self.assertRaisesRegex(FeatureValidationError, "Unknown feature set"):
            split_features_target(
                modeling_frame(), "full", model_config()["feature_sets"]
            )

    def test_identifier_target_and_leakage_columns_are_never_predictors(self) -> None:
        frame = modeling_frame().assign(cdr=[0, 1], group=["Nondemented", "Demented"])
        config = model_config()
        for name in config["feature_sets"]:
            features, _ = split_features_target(
                frame, name, config["feature_sets"]
            )
            self.assertTrue(
                {"subject_id", "dementia", "cdr", "group"}.isdisjoint(
                    features.columns
                )
            )

    def test_invalid_sex_value_is_rejected(self) -> None:
        frame = modeling_frame()
        frame.loc[0, "sex"] = "unknown"
        with self.assertRaisesRegex(FeatureValidationError, "sex"):
            split_features_target(
                frame, "clinical", model_config()["feature_sets"]
            )
