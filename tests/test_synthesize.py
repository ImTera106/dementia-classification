"""Tests for leakage-safe, feature-set-specific synthetic augmentation."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from src.synthesize import build_metadata, generate_synthetic_subjects
from src.tune import _fold_augmented_data


FEATURE_SETS = {
    "clinical": {
        "numeric": ["age", "education_years", "ses", "mmse"],
        "categorical": ["sex"],
    },
    "clinical_imaging": {
        "numeric": [
            "age", "education_years", "ses", "mmse", "etiv", "nwbv", "asf"
        ],
        "categorical": ["sex"],
    },
}


def synthesis_config() -> dict:
    return {
        "training_condition": "real_plus_synthetic",
        "required_sdv_version": "1.37.3",
        "synthesizer": {
            "method": "gaussian_copula",
            "synthetic_to_real_ratio": 1.0,
            "preserve_target_counts": True,
            "enforce_min_max_values": True,
            "enforce_rounding": True,
        },
        "validation": {
            "require_unique_subject_ids": True,
            "reject_exact_real_matches": True,
            "require_sdv_diagnostic_score": 1.0,
        },
    }


def training_frame() -> pd.DataFrame:
    rows = 20
    return pd.DataFrame(
        {
            "subject_id": [f"S{i:02d}" for i in range(rows)],
            "sex": ["female", "male"] * 10,
            "age": list(range(65, 85)),
            "education_years": [10, 12, 14, 16] * 5,
            "ses": [1, 2, 3, 4, 5] * 4,
            "mmse": [29, 20, 28, 19, 27] * 4,
            "etiv": list(range(1300, 1320)),
            "nwbv": [0.60 + i / 100 for i in range(rows)],
            "asf": [1.0 + i / 100 for i in range(rows)],
            "dementia": [0, 1] * 10,
        }
    )


def test_clinical_synthesis_excludes_imaging_and_preserves_target_counts() -> None:
    real = training_frame()
    synthetic, report = generate_synthetic_subjects(
        real,
        "clinical",
        FEATURE_SETS,
        synthesis_config(),
        target_column="dementia",
        subject_id_column="subject_id",
        evaluate_reports=False,
    )
    assert len(synthetic) == len(real)
    assert synthetic["dementia"].value_counts().to_dict() == {0: 10, 1: 10}
    assert not {"etiv", "nwbv", "asf"}.intersection(synthetic.columns)
    assert set(synthetic["subject_id"]).isdisjoint(real["subject_id"])
    assert report["exact_real_matches"] == 0


def test_raw_education_years_and_ses_metadata_remain_numerical() -> None:
    data = training_frame()[
        ["age", "education_years", "ses", "mmse", "sex", "dementia"]
    ]
    metadata = build_metadata(data, target_column="dementia").to_dict()
    columns = metadata["tables"]["subjects"]["columns"]
    assert columns["education_years"]["sdtype"] == "numerical"
    assert columns["ses"]["sdtype"] == "numerical"
    assert columns["dementia"]["sdtype"] == "categorical"


def test_fold_synthesis_receives_only_real_fold_training_rows() -> None:
    real = training_frame()
    seen_ids: list[set[str]] = []

    def fake_generate(fold: pd.DataFrame, *_: object, **kwargs: object):
        seen_ids.append(set(fold["subject_id"]))
        columns = ["age", "education_years", "ses", "mmse", "sex", "dementia"]
        synthetic = fold.loc[:, columns].copy()
        synthetic.insert(0, "subject_id", [f"SYN_{i}" for i in range(len(fold))])
        return synthetic, {}

    splits = [(list(range(10)), list(range(10, 20))), (list(range(10, 20)), list(range(10)))]
    with patch("src.tune.generate_synthetic_subjects", side_effect=fake_generate):
        folds = _fold_augmented_data(
            real,
            feature_set_name="clinical",
            feature_sets_config=FEATURE_SETS,
            synthesis_config=synthesis_config(),
            target_column="dementia",
            subject_id_column="subject_id",
            cv_splits=splits,
        )
    assert seen_ids == [set(real.iloc[:10]["subject_id"]), set(real.iloc[10:]["subject_id"])]
    assert all(len(augmented_x) == 20 for augmented_x, _, _, _ in folds)
    assert all(len(validation_x) == 10 for _, _, validation_x, _ in folds)
