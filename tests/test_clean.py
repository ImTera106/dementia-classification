"""Tests for the leakage-safe Phase 1 data foundation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.clean import (
    DataValidationError,
    assess_data,
    clean_longitudinal,
    collapse_to_last_visit,
    run_cleaning_pipeline,
    split_subjects,
)


def raw_frame(batch: int = 0) -> pd.DataFrame:
    """Return three subjects with repeated visits and deliberate missingness."""
    suffix = f"_{batch}"
    return pd.DataFrame(
        {
            "Subject ID": [f"S1{suffix}", f"S1{suffix}", f"S2{suffix}", f"S2{suffix}", f"S3{suffix}"],
            "MRI ID": [f"M{i}{suffix}" for i in range(1, 6)],
            "Group": ["Nondemented", "Nondemented", "Converted", "Converted", "Demented"],
            "Visit": [1, 2, 1, 3, 1],
            "MR Delay": [0, 400, 0, 800, 0],
            "M/F": ["M", "M", "F", "F", "M"],
            "Hand": ["R"] * 5,
            "Age": [70, 71, 75, 77, 80],
            "EDUC": [11, 12, 15, 16, 17],
            "SES": [2, pd.NA, 3, 3, 1],
            "MMSE": [29, 28, 26, pd.NA, 20],
            "CDR": [0, 0, 0, 0.5, 1],
            "eTIV": [1500, 1510, 1400, 1410, 1300],
            "nWBV": [0.75, 0.74, 0.72, 0.70, 0.68],
            "ASF": [1.1, 1.09, 1.2, 1.19, 1.3],
        }
    )


class CleanTests(unittest.TestCase):
    """Verify schema, deterministic cleaning, splitting, and persistence."""

    def test_collapse_preserves_whole_last_visit_row(self) -> None:
        collapsed = collapse_to_last_visit(raw_frame())
        s1 = collapsed.loc[collapsed["Subject ID"] == "S1_0"].iloc[0]
        self.assertEqual(s1["Visit"], 2)
        self.assertTrue(pd.isna(s1["SES"]))

    def test_clean_reproduces_confirmed_target_and_features(self) -> None:
        cleaned = clean_longitudinal(raw_frame())
        self.assertEqual(cleaned["dementia"].tolist(), [0, 1, 1])
        self.assertEqual(cleaned["sex"].tolist(), ["male", "female", "male"])
        self.assertEqual(cleaned["education_years"].tolist(), [12, 16, 17])
        self.assertNotIn("cdr", cleaned.columns)

    def test_cleaning_preserves_missing_values(self) -> None:
        cleaned = clean_longitudinal(raw_frame())
        self.assertEqual(int(cleaned["ses"].isna().sum()), 1)
        self.assertEqual(int(cleaned["mmse"].isna().sum()), 1)

    def test_unknown_group_fails_closed(self) -> None:
        frame = raw_frame()
        frame.loc[frame["Subject ID"] == "S3_0", "Group"] = "Unknown"
        with self.assertRaisesRegex(DataValidationError, "Group"):
            clean_longitudinal(frame)

    def test_missing_required_column_is_reported(self) -> None:
        with self.assertRaisesRegex(DataValidationError, "MMSE"):
            clean_longitudinal(raw_frame().drop(columns="MMSE"))

    def test_assessment_is_descriptive_only(self) -> None:
        frame = raw_frame()
        assessment = assess_data(frame, stage="raw")
        self.assertEqual(assessment["rows"], 5)
        self.assertEqual(assessment["unique_subjects"], 3)
        self.assertTrue(pd.isna(frame.loc[1, "SES"]))

    def test_split_is_reproducible_stratified_and_disjoint(self) -> None:
        subject_level = clean_longitudinal(
            pd.concat([raw_frame(i) for i in range(10)], ignore_index=True)
        )
        first_train, first_test = split_subjects(
            subject_level, test_size=0.2, random_state=7, stratify=True
        )
        second_train, second_test = split_subjects(
            subject_level, test_size=0.2, random_state=7, stratify=True
        )
        self.assertEqual(first_train["subject_id"].tolist(), second_train["subject_id"].tolist())
        self.assertEqual(first_test["subject_id"].tolist(), second_test["subject_id"].tolist())
        self.assertTrue(set(first_train["subject_id"]).isdisjoint(first_test["subject_id"]))
        self.assertEqual(first_test["dementia"].value_counts().to_dict(), {1: 4, 0: 2})

    def test_pipeline_saves_unimputed_contract(self) -> None:
        raw = pd.concat([raw_frame(i) for i in range(10)], ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "raw.csv"
            raw.to_csv(input_path, index=False)
            paths = run_cleaning_pipeline(
                input_path,
                root / "cleaned" / "subject_level.csv",
                root / "processed" / "train.csv",
                root / "processed" / "test.csv",
                root / "outputs" / "assessment.json",
                root / "outputs" / "split.json",
                test_size=0.2,
                random_state=7,
                stratify=True,
            )
            self.assertTrue(all(path.is_file() for path in paths.values()))
            saved_subjects = pd.read_csv(paths["subject_level"])
            self.assertGreater(int(saved_subjects[["ses", "mmse"]].isna().sum().sum()), 0)
            summary = json.loads(paths["split_summary"].read_text(encoding="utf-8"))
            self.assertEqual(summary["subject_overlap_count"], 0)
            self.assertNotIn("imputation", paths)
