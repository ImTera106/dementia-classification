"""Build leakage-safe subject-level OASIS-2 train and test partitions.

Only deterministic operations occur here: schema validation, whole-row
last-visit selection, recoding, assessment, and subject-level splitting.
Missing values are intentionally preserved for future sklearn pipelines.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Final

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import load_yaml_config, sha256_file

LOGGER = logging.getLogger(__name__)

DEFAULT_PATHS_CONFIG: Final[Path] = Path("config/paths.yaml")
DEFAULT_MODEL_CONFIG: Final[Path] = Path("config/model_config.yaml")

RAW_TO_CLEAN: Final[dict[str, str]] = {
    "Subject ID": "subject_id",
    "MRI ID": "mri_id",
    "Group": "group",
    "Visit": "visit",
    "MR Delay": "mr_delay",
    "M/F": "sex",
    "Hand": "hand",
    "Age": "age",
    "EDUC": "education_years",
    "SES": "ses",
    "MMSE": "mmse",
    "CDR": "cdr",
    "eTIV": "etiv",
    "nWBV": "nwbv",
    "ASF": "asf",
}
REQUIRED_RAW_COLUMNS: Final[frozenset[str]] = frozenset(RAW_TO_CLEAN)
TARGET_MAPPING: Final[dict[str, int]] = {
    "Nondemented": 0,
    "Demented": 1,
    "Converted": 1,
}
SEX_MAPPING: Final[dict[str, str]] = {"M": "male", "F": "female"}

NON_MODEL_COLUMNS = (
    "mri_id",
    "group",
    "visit",
    "mr_delay",
    "hand",
    "cdr",
)

CLEAN_COLUMNS = (
    "subject_id",
    "sex",
    "age",
    "education_years",
    "ses",
    "mmse",
    "etiv",
    "nwbv",
    "asf",
    "dementia",
)


class DataValidationError(ValueError):
    """Raised when OASIS-2 data violate an expected schema or invariant."""


def validate_raw_schema(df: pd.DataFrame) -> None:
    """Require the source columns used by deterministic Phase 1 cleaning."""
    missing = sorted(REQUIRED_RAW_COLUMNS.difference(df.columns))
    if missing:
        raise DataValidationError(f"Missing required raw columns: {missing}")


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load an immutable raw CSV and validate its required schema."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Raw OASIS-2 CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    validate_raw_schema(df)
    LOGGER.info(
        "Loaded %d visits from %d subjects",
        len(df),
        df["Subject ID"].nunique(),
    )
    return df


def collapse_to_last_visit(df: pd.DataFrame) -> pd.DataFrame:
    """Select each subject's whole row at its highest numeric visit."""
    validate_raw_schema(df)
    visits = pd.to_numeric(df["Visit"], errors="coerce")
    if visits.isna().any():
        raise DataValidationError("Visit must contain only numeric values")
    indices = visits.groupby(df["Subject ID"], sort=False).idxmax()
    result = df.loc[indices].copy().reset_index(drop=True)
    if result["Subject ID"].duplicated().any():
        raise DataValidationError("Last-visit selection did not produce unique subjects")
    LOGGER.info("Collapsed %d visits to %d subjects", len(df), len(result))
    return result


def clean_longitudinal(df: pd.DataFrame) -> pd.DataFrame:
    """Create the deterministic one-row-per-subject classification table."""
    validate_raw_schema(df)
    selected = collapse_to_last_visit(df).rename(columns=RAW_TO_CLEAN)
    unknown_groups = sorted(set(selected["group"].dropna()) - set(TARGET_MAPPING))
    if unknown_groups or selected["group"].isna().any():
        raise DataValidationError(
            f"Unexpected or missing Group values: {unknown_groups or ['<missing>']}"
        )
    unknown_sexes = sorted(set(selected["sex"].dropna()) - set(SEX_MAPPING))
    if unknown_sexes or selected["sex"].isna().any():
        raise DataValidationError(
            f"Unexpected or missing M/F values: {unknown_sexes or ['<missing>']}"
        )

    selected["dementia"] = selected["group"].map(TARGET_MAPPING).astype("int8")
    selected["sex"] = selected["sex"].map(SEX_MAPPING).astype("string")

    cleaned = selected.drop(columns=list(NON_MODEL_COLUMNS))
    cleaned = cleaned.loc[:, CLEAN_COLUMNS].sort_values("subject_id").reset_index(drop=True)
    validate_subject_table(cleaned)
    LOGGER.info("Cleaned data contains %d subjects and %d columns", *cleaned.shape)
    return cleaned


def validate_subject_table(df: pd.DataFrame) -> None:
    """Validate invariants of the deterministic subject-level table."""
    missing = sorted(set(CLEAN_COLUMNS).difference(df.columns))
    if missing:
        raise DataValidationError(f"Missing cleaned columns: {missing}")
    if df["subject_id"].isna().any() or df["subject_id"].duplicated().any():
        raise DataValidationError("subject_id must be non-null and unique")
    target_values = set(df["dementia"].dropna().unique())
    if df["dementia"].isna().any() or target_values != {0, 1}:
        raise DataValidationError("dementia must contain both binary classes 0 and 1")


def assess_data(df, *, stage):
    """Return a JSON-serializable structural assessment without mutation."""
    assessment = {
        "stage": stage,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": df.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_by_column": {
            column: int(count) for column, count in df.isna().sum().items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }
    subject_column = "subject_id" if "subject_id" in df else "Subject ID"
    if subject_column in df:
        assessment["unique_subjects"] = int(df[subject_column].nunique())
    target_column = "dementia" if "dementia" in df else "Group"
    if target_column in df:
        assessment["target_counts"] = {
            str(label): int(count)
            for label, count in df[target_column].value_counts(dropna=False).items()
        }
    return assessment


def split_subjects(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    stratify: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a reproducible stratified split of unique subject rows."""
    validate_subject_table(df)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1")
    class_counts = df["dementia"].value_counts()
    if (class_counts < 2).any():
        raise DataValidationError("Each target class needs at least two subjects")
    stratify_values = df["dementia"] if stratify else None
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )
    train = train.sort_values("subject_id").reset_index(drop=True)
    test = test.sort_values("subject_id").reset_index(drop=True)
    overlap = set(train["subject_id"]).intersection(test["subject_id"])
    if overlap:
        raise DataValidationError(f"Subjects appear in both splits: {sorted(overlap)[:5]}")
    LOGGER.info("Created stratified split: %d train, %d test", len(train), len(test))
    return train, test


def build_split_summary(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    stratify: bool,
) -> dict[str, Any]:

    overlap = sorted(set(train["subject_id"]).intersection(test["subject_id"]))
    return {
        "test_size": test_size,
        "random_state": random_state,
        "stratified": stratify,
        "train_subjects": int(len(train)),
        "test_subjects": int(len(test)),
        "subject_overlap_count": len(overlap),
        "train_target_counts": {
            str(key): int(value) for key, value in train["dementia"].value_counts().items()
        },
        "test_target_counts": {
            str(key): int(value) for key, value in test["dementia"].value_counts().items()
        },
        "train_missing_by_column": {
            key: int(value) for key, value in train.isna().sum().items()
        },
        "test_missing_by_column": {
            key: int(value) for key, value in test.isna().sum().items()
        },
    }


def run_cleaning_pipeline(
    input_path: str | Path,
    subject_level_path: str | Path,
    train_path: str | Path,
    test_path: str | Path,
    assessment_path: str | Path,
    split_summary_path: str | Path,
    *,
    test_size: float,
    random_state: int,
    stratify: bool,
) -> dict[str, Path]:
    """Assess, deterministically clean, split, and save OASIS-2 data."""
    raw = load_raw(input_path)
    subject_level = clean_longitudinal(raw)
    train, test = split_subjects(
        subject_level,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    paths = {
        "subject_level": Path(subject_level_path),
        "train": Path(train_path),
        "test": Path(test_path),
        "assessment": Path(assessment_path),
        "split_summary": Path(split_summary_path),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    assessment = {
        "raw": assess_data(raw, stage="raw"),
        "subject_level": assess_data(subject_level, stage="subject_level"),
    }
    summary = build_split_summary(
        train,
        test,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    try:
        subject_level.to_csv(paths["subject_level"], index=False)
        train.to_csv(paths["train"], index=False)
        test.to_csv(paths["test"], index=False)
        summary["file_sha256"] = {
            "train": sha256_file(paths["train"]),
            "test": sha256_file(paths["test"]),
        }
        paths["assessment"].write_text(json.dumps(assessment, indent=2), encoding="utf-8")
        paths["split_summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Could not save Phase 1 outputs: {exc}") from exc
    LOGGER.info("Saved deterministic subject table and persistent train/test split")
    return paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments, defaulting paths/params from YAML config."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    config_parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    config_args, remaining_argv = config_parser.parse_known_args()

    paths_config = load_yaml_config(config_args.paths_config)
    model_config = load_yaml_config(config_args.model_config)

    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument("--input", type=Path, default=Path(paths_config["data"]["raw"]))
    parser.add_argument(
        "--subject-level", type=Path, default=Path(paths_config["data"]["cleaned"])
    )
    parser.add_argument("--train", type=Path, default=Path(paths_config["data"]["train"]))
    parser.add_argument("--test", type=Path, default=Path(paths_config["data"]["test"]))
    parser.add_argument(
        "--assessment", type=Path, default=Path(paths_config["outputs"]["data_assessment"])
    )
    parser.add_argument(
        "--split-summary", type=Path, default=Path(paths_config["outputs"]["split_summary"])
    )
    parser.add_argument(
        "--test-size", type=float, default=float(model_config["split"]["test_size"])
    )
    parser.add_argument(
        "--random-state", type=int, default=int(model_config["random_state"])
    )
    parser.add_argument(
        "--stratify",
        action=argparse.BooleanOptionalAction,
        default=bool(model_config["split"]["stratify"]),
    )
    return parser.parse_args(remaining_argv)


def main() -> int:
    """Run the Phase 1 cleaning command-line interface."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    try:
        run_cleaning_pipeline(
            args.input,
            args.subject_level,
            args.train,
            args.test,
            args.assessment,
            args.split_summary,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=args.stratify,
        )
    except (FileNotFoundError, DataValidationError, OSError, ValueError) as exc:
        LOGGER.error("Cleaning failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



# %%
