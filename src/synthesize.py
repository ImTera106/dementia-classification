"""Generate and validate subject-level synthetic training data without test access."""

from __future__ import annotations

import logging
from importlib.metadata import version
from math import floor
from typing import Any

import pandas as pd
from sdmetrics.reports import DiagnosticReport, QualityReport
from sdv.metadata import Metadata
from sdv.sampling import Condition
from sdv.single_table import GaussianCopulaSynthesizer

from src.features import get_feature_set
from src.utils.io import require_package_version

LOGGER = logging.getLogger(__name__)


class SynthesisError(ValueError):
    """Raised when synthetic configuration or generated data is invalid."""


def resolve_synthesis_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize Phase 8 synthesis settings."""
    if config.get("training_condition") != "real_plus_synthetic":
        raise SynthesisError("Synthetic training condition must be real_plus_synthetic")
    required_version = config.get("required_sdv_version")
    synthesizer = config.get("synthesizer")
    validation = config.get("validation")
    if not isinstance(required_version, str) or not required_version:
        raise SynthesisError("required_sdv_version must be a version string")
    require_package_version("sdv", required_version, context="Phase 8 synthesis")
    if not isinstance(synthesizer, dict) or not isinstance(validation, dict):
        raise SynthesisError("synthesizer and validation must be YAML mappings")
    if synthesizer.get("method") != "gaussian_copula":
        raise SynthesisError("Phase 8 supports gaussian_copula only")
    ratio = float(synthesizer.get("synthetic_to_real_ratio", 0))
    if ratio <= 0:
        raise SynthesisError("synthetic_to_real_ratio must be positive")
    if synthesizer.get("preserve_target_counts") is not True:
        raise SynthesisError("preserve_target_counts must remain true")
    if validation.get("require_unique_subject_ids") is not True:
        raise SynthesisError("require_unique_subject_ids must remain true")
    return {
        "required_sdv_version": required_version,
        "ratio": ratio,
        "enforce_min_max_values": bool(synthesizer["enforce_min_max_values"]),
        "enforce_rounding": bool(synthesizer["enforce_rounding"]),
        "reject_exact_real_matches": bool(validation["reject_exact_real_matches"]),
        "diagnostic_threshold": float(validation["require_sdv_diagnostic_score"]),
    }


def modeling_columns(
    feature_set_name: str,
    feature_sets_config: dict[str, Any],
    *,
    target_column: str,
) -> list[str]:
    """Return the feature-set-specific synthesis schema plus the target."""
    feature_set = get_feature_set(feature_sets_config, feature_set_name)
    return [*feature_set.numeric, *feature_set.categorical, target_column]


def build_metadata(data: pd.DataFrame, *, target_column: str) -> Metadata:
    """Build explicit single-table metadata without modeling subject identifiers."""
    metadata = Metadata.detect_from_dataframe(
        data, table_name="subjects", infer_keys=None
    )
    for column in data.columns:
        if column == target_column or pd.api.types.is_object_dtype(data[column]):
            metadata.update_column(column_name=str(column), sdtype="categorical")
        else:
            metadata.update_column(column_name=str(column), sdtype="numerical")
    metadata.validate()
    return metadata


def _target_sample_counts(target: pd.Series, ratio: float) -> dict[Any, int]:
    """Scale each observed class count while preserving training prevalence."""
    raw = target.value_counts().sort_index()
    exact = raw.astype(float) * ratio
    counts = exact.map(floor).astype(int)
    remainder = int(round(len(target) * ratio)) - int(counts.sum())
    if remainder > 0:
        fractions = (exact - counts).sort_values(ascending=False, kind="stable")
        for label in fractions.index[:remainder]:
            counts.loc[label] += 1
    if (counts < 1).any():
        raise SynthesisError("Every observed target class must receive a synthetic row")
    return counts.to_dict()


def _exact_match_count(real: pd.DataFrame, synthetic: pd.DataFrame) -> int:
    """Count complete synthetic records duplicated from the real training input."""
    comparable = list(real.columns)
    real_keys = real[comparable].astype(object).where(real[comparable].notna(), "<NA>")
    synthetic_keys = synthetic[comparable].astype(object).where(
        synthetic[comparable].notna(), "<NA>"
    )
    merged = synthetic_keys.merge(real_keys.drop_duplicates(), how="inner", on=comparable)
    return int(len(merged))


def _report_properties(report: Any) -> dict[str, float]:
    """Normalize current SDMetrics report properties to a JSON mapping."""
    properties = report.get_properties()
    if isinstance(properties, pd.DataFrame):
        return {
            str(row["Property"]): float(row["Score"])
            for _, row in properties.iterrows()
        }
    return {str(key): float(value) for key, value in properties.items()}


def _sample_without_exact_matches(
    synthesizer: GaussianCopulaSynthesizer,
    real: pd.DataFrame,
    *,
    target_column: str,
    counts: dict[Any, int],
    reject_exact_real_matches: bool,
) -> pd.DataFrame:
    """Conditionally sample exact class counts, excluding duplicates and real copies."""
    accepted: list[pd.DataFrame] = []
    for label, required in counts.items():
        label_rows: list[pd.DataFrame] = []
        remaining = required
        for _ in range(5):
            candidate = synthesizer.sample_from_conditions(
                [
                    Condition(
                        num_rows=max(remaining * 2, remaining + 5),
                        column_values={target_column: label},
                    )
                ]
            )
            pool = pd.concat([*label_rows, candidate], ignore_index=True).drop_duplicates()
            if reject_exact_real_matches:
                joined = pool.merge(
                    real.drop_duplicates(), how="left", on=list(real.columns), indicator=True
                )
                pool = joined.loc[joined["_merge"] == "left_only", real.columns]
            label_rows = [pool]
            remaining = required - len(pool)
            if remaining <= 0:
                accepted.append(pool.iloc[:required].copy())
                break
        else:
            raise SynthesisError(
                f"Could not generate {required} unique nonmatching rows for target {label}"
            )
    return pd.concat(accepted, ignore_index=True)


def generate_synthetic_subjects(
    real_frame: pd.DataFrame,
    feature_set_name: str,
    feature_sets_config: dict[str, Any],
    synthesis_config: dict[str, Any],
    *,
    target_column: str,
    subject_id_column: str,
    id_prefix: str = "SYN",
    evaluate_reports: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit on real training subjects and conditionally sample target-matched rows."""
    settings = resolve_synthesis_settings(synthesis_config)
    columns = modeling_columns(
        feature_set_name, feature_sets_config, target_column=target_column
    )
    missing = sorted(set(columns).difference(real_frame.columns))
    if missing:
        raise SynthesisError(f"Real synthesis frame is missing columns: {missing}")
    real = real_frame.loc[:, columns].copy()
    if real[target_column].isna().any() or set(real[target_column]) != {0, 1}:
        raise SynthesisError("Synthesis target must be complete and contain classes 0 and 1")

    metadata = build_metadata(real, target_column=target_column)
    synthesizer = GaussianCopulaSynthesizer(
        metadata,
        enforce_min_max_values=settings["enforce_min_max_values"],
        enforce_rounding=settings["enforce_rounding"],
    )
    synthesizer.fit(real)
    counts = _target_sample_counts(real[target_column], settings["ratio"])
    synthetic = _sample_without_exact_matches(
        synthesizer,
        real,
        target_column=target_column,
        counts=counts,
        reject_exact_real_matches=settings["reject_exact_real_matches"],
    ).loc[:, columns]
    synthetic[target_column] = synthetic[target_column].astype(real[target_column].dtype)
    expected_rows = sum(counts.values())
    if len(synthetic) != expected_rows:
        raise SynthesisError(
            f"Expected {expected_rows} synthetic rows, generated {len(synthetic)}"
        )
    if synthetic.duplicated().any():
        raise SynthesisError("Synthetic output contains duplicate complete records")
    exact_matches = _exact_match_count(real, synthetic)
    if settings["reject_exact_real_matches"] and exact_matches:
        raise SynthesisError(
            f"Synthetic output contains {exact_matches} exact real-training matches"
        )

    synthetic.insert(
        0,
        subject_id_column,
        [f"{id_prefix}_{feature_set_name}_{index:04d}" for index in range(1, len(synthetic) + 1)],
    )
    if synthetic[subject_id_column].isna().any() or not synthetic[
        subject_id_column
    ].is_unique:
        raise SynthesisError("Synthetic subject identifiers must be unique and complete")
    report = {
        "feature_set": feature_set_name,
        "training_condition": "real_plus_synthetic",
        "real_rows": int(len(real)),
        "synthetic_rows": int(len(synthetic)),
        "target_counts": {str(key): int(value) for key, value in counts.items()},
        "exact_real_matches": exact_matches,
        "versions": {"sdv": version("sdv")},
        "metadata": metadata.to_dict(),
    }
    if evaluate_reports:
        diagnostic = DiagnosticReport()
        diagnostic.generate(
            real_data={"subjects": real},
            synthetic_data={"subjects": synthetic[columns]},
            metadata=metadata.to_dict(),
            verbose=False,
        )
        diagnostic_score = float(diagnostic.get_score())
        if diagnostic_score < settings["diagnostic_threshold"]:
            raise SynthesisError(
                f"SDV diagnostic score {diagnostic_score:.4f} is below required "
                f"{settings['diagnostic_threshold']:.4f}"
            )
        quality = QualityReport()
        quality.generate(
            real_data={"subjects": real},
            synthetic_data={"subjects": synthetic[columns]},
            metadata=metadata.to_dict(),
            verbose=False,
        )
        report.update(
            {
                "diagnostic_score": diagnostic_score,
                "diagnostic_properties": _report_properties(diagnostic),
                "quality_score": float(quality.get_score()),
                "quality_properties": _report_properties(quality),
            }
        )
    return synthetic, report


def combine_real_and_synthetic(
    real_frame: pd.DataFrame,
    synthetic_frame: pd.DataFrame,
    *,
    columns: list[str],
) -> pd.DataFrame:
    """Combine explicitly selected real and synthetic training columns."""
    missing = sorted(
        set(columns).difference(real_frame.columns).union(
            set(columns).difference(synthetic_frame.columns)
        )
    )
    if missing:
        raise SynthesisError(f"Combined training data is missing columns: {missing}")
    return pd.concat(
        [real_frame.loc[:, columns], synthetic_frame.loc[:, columns]],
        ignore_index=True,
    )
