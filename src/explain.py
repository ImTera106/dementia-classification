"""Explain fixed Phase 4 models without refitting or selecting new models."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_is_fitted

from src.features import split_features_target
from src.tune import REQUIRED_ALGORITHMS
from src.utils.io import (
    load_joblib,
    load_json,
    load_yaml_config,
    require_package_version,
)
from src.utils.plotting import (
    plot_importance_matrix,
    plot_logistic_coefficients,
    plot_shap_importance,
)

# Import after plotting configures a writable non-interactive Matplotlib cache.
import shap  # noqa: E402

LOGGER = logging.getLogger(__name__)

DEFAULT_PATHS_CONFIG: Final[Path] = Path("config/paths.yaml")
DEFAULT_MODEL_CONFIG: Final[Path] = Path("config/model_config.yaml")
DEFAULT_EXPLAINABILITY_CONFIG: Final[Path] = Path(
    "config/explainability_config.yaml"
)
FEATURE_SETS: Final[tuple[str, str]] = ("clinical", "clinical_imaging")


class ExplainabilityError(ValueError):
    """Raised when saved models or explanation settings are invalid."""


def resolve_explainability_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the approved Phase 6 explanation settings."""
    permutation = config.get("permutation_importance")
    logistic = config.get("logistic_coefficients")
    tree_shap = config.get("tree_shap")
    figures = config.get("figures")
    if not all(
        isinstance(value, dict)
        for value in (permutation, logistic, tree_shap, figures)
    ):
        raise ExplainabilityError(
            "permutation_importance, logistic_coefficients, tree_shap, and "
            "figures must be mappings"
        )
    if config.get("training_condition") != "real_only":
        raise ExplainabilityError("Phase 6 training_condition must be real_only")
    if permutation.get("scoring") != "balanced_accuracy":
        raise ExplainabilityError(
            "Permutation importance must use balanced_accuracy"
        )
    n_repeats = int(permutation.get("n_repeats", 0))
    if n_repeats < 1:
        raise ExplainabilityError("Permutation n_repeats must be positive")
    if logistic.get("algorithm") != "logistic_regression":
        raise ExplainabilityError("Coefficient explanations require logistic_regression")
    if logistic.get("numeric_unit") != "one_training_standard_deviation":
        raise ExplainabilityError(
            "Numeric logistic coefficients must use training-standard-deviation units"
        )
    if logistic.get("categorical_contrast") != "male_minus_female":
        raise ExplainabilityError("Sex coefficient contrast must be male_minus_female")
    shap_algorithms = tree_shap.get("algorithms")
    expected_tree_algorithms = {"decision_tree", "random_forest", "xgboost"}
    if not isinstance(shap_algorithms, list) or set(shap_algorithms) != expected_tree_algorithms:
        raise ExplainabilityError(
            "Tree SHAP algorithms must be decision_tree, random_forest, and xgboost"
        )
    if tree_shap.get("class_label") != 1:
        raise ExplainabilityError("Tree SHAP must explain class label 1")
    if tree_shap.get("aggregate_encoded_features") is not True:
        raise ExplainabilityError("Encoded SHAP features must be aggregated")
    if tree_shap.get("check_additivity") is not True:
        raise ExplainabilityError("Tree SHAP additivity checking must remain enabled")
    expected_subject_count = int(config.get("expected_subject_count", 0))
    if expected_subject_count < 2:
        raise ExplainabilityError("expected_subject_count must be at least 2")
    return {
        "training_condition": "real_only",
        "expected_subject_count": expected_subject_count,
        "random_state": int(config["random_state"]),
        "n_repeats": n_repeats,
        "n_jobs": int(permutation["n_jobs"]),
        "scoring": "balanced_accuracy",
        "shap_algorithms": list(shap_algorithms),
        "dpi": int(figures["dpi"]),
    }


def validate_explanation_contract(
    manifest: dict[str, Any],
    model_config: dict[str, Any],
    settings: dict[str, Any],
    test_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Require canonical runtime, models, feature sets, and held-out subjects."""
    versions = manifest.get("versions")
    if not isinstance(versions, dict) or not isinstance(
        versions.get("scikit_learn"), str
    ):
        raise ExplainabilityError(
            "Model manifest must record the scikit-learn version"
        )
    if not isinstance(versions.get("xgboost"), str):
        raise ExplainabilityError("Model manifest must record the XGBoost version")
    require_package_version(
        "scikit-learn",
        versions["scikit_learn"],
        context="Saved-model explainability",
    )
    require_package_version(
        "xgboost", versions["xgboost"], context="Saved-model explainability"
    )
    if manifest.get("training_condition") != settings["training_condition"]:
        raise ExplainabilityError("Model manifest training condition must be real_only")
    feature_sets = model_config.get("feature_sets")
    if not isinstance(feature_sets, dict) or tuple(feature_sets) != FEATURE_SETS:
        raise ExplainabilityError(
            "Model configuration must define clinical then clinical_imaging"
        )
    if len(test_frame) != settings["expected_subject_count"]:
        raise ExplainabilityError(
            f"Expected {settings['expected_subject_count']} held-out subjects, "
            f"found {len(test_frame)}"
        )
    records = manifest.get("models")
    if not isinstance(records, list):
        raise ExplainabilityError("Model manifest models must be a list")
    expected = {
        (algorithm, feature_set)
        for feature_set in FEATURE_SETS
        for algorithm in REQUIRED_ALGORITHMS
    }
    actual = {
        (str(record.get("algorithm")), str(record.get("feature_set")))
        for record in records
        if isinstance(record, dict)
    }
    if len(records) != 10 or actual != expected:
        raise ExplainabilityError("Manifest must contain the exact 10 final models")
    return records


def _experiment_data(
    test_frame: pd.DataFrame,
    feature_set: str,
    model_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    split = model_config["split"]
    return split_features_target(
        test_frame,
        feature_set,
        model_config["feature_sets"],
        target_column=str(split["target_column"]),
        subject_id_column=str(split["subject_id_column"]),
    )


def calculate_permutation_importance(
    test_frame: pd.DataFrame,
    records: list[dict[str, Any]],
    model_config: dict[str, Any],
    settings: dict[str, Any],
) -> pd.DataFrame:
    """Calculate original-feature importance for all fixed models without fitting."""
    rows: list[dict[str, Any]] = []
    for record in records:
        algorithm = str(record["algorithm"])
        feature_set = str(record["feature_set"])
        features, target = _experiment_data(test_frame, feature_set, model_config)
        if list(features.columns) != record.get("feature_columns"):
            raise ExplainabilityError(
                f"Feature columns do not match manifest for {(algorithm, feature_set)}"
            )
        pipeline = load_joblib(record["path"])
        check_is_fitted(pipeline)
        result = permutation_importance(
            pipeline,
            features,
            target,
            scoring=settings["scoring"],
            n_repeats=settings["n_repeats"],
            random_state=settings["random_state"],
            n_jobs=settings["n_jobs"],
        )
        for index, feature in enumerate(features.columns):
            rows.append(
                {
                    "algorithm": algorithm,
                    "feature_set": feature_set,
                    "training_condition": "real_only",
                    "feature": feature,
                    "scoring": settings["scoring"],
                    "importance_mean": float(result.importances_mean[index]),
                    "importance_std": float(result.importances_std[index]),
                    "n_repeats": settings["n_repeats"],
                }
            )
    return pd.DataFrame.from_records(rows)


def calculate_logistic_coefficients(
    records: list[dict[str, Any]], model_config: dict[str, Any]
) -> pd.DataFrame:
    """Extract standardized numeric effects and the identifiable sex contrast."""
    rows: list[dict[str, Any]] = []
    for record in records:
        if record["algorithm"] != "logistic_regression":
            continue
        feature_set = str(record["feature_set"])
        pipeline = load_joblib(record["path"])
        check_is_fitted(pipeline)
        preprocessor = pipeline.named_steps["preprocess"]
        estimator = pipeline.named_steps["model"]
        names = list(preprocessor.get_feature_names_out())
        coefficients = np.asarray(estimator.coef_).reshape(-1)
        if len(names) != len(coefficients):
            raise ExplainabilityError(
                f"Coefficient/feature mismatch for logistic {feature_set}"
            )
        coefficient_map = dict(zip(names, coefficients, strict=True))
        numeric_features = model_config["feature_sets"][feature_set]["numeric"]
        for feature in numeric_features:
            coefficient = float(coefficient_map[f"numeric__{feature}"])
            rows.append(
                {
                    "algorithm": "logistic_regression",
                    "feature_set": feature_set,
                    "training_condition": "real_only",
                    "feature": feature,
                    "term": feature,
                    "coefficient": coefficient,
                    "odds_ratio": float(np.exp(coefficient)),
                    "unit": "one_training_standard_deviation",
                }
            )
        try:
            sex_contrast = float(
                coefficient_map["categorical__sex_male"]
                - coefficient_map["categorical__sex_female"]
            )
        except KeyError as exc:
            raise ExplainabilityError(
                "Logistic pipeline must contain female and male sex indicators"
            ) from exc
        rows.append(
            {
                "algorithm": "logistic_regression",
                "feature_set": feature_set,
                "training_condition": "real_only",
                "feature": "sex",
                "term": "sex_male_vs_female",
                "coefficient": sex_contrast,
                "odds_ratio": float(np.exp(sex_contrast)),
                "unit": "male_vs_female",
            }
        )
    return pd.DataFrame.from_records(rows)


def _original_feature_name(transformed_name: str) -> str:
    """Map one fitted preprocessor output back to its original feature."""
    if transformed_name.startswith("numeric__"):
        return transformed_name.removeprefix("numeric__")
    if transformed_name.startswith("categorical__sex_"):
        return "sex"
    raise ExplainabilityError(f"Unsupported transformed feature: {transformed_name}")


def _positive_class_shap(
    estimator: Any, transformed: np.ndarray, *, check_additivity: bool
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return class-1 SHAP values and base values with an additivity check."""
    explanation = shap.TreeExplainer(estimator)(
        transformed, check_additivity=check_additivity
    )
    values = np.asarray(explanation.values)
    base_values = np.asarray(explanation.base_values)
    if values.ndim == 3:
        if values.shape[2] != 2:
            raise ExplainabilityError("Tree SHAP must expose two binary outputs")
        values = values[:, :, 1]
        base_values = base_values[:, 1]
        expected = np.asarray(estimator.predict_proba(transformed))[:, 1]
        output_scale = "class_1_probability"
    elif values.ndim == 2:
        base_values = np.broadcast_to(base_values, len(transformed)).astype(float)
        try:
            expected = np.asarray(
                estimator.predict(transformed, output_margin=True)
            )
            output_scale = "raw_margin"
        except TypeError as exc:
            raise ExplainabilityError(
                "Two-dimensional Tree SHAP output requires raw-margin prediction"
            ) from exc
    else:
        raise ExplainabilityError(f"Unexpected Tree SHAP shape: {values.shape}")
    reconstructed = base_values + values.sum(axis=1)
    if not np.allclose(reconstructed, expected, rtol=1e-5, atol=1e-6):
        raise ExplainabilityError("Tree SHAP additivity check failed")
    return values.astype(float), base_values.astype(float), output_scale


def calculate_tree_shap(
    test_frame: pd.DataFrame,
    records: list[dict[str, Any]],
    model_config: dict[str, Any],
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate aggregated original-feature SHAP values for six tree models."""
    value_rows: list[dict[str, Any]] = []
    subject_column = str(model_config["split"]["subject_id_column"])
    for record in records:
        algorithm = str(record["algorithm"])
        if algorithm not in settings["shap_algorithms"]:
            continue
        feature_set = str(record["feature_set"])
        features, _ = _experiment_data(test_frame, feature_set, model_config)
        pipeline = load_joblib(record["path"])
        check_is_fitted(pipeline)
        preprocessor = pipeline.named_steps["preprocess"]
        estimator = pipeline.named_steps["model"]
        transformed = preprocessor.transform(features)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed, dtype=float)
        transformed_names = list(preprocessor.get_feature_names_out())
        if transformed.shape[1] != len(transformed_names):
            raise ExplainabilityError(
                f"SHAP feature mismatch for {(algorithm, feature_set)}"
            )
        shap_values, base_values, output_scale = _positive_class_shap(
            estimator, transformed, check_additivity=True
        )
        original_names = [_original_feature_name(name) for name in transformed_names]
        subject_ids = test_frame[subject_column].astype(str).reset_index(drop=True)
        for row_index, subject_id in enumerate(subject_ids):
            for feature in dict.fromkeys(original_names):
                columns = [
                    index for index, name in enumerate(original_names) if name == feature
                ]
                value_rows.append(
                    {
                        "algorithm": algorithm,
                        "feature_set": feature_set,
                        "training_condition": "real_only",
                        "subject_id": subject_id,
                        "feature": feature,
                        "shap_value": float(shap_values[row_index, columns].sum()),
                        "base_value": float(base_values[row_index]),
                        "output_scale": output_scale,
                    }
                )
    values = pd.DataFrame.from_records(value_rows)
    importance = (
        values.assign(absolute_shap=lambda frame: frame["shap_value"].abs())
        .groupby(
            [
                "algorithm",
                "feature_set",
                "training_condition",
                "feature",
                "output_scale",
            ],
            as_index=False,
            sort=True,
        )["absolute_shap"]
        .mean()
        .rename(columns={"absolute_shap": "mean_absolute_shap"})
    )
    return importance, values


def save_explanation_outputs(
    permutation: pd.DataFrame,
    coefficients: pd.DataFrame,
    shap_importance: pd.DataFrame,
    shap_values: pd.DataFrame,
    *,
    output_paths: dict[str, Any],
    dpi: int,
) -> dict[str, Path]:
    """Save explanation tables and model-comparison visualizations."""
    paths = {
        "permutation": Path(output_paths["permutation_importance"]),
        "coefficients": Path(output_paths["logistic_coefficients"]),
        "shap_importance": Path(output_paths["tree_shap_importance"]),
        "shap_values": Path(output_paths["tree_shap_values"]),
        "permutation_clinical_figure": Path(
            output_paths["permutation_importance_clinical_figure"]
        ),
        "permutation_imaging_figure": Path(
            output_paths["permutation_importance_clinical_imaging_figure"]
        ),
        "coefficient_figure": Path(output_paths["logistic_coefficients_figure"]),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    permutation.to_csv(paths["permutation"], index=False)
    coefficients.to_csv(paths["coefficients"], index=False)
    shap_importance.to_csv(paths["shap_importance"], index=False)
    shap_values.to_csv(paths["shap_values"], index=False)
    maximum_absolute_importance = float(
        permutation["importance_mean"].abs().max()
    )
    if maximum_absolute_importance == 0:
        maximum_absolute_importance = 1.0
    global_min = -maximum_absolute_importance
    global_max = maximum_absolute_importance
    plot_importance_matrix(
        permutation,
        paths["permutation_clinical_figure"],
        feature_set="clinical",
        value_column="importance_mean",
        title="Permutation-importance comparison",
        color_label="Balanced-accuracy decrease",
        vmin=global_min,
        vmax=global_max,
        dpi=dpi,
    )
    plot_importance_matrix(
        permutation,
        paths["permutation_imaging_figure"],
        feature_set="clinical_imaging",
        value_column="importance_mean",
        title="Permutation-importance comparison",
        color_label="Balanced-accuracy decrease",
        vmin=global_min,
        vmax=global_max,
        dpi=dpi,
    )
    plot_logistic_coefficients(coefficients, paths["coefficient_figure"], dpi=dpi)
    shap_root = Path(output_paths["tree_shap_figures_dir"])
    for (algorithm, feature_set), _ in shap_importance.groupby(
        ["algorithm", "feature_set"], sort=True
    ):
        key = f"shap_{algorithm}_{feature_set}"
        path = shap_root / f"{algorithm}_{feature_set}.png"
        paths[key] = plot_shap_importance(
            shap_importance,
            path,
            algorithm=str(algorithm),
            feature_set=str(feature_set),
            dpi=dpi,
        )
    return paths


def run_explanation_pipeline(
    test_path: str | Path,
    *,
    manifest: dict[str, Any],
    model_config: dict[str, Any],
    explainability_config: dict[str, Any],
    output_paths: dict[str, Any],
) -> dict[str, Path]:
    """Run fixed-model explanations and save all Phase 6 artifacts."""
    path = Path(test_path)
    if not path.is_file():
        raise FileNotFoundError(f"Real held-out partition not found: {path}")
    test_frame = pd.read_csv(path)
    settings = resolve_explainability_settings(explainability_config)
    records = validate_explanation_contract(
        manifest, model_config, settings, test_frame
    )
    permutation = calculate_permutation_importance(
        test_frame, records, model_config, settings
    )
    coefficients = calculate_logistic_coefficients(records, model_config)
    shap_importance, shap_values = calculate_tree_shap(
        test_frame, records, model_config, settings
    )
    paths = save_explanation_outputs(
        permutation,
        coefficients,
        shap_importance,
        shap_values,
        output_paths=output_paths,
        dpi=settings["dpi"],
    )
    LOGGER.info(
        "Saved explanations for 10 permutation, 2 logistic, and 6 tree experiments"
    )
    return paths


def parse_args() -> argparse.Namespace:
    """Parse configuration paths for fixed-model explainability."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument(
        "--explainability-config",
        type=Path,
        default=DEFAULT_EXPLAINABILITY_CONFIG,
    )
    return parser.parse_args()


def main() -> int:
    """Explain fixed models without fitting or changing the experiment design."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    try:
        paths = load_yaml_config(args.paths_config)
        run_explanation_pipeline(
            paths["data"]["test"],
            manifest=load_json(paths["model_manifest"]),
            model_config=load_yaml_config(args.model_config),
            explainability_config=load_yaml_config(args.explainability_config),
            output_paths=paths["outputs"],
        )
    except (
        ExplainabilityError,
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        LOGGER.error("Explainability failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
