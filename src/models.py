"""Construct leakage-safe preprocessing and baseline model pipelines."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.features import get_feature_set


class ModelConfigurationError(ValueError):
    """Raised when model configuration is unsupported or incomplete."""


def build_preprocessor(
    feature_set_name: str,
    preprocessing_config: dict[str, Any],
    feature_sets_config: dict[str, Any],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    """Build unfitted preprocessing for one approved feature set."""
    feature_set = get_feature_set(feature_sets_config, feature_set_name)
    try:
        numeric_strategy = preprocessing_config["numeric_imputer"]["strategy"]
        categorical_strategy = preprocessing_config["categorical_imputer"]["strategy"]
        handle_unknown = preprocessing_config["one_hot_encoder"]["handle_unknown"]
    except (KeyError, TypeError) as exc:
        raise ModelConfigurationError(
            "preprocessing configuration must define numeric/categorical imputation, "
            "and one-hot handling"
        ) from exc
    if not isinstance(scale_numeric, bool):
        raise ModelConfigurationError("scale_numeric must be an explicit boolean")

    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy=numeric_strategy))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=categorical_strategy)),
            ("encoder", OneHotEncoder(handle_unknown=handle_unknown)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(steps=numeric_steps), list(feature_set.numeric)),
            (
                "categorical",
                categorical_pipeline,
                list(feature_set.categorical),
            ),
        ],
        remainder="drop",
    )


def build_baseline_estimator(
    algorithm: str,
    algorithm_config: dict[str, Any],
    *,
    random_state: int,
) -> DummyClassifier | LogisticRegression:
    """Construct one supported baseline estimator from YAML parameters."""
    if algorithm == "dummy":
        try:
            return DummyClassifier(strategy=algorithm_config["strategy"])
        except KeyError as exc:
            raise ModelConfigurationError("dummy.strategy is required") from exc
    if algorithm == "logistic_regression":
        required = {"regularization", "solver", "C", "max_iter"}
        missing = sorted(required.difference(algorithm_config))
        if missing:
            raise ModelConfigurationError(
                f"logistic_regression is missing parameters: {missing}"
            )
        if algorithm_config["regularization"] != "l2":
            raise ModelConfigurationError(
                "Phase 2 logistic_regression.regularization must be l2"
            )
        return LogisticRegression(
            solver=algorithm_config["solver"],
            C=float(algorithm_config["C"]),
            max_iter=int(algorithm_config["max_iter"]),
            random_state=random_state,
        )
    raise ModelConfigurationError(
        f"Unsupported baseline algorithm {algorithm!r}; choose dummy or "
        "logistic_regression"
    )


def build_baseline_pipeline(
    algorithm: str,
    feature_set_name: str,
    *,
    preprocessing_config: dict[str, Any],
    feature_sets_config: dict[str, Any],
    algorithm_config: dict[str, Any],
    random_state: int,
) -> Pipeline:
    """Return an unfitted pipeline so preprocessing is learned per CV train fold."""
    scale_numeric = preprocessing_config.get("scale_numeric")
    if not isinstance(scale_numeric, bool):
        raise ModelConfigurationError(
            "preprocessing.scale_numeric must be an explicit boolean"
        )
    return Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(
                    feature_set_name,
                    preprocessing_config,
                    feature_sets_config,
                    scale_numeric=scale_numeric,
                ),
            ),
            (
                "model",
                build_baseline_estimator(
                    algorithm, algorithm_config, random_state=random_state
                ),
            ),
        ]
    )


def build_candidate_estimator(
    algorithm: str,
    estimator_config: dict[str, Any],
    *,
    random_state: int,
) -> Any:
    """Construct one unfitted Phase 3 candidate estimator."""
    parameters = dict(estimator_config)
    if algorithm == "logistic_regression":
        return LogisticRegression(random_state=random_state, **parameters)
    if algorithm == "svm":
        return SVC(random_state=random_state, **parameters)
    if algorithm == "decision_tree":
        return DecisionTreeClassifier(random_state=random_state, **parameters)
    if algorithm == "random_forest":
        return RandomForestClassifier(random_state=random_state, **parameters)
    if algorithm == "xgboost":
        return XGBClassifier(random_state=random_state, **parameters)
    raise ModelConfigurationError(
        f"Unsupported tuning algorithm {algorithm!r}; choose logistic_regression, "
        "svm, decision_tree, random_forest, or xgboost"
    )


def build_model_pipeline(
    algorithm: str,
    feature_set_name: str,
    *,
    preprocessing_config: dict[str, Any],
    feature_sets_config: dict[str, Any],
    estimator_config: dict[str, Any],
    scale_numeric: bool,
    random_state: int,
) -> Pipeline:
    """Build an unfitted algorithm-specific preprocessing and model pipeline."""
    return Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(
                    feature_set_name,
                    preprocessing_config,
                    feature_sets_config,
                    scale_numeric=scale_numeric,
                ),
            ),
            (
                "model",
                build_candidate_estimator(
                    algorithm,
                    estimator_config,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_tuning_pipeline(
    algorithm: str,
    feature_set_name: str,
    *,
    preprocessing_config: dict[str, Any],
    feature_sets_config: dict[str, Any],
    estimator_config: dict[str, Any],
    scale_numeric: bool,
    random_state: int,
) -> Pipeline:
    """Build a model pipeline for fold-safe Phase 3 hyperparameter search."""
    return build_model_pipeline(
        algorithm,
        feature_set_name,
        preprocessing_config=preprocessing_config,
        feature_sets_config=feature_sets_config,
        estimator_config=estimator_config,
        scale_numeric=scale_numeric,
        random_state=random_state,
    )
