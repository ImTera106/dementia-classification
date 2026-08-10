"""Final repository-level contracts for reproducibility and data safety."""

from __future__ import annotations

import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = {
    "logistic_regression", "svm", "decision_tree", "random_forest", "xgboost"
}
CONDITIONS = {"real_only", "real_plus_synthetic"}


def skip_when_all_absent(paths: list[Path], reason: str):
    """Skip local-artifact contracts only in a source-only checkout."""
    return pytest.mark.skipif(not any(path.exists() for path in paths), reason=reason)


REQUIRES_MODEL_ARTIFACTS = skip_when_all_absent(
    [
        ROOT / "models/real_only/model_manifest.json",
        ROOT / "models/real_plus_synthetic/model_manifest.json",
    ],
    "saved model artifacts are not present in this source-only checkout",
)
REQUIRES_DATA_ARTIFACTS = skip_when_all_absent(
    [
        ROOT / "data/processed/train_real.csv",
        ROOT / "data/processed/test_real.csv",
        ROOT / "data/synthetic/train_synthetic_clinical.csv",
        ROOT / "data/synthetic/train_synthetic_clinical_imaging.csv",
    ],
    "processed and synthetic data are not present in this source-only checkout",
)
REQUIRES_METRIC_ARTIFACTS = skip_when_all_absent(
    [ROOT / "outputs/metrics/training_condition_comparison.csv"],
    "saved metric artifacts are not present in this source-only checkout",
)


def load_yaml(relative: str) -> dict:
    value = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@REQUIRES_MODEL_ARTIFACTS
def test_manifests_cover_exact_experiment_grid_and_yaml_features() -> None:
    model_config = load_yaml("config/model_config.yaml")
    paths = load_yaml("config/paths.yaml")
    manifests = {
        "real_only": ROOT / paths["model_manifest"],
        "real_plus_synthetic": ROOT / paths["synthetic_model_manifest"],
    }
    expected = {
        (algorithm, feature_set)
        for algorithm in ALGORITHMS
        for feature_set in model_config["feature_sets"]
    }
    for condition, path in manifests.items():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["training_condition"] == condition
        assert manifest["versions"]["scikit_learn"] == version("scikit-learn")
        assert manifest["versions"]["xgboost"] == version("xgboost")
        records = manifest["models"]
        assert {(r["algorithm"], r["feature_set"]) for r in records} == expected
        for record in records:
            definition = model_config["feature_sets"][record["feature_set"]]
            expected_columns = [*definition["numeric"], *definition["categorical"]]
            assert record["training_condition"] == condition
            assert record["feature_columns"] == expected_columns
            assert (ROOT / record["path"]).is_file()


@REQUIRES_DATA_ARTIFACTS
def test_subject_boundaries_and_synthetic_counts_remain_intact() -> None:
    real_train = pd.read_csv(ROOT / "data/processed/train_real.csv")
    real_test = pd.read_csv(ROOT / "data/processed/test_real.csv")
    assert len(real_train) == 120 and len(real_test) == 30
    assert set(real_train["subject_id"]).isdisjoint(real_test["subject_id"])
    for feature_set in ("clinical", "clinical_imaging"):
        synthetic = pd.read_csv(
            ROOT / f"data/synthetic/train_synthetic_{feature_set}.csv"
        )
        assert len(synthetic) == 120
        assert synthetic["subject_id"].is_unique
        assert synthetic["dementia"].value_counts().sort_index().to_dict() == {
            0: 58,
            1: 62,
        }
        assert set(synthetic["subject_id"]).isdisjoint(real_train["subject_id"])
        assert set(synthetic["subject_id"]).isdisjoint(real_test["subject_id"])


@REQUIRES_METRIC_ARTIFACTS
def test_saved_metrics_cover_twenty_experiments_without_ordinary_accuracy() -> None:
    comparison = pd.read_csv(
        ROOT / "outputs/metrics/training_condition_comparison.csv"
    )
    assert len(comparison) == 20
    assert set(comparison["algorithm"]) == ALGORITHMS
    assert set(comparison["feature_set"]) == {"clinical", "clinical_imaging"}
    assert set(comparison["training_condition"]) == CONDITIONS
    assert "accuracy" not in comparison.columns
    assert not comparison.duplicated(
        ["algorithm", "feature_set", "training_condition"]
    ).any()
    intervals = pd.read_csv(
        ROOT
        / "outputs/metrics/test_metric_bootstrap_intervals_real_plus_synthetic.csv"
    )
    differences = pd.read_csv(
        ROOT / "outputs/tables/training_condition_balanced_accuracy_differences.csv"
    )
    assert len(intervals) == 60
    assert set(intervals["training_condition"]) == {"real_plus_synthetic"}
    assert set(intervals["metric"]) == {
        "balanced_accuracy",
        "roc_auc",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
    }
    assert len(differences) == 10
    assert not differences.duplicated(["algorithm", "feature_set"]).any()
    augmented = comparison.loc[
        comparison["training_condition"] == "real_plus_synthetic"
    ].set_index(["algorithm", "feature_set"])
    paired = differences.set_index(["algorithm", "feature_set"])
    pd.testing.assert_series_equal(
        paired["estimate"].sort_index(),
        augmented["balanced_accuracy_delta_vs_real_only"].sort_index(),
        check_names=False,
    )


def test_repository_paths_are_relative_and_have_no_user_home_literals() -> None:
    paths = load_yaml("config/paths.yaml")

    def strings(value: object):
        if isinstance(value, dict):
            for nested in value.values():
                yield from strings(nested)
        elif isinstance(value, str):
            yield value

    assert all(not Path(value).is_absolute() for value in strings(paths))
    checked = [
        *ROOT.glob("src/**/*.py"),
        *ROOT.glob("config/*.yaml"),
        ROOT / "README.md",
        ROOT / "AGENTS.md",
        ROOT / "report/report.qmd",
    ]
    for path in checked:
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text
        assert "C:\\Users\\" not in text


def test_sensitive_and_generated_artifacts_are_gitignored() -> None:
    examples = [
        "data/raw/longitudinal.csv",
        "data/processed/train_real.csv",
        "data/synthetic/train_synthetic_clinical.csv",
        "models/real_only/clinical/logistic_regression.joblib",
        "models/real_only/model_manifest.json",
        "outputs/predictions/test_predictions.csv",
        "report/report.html",
        ".vscode/settings.json",
    ]
    for relative in examples:
        result = subprocess.run(
            ["git", "check-ignore", "-q", relative], cwd=ROOT, check=False
        )
        assert result.returncode == 0, relative
    assert not (ROOT / "report/report.html").exists()


def test_documented_python_entry_points_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    modules = (
        "clean", "baseline", "tune", "train", "evaluate", "validate",
        "explain", "phase8", "predict",
    )
    for module in modules:
        assert (ROOT / f"src/{module}.py").is_file()
        assert f"python -m src.{module}" in readme
