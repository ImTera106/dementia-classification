"""Tests for the single complete-set held-out evaluation boundary."""

from __future__ import annotations

import ast
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from src.final_evaluate import run_final_evaluation, validate_frozen_experiment_set
from src.freeze_experiment import build_frozen_manifest, manifest_sha256
from src.utils.io import sha256_file
from tests.test_evaluate import evaluation_config, held_out_frame, manifest, model_config

ROOT = Path(__file__).resolve().parents[1]
CLEAN_GIT_STATE = {
    "commit": "abc123",
    "dirty": False,
    "status_sha256": "status",
    "tracked_diff_sha256": "diff",
}


def _condition_manifest(root: Path, condition: str) -> dict:
    value = manifest(root / condition)
    value["training_condition"] = condition
    for record in value["models"]:
        record["training_condition"] = condition
    return value


def _evaluation_configs() -> dict:
    real = evaluation_config()
    augmented = evaluation_config()
    augmented["training_condition"] = "real_plus_synthetic"
    return {"real_only": real, "real_plus_synthetic": augmented}


def _output_paths(root: Path) -> dict:
    return {
        "final_evaluation_dir": root / "final_evaluation",
        "test_metrics": root / "real_metrics.csv",
        "test_predictions": root / "real_predictions.csv",
        "test_confusion_matrices": root / "real_confusion.csv",
        "test_balanced_accuracy_figure": root / "real_balanced.png",
        "test_roc_figure": root / "real_roc.png",
        "synthetic_test_metrics": root / "augmented_metrics.csv",
        "synthetic_test_predictions": root / "augmented_predictions.csv",
        "synthetic_test_confusion_matrices": root / "augmented_confusion.csv",
        "synthetic_test_balanced_accuracy_figure": root / "augmented_balanced.png",
        "synthetic_test_roc_figure": root / "augmented_roc.png",
        "training_condition_comparison": root / "comparison.csv",
        "training_condition_comparison_figure": root / "comparison.png",
    }


def _frozen_fixture(root: Path) -> tuple[dict, Path, dict[str, Path]]:
    test_path = root / "test.csv"
    train_path = root / "train.csv"
    held_out_frame().to_csv(test_path, index=False)
    held_out_frame().to_csv(train_path, index=False)
    config_values = {
        "paths": {"project_root": "."},
        "model": model_config(),
        "real_tuning": {"cross_validation": {"n_splits": 2}},
        "augmented_tuning": {"cross_validation": {"n_splits": 2}},
        "synthesis": {"synthesizer": {"synthetic_to_real_ratio": 1.0}},
        "real_evaluation": evaluation_config(),
        "augmented_evaluation": _evaluation_configs()["real_plus_synthetic"],
    }
    config_paths: dict[str, Path] = {}
    for name, value in config_values.items():
        path = root / "config" / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        config_paths[name] = path
    manifests = {
        condition: _condition_manifest(root, condition)
        for condition in model_config()["training_conditions"]
    }
    with patch("src.freeze_experiment.git_state", return_value=CLEAN_GIT_STATE):
        frozen = build_frozen_manifest(
            condition_manifests=manifests,
            model_config=model_config(),
            train_path=train_path,
            split_summary={"file_sha256": {"test": sha256_file(test_path)}},
            configuration_paths=config_paths,
            project_root=root,
        )
    return frozen, test_path, config_paths


def test_evaluator_has_no_development_imports_or_calls() -> None:
    for relative in ("src/final_evaluate.py", "src/evaluate.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert imported_modules.isdisjoint({"src.train", "src.tune", "src.synthesize"})
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        called_attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert (called_names | called_attributes).isdisjoint(
            {"fit", "GridSearchCV", "RandomizedSearchCV", "generate_synthetic_subjects"}
        )
    evaluate_source = (ROOT / "src/evaluate.py").read_text(encoding="utf-8")
    assert "run_evaluation_pipeline" not in evaluate_source
    assert "pd.read_csv" not in evaluate_source
    for relative in ("src/analyze_release.py", "src/validate.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "generate_saved_predictions" not in source
        assert "load_joblib" not in source
        assert 'paths["data"]["test"]' not in source


def test_incomplete_manifest_set_fails_before_test_is_opened() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        frozen, test_path, _ = _frozen_fixture(root)
        del frozen["condition_manifests"]["real_plus_synthetic"]
        frozen["manifest_sha256"] = manifest_sha256(frozen)
        with patch("src.final_evaluate.git_state", return_value=CLEAN_GIT_STATE), patch(
            "src.final_evaluate.load_real_test_data"
        ) as load_test:
            try:
                run_final_evaluation(
                    test_path,
                    frozen_manifest=frozen,
                    model_config=model_config(),
                    evaluation_configs=_evaluation_configs(),
                    output_paths=_output_paths(root),
                    project_root=root,
                )
            except ValueError as exc:
                assert "all training conditions" in str(exc)
            else:
                raise AssertionError("Incomplete manifests must be rejected")
            load_test.assert_not_called()


def test_changed_frozen_manifest_fails_before_test_is_opened() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        frozen, test_path, _ = _frozen_fixture(root)
        frozen["experiments"][0]["random_state"] = 999
        with patch("src.final_evaluate.load_real_test_data") as load_test:
            try:
                run_final_evaluation(
                    test_path,
                    frozen_manifest=frozen,
                    model_config=model_config(),
                    evaluation_configs=_evaluation_configs(),
                    output_paths=_output_paths(root),
                    project_root=root,
                )
            except ValueError as exc:
                assert "manifest fingerprint mismatch" in str(exc)
            else:
                raise AssertionError("Changed frozen manifest must be rejected")
            load_test.assert_not_called()


def test_complete_frozen_set_is_evaluated_from_one_test_load() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        frozen, test_path, _ = _frozen_fixture(root)
        with patch("src.final_evaluate.git_state", return_value=CLEAN_GIT_STATE):
            validate_frozen_experiment_set(
                frozen, model_config(), _evaluation_configs(), project_root=root
            )
        with patch("src.final_evaluate.git_state", return_value=CLEAN_GIT_STATE), patch(
            "src.final_evaluate.load_real_test_data", return_value=held_out_frame()
        ) as load_test:
            paths = run_final_evaluation(
                test_path,
                frozen_manifest=frozen,
                model_config=model_config(),
                evaluation_configs=_evaluation_configs(),
                output_paths=_output_paths(root),
                project_root=root,
            )
        load_test.assert_called_once_with(test_path)
        assert len(paths) == 3
        assert paths["release_dir"].is_dir()
        assert all(
            path.is_file() for name, path in paths.items() if name != "release_dir"
        )
        canonical = __import__("pandas").read_csv(paths["predictions"])
        assert len(canonical) == 80
        assert canonical["experiment_id"].nunique() == 20


def test_changed_model_fails_before_test_is_opened() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        frozen, test_path, _ = _frozen_fixture(root)
        model_path = Path(frozen["experiments"][0]["model_artifact_path"])
        model_path.write_bytes(model_path.read_bytes() + b"changed")
        with patch("src.final_evaluate.git_state", return_value=CLEAN_GIT_STATE), patch(
            "src.final_evaluate.load_real_test_data"
        ) as load_test:
            try:
                run_final_evaluation(
                    test_path,
                    frozen_manifest=frozen,
                    model_config=model_config(),
                    evaluation_configs=_evaluation_configs(),
                    output_paths=_output_paths(root),
                    project_root=root,
                )
            except ValueError as exc:
                assert "model fingerprint mismatch" in str(exc)
            else:
                raise AssertionError("Changed model must be rejected")
            load_test.assert_not_called()


def test_changed_training_data_fails_before_test_is_opened() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        frozen, test_path, _ = _frozen_fixture(root)
        Path(frozen["training_data"]["path"]).write_text("changed", encoding="utf-8")
        with patch("src.final_evaluate.git_state", return_value=CLEAN_GIT_STATE), patch(
            "src.final_evaluate.load_real_test_data"
        ) as load_test:
            try:
                run_final_evaluation(
                    test_path,
                    frozen_manifest=frozen,
                    model_config=model_config(),
                    evaluation_configs=_evaluation_configs(),
                    output_paths=_output_paths(root),
                    project_root=root,
                )
            except ValueError as exc:
                assert "Training-data fingerprint mismatch" in str(exc)
            else:
                raise AssertionError("Changed training data must be rejected")
            load_test.assert_not_called()


def test_changed_configuration_fails_before_test_is_opened() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        frozen, test_path, config_paths = _frozen_fixture(root)
        config_paths["synthesis"].write_text("changed: true\n", encoding="utf-8")
        with patch("src.final_evaluate.git_state", return_value=CLEAN_GIT_STATE), patch(
            "src.final_evaluate.load_real_test_data"
        ) as load_test:
            try:
                run_final_evaluation(
                    test_path,
                    frozen_manifest=frozen,
                    model_config=model_config(),
                    evaluation_configs=_evaluation_configs(),
                    output_paths=_output_paths(root),
                    project_root=root,
                )
            except ValueError as exc:
                assert "Configuration fingerprint mismatch" in str(exc)
            else:
                raise AssertionError("Changed configuration must be rejected")
            load_test.assert_not_called()


def test_changed_test_fails_after_open_but_before_prediction() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        frozen, test_path, _ = _frozen_fixture(root)
        test_path.write_text(test_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with patch("src.final_evaluate.git_state", return_value=CLEAN_GIT_STATE), patch(
            "src.final_evaluate.generate_saved_predictions"
        ) as predict:
            try:
                run_final_evaluation(
                    test_path,
                    frozen_manifest=frozen,
                    model_config=model_config(),
                    evaluation_configs=_evaluation_configs(),
                    output_paths=_output_paths(root),
                    project_root=root,
                )
            except ValueError as exc:
                assert "test fingerprint mismatch" in str(exc)
            else:
                raise AssertionError("Changed test split must be rejected")
            predict.assert_not_called()


def test_existing_release_fails_before_test_is_opened() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        frozen, test_path, _ = _frozen_fixture(root)
        release = _output_paths(root)["final_evaluation_dir"] / frozen[
            "manifest_sha256"
        ][:16]
        release.mkdir(parents=True)
        with patch("src.final_evaluate.git_state", return_value=CLEAN_GIT_STATE), patch(
            "src.final_evaluate.load_real_test_data"
        ) as load_test:
            try:
                run_final_evaluation(
                    test_path,
                    frozen_manifest=frozen,
                    model_config=model_config(),
                    evaluation_configs=_evaluation_configs(),
                    output_paths=_output_paths(root),
                    project_root=root,
                )
            except FileExistsError as exc:
                assert "already exists" in str(exc)
            else:
                raise AssertionError("An existing canonical release must not be overwritten")
            load_test.assert_not_called()
