"""Tests for development-side experiment freezing."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import patch

import pytest

from src.freeze_experiment import FreezeError, build_frozen_manifest

ROOT = Path(__file__).resolve().parents[1]


def test_freeze_interface_cannot_receive_test_path() -> None:
    assert "test_path" not in inspect.signature(build_frozen_manifest).parameters


def test_freeze_module_never_opens_configured_test_data() -> None:
    source = (ROOT / "src/freeze_experiment.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert 'paths["data"]["test"]' not in source
    function_calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "load_real_test_data" not in function_calls


def test_dirty_worktree_is_rejected_by_default() -> None:
    with patch(
        "src.freeze_experiment.git_state",
        return_value={"commit": "abc", "dirty": True},
    ), pytest.raises(FreezeError, match="dirty Git worktree"):
        build_frozen_manifest(
            condition_manifests={},
            model_config={},
            train_path="unused",
            split_summary={},
            configuration_paths={},
            project_root=ROOT,
        )
