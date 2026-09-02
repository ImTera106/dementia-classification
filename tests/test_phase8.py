"""Tests enforcing the development-only Phase 8 boundary."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from src.phase8 import run_phase8

ROOT = Path(__file__).resolve().parents[1]


def test_development_interface_cannot_receive_test_or_evaluation_inputs() -> None:
    parameters = inspect.signature(run_phase8).parameters
    assert "test_path" not in parameters
    assert "evaluation_config" not in parameters


def test_development_module_has_no_test_or_evaluation_dependencies() -> None:
    source = (ROOT / "src/phase8.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "src.evaluate" not in imported_modules
    accessed_keys = {
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    assert "test" not in accessed_keys
