"""Shared configuration, JSON, and trusted local model artifact I/O."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import yaml


def require_package_version(package: str, expected: str, *, context: str) -> str:
    """Require an exact installed distribution version for reproducible artifacts."""
    try:
        actual = version(package)
    except PackageNotFoundError as exc:
        raise ValueError(
            f"{context} requires {package}=={expected}, but it is not installed"
        ) from exc
    if actual != expected:
        raise ValueError(
            f"{context} requires {package}=={expected}, but runtime has {actual}"
        )
    return actual


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping with an actionable path or structure error."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"YAML config must contain a top-level mapping: {config_path}")
    return config


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON mapping from a required local artifact."""
    json_path = Path(path)
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON artifact not found: {json_path}")
    try:
        value = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON artifact {json_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain a mapping: {json_path}")
    return value


def save_json(value: dict[str, Any], path: str | Path) -> Path:
    """Save a JSON mapping using stable, readable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Could not save JSON artifact {output_path}: {exc}") from exc
    return output_path


def save_joblib(value: Any, path: str | Path) -> Path:
    """Serialize a trusted local Python artifact with joblib."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(value, output_path)
    except OSError as exc:
        raise OSError(f"Could not save model artifact {output_path}: {exc}") from exc
    return output_path


def load_joblib(path: str | Path) -> Any:
    """Load a trusted local joblib artifact produced by this project."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
    try:
        return joblib.load(artifact_path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load model artifact {artifact_path}: {exc}") from exc
