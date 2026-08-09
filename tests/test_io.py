"""Tests for shared configuration and model artifact I/O."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.utils.io import (
    load_joblib,
    load_json,
    load_yaml_config,
    save_joblib,
    save_json,
)


class IoTests(unittest.TestCase):
    """Verify typed local artifact round trips and actionable failures."""

    def test_yaml_json_and_joblib_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            yaml_path = root / "config.yaml"
            yaml_path.write_text("value: 7\n", encoding="utf-8")
            self.assertEqual(load_yaml_config(yaml_path), {"value": 7})

            json_path = save_json({"value": [1, 2]}, root / "value.json")
            self.assertEqual(load_json(json_path), {"value": [1, 2]})

            model_path = save_joblib({"fitted": True}, root / "model.joblib")
            self.assertEqual(load_joblib(model_path), {"fitted": True})

    def test_missing_artifact_is_reported(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_json("does-not-exist.json")
