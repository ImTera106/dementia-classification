"""Tests for the read-only Phase 7 report contract."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
REPORT_SOURCE = ROOT / "report" / "report.qmd"
README = ROOT / "README.md"
LOCAL_REPORT_ARTIFACTS = [
    ROOT / "outputs/metrics/test_metrics.csv",
    ROOT / "outputs/metrics/test_metrics_real_plus_synthetic.csv",
]
REQUIRES_REPORT_ARTIFACTS = pytest.mark.skipif(
    not any(path.exists() for path in LOCAL_REPORT_ARTIFACTS),
    reason="saved report artifacts are not present in this source-only checkout",
)


class ReportTests(unittest.TestCase):
    """Verify artifact-driven reporting without model execution."""

    def test_report_code_is_read_only(self) -> None:
        source = REPORT_SOURCE.read_text(encoding="utf-8")
        python_chunks = "\n".join(
            re.findall(r"```\{python\}(.*?)```", source, flags=re.DOTALL)
        )
        forbidden = (
            "from src",
            "import src",
            ".fit(",
            "joblib",
            "pickle",
            "RandomizedSearchCV",
            "GridSearchCV",
            "train_test_split",
        )
        for token in forbidden:
            self.assertNotIn(token, python_chunks)
        self.assertIn("pd.read_csv", python_chunks)
        self.assertIn("read_text", python_chunks)

    @REQUIRES_REPORT_ARTIFACTS
    def test_report_references_required_saved_artifacts(self) -> None:
        source = REPORT_SOURCE.read_text(encoding="utf-8")
        required = (
            "outputs/data/data_assessment.json",
            "outputs/data/split_summary.json",
            "outputs/metrics/baseline_cv_summary.csv",
            "outputs/tuning/tuning_summary.csv",
            "outputs/metrics/test_metrics.csv",
            "outputs/metrics/test_metric_bootstrap_intervals.csv",
            "outputs/tables/feature_set_balanced_accuracy_differences.csv",
            "outputs/tables/permutation_importance.csv",
            "outputs/tables/logistic_coefficients.csv",
            "outputs/tables/tree_shap_importance.csv",
            "outputs/metrics/synthetic_quality.json",
            "outputs/tuning/real_plus_synthetic/tuning_summary.csv",
            "outputs/metrics/test_metrics_real_plus_synthetic.csv",
            "outputs/metrics/training_condition_comparison.csv",
            "outputs/metrics/test_metric_bootstrap_intervals_real_plus_synthetic.csv",
            "outputs/tables/training_condition_balanced_accuracy_differences.csv",
        )
        for relative in required:
            self.assertIn(relative, source)
            self.assertTrue((ROOT / relative).is_file(), relative)

    @REQUIRES_REPORT_ARTIFACTS
    def test_readme_results_match_saved_metrics(self) -> None:
        readme = README.read_text(encoding="utf-8")
        report = REPORT_SOURCE.read_text(encoding="utf-8")
        metrics = pd.read_csv(ROOT / "outputs/metrics/test_metrics.csv")
        best = metrics.groupby("feature_set")["balanced_accuracy"].max()
        self.assertIn(f"{best['clinical']:.3f}", readme)
        self.assertIn(f"{best['clinical_imaging']:.3f}", readme)
        self.assertIn(f"{best['clinical']:.3f}", report)
        self.assertIn(f"{best['clinical_imaging']:.3f}", report)
        self.assertIn("Python 3.12", readme)
        self.assertIn("not suitable for clinical use", readme)

    @REQUIRES_REPORT_ARTIFACTS
    def test_report_image_sources_exist(self) -> None:
        source = REPORT_SOURCE.read_text(encoding="utf-8")
        relative_images = re.findall(r"!\[[^]]*\]\((\.\./outputs/[^)]+\.png)\)", source)
        self.assertEqual(len(relative_images), 12)
        for relative in relative_images:
            self.assertTrue((REPORT_SOURCE.parent / relative).resolve().is_file(), relative)

    @REQUIRES_REPORT_ARTIFACTS
    def test_phase8_headlines_match_saved_augmented_metrics(self) -> None:
        source = REPORT_SOURCE.read_text(encoding="utf-8")
        readme = README.read_text(encoding="utf-8")
        metrics = pd.read_csv(
            ROOT / "outputs/metrics/test_metrics_real_plus_synthetic.csv"
        )
        best = metrics.groupby("feature_set")["balanced_accuracy"].max()
        for text in (source, readme):
            self.assertIn(f"{best['clinical']:.3f}", text)
            self.assertIn(f"{best['clinical_imaging']:.3f}", text)
        differences = pd.read_csv(
            ROOT
            / "outputs/tables/training_condition_balanced_accuracy_differences.csv"
        )
        excluded = differences.loc[~differences["interval_includes_zero"]]
        self.assertEqual(len(excluded), 1)
        row = excluded.iloc[0]
        self.assertEqual(row["algorithm"], "decision_tree")
        self.assertEqual(row["feature_set"], "clinical_imaging")
