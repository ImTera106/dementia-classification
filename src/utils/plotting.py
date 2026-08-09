"""Shared non-interactive plots for saved model evaluation artifacts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_MPL_CACHE = Path(tempfile.gettempdir()) / "ad-matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402


def plot_balanced_accuracy(
    metrics: pd.DataFrame, path: str | Path, *, dpi: int
) -> Path:
    """Plot held-out balanced accuracy by algorithm and feature set."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = metrics.pivot(
        index="algorithm", columns="feature_set", values="balanced_accuracy"
    )
    axis = pivot.plot(kind="bar", figsize=(9, 5), ylim=(0, 1), rot=25)
    axis.set_title("Held-out balanced accuracy")
    axis.set_xlabel("Algorithm")
    axis.set_ylabel("Balanced accuracy")
    axis.legend(title="Feature set")
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(axis.figure)
    return output_path


def plot_roc_curves(
    predictions: pd.DataFrame, path: str | Path, *, dpi: int
) -> Path:
    """Plot held-out ROC curves in one panel per feature set."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_sets = list(dict.fromkeys(predictions["feature_set"]))
    figure, axes = plt.subplots(
        1, len(feature_sets), figsize=(7 * len(feature_sets), 5), squeeze=False
    )
    for axis, feature_set in zip(axes[0], feature_sets, strict=True):
        feature_predictions = predictions.loc[
            predictions["feature_set"] == feature_set
        ]
        for algorithm, group in feature_predictions.groupby("algorithm", sort=False):
            false_positive_rate, true_positive_rate, _ = roc_curve(
                group["target"], group["score"]
            )
            axis.plot(false_positive_rate, true_positive_rate, label=algorithm)
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        axis.set_title(feature_set)
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Held-out ROC curves")
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_training_condition_comparison(
    metrics: pd.DataFrame, path: str | Path, *, dpi: int
) -> Path:
    """Plot held-out balanced accuracy across feature and training conditions."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    required = {"algorithm", "feature_set", "training_condition", "balanced_accuracy"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"Training-condition comparison is missing columns: {missing}")
    data = metrics.copy()
    data["condition"] = (
        data["feature_set"].str.replace("_", " ")
        + " | "
        + data["training_condition"].str.replace("_", " ")
    )
    pivot = data.pivot(
        index="algorithm", columns="condition", values="balanced_accuracy"
    )
    axis = pivot.plot(kind="bar", figsize=(11, 6), ylim=(0, 1), rot=25)
    axis.set_title("Held-out balanced accuracy by training condition")
    axis.set_xlabel("Algorithm")
    axis.set_ylabel("Balanced accuracy")
    axis.legend(title="Feature set | training condition", fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(axis.figure)
    return output_path


def plot_balanced_accuracy_intervals(
    intervals: pd.DataFrame, path: str | Path, *, dpi: int
) -> Path:
    """Plot held-out balanced accuracy with bootstrap percentile intervals."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = intervals.loc[intervals["metric"] == "balanced_accuracy"].copy()
    data["label"] = data["algorithm"] + " | " + data["feature_set"]
    positions = np.arange(len(data))
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.errorbar(
        data["estimate"],
        positions,
        xerr=np.vstack(
            [
                data["estimate"] - data["lower_bound"],
                data["upper_bound"] - data["estimate"],
            ]
        ),
        fmt="o",
        capsize=3,
    )
    axis.set_yticks(positions, data["label"])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Balanced accuracy")
    axis.set_title("Held-out balanced accuracy with bootstrap intervals")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_feature_set_differences(
    differences: pd.DataFrame, path: str | Path, *, dpi: int
) -> Path:
    """Plot paired imaging-minus-clinical balanced-accuracy differences."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.arange(len(differences))
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.errorbar(
        differences["estimate"],
        positions,
        xerr=np.vstack(
            [
                differences["estimate"] - differences["lower_bound"],
                differences["upper_bound"] - differences["estimate"],
            ]
        ),
        fmt="o",
        capsize=3,
    )
    axis.axvline(0, color="gray", linestyle="--", linewidth=1)
    axis.set_yticks(positions, differences["algorithm"])
    axis.set_xlabel("Balanced accuracy difference (imaging - clinical)")
    axis.set_title("Paired feature-set differences")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_importance_matrix(
    importance: pd.DataFrame,
    path: str | Path,
    *,
    feature_set: str,
    value_column: str,
    title: str,
    color_label: str,
    vmin: float,
    vmax: float,
    dpi: int,
) -> Path:
    """Plot a feature-by-algorithm importance matrix for one feature set."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = importance.loc[importance["feature_set"] == feature_set]
    matrix = data.pivot(index="feature", columns="algorithm", values=value_column)
    figure, axis = plt.subplots(
        figsize=(max(8, 1.5 * len(matrix.columns)), max(4, 0.65 * len(matrix.index)))
    )
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    image = axis.imshow(
        matrix.to_numpy(), aspect="auto", cmap="coolwarm", norm=norm
    )
    axis.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(matrix.index)), matrix.index)
    axis.set_title(f"{title}: {feature_set}")
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            value = matrix.iloc[row, column]
            normalized = float(norm(value))
            axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if normalized < 0.18 or normalized > 0.82 else "black",
                fontsize=8,
            )
    figure.colorbar(image, ax=axis, label=color_label)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_logistic_coefficients(
    coefficients: pd.DataFrame, path: str | Path, *, dpi: int
) -> Path:
    """Plot signed logistic coefficients in one panel per feature set."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_sets = list(dict.fromkeys(coefficients["feature_set"]))
    figure, axes = plt.subplots(
        1,
        len(feature_sets),
        figsize=(7 * len(feature_sets), 5),
        squeeze=False,
        sharex=True,
    )
    for axis, feature_set in zip(axes[0], feature_sets, strict=True):
        data = coefficients.loc[coefficients["feature_set"] == feature_set].sort_values(
            "coefficient"
        )
        axis.barh(data["term"], data["coefficient"])
        axis.axvline(0, color="gray", linestyle="--", linewidth=1)
        axis.set_title(feature_set)
        axis.set_xlabel("Coefficient (log-odds scale)")
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle("Logistic-regression coefficient comparison")
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_shap_importance(
    importance: pd.DataFrame,
    path: str | Path,
    *,
    algorithm: str,
    feature_set: str,
    dpi: int,
) -> Path:
    """Plot mean absolute SHAP contribution for one fixed tree model."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = importance.loc[
        (importance["algorithm"] == algorithm)
        & (importance["feature_set"] == feature_set)
    ].sort_values("mean_absolute_shap")
    figure, axis = plt.subplots(figsize=(7, max(4, 0.55 * len(data))))
    axis.barh(data["feature"], data["mean_absolute_shap"])
    output_scale = data["output_scale"].iloc[0]
    axis.set_xlabel(f"Mean absolute SHAP contribution ({output_scale})")
    axis.set_title(f"Tree SHAP: {algorithm} | {feature_set}")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path
