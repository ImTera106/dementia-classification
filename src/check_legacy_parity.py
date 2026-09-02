"""Check legacy outputs against a canonical release without claiming independence."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluate import calculate_evaluation_tables
from src.evaluation_release import load_evaluation_release
from src.utils.io import load_json

LOGGER = logging.getLogger(__name__)


class ParityError(ValueError):
    """Raised when legacy and canonical results are not computationally equivalent."""


def verify_legacy_parity(
    release_dir: str | Path,
    frozen_manifest: dict[str, Any],
    *,
    legacy_prediction_paths: dict[str, str | Path],
    legacy_metric_paths: dict[str, str | Path],
) -> dict[str, int]:
    """Require exact labels and numerically equivalent scores and metrics."""
    canonical, _ = load_evaluation_release(release_dir, frozen_manifest)
    key = ["experiment_id", "subject_id"]
    legacy_frames: list[pd.DataFrame] = []
    for condition in ("real_only", "real_plus_synthetic"):
        path = Path(legacy_prediction_paths[condition])
        legacy = pd.read_csv(path, dtype={"subject_id": str})
        legacy["experiment_id"] = (
            legacy["training_condition"].astype(str)
            + "__" + legacy["feature_set"].astype(str)
            + "__" + legacy["algorithm"].astype(str)
        )
        legacy_frames.append(legacy)
    legacy = pd.concat(legacy_frames, ignore_index=True)
    columns = [
        *key, "algorithm", "feature_set", "training_condition",
        "target", "prediction", "score",
    ]
    left = canonical[columns].sort_values(key).reset_index(drop=True)
    right = legacy[columns].sort_values(key).reset_index(drop=True)
    if len(left) != len(right) or not left[key].equals(right[key]):
        raise ParityError("Legacy and canonical prediction identities differ")
    exact = ["algorithm", "feature_set", "training_condition", "target", "prediction"]
    if not left[exact].equals(right[exact]):
        raise ParityError("Legacy and canonical prediction labels differ")
    if not np.allclose(left["score"], right["score"], rtol=1e-12, atol=1e-12):
        raise ParityError("Legacy and canonical prediction scores differ")

    canonical_metrics, _ = calculate_evaluation_tables(
        canonical.drop(columns="experiment_id")
    )
    metric_columns = [
        "balanced_accuracy", "roc_auc", "sensitivity", "specificity", "precision",
        "f1", "tn", "fp", "fn", "tp",
    ]
    identity = ["algorithm", "feature_set", "training_condition"]
    for condition in ("real_only", "real_plus_synthetic"):
        saved = pd.read_csv(legacy_metric_paths[condition])
        expected = canonical_metrics.loc[
            canonical_metrics["training_condition"] == condition
        ]
        joined = expected.merge(
            saved, on=identity, suffixes=("_canonical", "_legacy"), validate="one_to_one"
        )
        if len(joined) != len(expected) or any(
            not np.allclose(
                joined[f"{name}_canonical"], joined[f"{name}_legacy"],
                rtol=1e-12, atol=1e-12,
            )
            for name in metric_columns
        ):
            raise ParityError(f"Legacy metrics differ for {condition}")
    return {
        "prediction_rows": len(canonical),
        "experiments": int(canonical["experiment_id"].nunique()),
        "subjects": int(canonical["subject_id"].nunique()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="A match proves computational parity only; it does not restore test independence.",
    )
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--real-predictions", type=Path, required=True)
    parser.add_argument("--augmented-predictions", type=Path, required=True)
    parser.add_argument("--real-metrics", type=Path, required=True)
    parser.add_argument("--augmented-metrics", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    try:
        summary = verify_legacy_parity(
            args.release_dir,
            load_json(args.frozen_manifest),
            legacy_prediction_paths={
                "real_only": args.real_predictions,
                "real_plus_synthetic": args.augmented_predictions,
            },
            legacy_metric_paths={
                "real_only": args.real_metrics,
                "real_plus_synthetic": args.augmented_metrics,
            },
        )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        LOGGER.error("Legacy parity check failed: %s", exc)
        return 1
    LOGGER.info(
        "Computational parity confirmed for %d experiments and %d subjects; "
        "this does not establish renewed test independence",
        summary["experiments"], summary["subjects"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
