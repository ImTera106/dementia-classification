"""Develop real-plus-synthetic models without access to held-out test data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Final

import pandas as pd

from src.synthesize import modeling_columns
from src.train import train_augmented_models
from src.tune import save_tuning_outputs, tune_augmented_candidates
from src.utils.io import load_yaml_config, save_json

LOGGER = logging.getLogger(__name__)
DEFAULT_PATHS_CONFIG: Final[Path] = Path("config/paths.yaml")
DEFAULT_MODEL_CONFIG: Final[Path] = Path("config/model_config.yaml")
DEFAULT_TUNING_CONFIG: Final[Path] = Path("config/tuning_synthetic_config.yaml")
DEFAULT_SYNTHETIC_CONFIG: Final[Path] = Path("config/synthetic_config.yaml")


def _save_training_data(
    real_train: pd.DataFrame,
    synthetic_frames: dict[str, pd.DataFrame],
    *,
    model_config: dict[str, Any],
    data_paths: dict[str, Any],
) -> None:
    """Save feature-set-specific synthetic and combined training tables."""
    target = str(model_config["split"]["target_column"])
    subject = str(model_config["split"]["subject_id_column"])
    for feature_set, synthetic in synthetic_frames.items():
        synthetic_path = Path(data_paths[f"synthetic_{feature_set}"])
        synthetic_path.parent.mkdir(parents=True, exist_ok=True)
        synthetic.to_csv(synthetic_path, index=False)
        columns = [
            subject,
            *modeling_columns(
                feature_set, model_config["feature_sets"], target_column=target
            ),
        ]
        combined = pd.concat(
            [real_train.loc[:, columns], synthetic.loc[:, columns]], ignore_index=True
        )
        combined_path = Path(data_paths[f"combined_train_{feature_set}"])
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(combined_path, index=False)


def run_phase8(
    paths_config: dict[str, Any],
    model_config: dict[str, Any],
    tuning_config: dict[str, Any],
    synthesis_config: dict[str, Any],
) -> dict[str, Path]:
    """Tune, synthesize, and fit augmented models using development data only."""
    data_paths = paths_config["data"]
    output_paths = paths_config["outputs"]
    real_train = pd.read_csv(data_paths["train"])
    cv_results, tuning_summary, best_parameters = tune_augmented_candidates(
        real_train, model_config, tuning_config, synthesis_config
    )
    saved = save_tuning_outputs(
        cv_results,
        tuning_summary,
        best_parameters,
        cv_results_path=output_paths["synthetic_tuning_cv_results"],
        summary_path=output_paths["synthetic_tuning_summary"],
        best_parameters_path=output_paths["synthetic_tuning_best_parameters"],
    )
    manifest, synthetic_frames, synthesis_reports = train_augmented_models(
        real_train,
        model_config,
        tuning_config,
        synthesis_config,
        best_parameters,
        models_dir=paths_config["models_dir"],
    )
    _save_training_data(
        real_train,
        synthetic_frames,
        model_config=model_config,
        data_paths=data_paths,
    )
    save_json(
        {"training_condition": "real_plus_synthetic", "reports": synthesis_reports},
        output_paths["synthetic_quality"],
    )
    save_json(manifest, paths_config["synthetic_model_manifest"])
    return {
        **saved,
        "manifest": Path(paths_config["synthetic_model_manifest"]),
    }


def parse_args() -> argparse.Namespace:
    """Parse Phase 8 configuration paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--tuning-config", type=Path, default=DEFAULT_TUNING_CONFIG)
    parser.add_argument("--synthetic-config", type=Path, default=DEFAULT_SYNTHETIC_CONFIG)
    return parser.parse_args()


def main() -> int:
    """Run Phase 8 from YAML configuration."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    try:
        run_phase8(
            load_yaml_config(args.paths_config),
            load_yaml_config(args.model_config),
            load_yaml_config(args.tuning_config),
            load_yaml_config(args.synthetic_config),
        )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        LOGGER.error("Phase 8 failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
