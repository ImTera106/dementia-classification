# Dementia Classification With and Without MRI — Codex Instructions

## Project objective
Build a reproducible Python Data Science portfolio project using OASIS-2.

Supported runtime: Python 3.12 or newer. This is required by the canonical
XGBoost 3.4.0 dependency.

Primary task: subject-level dementia classification comparing:
- `clinical`: demographic, socioeconomic, and cognitive predictors without MRI.
- `clinical_imaging`: the same predictors plus structured MRI-derived variables.

Algorithms:
- logistic_regression
- svm
- decision_tree
- random_forest
- xgboost
  
Training conditions:
- real_only
- real_plus_synthetic


All experiments must use the same held-out, entirely real 
test set.

## Data rules
- Never overwrite `data/raw/`.
- Main modeling table: one row per subject, selected at the highest visit number.
- Retain raw numeric `education_years` as a predictor; do not bin it or create  an `education_level` feature.
- Trajectory features are deferred as an optional extension.
- Split by subject before learned preprocessing or synthesis.
- The same subject must never appear in both train and test.
- Generate synthetic data from real subject-level training data only.
- Use separate generators for `clinical` and `clinical_imaging` so clinical-only
  synthesis never learns from MRI-derived variables.
- During augmented tuning, fit synthesis on each real fold-training subset only
  and score exclusively on that fold's real validation subjects.
- Never synthesize test observations or combine train and test data.
- Do not invent target definitions, feature meanings, thresholds, or leakage rules.

## Leakage rules
- Deterministic cleaning may occur before splitting.
- Fit imputation, scaling, encoding, feature selection, resampling, and synthesis using training data only.
- Keep learned preprocessing inside cross-validation with sklearn `Pipeline` and `ColumnTransformer` where appropriate.
- Keep test data free of fitted preprocessing until final evaluation.
- Inference must require an explicit algorithm, feature set, and training
  condition; never select a deployment model from held-out test performance.
- Inference must load an already-fitted trusted local artifact and must not fit,
  tune, read outcomes, or calculate evaluation metrics.
- Every evaluation, explanation, or inference path that loads a saved sklearn
  pipeline must require the manifest's exact scikit-learn runtime version, plus
  the exact model-library version where applicable.
- Training-condition uncertainty comparisons must align the same held-out
  subjects and targets, reuse identical stratified bootstrap draws, and remain
  read-only with respect to fitted models and predictions.

## Repository conventions
- Reusable code: `src/`
- Exploration: `notebooks/`
- Unit tests: `tests/`
- Configuration: `config/`
- Saved estimators: `models/`
- Generated artifacts: `outputs/`
- Reporting: `report/`
- Do not modify the original Rmd.

## Module responsibilities
- `models.py`: construct estimators and model-specific pipelines.
- `tune.py`: cross-validated hyperparameter search.
- `train.py`: fit and save final models.
- `evaluate.py`: pure helpers for scoring already-fitted models.
- `freeze_experiment.py`: freeze development artifacts without test access.
- `final_evaluate.py`: verify the frozen experiment and evaluate the real test data.
- `evaluation_release.py`: atomically publish and verify canonical test predictions.
- `analyze_release.py`: regenerate fingerprinted formal results from frozen predictions.
- `check_legacy_parity.py`: compare legacy and canonical outputs without test access.
- `publish_verification.py`: publish only verified, privacy-safe aggregate evidence.
- `validate.py`: analyze only the verified canonical prediction release.
- `explain.py`: feature importance and SHAP.
- `predict.py`: inference.
- `main.py`: orchestration only.
- `utils/io.py`: shared loading and saving.
- `utils/metrics.py`: shared metrics.
- `utils/plotting.py`: shared plots.

Treat algorithm, feature set, and training condition as separate experiment dimensions. Avoid names such as `model1` or `rf_model1`.

## Working rules
- Inspect before editing.
- Implement one phase at a time.
- Do not implement unrelated future phases.
- Prefer small functions over long scripts.
- Avoid unnecessary classes and infrastructure.
- Add type hints, docstrings, logging, actionable errors, and targeted tests.
- Run relevant tests after changes.
- At task end, update `docs/handoff.md` and report files changed, decisions, tests, unresolved issues, and exactly one next task.
