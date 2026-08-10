# Project Handoff

## Current phase status
Phase 12 — Preserve the Phase 8 paired-bootstrap correction: complete.

## Completed
- Added a read-only validation mode comparing saved `real_plus_synthetic` and
  `real_only` held-out predictions without loading or refitting models.
- Required exact agreement in subject IDs, targets, algorithm grid, feature-set
  grid, prediction completeness, and training-condition labels before comparing
  conditions.
- Generated one set of 5,000 class-stratified subject bootstrap draws and reused
  the exact same indices for every model and both training conditions.
- Added 95% percentile-bootstrap intervals for all six approved metrics across
  the ten augmented experiments.
- Added ten paired augmented-minus-real-only balanced-accuracy intervals, one
  per algorithm and feature set, with an explicit interval-includes-zero flag.
- Added augmented absolute-interval and paired training-condition-difference
  figures.
- Updated the README and Quarto source with the new results, interpretation, and
  reproducibility command; no HTML was generated.
- Reviewed and committed the complete correction as a source-only Git change;
  ignored predictions, metrics, tables, figures, models, and data remained local.

## Files changed
- `AGENTS.md`
- `README.md`
- `config/paths.yaml`
- `config/validation_config.yaml`
- `src/validate.py`
- `src/utils/plotting.py`
- `tests/test_validate.py`
- `tests/test_repository_contract.py`
- `tests/test_report.py`
- `report/report.qmd`
- `docs/handoff.md`
- generated `outputs/metrics/test_metric_bootstrap_intervals_real_plus_synthetic.csv`
- generated `outputs/tables/training_condition_balanced_accuracy_differences.csv`
- generated `outputs/figures/test_balanced_accuracy_intervals_real_plus_synthetic.png`
- generated `outputs/figures/training_condition_balanced_accuracy_differences.png`

## Decisions
- The comparison is paired by held-out subject because both training conditions
  were evaluated on the same 30 entirely real subjects.
- The comparison direction is always `real_plus_synthetic - real_only` and the
  comparison metric remains balanced accuracy; ordinary accuracy is excluded.
- Point differences are recomputed from the two prediction artifacts and tested
  against the existing Phase 8 comparison table rather than copied from it.
- Percentile intervals use the existing 5,000-resample, 95%, random-state-123,
  class-stratified bootstrap configuration.
- The intervals are unadjusted for ten comparisons and conditional on fixed
  models, one synthetic run, and the previously inspected held-out sample. They
  do not measure training, tuning, synthesis, or external-population variability.
- No model selection, threshold selection, training, tuning, synthesis, or
  held-out prediction regeneration occurred.
- The Phase 12 commit uses an explicit tracked-file allowlist and does not add
  ignored generated artifacts needed only for local report reproduction.

## Results
- All ten augmented balanced-accuracy point estimates improved or tied their
  paired real-only estimate.
- Nine of ten paired 95% percentile intervals included zero.
- Decision-tree/clinical-imaging was the only interval excluding zero: +0.1875,
  95% interval [+0.0446, +0.3482].
- Random-forest/clinical-imaging had an exact zero point difference and interval.
- The remaining eight positive point differences had intervals that included or
  touched zero.
- The result does not establish a general augmentation benefit because it is
  unadjusted, conditional, based on 30 reused subjects, and lacks external
  validation.

## Tests
- `python -m pytest tests/test_validate.py -q`: 6 passed and 2 subtests passed.
- Validation/report/repository contracts: 17 passed and 2 subtests passed.
- `python -m pytest -q`: 65 passed, 2 subtests passed, with four non-failing
  third-party warnings.
- `python -m compileall -q src tests`: passed.
- `git diff --check`: passed.
- All 16 Python chunks in `report/report.qmd` executed successfully in sequence
  and read saved artifacts only.
- Both new PNG figures were visually inspected; no report HTML was produced.

## Unresolved issues
- The held-out partition contains only 30 subjects and has already been
  inspected across both training conditions.
- The paired intervals do not include retraining or synthetic-generation
  variability and are not adjusted for the ten simultaneous comparisons.
- No independent external cohort has validated the reported differences.
- Synthetic data provide no formal privacy guarantee and may reproduce source
  biases.
- This remains contemporaneous classification, not early detection, prognosis,
  clinical validation, or a deployable diagnostic system.
- SHAP and SDV emit four upstream deprecation/metadata warnings during tests;
  they do not cause failures.
- No source-code license has been selected, and no remote repository is
  configured.

## Next task
Choose and add an explicit source-code license before publishing the repository.
