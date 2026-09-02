# Project Handoff

## Current phase status
Phase 19 — Canonical migration, legacy parity, and public release: complete.

## Completed
- Added SHA-256 fingerprints for the persistent train and test CSV files when
  `src.clean` creates the split; the fingerprints are stored in the split summary.
- Added `src.freeze_experiment`, which cannot receive a test path and refuses a
  dirty Git worktree by default.
- The frozen manifest records the configuration-derived experiment grid, stable
  experiment IDs, selected hyperparameters, preprocessing, CV and augmentation
  settings, random state, exact model paths and hashes, training-data hash,
  configuration hashes, embedded condition manifests, Git state, and its own
  canonical-JSON fingerprint.
- Updated `src.final_evaluate` to verify the frozen manifest, Git state, training
  data, configurations, complete experiment set, and every model artifact before
  opening the test file. It then verifies the test fingerprint before prediction.
- Added temporary-fixture tests proving that manifest, model, training-data, and
  configuration mismatches prevent test access, while a changed test file is
  detected after opening and before prediction.
- Removed every held-out test read, prediction, metric, and comparison operation
  from `src.phase8`; its public interface now accepts development configuration
  only and performs augmented tuning, synthesis, and fitting.
- Converted `src.evaluate` into a non-executable evaluation helper module and
  removed its dependency on `src.tune`.
- Added `src.final_evaluate` as the single supported held-out evaluation command.
  It validates the complete configured manifest/model set before loading the
  test partition once and scoring both training conditions.
- Added source-boundary tests that reject test-path access from development and
  fitting, tuning, or synthesis dependencies in evaluation.
- Added an end-to-end evaluation test using temporary fabricated models and test
  data; no real model, test prediction, metric, or split artifact was regenerated.
- Removed the legacy `run_evaluation_pipeline` bypass. `src.evaluate` no longer
  opens CSV files; only the canonical final evaluator opens the formal test path.
- Added `src.analyze_release`, a safely rerunnable analysis-only command that
  verifies frozen predictions and regenerates formal metrics, confusion tables,
  bootstrap summaries, comparisons, and figures without models or test data.
- Added an analysis manifest linked to the frozen manifest and prediction digest;
  it fingerprints every formal release-derived table and figure and is written
  only after the complete analysis succeeds.
- Added `src.check_legacy_parity`, which compares legacy predictions and metrics
  with a canonical release without reopening the test data or claiming renewed
  independence.
- Marked held-out permutation importance as outcome-aware post-hoc interpretation,
  separate from the one formal prediction release.
- Added `src.publish_verification`, which requires a verified canonical release,
  complete analysis manifest, and release-identifying rendered report before it
  can publish any result.
- Public packages preserve verified formal aggregates plus approved CV,
  synthetic-quality, aggregate-importance, figure, and report artifacts while
  rejecting subject-level columns, likely OASIS identifiers, predictions,
  fitted models, synthetic rows, and unapproved source roots.
- Each non-overwriting package includes a self-verifying manifest with release,
  analysis, prediction, test-data, configuration, Git, runtime, split-count, and
  artifact fingerprints. Independent package verification rejects changed,
  missing, or unrecorded files.
- Re-ran deterministic cleaning solely to add split fingerprints; train and test
  CSV hashes remained unchanged, with 120/30 subjects, zero overlap, and held-out
  class counts 14/16.
- Froze all 20 existing experiments at source commit `df43b38` and created
  canonical migration release `ec26697ccaf9b788` from the previously inspected
  test partition.
- Confirmed legacy computational parity for all 600 prediction rows, 20
  experiments, 30 subjects, scores, labels, and model-level metrics. This is
  migration equivalence, not a new independent evaluation.
- Regenerated and fingerprinted 18 formal analysis artifacts from canonical
  predictions, reran explicitly post-hoc explanations, and rendered an HTML
  report containing the exact release and analysis identifiers.
- Published and independently verified 37 privacy-safe aggregate/report
  artifacts in `public_results/ec26697ccaf9b788/`.
- Added a standalone, beginner-friendly end-to-end ML guideline covering problem
  definition, data boundaries, preprocessing, model assumptions, tuning,
  evaluation, inference, deployment, monitoring, and retraining.
- Used a consistent concept/importance/mistakes/recommended-practice/project-status/
  limitation structure throughout the guide.
- Added a dedicated leakage discussion covering preprocessing, subject, temporal,
  synthetic, feature-selection, threshold-selection, and human-in-the-loop
  test-set overuse.
- Documented the objectives, assumptions, risks, and important hyperparameters of
  logistic regression, RBF SVM, decision tree, random forest, and XGBoost.
- Added a project-specific audit matrix distinguishing implemented safeguards,
  partial checks, deferred analyses, and work outside the current scope.
- Added explicit supported and unsupported claim sections, a reusable lifecycle
  checklist, and a plain-language glossary.
- Preserved the current project scope: no model, data, configuration, metric,
  threshold, feature, or evaluation changes were made.

## Files changed
- `.gitignore`
- `src/utils/io.py`
- `src/clean.py`
- `src/freeze_experiment.py`
- `src/evaluation_release.py`
- `src/analyze_release.py`
- `src/check_legacy_parity.py`
- `src/publish_verification.py`
- `public_results/README.md`
- `config/paths.yaml`
- `tests/test_clean.py`
- `tests/test_freeze_experiment.py`
- `tests/test_evaluation_release.py`
- `tests/test_analyze_release.py`
- `tests/test_legacy_parity.py`
- `tests/test_publish_verification.py`
- `config/model_config.yaml`
- `src/phase8.py`
- `src/evaluate.py`
- `src/final_evaluate.py`
- `tests/test_phase8.py`
- `tests/test_evaluate.py`
- `tests/test_final_evaluate.py`
- `tests/test_repository_contract.py`
- `README.md`
- `docs/end_to_end_ml_guideline.md`
- `docs/handoff.md`

## Decisions
- File fingerprints use SHA-256 over exact file bytes. The frozen manifest uses
  canonical sorted JSON excluding its own digest field.
- The freeze interface receives only the training path and the expected test
  digest already recorded by cleaning; it never receives or opens the test CSV.
- Final evaluation checks development-side fingerprints before test access, then
  loads the test partition, checks its digest, and only then predicts.
- A clean Git worktree is required for a real freeze. `--allow-dirty` exists for
  controlled development only and records the dirty state and diff fingerprints.
- The frozen-manifest digest detects accidental modification but is not a digital
  signature or defense against a malicious actor who rewrites all provenance.
- The expected experiment count is derived from the configured algorithms,
  feature sets, and training conditions; the evaluator is not intrinsically
  hard-coded to twenty experiments.
- The current 30-subject partition remains a previously inspected test
  partition. Reorganization prevents future accidental development access but
  does not restore historical independence.
- Existing local predictions remain suitable only for a future computational
  parity check, not a new independent evaluation.
- The final evaluator publishes one manifest-addressed canonical prediction
  release and refuses a rerun before opening the held-out test file.
- Each release contains the complete experiment set in `frozen_predictions.csv`
  and an `evaluation_receipt.json` recording manifest, test, prediction, count,
  and runtime provenance.
- Publication stages a sibling directory and atomically renames it. This is a
  supported-command non-overwrite guarantee, not OS write protection or a
  digital signature.
- Metrics are derived after reloading and verifying the canonical predictions;
  bootstrap analyses filter the same verified release instead of reading two
  condition-specific prediction files.
- Report rendering verifies the release receipt and prediction digest before
  consuming aggregate results and does not publish subject-level predictions.
- Formal aggregate artifacts are now linked to the canonical release by an
  analysis manifest. Failed analysis can be rerun without regenerating or
  changing frozen predictions.
- Explainability remains a separate post-hoc path: permutation importance reads
  held-out outcomes, so the defensible claim is one formal prediction release,
  not one lifetime opening of the test file.
- Public verification packages are generated only from the new canonical
  release lineage. Existing legacy aggregate files are not eligible by
  themselves.
- The report must embed the exact release ID and analysis-manifest digest before
  the publisher accepts it.
- The migration deliberately reused the historically inspected test partition.
  Its purpose is provenance and parity, not renewed statistical independence.
- The guide treats this repository as a reproducible offline ML experiment with
  controlled batch inference, not as a deployed or clinically validated system.
- A missing analysis is labeled as a limitation or deferred check rather than a
  coding error when the existing implementation remains valid for its stated
  scope.
- Repeated inspection of the 30-subject held-out set is described as limiting its
  future independence, not as retroactive contamination of already-fitted models.
- Cross-validation remains the recommended validation mechanism for the
  120-subject development partition; a new independent cohort, not another small
  internal split, is needed for fresh confirmation.
- Subgroup and calibration analyses are documented as absent without claiming
  that the current sample is large enough to estimate them reliably.
- Local rendered report HTML may be retained for review but remains Git-ignored;
  its presence is no longer treated as a repository-contract failure.

## Tests
- Canonical release `ec26697ccaf9b788`: parity confirmed for 20 experiments and
  30 subjects; the analysis manifest records 18 artifacts; the public manifest
  records 37 artifacts.
- `quarto render report/report.qmd`: all 17 Python cells executed and the linked
  HTML rendered successfully.
- `python -m pytest -q -p no:cacheprovider`: 85 passed, 2 subtests passed, with
  four non-failing third-party warnings after migration artifacts were present.
- `python -m pytest -q -p no:cacheprovider`: 85 passed, 2 subtests passed, with
  four non-failing third-party warnings. The retained local `report/report.html`
  is Git-ignored and no longer causes a contract failure.
- `python -m pytest -q -p no:cacheprovider`: 81 passed, 1 failed, 2 subtests
  passed. The sole failure remains the pre-existing ignored local
  `report/report.html` repository-contract conflict; all new boundary, recovery,
  lineage, and parity tests passed.
- Boundary, release, recovery, parity, validation, report, and repository tests:
  35 passed, 1 failed, 2 subtests passed. The only failure is the pre-existing
  ignored local `report/report.html` contract conflict; no real test evaluation,
  analysis release, or parity run was performed.
- `python -m pytest -q -p no:cacheprovider tests/test_evaluation_release.py tests/test_final_evaluate.py tests/test_evaluate.py tests/test_validate.py`:
  23 passed, 2 subtests passed.
- `python -m pytest -q -p no:cacheprovider`: 79 passed, 1 failed, 2 subtests
  passed. The sole failure remains the pre-existing ignored local
  `report/report.html` repository-contract conflict; no real test evaluation or
  canonical release was run.
- `python -m pytest -q -p no:cacheprovider tests/test_clean.py tests/test_freeze_experiment.py tests/test_final_evaluate.py tests/test_evaluate.py`:
  25 passed.
- `python -m pytest -q -p no:cacheprovider`: 76 passed, 1 failed, 2 subtests
  passed. The sole failure remains the pre-existing local `report/report.html`
  repository-contract conflict; neither freezing nor final evaluation was run on
  the real project artifacts.
- `python -m pytest -q -p no:cacheprovider tests/test_phase8.py tests/test_evaluate.py tests/test_final_evaluate.py`:
  11 passed.
- `python -m pytest -q -p no:cacheprovider`: 68 passed, 1 failed, 2 subtests
  passed. The sole failure remains the pre-existing local `report/report.html`
  repository-contract conflict; no final evaluation was run.
- `wc -l -w docs/end_to_end_ml_guideline.md`: 1,388 lines and 7,048 words.
- Manual structure review: all approved sections, project audit matrix, claim
  boundaries, checklist, and glossary are present.
- `python -m pytest -q`: 64 passed, 1 failed, 2 subtests passed. The pre-existing
  failure is `test_sensitive_and_generated_artifacts_are_gitignored` because the
  ignored local `report/report.html` file exists.
- `git diff --check`: the new guide has no reported whitespace errors; the command
  still reports pre-existing trailing whitespace in the user-modified `AGENTS.md`.

## Unresolved issues
- The held-out partition contains only 30 subjects and has already been inspected
  across both training conditions.
- No independent external cohort validates transportability.
- Calibration, subgroup performance, formal outlier/influence analysis, logit
  linearity, and collinearity diagnostics were not performed.
- Raw-data validation does not explicitly reject exact duplicate rows, tied
  highest visits, or values outside domain-approved plausibility ranges.
- Fixed-prediction bootstrap intervals exclude retraining, tuning,
  synthetic-generation, and external-population variability and are unadjusted
  for the ten simultaneous training-condition comparisons.
- Synthetic data provide no formal privacy guarantee and may reproduce source
  bias.
- No source-code license has been selected, and no remote repository is
  configured.
- Unrelated existing edits in `AGENTS.md` and `README.md` were preserved.

## Next task
Perform a final résumé-facing documentation review against the committed public
verification package and remove stale phase language without changing results.
