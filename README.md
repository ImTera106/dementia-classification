# Dementia Classification With and Without MRI-Derived Features

This portfolio project uses the longitudinal OASIS-2 release to compare
subject-level dementia classification with two predictor sets:

- `clinical`: age, raw years of education, socioeconomic status, MMSE, and sex;
- `clinical_imaging`: the clinical predictors plus eTIV, nWBV, and ASF.

Five algorithms are evaluated under both feature settings: logistic
regression, radial SVM, decision tree, random forest, and XGBoost. Balanced
accuracy is the primary metric; ordinary accuracy is intentionally excluded.

## Main finding

With real-only training, the highest balanced accuracy on the fixed 30-subject
real held-out partition was 0.866 for clinical features and 0.839 for
clinical-imaging features. With 1:1 synthetic augmentation, the highest values
were 0.902 and 0.933, respectively. All ten augmentation comparisons improved
or tied their corresponding real-only point estimate. Nine of ten paired 95%
bootstrap intervals included zero; only decision-tree/clinical-imaging excluded
zero (+0.188, 95% percentile interval +0.045 to +0.348). These unadjusted,
conditional results come from the same small, previously inspected test set and
do not establish generalization or clinical benefit.

Inspect the current [Quarto report source](report/report.qmd). Generated HTML is
intentionally not retained in the repository.

## Study design

The raw release contains 373 visits from 150 subjects. The modeling table uses
one complete row per subject, selected at the highest visit number. It is a
contemporaneous classification design—not early detection or forecasting.

The persistent split contains 120 real training subjects and 30 entirely real
test subjects. Learned preprocessing stays inside scikit-learn pipelines and
is fitted only on training data or training folds. The held-out outcomes never
drive preprocessing, tuning, feature selection, threshold selection, or
dependency selection.

The OASIS-2 dataset is described by [Marcus et al.
(2010)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2895005/). Data access remains
subject to the OASIS data-use terms; raw data are not stored in this repository.

## Repository layout

```text
config/          YAML experiment and path configuration
data/            Local raw, cleaned, and processed data (ignored)
docs/            Project handoff and migration decisions
models/          Serialized fitted pipelines (ignored)
outputs/         Generated metrics, tables, predictions, and figures (ignored)
public_results/  Release-specific aggregate verification packages
report/          Quarto report source (generated HTML is not retained)
src/             Cleaning, modeling, evaluation, validation, and explanation
tests/           Targeted unit and artifact-contract tests
```

## Environment

Python 3.12 or newer is required by the canonical XGBoost 3.4.0, SHAP 0.52.0,
and SDV 1.37.3 environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

XGBoost on macOS also requires an available OpenMP runtime. Quarto is an
external report renderer and must be installed separately.

## Data placement

After obtaining OASIS-2 under its data-use agreement, place the longitudinal
CSV at:

```text
data/raw/longitudinal.csv
```

Never overwrite or modify the raw source file.

## Reproduce the pipeline

Run the phases in order:

```bash
python -m src.clean
python -m src.baseline
python -m src.tune
python -m src.train
python -m src.phase8
python -m src.freeze_experiment
python -m src.final_evaluate
python -m src.analyze_release
python -m src.explain
```

`src.phase8` is development-only: it fits synthetic generators only on real
training folds during tuning, scores on real validation folds, and fits the
augmented final models without accepting or resolving a test path.
`src.freeze_experiment` refuses a dirty Git worktree by default and records the
complete experiment grid, selected settings, Git commit, and SHA-256 fingerprints
for the training data, configurations, model artifacts, and original test split.
It copies the expected test fingerprint from the split summary without opening
the test file. `src.final_evaluate` verifies all development-side fingerprints,
refuses to overwrite an existing release, then opens the entirely real test
partition, verifies its fingerprint, and scores both training conditions
together. It atomically publishes one manifest-addressed
`frozen_predictions.csv` and an `evaluation_receipt.json` recording its
fingerprints. Do not iterate on models, features,
augmentation settings, thresholds, or environment versions after inspecting
held-out results.

`src.analyze_release` verifies that canonical prediction release and regenerates
all formal metrics, confusion tables, bootstrap summaries, comparisons, and
figures without test data or fitted models. It writes an analysis manifest that
fingerprints every derived formal-evaluation artifact, and it can be rerun to
recover from an interrupted analysis without regenerating predictions.

`src.explain` is a separate, explicitly post-hoc interpretation step. Its
permutation importance reopens the held-out features and outcomes, so the
project claims one formal prediction release—not that the test file is opened
only once for every downstream purpose. Explanations must not feed back into
model development.

Legacy output parity may be checked with `python -m src.check_legacy_parity`
and its required explicit artifact arguments. A match demonstrates computational
equivalence only; it does not make the previously inspected test partition
independent again.

Optionally render the report locally after the saved artifacts exist:

```bash
quarto render report/report.qmd
```

The report reads saved JSON, CSV, and PNG artifacts. Rendering does not rerun
cleaning, tuning, training, evaluation, bootstrap validation, or SHAP.

After rendering a report for the verified release, create a public aggregate
verification package with:

```bash
python -m src.publish_verification
```

Publication verifies the canonical release and every formal-analysis hash,
rejects subject-level columns and likely OASIS identifiers, and creates a
non-overwriting `public_results/<release-id>/` package. It includes aggregate
CV, held-out, bootstrap, synthetic-quality, and feature-importance evidence,
selected figures, the rendered report, and a verification manifest. It never
includes OASIS rows, subject IDs, predictions, split membership, synthetic
subjects, or fitted models. No package is created from legacy outputs alone.

The current migration package is available at
[`public_results/ec26697ccaf9b788/`](public_results/ec26697ccaf9b788/). Its
manifest identifies the previously inspected test status and fingerprints all
37 included aggregate/report artifacts. Subject-level predictions remain local.

Run the test suite with:

```bash
python -m pytest -q
```

In a source-only checkout, tests that validate ignored local data, fitted
models, saved metrics, or report figures are reported as skipped. After the
pipeline has generated any artifact group, its contract tests run normally and
fail if that group is incomplete or inconsistent. Unit tests and source-level
repository contracts always run.

## Version-control boundaries

Commit source code, YAML configuration, tests, documentation, the Quarto
source, and `.gitkeep` placeholders. Do not commit OASIS source files,
processed or synthetic subject data, fitted models or manifests, predictions,
metrics, figures, rendered reports, editor settings, or Python caches. These
local and generated files are excluded by `.gitignore`.

## Run inference

Prediction requires an explicit algorithm, feature set, and training condition;
the command never chooses a model from held-out results. For example:

```bash
python -m src.predict \
  --input path/to/new_subjects.csv \
  --output path/to/predictions.csv \
  --algorithm logistic_regression \
  --feature-set clinical \
  --training-condition real_only
```

The input must contain a unique, non-empty `subject_id` and every column named
by the selected feature set in `config/model_config.yaml`. Missing predictor
values may be imputed by the already-fitted pipeline, but required columns must
exist. The command does not require or read an outcome column, refit any model,
calculate evaluation metrics, or overwrite an existing output unless
`--overwrite` is supplied. It loads only the explicitly selected, trusted local
model artifact recorded in that training condition's manifest. The output
contains identifiers, the three experiment dimensions, binary predictions, and
the model's positive-class probability or decision score.

## Interpretation limits

- The held-out sample contains only 30 subjects.
- The outcome and MMSE are contemporaneous; this is not prospective risk
  prediction.
- Converted subjects are mapped to the dementia class by the confirmed project
  definition.
- Feature importance and SHAP describe model behavior, not causality or
  clinical importance.
- The work is not a diagnostic system and is not suitable for clinical use.
- Synthetic augmentation is not privacy protection; generated rows are kept
  local under the same data-governance assumptions as the source training data.

Implementation references include [scikit-learn](https://www.jmlr.org/papers/v12/pedregosa11a.html),
[XGBoost](https://arxiv.org/abs/1603.02754), and
[SHAP](https://papers.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html).
