# End-to-End Machine Learning Project Guideline

## Purpose

This guide explains how to design, implement, evaluate, and maintain a trustworthy
end-to-end machine-learning project. It combines general guidance with a concrete
audit of this repository's OASIS-2 dementia-classification project.

The repository is a reproducible offline machine-learning portfolio project. It
is not a clinically validated diagnostic system, an online prediction service,
or a continuously retrained production system. Those distinctions determine
which practices are required now and which belong to a possible future system.

The central principle is:

> A model result is trustworthy only when the problem definition, data boundary,
> validation design, preprocessing, model selection, evaluation, and claims are
> all aligned.

Correct code cannot rescue the wrong scientific question. A high metric cannot
rescue contaminated evaluation. Deployment infrastructure cannot establish
clinical validity.

## 1. Four levels of an ML project

These levels are related but not interchangeable.

### 1.1 Offline experiment

An offline experiment uses a fixed dataset to compare modeling choices under a
controlled evaluation design. It should include reproducible cleaning, splitting,
cross-validation, training, evaluation, and reporting.

This repository primarily belongs at this level.

### 1.2 Reproducible batch inference

Batch inference loads a previously fitted model and predicts a supplied table. It
must reproduce the exact preprocessing learned during training and must not fit,
tune, read outcomes, or select a model from test performance.

This repository supports this level through `src/predict.py`.

### 1.3 Production ML service

A production service additionally needs deployment, authentication, availability,
input validation, audit logs, monitoring, rollback, incident handling, and model
version management. An API alone is not a complete production system.

This repository does not implement this level.

### 1.4 Validated clinical system

A clinical system additionally needs external validation, representative cohorts,
calibration, subgroup assessment, clinically justified thresholds, workflow and
utility studies, governance, privacy review, human oversight, and regulatory or
institutional approval where applicable.

This repository does not claim this level.

## 2. The end-to-end lifecycle

```text
Problem definition
    -> data provenance and ingestion
    -> deterministic validation and observation construction
    -> train/test boundary
    -> training-only preprocessing and feature engineering
    -> baselines
    -> cross-validation and hyperparameter selection
    -> final training
    -> one-time held-out evaluation
    -> uncertainty and robustness analysis
    -> explainability and reporting
    -> controlled inference
    -> deployment, monitoring, and governed retraining when required
```

At every arrow ask:

1. What information crosses this boundary?
2. Was any future, validation, or test information used to learn a decision?
3. Is the evaluation unit the same as the claimed prediction unit?
4. Does the reported result answer the original question?

## 3. Problem definition

### Concept

Define the target, observation unit, prediction time, available predictors,
population, intended use, and success criteria before modeling.

### Why it matters

Ambiguous questions produce ambiguous labels and accidental leakage. A model can
be technically accurate while answering a different question from the one stated.

### Common mistakes

- Describing contemporaneous classification as future prediction.
- Using a later measurement to claim early detection.
- Changing the target after inspecting results.
- Treating repeated visits as independent people.
- Claiming clinical utility from an internal benchmark.
- Selecting a convenient metric without connecting it to the task.

### Recommended practice

Write a prediction statement before implementation:

```text
Using [predictors available at time T], predict [target at time T or T+delta]
for [observation unit] in [population], evaluated using [primary metric].
```

Also record what the project explicitly does not claim.

### This repository

**Status: Implemented correctly, with a narrow scope.**

- Task: binary, subject-level dementia classification.
- Observation: one row per subject, selected at the highest visit number.
- Predictors and outcome are contemporaneous at that selected visit.
- Primary metric: balanced accuracy.
- Feature sets: predefined `clinical` and `clinical_imaging` sets.
- Training conditions: `real_only` and `real_plus_synthetic`.

### Limitation

The design does not support claims about early detection, future conversion,
prognosis, causal effects, or clinical diagnosis.

## 4. Data provenance, ingestion, and raw-data protection

### Concept

Data provenance records where data came from, what each row and column means,
how access is governed, and whether source files remain immutable.

### Why it matters

Silent changes to a source file can make an experiment irreproducible. Incorrect
schema, units, or labels can invalidate all later stages.

### Common mistakes

- Overwriting raw data during cleaning.
- Accepting missing required columns.
- Ignoring duplicate records or conflicting entity identifiers.
- Allowing inconsistent categories such as `M`, `Male`, and `male` without an
  explicit mapping.
- Failing to validate numeric types, units, or plausible ranges.
- Committing restricted or sensitive data.

### Recommended practice

- Treat `data/raw/` as immutable.
- Validate required columns, identifiers, categories, numeric parseability,
  uniqueness constraints, and domain-approved ranges.
- Save a structural assessment with row counts, subject counts, missingness,
  duplicates, types, and class counts.
- Separate detection from correction: never silently repair a suspicious value.
- Record data-use and version information without committing restricted data.

### This repository

**Status: Partially strong.**

Implemented:

- Raw data are not overwritten or committed.
- Required columns are checked.
- unexpected target and sex labels are rejected.
- Subject identifiers must be non-null and unique after construction.
- Structural assessments record types, missingness, duplicates, subject counts,
  and target counts.
- The current assessment reports 373 rows, 150 subjects, and zero exact raw
  duplicate rows.

Not checked:

- Domain-approved plausibility ranges for age, education, MMSE, SES, eTIV, nWBV,
  or ASF.
- Conflicting rows tied at the highest visit number for one subject.
- A strict failure on exact raw duplicate rows.
- Formal outlier or influential-observation assessment.

### Limitation

No evidence currently shows corrupt values, but the ingestion contract is not
strong enough to reject every plausible form of dirty data.

## 5. Observation-unit construction and entity boundaries

### Concept

The observation unit is what one model row represents: a person, visit, device,
transaction, household, or time window.

### Why it matters

Related rows are not independent. If one person's visits appear in both training
and test, the model can recognize person-specific patterns rather than generalize
to new people.

### Common mistakes

- Randomly splitting visits when the goal is generalization to new subjects.
- Treating multiple images from one patient as independent observations.
- Computing entity-level summaries after entities have crossed partitions.
- Using the latest visit for a claim about prediction from the first visit.

### Recommended practice

- Define the entity key before splitting.
- Keep every row belonging to an entity in one partition.
- Use grouped splitting if multiple rows per entity remain.
- Ensure observation construction is aligned with prediction timing.

### This repository

**Status: Implemented correctly for contemporaneous classification.**

The deterministic rule selects each subject's whole row at the highest numeric
visit, producing 150 unique subject rows before a subject-level split. Because
the selection is local to each subject and does not learn population statistics,
collapsing then splitting is equivalent to splitting subject IDs and applying the
same rule within each side.

### Limitation

Using the last visit is appropriate only for the stated contemporaneous task. It
would be temporally invalid for early detection or prognosis.

## 6. Train, validation, and test design

### Concept

- Training data fit model parameters.
- Validation data select features, hyperparameters, thresholds, or model designs.
- Test data estimate performance after development decisions are locked.

Cross-validation can provide the validation function without permanently
discarding a small fixed validation partition.

### Why it matters

Using test results to make development choices makes the final test estimate
optimistic, even when the test rows never appear directly in `model.fit()`.

### Common mistakes

- Tuning hyperparameters on the test set.
- Trying many feature sets and reporting only the best test result.
- Repeatedly adjusting a model after inspecting the same test subjects.
- Splitting related entities across partitions.
- Using random splits for forecasting or future-outcome tasks.
- Creating a tiny fixed validation set when repeated CV uses limited data more
  efficiently.

### Recommended practice

For small independent subject-level data:

```text
Development partition -> repeated stratified cross-validation
Held-out partition     -> evaluate only after choices are fixed
External cohort        -> fresh confirmation after further development
```

For time-dependent tasks, use temporal validation. For grouped data, use grouped
validation. The splitting strategy must match the intended generalization claim.

### This repository

**Status: Correct implementation; independence is now limited by inspection.**

- 120 real subjects form the development partition.
- 30 entirely real subjects form the persistent held-out partition.
- Tuning uses repeated stratified five-fold CV with three repeats.
- Test subjects are excluded from preprocessing, synthesis, tuning, and fitting.
- The same test subjects support the predefined factorial comparisons.

The 30 subjects have now been inspected across algorithms, feature sets, training
conditions, bootstrap analyses, figures, and reporting. That does not change the
already-fitted models, but future design decisions based on those results would
turn the test set into a de facto validation set.

### Limitation

Further development needs a new independent test cohort for a fresh unbiased
estimate. Splitting the existing 120 subjects again is not a substitute for new
external data.

## 7. Deterministic cleaning versus learned preprocessing

### Concept

Deterministic cleaning applies fixed rules. Learned preprocessing estimates
parameters from data.

Examples:

```text
Deterministic: rename columns, fixed label mapping, select highest visit.
Learned: median imputation, scaling, encoding learned categories, PCA,
feature selection, resampling, or synthetic generation.
```

### Why it matters

Learned preprocessing fitted before splitting transfers information from
validation or test data into training.

### Common mistakes

- Scaling the complete dataset before splitting.
- Filling missing values using the global median.
- Selecting features using all targets.
- Fitting PCA or a synthetic generator on all subjects.
- Fitting a separate scaler to the test set.

### Recommended practice

Put learned transformations inside an sklearn `Pipeline` and
`ColumnTransformer`. During CV, each pipeline clone learns only from its
fold-training rows.

At final evaluation:

```text
training: fit_transform with training statistics
test:     transform using the already-fitted training statistics
```

### This repository

**Status: Implemented correctly.**

Imputation, scaling, and encoding are inside model pipelines. Test data arrive in
raw feature units and are transformed using training-derived parameters.

## 8. Missing values

### Concept

Missingness can be noise, measurement failure, workflow information, or a signal
related to the target.

### Why it matters

Most estimators cannot accept missing values directly. Blindly dropping rows can
reduce power or introduce selection bias; blindly filling zero can create an
artificial value.

### Common mistakes

- Dropping all incomplete subjects without studying the pattern.
- Filling numeric values with zero when zero has substantive meaning.
- Computing imputation values on the full dataset.
- Assuming median imputation is universally harmless.
- Adding missing indicators without validating whether they improve robustness.

### Recommended practice

- Report missingness by partition and feature.
- Fit imputation within training folds.
- Choose a method consistent with feature meaning and sample size.
- Consider missingness indicators or sensitivity analysis when missingness may be
  informative.
- Test inference behavior for missing but present columns.

### This repository

**Status: Leakage-safe but incomplete.**

- Subject-level missingness: 8 SES values and 1 MMSE value.
- Numeric predictors use fold-local median imputation.
- Sex uses fold-local most-frequent imputation.
- Required columns must still exist at inference.
- No missingness indicators or missingness sensitivity analysis were performed.

### Limitation

The project does not establish whether missingness itself carries predictive
information. Given the very small number of missing values, elaborate methods
could also overfit.

## 9. Outliers and plausible ranges

### Concept

An outlier may be a data error, a unit error, or a legitimate rare subject.

### Why it matters

Linear and distance-based models can be influenced by extreme values. Automatic
removal can also erase clinically important cases.

### Common mistakes

- Removing observations only because they reduce performance.
- Defining outlier limits using train and test together.
- Winsorizing without preserving the fitted thresholds in the pipeline.
- Treating every statistically unusual clinical measurement as erroneous.

### Recommended practice

- First apply domain-approved plausibility checks.
- Investigate rather than silently alter suspicious records.
- If robust transformation or clipping is justified, learn thresholds on
  training folds and preserve them in the pipeline.
- Report sensitivity to influential observations when sample size permits.

### This repository

**Status: Not evaluated.**

There is no formal plausibility-range, outlier, leverage, or influence analysis.
This is a quality gap, not evidence that current observations are invalid.

## 10. Categorical encoding and identifiers

### Concept

Nominal categories have no inherent order. Ordinal categories have a meaningful
ordering. Identifiers are usually tracking fields, not predictors.

### Why it matters

Encoding a nominal category as `female=0`, `male=1`, `other=2` may invent a
distance or order. Including subject IDs can allow memorization.

### Common mistakes

- Label-encoding nominal predictors as ordered integers.
- One-hot encoding the target instead of supplying a binary target vector when
  the estimator expects class labels.
- Including IDs, post-outcome variables, or direct target proxies.
- Crashing inference on a previously unseen category.

### Recommended practice

- Use one-hot encoding for low-cardinality nominal predictors.
- Fit encoders within training folds.
- Define behavior for unseen categories.
- Preserve identifiers only for alignment and auditing.
- Maintain an explicit leakage-column denylist.

### This repository

**Status: Implemented correctly.**

- `sex` is most-frequent imputed and one-hot encoded.
- Unknown inference categories are ignored safely by the encoder.
- `dementia` remains a binary 0/1 target.
- `subject_id` is preserved for alignment but excluded from predictors.
- Group, CDR, visit, MRI ID, delay, and hand are excluded as identifiers or
  leakage columns.
- There are no free-text predictors.

## 11. Scaling and normalization

### Concept

Standardization transforms a numeric feature using training mean and standard
deviation:

```text
z = (x - training_mean) / training_standard_deviation
```

Normalization often refers to mapping values into a fixed interval such as
0 to 1. They are not the same operation.

### Why it matters

Scale-sensitive models can let large-unit features dominate geometry or
optimization. Trees generally depend on ordering, not feature units.

### Common mistakes

- Scaling before splitting.
- Fitting the test scaler separately.
- Assuming all algorithms require scaling.
- Interpreting standardized coefficients as effects per original measurement
  unit.

### Recommended practice

- Standardize numeric features for logistic regression, SVM, and other
  scale-sensitive methods when justified.
- Usually omit scaling for decision trees and tree ensembles.
- Keep the scaler in the persisted model pipeline.

### This repository

**Status: Implemented correctly.**

- Logistic regression and RBF SVM standardize numeric predictors.
- Decision tree, random forest, and XGBoost do not.
- No min-max normalization is used.
- Test rows use training-derived statistics through the saved pipeline.

## 12. Class imbalance, metrics, and decision thresholds

### Concept

Class imbalance occurs when one target class is substantially more common.
Accuracy can hide failure on the less common class.

Balanced accuracy is:

```text
(sensitivity + specificity) / 2
```

### Why it matters

A majority-class model can achieve high ordinary accuracy while having zero
sensitivity to the minority class.

### Common mistakes

- Reporting only accuracy.
- Resampling before the train/test split.
- Applying resampling to validation or test rows.
- Selecting a classification threshold using the test set.
- Treating ROC AUC as proof that probabilities are calibrated.

### Recommended practice

- Inspect class counts in every partition.
- Use stratified splits where appropriate.
- Report sensitivity and specificity alongside a task-appropriate primary
  metric.
- Tune class weights or resampling inside CV.
- Select any operational threshold using training/validation data and a stated
  cost or utility; never optimize it on the test set.

### This repository

**Status: Implemented correctly for internal discrimination.**

- Subject-level classes are mildly imbalanced: 78 dementia and 72 nondemented.
- The split and CV are stratified.
- Balanced accuracy is primary; ordinary accuracy is intentionally excluded.
- Sensitivity, specificity, ROC AUC, precision, and F1 are secondary metrics.
- Class weights are candidate hyperparameters.
- No threshold is selected from the test data.

### Limitation

The default classification decisions are not an externally validated clinical
threshold.

## 13. Baselines and model complexity

### Concept

A baseline establishes how much value a more complex method adds.

### Why it matters

Without a baseline, an impressive-looking metric may not beat a trivial rule or
a simple interpretable model.

### Common mistakes

- Starting with neural networks or complex ensembles on a small table.
- Comparing only complex models.
- Giving the baseline weaker preprocessing or validation.
- Treating complexity as evidence of quality.

### Recommended practice

Begin with a dummy predictor and a simple plausible model under the same CV and
metric design. Add complexity only when it tests a meaningful hypothesis.

### This repository

**Status: Implemented correctly.**

The project includes a most-frequent dummy baseline and logistic regression
before comparing SVM and tree-based models. All use controlled preprocessing and
training-only CV.

## 14. Model objectives, assumptions, and diagnostics

### 14.1 Loss versus evaluation metric

A loss or fitting objective determines how a model learns parameters. An
evaluation metric determines how fitted candidates are compared. They need not
be identical.

This repository selects candidates by balanced accuracy while algorithms use
their own fitting objectives.

### 14.2 Logistic regression

**Objective:** binary log loss plus L2 regularization.

**Important assumptions and risks:**

- Continuous predictors have an approximately linear relationship with log-odds
  unless transformations are modeled.
- Observations are independent at the modeling unit.
- Strong multicollinearity can destabilize individual coefficients.
- Extreme or influential subjects may affect coefficients.
- Probabilities may require calibration assessment.

**Hyperparameter:** `C`, the inverse regularization strength. Smaller `C` means
stronger regularization; larger `C` allows more flexible coefficients.

**Repository status:** independence, scaling, regularization, and binary target
handling are addressed. Log-odds linearity, multicollinearity, influence, and
calibration are not formally diagnosed.

### 14.3 RBF SVM

**Objective:** soft-margin, hinge-loss-based kernel classification.

**Important assumptions and risks:**

- Numeric scales must be comparable.
- `C` controls the penalty for margin violations.
- `gamma` controls how local the RBF influence is.
- Large `C` and gamma can create an overly detailed boundary.
- Outliers can affect the boundary.
- Decision scores are not automatically probabilities.

**Repository status:** scaling and systematic tuning are addressed; outlier
sensitivity and probability calibration are not.

### 14.4 Decision tree

**Objective:** choose splits using Gini impurity or entropy reduction.

**Important assumptions and risks:**

- Trees do not require normal predictors or linear effects.
- Deep trees and tiny leaves can overfit.
- Small data changes can produce different splits.
- Impurity-based importance can be misleading.

**Repository status:** depth, leaf size, split size, feature sampling, criterion,
and class weight are tuned. External stability remains unknown.

### 14.5 Random forest

**Objective:** aggregate many randomized decision trees.

**Important assumptions and risks:**

- More trees reduce Monte Carlo variability but do not create new information.
- Correlated predictors can share or mask importance.
- Small datasets can still produce unstable generalization estimates.

**Repository status:** tree count, depth, leaf size, split size, feature sampling,
and class weight are tuned. Correlated-importance limitations are documented.

### 14.6 XGBoost

**Objective:** binary logistic loss. Each new tree contributes:

```text
updated score = current score + learning_rate * new tree contribution
```

**Important hyperparameters:**

- `learning_rate`: contribution of each new tree.
- `n_estimators`: number of trees.
- `max_depth`: interaction complexity.
- `subsample`: fraction of rows used per tree.
- `colsample_bytree`: fraction of predictors used per tree.
- `min_child_weight`: minimum support required for child structure.
- `reg_alpha` and `reg_lambda`: L1 and L2 regularization.

**Repository status:** these are tuned jointly rather than selecting a learning
rate in isolation. Training uses binary logistic loss, while candidates are
ranked by CV balanced accuracy.

### 14.7 Shared assumption

All models assume that future subjects are sufficiently comparable to the
development population:

```text
training distribution approximately matches the intended-use distribution
```

This is not established without external validation.

## 15. Hyperparameter tuning

### Concept

Hyperparameters govern model complexity or learning behavior but are not learned
as ordinary fitted coefficients.

### Why it matters

Poor choices can underfit, overfit, or make optimization unstable. Test-driven
choices invalidate the test estimate.

### Common mistakes

- Manually trying settings on the test set.
- Tuning preprocessing outside CV.
- Selecting one parameter while ignoring interactions.
- Reporting the best of many noisy candidates without uncertainty.
- Refitting during search when the pipeline design intends selection only.

### Recommended practice

1. Define justified candidate ranges in configuration.
2. Use shared, reproducible CV splits for fair comparison.
3. Fit the entire pipeline in every fold.
4. Rank by the approved primary metric.
5. Resolve ties using a predefined rule.
6. Save all results and selected parameters.
7. Fit final models only after selection.

Example XGBoost selection:

```text
Candidate A: learning_rate=.01, trees=400, depth=3 -> mean CV BA=.77
Candidate B: learning_rate=.05, trees=200, depth=4 -> mean CV BA=.74
Candidate C: learning_rate=.10, trees=100, depth=5 -> mean CV BA=.71
```

Candidate A wins as a complete parameter combination. The learning rate does not
win independently of tree count, depth, sampling, and regularization.

### This repository

**Status: Implemented correctly.**

- Randomized search is systematic and reproducible.
- Five folds repeated three times produce 15 validation measurements per
  candidate.
- Hyperparameters are selected separately for algorithm, feature set, and
  training condition.
- Test data are absent from selection.

### Limitation

Repeated CV estimates split variability in a small dataset; it does not create
new independent subjects or establish external generalization.

## 16. Synthetic augmentation

### Concept

Synthetic augmentation learns a distribution from real training data and samples
additional training-like rows.

### Why it matters

Synthetic rows can leak validation information if the generator sees validation
subjects. They can reproduce biases and are not equivalent to new independent
patients.

### Common mistakes

- Fitting the generator before the train/test split.
- Fitting it once on all development data before CV.
- Evaluating on synthetic validation or test observations.
- Letting a clinical-only generator learn MRI variables indirectly.
- Claiming synthetic data provide privacy without a formal guarantee.
- Treating synthetic sample size as an increase in independent evidence.

### Recommended practice

During augmented tuning:

```text
real fold training -> fit generator -> real + synthetic model fitting
real fold validation ----------------> scoring only
```

For final training, fit a new generator on the full real development partition.
Keep the held-out test set entirely real. Use separate generators for feature
schemas with different information access.

### This repository

**Status: Implemented strongly.**

- Clinical and clinical-imaging generators are separate.
- Each CV generator sees only real fold-training subjects.
- Validation and test subjects remain real.
- Final generators see only the 120 real development subjects.
- Exact real-row matches are checked and synthetic IDs are separate.
- Training conditions are compared on aligned held-out subjects.

### Limitation

The synthetic data come from only 120 real subjects, provide no formal privacy
guarantee, may reproduce source bias, and add no independent external evidence.

## 17. Final training, artifacts, and version compatibility

### Concept

Final training fits the selected pipeline on all approved development data and
persists the fitted estimator plus enough metadata to reproduce its identity.

### Common mistakes

- Retuning during final training.
- Saving only the estimator but not preprocessing.
- Loading arbitrary untrusted serialized files.
- Ignoring package-version incompatibility.
- Naming models ambiguously, such as `model1`.

### Recommended practice

- Persist the complete preprocessing-and-model pipeline.
- Record algorithm, feature set, training condition, feature schema, selected
  parameters, data count, random state, and exact runtime versions.
- Treat serialized model files as trusted local artifacts, because loading
  joblib/pickle from untrusted sources is unsafe.
- Validate manifest completeness before evaluation or inference.

### This repository

**Status: Implemented correctly.**

Manifests separate algorithm, feature set, and training condition, and evaluation,
explanation, and inference require exact scikit-learn and relevant model-library
versions.

## 18. Held-out evaluation and uncertainty

### Concept

Held-out evaluation estimates performance of a locked development procedure on
unseen observations. Point estimates need uncertainty context.

### Common mistakes

- Reporting only the highest metric.
- Selecting a deployment model from test performance.
- Treating overlapping intervals as a formal paired comparison.
- Bootstrapping experiments with different subjects as though they were paired.
- Treating bootstrap intervals over fixed predictions as retraining variability.
- Ignoring multiple comparisons.

### Recommended practice

- Evaluate all predefined experiments on the same real test subjects when paired
  comparison is intended.
- Save subject-level predictions.
- Align subjects and targets before paired analysis.
- Reuse identical bootstrap indices across paired conditions.
- State exactly what uncertainty includes and excludes.
- Keep evaluation code read-only with respect to fitted models.

### This repository

**Status: Implemented carefully, with major sample limitations.**

- All experiments use the same 30 real subjects.
- Six approved metrics and confusion counts are saved.
- Fixed-prediction percentile bootstrap intervals use 5,000 class-stratified
  resamples.
- Paired feature-set and training-condition comparisons reuse aligned draws.
- Nine of ten augmented-minus-real-only intervals include zero.
- Comparisons are explicitly described as unadjusted and conditional.

### Limitation

The intervals do not include retraining, tuning, synthesis-run, cohort-selection,
or external-population variability. Thirty subjects produce broad, sample-specific
uncertainty.

## 19. Calibration and probabilities

### Concept

Discrimination asks whether a model ranks or classifies subjects correctly.
Calibration asks whether a predicted probability has the correct frequency
meaning.

For example, among many subjects assigned probability 0.80, approximately 80%
should be positive for the probabilities to be well calibrated.

### Common mistakes

- Treating high ROC AUC as calibrated probability.
- Interpreting an SVM decision score as a probability.
- Fitting a calibrator on test predictions.
- Selecting calibration method or threshold using the test set.
- Estimating detailed calibration curves from a tiny sample.

### Recommended practice

- Decide whether calibrated probabilities are required for the use case.
- Fit any calibrator inside training-only CV.
- Assess Brier score, calibration curves, intercept, and slope on independent
  data when sample size permits.
- Avoid clinical risk language for unvalidated scores.

### This repository

**Status: Not evaluated.**

The project assesses discrimination, not calibration. SVC supplies decision
scores rather than probabilities. No calibration model or study is present.

### Limitation

Scores should not be interpreted as validated individual dementia risk.

## 20. Subgroup performance, bias, and fairness

### Concept

Aggregate performance can hide substantial differences between populations.

### Common mistakes

- Assuming overall accuracy implies equal subgroup performance.
- Reporting tiny subgroup estimates without uncertainty.
- selecting subgroup definitions after seeing favorable results.
- Treating removal of a protected feature as proof of fairness.
- Using synthetic test subjects to claim subgroup validity.

### Recommended practice

- Predefine relevant groups with domain and ethical input.
- Check representation in training and evaluation data.
- Report subgroup sensitivity, specificity, calibration, and uncertainty when
  sample size supports them.
- Investigate measurement and selection bias, not only metric parity.
- Avoid claims when groups are too small.

### This repository

**Status: Deliberately not evaluated.**

No subgroup analysis was performed. The 30-subject test set is too small for
reliable fine-grained subgroup conclusions.

### Limitation

The project cannot claim comparable performance across sex, age, education, SES,
or other demographic groups. Deferring this analysis is acceptable if the gap is
stated and fairness is not claimed.

## 21. Explainability

### Concept

Explainability describes how a fitted model uses inputs. It does not establish
causality, biological mechanism, fairness, or clinical importance.

### Common mistakes

- Calling feature importance a causal effect.
- Comparing importance scales from incompatible methods without qualification.
- Ignoring correlated predictors.
- Explaining models with a different runtime or feature pipeline.
- Selecting a model because its test-set explanation looks attractive.

### Recommended practice

- Match methods to model families.
- Use coefficients or odds ratios carefully for regularized logistic models.
- Use permutation importance as predictive sensitivity, not causality.
- Use SHAP to describe model contributions and state the output scale.
- Discuss instability and correlated-feature effects.

### This repository

**Status: Implemented appropriately for an exploratory portfolio.**

The project reports logistic coefficients, permutation importance, and tree SHAP,
and explicitly warns that importance is not causality or clinical validity.

### Limitation

No external stability analysis shows that explanations reproduce in another
cohort.

## 22. Inference

### Concept

Inference applies an already-fitted model to new predictors. It is not training or
evaluation.

### Common mistakes

- Automatically choosing the model with the best held-out score.
- Requiring the target column during inference.
- Refitting an imputer, scaler, or model on incoming data.
- Calculating evaluation metrics without outcomes.
- Loading an incompatible or untrusted model artifact.
- Silently overwriting prediction outputs.

### Recommended practice

- Require explicit model identity.
- Validate identifiers, required raw feature columns, and categories.
- Load one trusted fitted pipeline and verify it is fitted.
- Apply `transform`, never `fit`, to incoming data.
- Return identifiers, experiment identity, prediction, and score.
- Record model and runtime versions.

### This repository

**Status: Implemented correctly for local batch inference.**

Inference requires algorithm, feature set, and training condition. It does not
read outcomes, tune, refit, calculate evaluation metrics, or automatically choose
a deployment model from test performance.

### Limitation

This is not an authenticated online service and is not clinically validated.

## 23. External validation

### Concept

External validation evaluates a locked model or procedure on an independently
collected cohort not used for development.

### Why it matters

Internal splits share dataset recruitment, measurement, scanner, and labeling
conditions. Performance can fall under population or measurement shift.

### Common mistakes

- Calling an internal random split external validation.
- Reusing the same cohort after extensive model iteration.
- Assuming synthetic samples are an external population.
- Changing preprocessing or thresholds after seeing external results and still
  calling them a final test.

### Recommended practice

Lock the model, schema, preprocessing, and evaluation plan before accessing the
external cohort. Document differences in setting, prevalence, measurement, and
missingness.

### This repository

**Status: Not performed; major limitation.**

All real development and evaluation subjects come from OASIS-2. No independent
cohort confirms transportability.

## 24. Deployment, monitoring, and retraining

### Concept

A production feedback loop consists of governed operations, not merely an arrow
from monitoring to automatic retraining.

### Common mistakes

- Deploying before external and use-case validation.
- Monitoring only latency and not data quality.
- Measuring accuracy without reliable delayed outcomes.
- Retraining automatically on unlabeled or weakly labeled feedback.
- Promoting a model because it beats the old model on an overused test set.
- Failing to support rollback and auditability.

### Recommended practice

Monitor:

- schema failures and missingness rates;
- input ranges and unexpected categories;
- population and feature drift;
- prediction and score distributions;
- calibration and performance once trustworthy outcomes arrive;
- subgroup performance where appropriate;
- runtime versions, failures, and latency.

Use a governed loop:

```text
monitoring alert
    -> investigation
    -> approved data snapshot
    -> training-only development
    -> independent validation
    -> human approval
    -> controlled deployment with rollback
```

### This repository

**Status: Outside current scope.**

There is no API, continuous deployment, monitoring service, automatic retraining,
or model-feedback loop. Adding those would not resolve missing external clinical
validation.

## 25. Documentation, artifacts, and version control

### Concept

Reproducibility requires code, configuration, environment information, experiment
identity, outputs, and limitations to agree.

### Common mistakes

- Hard-coding paths and parameters in notebooks.
- Committing raw restricted data or fitted artifacts unintentionally.
- Failing to record selected hyperparameters and library versions.
- Allowing reports to retrain models while rendering.
- Leaving generated outputs that violate repository policy.
- Publishing without a license.

### Recommended practice

- Keep reusable code, configuration, tests, models, outputs, and reports separate.
- Generate reports from saved artifacts without rerunning training.
- Test repository contracts and ignored-file boundaries.
- Update a handoff after each phase.
- Use explicit version compatibility for serialized pipelines.

### This repository

**Status: Strong, with local housekeeping issues.**

Implemented:

- Configuration-driven experiments.
- Modular source responsibilities.
- Saved tuning results, manifests, predictions, metrics, and figures.
- Exact runtime checks for loaded model pipelines.
- Source-only reporting that reads saved artifacts.
- Tests for leakage, experiment coverage, inference, reporting, and ignored data.

Current issues:

- A local generated `report/report.html` may be retained for review and remains
  Git-ignored; repository contracts verify the ignore boundary rather than
  requiring deletion of the local report.
- No explicit source-code license is selected.

These are repository-governance issues, not evidence of invalid model fitting.

## 26. Project audit matrix

| Requirement | Status | Evidence or design | Remaining risk |
|---|---|---|---|
| Clear binary target | Implemented | Fixed 0/1 mapping | Converted-subject definition limits interpretation |
| Explicit observation unit | Implemented | One highest-visit row per subject | Cross-sectional, not future prediction |
| Immutable raw data | Implemented | Raw path protected and ignored | Provenance depends on local source handling |
| Schema validation | Partial | Required columns and categories checked | No full range or tied-visit contract |
| Duplicate handling | Partial | Assessment reports zero exact duplicates | Raw duplicates and tied highest visits do not fail explicitly |
| Missing-data handling | Implemented/partial | Fold-local median and mode imputation | Missingness signal not studied |
| Outlier assessment | Not evaluated | No formal check | Influence and plausibility unknown |
| Subject-safe split | Implemented | Unique IDs and zero overlap | Only one small internal test cohort |
| Scaling | Implemented | Fold-local standardization for LR/SVM | None identified in implementation |
| Categorical encoding | Implemented | One-hot sex encoding | Only binary sex categories represented in source |
| Leakage-column exclusion | Implemented | Explicit denylist | Domain review still required for new features |
| Class imbalance | Implemented | Stratification, balanced accuracy, class-weight tuning | Test counts remain small |
| Baselines | Implemented | Dummy and logistic baselines | Baseline does not prove external usefulness |
| Cross-validation | Implemented | Repeated stratified 5-fold CV | Does not capture external shift |
| Hyperparameter tuning | Implemented | Training-only randomized search | Candidate winner uncertainty remains |
| Model assumptions | Partial | Key algorithm needs respected | No formal logit-linearity, influence, or collinearity study |
| Synthetic leakage controls | Implemented | Fold-local, feature-set-specific generators | Synthetic rows are not independent evidence or private by default |
| Test-set independence | Implemented historically | Test excluded from all fitting | Repeated inspection limits future reuse |
| Uncertainty | Partial/strong | Paired stratified fixed-prediction bootstrap | Excludes retraining and external variability |
| Calibration | Not evaluated | No calibration workflow | Scores are not validated risks |
| Subgroup performance | Deferred | Explicitly disclosed | Fairness unknown |
| Explainability | Implemented | Coefficients, permutation, SHAP | Not causal or externally stable |
| External validation | Not performed | OASIS-2 only | Major transportability limitation |
| Batch inference | Implemented | Explicit trusted model selection | Not an online or clinical service |
| Deployment and monitoring | Outside scope | None | No operational guarantees |
| Reproducibility | Strong | Configs, manifests, tests, report | Exact reruns still require restricted source data and compatible tooling |
| Source license | Missing | Documented in handoff | Publication and reuse terms unclear |

## 27. What this repository can claim

The project can support these carefully scoped statements:

- It is a reproducible, leakage-aware internal OASIS-2 experiment.
- It compares five algorithms across two predefined feature sets and two training
  conditions.
- Its modeling table contains one selected visit per subject.
- Learned preprocessing and synthesis respect training and CV boundaries.
- All final comparisons use the same 30 entirely real held-out subjects.
- It reports classification discrimination, fixed-prediction bootstrap
  uncertainty, and model-behavior explanations.
- It provides controlled local batch inference from trusted fitted artifacts.

## 28. What this repository cannot claim

The evidence does not support claims of:

- early dementia detection;
- future conversion prediction or prognosis;
- causal effects of clinical or MRI variables;
- calibrated individual risk probabilities;
- equal performance across demographic subgroups;
- external generalization to another cohort or institution;
- clinical utility, safety, or diagnostic validity;
- privacy protection from synthetic data;
- production deployment readiness;
- a universal benefit from synthetic augmentation.

## 29. Reusable end-to-end checklist

### Before modeling

- [ ] State the target, positive class, observation unit, prediction time, and
  intended population.
- [ ] State whether the task is contemporaneous, prospective, temporal, grouped,
  or cross-sectional.
- [ ] Define allowed predictors using information available at prediction time.
- [ ] Identify IDs, post-outcome variables, direct proxies, and leakage columns.
- [ ] Protect the raw source and document provenance and data-use restrictions.
- [ ] Validate schema, types, categories, duplicates, entity uniqueness, and
  domain-approved ranges.
- [ ] Report missingness, class counts, and representation before correction.

### Before splitting

- [ ] Define the entity or grouping boundary.
- [ ] Decide whether temporal, grouped, or stratified splitting is required.
- [ ] Keep deterministic cleaning separate from learned preprocessing.
- [ ] Confirm no entity can cross partitions.
- [ ] Freeze and document the split or splitting procedure.

### Before tuning

- [ ] Put imputation, scaling, encoding, selection, and resampling inside the
  training pipeline.
- [ ] Establish dummy and simple-model baselines.
- [ ] Define the primary metric before comparing candidates.
- [ ] Choose hyperparameter ranges for a stated reason.
- [ ] Use validation folds that match the intended generalization boundary.
- [ ] Fit synthetic generators or resamplers only on fold-training data.
- [ ] Score only on real validation observations when evaluating augmentation.

### Before final training

- [ ] Save selected hyperparameters and CV summaries.
- [ ] Verify that test outcomes have not influenced design choices.
- [ ] Lock feature sets, thresholds, algorithms, and runtime versions.
- [ ] Fit final pipelines only on approved development data.
- [ ] Save the entire fitted preprocessing-and-model pipeline.
- [ ] Create a manifest with exact experiment identity and versions.

### Before test evaluation

- [ ] Confirm zero entity overlap with training.
- [ ] Load only already-fitted trusted artifacts.
- [ ] Use the predefined metrics and thresholds.
- [ ] Save subject-level predictions and continuous scores.
- [ ] Evaluate every predefined experiment, not only favorable results.
- [ ] Use paired methods when experiments share subjects.
- [ ] State multiplicity and uncertainty limitations.
- [ ] Do not feed results back into development without retiring the test set.

### Before reporting

- [ ] Distinguish training objective, selection metric, and test metric.
- [ ] Report class counts, confusion components, and uncertainty.
- [ ] Separate discrimination from calibration.
- [ ] Separate model explanation from causal interpretation.
- [ ] State subgroup analyses performed or explicitly deferred.
- [ ] State whether external validation exists.
- [ ] Match every conclusion to the actual prediction timing and population.
- [ ] Render reports from saved artifacts without retraining.

### Before inference or deployment

- [ ] Require explicit model identity and validate the raw input schema.
- [ ] Never fit or tune during inference.
- [ ] Check model-library compatibility and artifact trust.
- [ ] Define behavior for missing values, unknown categories, and range failures.
- [ ] Establish calibration and threshold validity for the intended use.
- [ ] Complete external and subgroup validation appropriate to the risk.
- [ ] Add access control, audit logs, monitoring, rollback, and incident response.
- [ ] Obtain domain, privacy, governance, and regulatory review where applicable.

### Before retraining

- [ ] Define what monitoring event can trigger investigation.
- [ ] Verify the quality and provenance of new labels.
- [ ] Preserve a new independent evaluation set.
- [ ] Compare against the incumbent using predefined promotion criteria.
- [ ] Require human approval for promotion.
- [ ] Version data, code, configuration, models, and decisions.
- [ ] Maintain rollback capability.

## 30. Glossary

**Algorithm:** A general learning method, such as logistic regression, SVM, or
XGBoost.

**Balanced accuracy:** The average of sensitivity and specificity.

**Calibration:** Agreement between predicted probabilities and observed event
frequencies.

**Class weight:** A multiplier that changes how strongly errors from a class
affect fitting.

**Cross-validation:** Repeatedly fitting on subsets of development data and
scoring on held-out development folds.

**`C`:** In logistic regression and SVM, a parameter related to regularization.
Larger `C` generally means weaker regularization or a stronger penalty for
training violations.

**Data drift:** A change in predictor distributions after development.

**Decision threshold:** The score or probability cutoff used to convert a
continuous output into a class prediction.

**External validation:** Evaluation on an independently collected cohort.

**Feature set:** A predefined collection of predictors available to a model.

**`gamma`:** In an RBF SVM, the parameter controlling how local each training
observation's influence is.

**Hyperparameter:** A model or pipeline setting selected outside ordinary
parameter fitting.

**Inference:** Applying a previously fitted model to new predictors.

**Learning rate:** In boosting, the multiplier controlling each new tree's
contribution.

**Leakage:** Information entering model development from a source that would not
be legitimately available at the relevant training or prediction boundary.

**Loss/objective:** The mathematical quantity an algorithm optimizes while
fitting.

**Metric:** A quantity used to evaluate or compare predictions.

**Model parameter:** A value learned during fitting, such as a regression
coefficient or tree split.

**Normalization:** Often, transformation into a fixed interval such as 0 to 1.

**Observation unit:** What one modeling row represents.

**Regularization:** A constraint or penalty that discourages excessive model
complexity.

**Standardization:** Centering and scaling using a mean and standard deviation.

**Subgroup analysis:** Evaluation separately within predefined populations.

**Synthetic augmentation:** Adding generated training rows sampled from a model
fitted to real training data.

**Test-set overuse:** Allowing repeated test results to influence later design or
selection decisions, weakening the test set's independence.

**Training condition:** The source composition used for fitting, such as
`real_only` or `real_plus_synthetic`.

## Final perspective

A strong ML project is not defined by the most complex model or the highest test
score. It is defined by a defensible chain of evidence:

```text
correct question
-> appropriate data and observation unit
-> protected information boundaries
-> reproducible training and selection
-> honest evaluation and uncertainty
-> claims limited to the evidence
```

This repository implements that chain well for an exploratory internal OASIS-2
comparison. Its most important remaining limitations are external validation,
test-set reuse for future development, calibration, subgroup evidence, and more
complete deterministic data-quality checks. Those limitations should guide future
work; they should not be hidden by adding unnecessary deployment infrastructure.
