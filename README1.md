# Dementia Classification With and Without MRI-Derived Features

This portfolio project uses the OASIS-2 longitudinal dataset and transforms repeated visit-level records into a subject-level dataset, where each row represents one patient at their selected visit. The project evaluates whether MRI-derived features improve contemporaneous dementia classification beyond a compact set of clinical variables.

Two predictor sets are compared:

- `clinical`: age, raw years of education, socioeconomic status, MMSE, and sex;
- `clinical_imaging`: the clinical predictors plus eTIV, nWBV, and ASF.

Five classification algorithms are evaluated across both feature sets: logistic regression, radial SVM, decision tree, random forest, and XGBoost. Balanced accuracy is used as the primary evaluation metric, with ROC-AUC, sensitivity, specificity, precision, and F1-score reported as secondary metrics.

## Main finding


## Study design

## Reproduce the pipeline

## Interpretation limits

