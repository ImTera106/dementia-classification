# Dementia Classification With and Without MRI — Migration Map

## Source audit

The primary source is `oldversion/MLAD.Rmd` in the supplied project directory. It reads the OASIS-2 longitudinal CSV (373 visits from 150 subjects), retains the row with the highest visit number for each subject, bins years of education, merges `Converted` with `Demented`, and compares five classifiers with and without three structured MRI measurements. The Python project now retains numeric education years by explicit project decision instead of reproducing that binning. The repository structure was reconciled with `Alzheimers_Engineering_Architecture_Guide_v1.docx` when it became available.

| Original R section | Python destination | Phase 1 status |
|---|---|---|
| Data import and last-visit filtering | `src/clean.py` | Implemented |
| Education retention, sex recoding, and target recoding | `src/clean.py` | Implemented |
| Feature selection and Full/NoMRI formulas | `src/features.py`, `config/model_config.yaml` | Implemented as definitions only |
| Data assessment | `src/clean.py` | Implemented; saved as JSON |
| Train/test split | `src/clean.py` | Implemented; stratified after one-row-per-subject collapse |
| Median imputation | Future sklearn preprocessing pipeline | Deferred; fit independently inside each CV fold |
| Scaling and encoding | Future sklearn `Pipeline`/`ColumnTransformer` | Deferred; fit inside training/CV only |
| Repeated cross-validation and algorithm grids | Future `src/tune.py` | Deferred |
| Final fitting | Future `src/train.py` | Deferred |
| Held-out metrics/confusion matrices | Future `src/evaluate.py` | Deferred |
| EDA and statistical tests | Exploration notebooks | Deferred |
| Feature importance/SHAP | Future explainability module | Deferred |
| Results/report rendering | `report/report.qmd` loading saved artifacts | Deferred |

## Model-name migration

| R object | Algorithm | R feature formula | Proposed feature setting | Future artifact path |
|---|---|---|---|---|
| `logit_model` | logistic regression (`glmnet`) | all selected columns | `clinical_imaging` | `models/logistic_regression/clinical_imaging.pkl` |
| `logit_model1` | logistic regression (`glmnet`) | minus MRI | `clinical` | `models/logistic_regression/clinical.pkl` |
| `svm_model` | radial SVM | all selected columns | `clinical_imaging` | `models/svm/clinical_imaging.pkl` |
| `svm_model1` | radial SVM | minus MRI | `clinical` | `models/svm/clinical.pkl` |
| `tree_model` | decision tree | all selected columns | `clinical_imaging` | `models/decision_tree/clinical_imaging.pkl` |
| `tree_model1` | decision tree | minus MRI | `clinical` | `models/decision_tree/clinical.pkl` |
| `rf_model` | random forest | all selected columns | `clinical_imaging` | `models/random_forest/clinical_imaging.pkl` |
| `rf_model1` | random forest | minus MRI | `clinical` | `models/random_forest/clinical.pkl` |
| `xgb_model` | XGBoost | all selected columns | `clinical_imaging` | `models/xgboost/clinical_imaging.pkl` |
| `xgb_model1` | XGBoost | minus MRI | `clinical` | `models/xgboost/clinical.pkl` |

## Confirmed definitions

- Target: `Nondemented = 0`; `Demented = 1`; `Converted = 1`, matching the Rmd's last-visit diagnostic framing.
- Subject unit: one whole row per `Subject ID`, selected by maximum `Visit`, before splitting.
- `clinical`: `sex`, `age`, numeric `education_years`, `ses`, and `mmse`.
- `clinical_imaging`: all clinical predictors plus `etiv`, `nwbv`, and `asf`.
- Exclusions: identifiers, original group label, visit metadata, handedness, and `cdr`. The Rmd explicitly identifies CDR as outcome-defining leakage.

## Unresolved questions and inconsistencies

1. The Rmd calls the task “early detection,” but selects the final visit and uses contemporaneous MMSE and final diagnostic status. The Python project is therefore named “Dementia Classification With and Without MRI-Derived Features.”
2. The Rmd groups SES imputation by education in training, imputes test SES with one overall train median, and does not consistently impute test MMSE. The Python data foundation preserves missing values; a future sklearn pipeline will fit one median per numeric field independently inside each CV fold.
3. The Rmd's “logistic regression” uses `glmnet`, which is regularized. The future Python constructor should document the chosen penalty and map its tuning semantics explicitly.
4. The Rmd creates CV folds once and passes them into repeated CV; this may not yield genuinely distinct repeated folds. The shared Python tuning workflow should define repeated, stratified subject-level folds deterministically.
5. The Kaggle metadata describes the fields but does not establish clinical deployment thresholds. Risk bands remain undefined until a later calibration and threshold-validation phase; they will not be selected using the test set.
