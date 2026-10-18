# Predicting In-Hospital Recurrent Myocardial Infarction

Comparative benchmarking of 10 machine-learning models for predicting **recurrent myocardial infarction (`REC_IM`)** during hospitalization, using admission-time data from the UCI *Myocardial Infarction Complications* dataset.

## Study design
Retrospective prediction-model development and internal validation study (secondary analysis of an existing dataset). Reporting follows TRIPOD+AI.

- **Patients:** 1,700 (159 recurrent MI, 9.4%)
- **Predictors:** 100 admission-time variables (post-admission and outcome variables excluded to prevent leakage)
- **Split:** 80% training / 20% held-out test, stratified, `random_state=2026`
- **Validation:** 5-fold CV for tuning, nested 5×5 CV for honest model comparison, held-out test set for final evaluation

## Models
KNN · Gaussian Naive Bayes · Logistic regression (ridge, lasso, elastic net) · LDA · QDA · Spline logistic regression · Random forest · Gradient boosting

## Evaluation
12 metrics (ROC-AUC with bootstrap 95% CI, PR-AUC, Brier, log loss, accuracy, balanced accuracy, sensitivity, specificity, PPV, NPV, F1, MCC), ROC/PR/calibration curves, SHAP feature importance for the top three models, and data-leakage and over/underfitting diagnostics.

## Key results
| Model | Nested CV AUC | Test AUC |
|---|---|---|
| Logistic (ridge) | 0.653 | 0.612 |
| LDA | 0.653 | 0.613 |
| Random forest | 0.651 | 0.583 |
| Spline logistic | 0.648 | 0.634 |

- Admission-time data predict recurrent MI only modestly (AUC ≈ 0.65).
- Angina history and angina severity were the most consistent predictors across models (SHAP).
- Simple regularized linear models matched or beat flexible nonlinear models.

## How to run
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/MI_RecurrentMI_Benchmark.ipynb
```
Full run takes about 7–10 minutes.

## Repository structure
```
data/mi.csv                               dataset ('?' already converted to missing)
notebooks/MI_RecurrentMI_Benchmark.ipynb  full analysis (with outputs)
figures/                                  supporting figures
requirements.txt                          Python packages
```

## Data source
Golovenkin S.E., Shulman V.A., Rossiev D.A., et al. *Myocardial infarction complications* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C53P5M (CC BY 4.0)

*For teaching and research only. Not for clinical decision-making.*
