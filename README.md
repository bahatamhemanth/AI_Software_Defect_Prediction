# AI-Based Software Defect Prediction and Test Prioritization

## Overview
This system predicts the likelihood that a software module contains a defect
from historical software metrics. Predicted defect probabilities are converted
into risk levels and used to prioritize testing effort.
## Objectives

- Analyze software complexity and code metrics.
- Compare reproducible machine-learning classifiers.
- Prioritize defective-module recall on imbalanced data.
- Generate probability-based risk classifications.
- Rank modules for testing using defect probability and complexity.
- Provide a Streamlit interface for batch predictions and results review.
## Dataset

The system uses the public NASA MDP PC1 dataset, distributed through PROMISE
and catalogued by OpenML. The unchanged source file is stored at
`data/raw/pc1.arff`.

- 1,109 original modules
- 21 numeric predictor features
- Target: `defects`
- 77 defective modules
- 1,032 non-defective modules
- Original defect rate: 6.94%
- 0 missing values and 0 infinite values
- 155 duplicate rows identified during EDA
The predictors contain McCabe complexity, Halstead measures, lines of code,
comment and blank-line counts, operator and operand counts, and branch counts.
Dataset provenance and feature names are documented in
`reports/dataset_selection.md`.

## Exploratory Data Analysis
The EDA found strong right skew in several metrics, especially `E`, `T`,
`ev(g)`, `iv(G)`, and `V`. `E` and `T` were perfectly correlated. Several
metrics had different distributions between defective and non-defective
modules. IQR analysis identified extreme observations; they were retained for
modeling because extreme software metrics are not automatically invalid.

The EDA notebook and generated figures are in `notebooks/01_eda.ipynb` and
`reports/figures/`.

## Methodology
```text
Dataset -> EDA -> preprocessing -> feature selection -> model training
	-> evaluation -> defect probability -> risk classification
	-> test prioritization -> Streamlit dashboard
```
Preprocessing removes duplicate rows for modeling, replaces infinite values,
excludes `T` because it is exactly redundant with `E`, and applies median
imputation and standard scaling inside each fitted pipeline. Plausible
outliers are retained. A stratified train/test split uses `random_state=42`.
Estimators that support it use balanced class weights; no resampling is used.
All transformations are fitted within the training pipeline to prevent data
leakage.

## Machine Learning Models
The implemented classifiers are:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

Five-fold stratified cross-validation selects the model by defective-class
recall, followed by average precision and F1-score. The held-out test set is
used for final reporting.

## Model Evaluation
The selected model is **Logistic Regression**. Its held-out test results are:

| Metric | Score |
| --- | ---: |
| Accuracy | 0.8010 |
| Precision | 0.2391 |
| Recall | 0.7857 |
| F1-score | 0.3667 |
| ROC-AUC | 0.9088 |
| PR-AUC | 0.4865 |

Recall is prioritized because missing a defective module is more costly than
flagging an additional non-defective module. The resulting trade-off is high
defective-class recall with lower precision, meaning the model catches many
defective modules while producing false positives that require review. The
model is not presented as an accuracy-led solution.

Detailed results, confusion matrices, curves, and feature importance are in
`reports/model_results.csv`, `reports/model_evaluation.md`, and
`reports/figures/`.

## Defect Prediction

The persisted pipeline at `models/best_model.joblib` validates the expected PC1
metric columns, applies the same fitted preprocessing, and returns a predicted
class and defect probability. Risk bands are transparent operational labels:

- Low: probability below 0.33
- Medium: probability from 0.33 to below 0.66
- High: probability at least 0.66

These bands are not claims that the probabilities are perfectly calibrated.

## Test Prioritization

Testing priority uses defect probability as the primary signal. It is combined
with a normalized complexity index from `loc`, `v(g)`, `N`, `V`, and
`branchCount`:

```text
Priority Score = Defect Probability * (0.5 + 0.5 * Complexity Index)
```

Modules are ranked from highest to lowest priority score. No business
criticality or historical test-cost information is invented because PC1 does
not provide those fields.

## Dashboard

`app/app.py` provides a Streamlit dashboard that loads the saved pipeline,
displays dataset and model results, accepts a CSV of PC1 metrics, generates
predictions, and presents a sortable testing-priority table.

## Project Structure

```text
AI_Software_Defect_Prediction/
├── data/raw/pc1.arff
├── data/processed/
├── notebooks/01_eda.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── prediction.py
│   ├── prioritization.py
│   ├── validate_pipeline.py
│   ├── validate_dashboard.py
│   └── generate_report.py
├── models/best_model.joblib
├── reports/
│   ├── figures/
│   ├── model_results.csv
│   ├── model_evaluation.md
│   └── final_report.docx
├── app/app.py
├── requirements.txt
└── README.md
```

## Installation

Python 3.12 is recommended. Python 3.14.7 is also verified for this project.
Use the virtual environment interpreter directly so the setup does not depend
on a global pip launcher:

```powershell
py -3.14 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe src\verify_environment.py
```

## Usage

Train models and regenerate all evaluation artifacts:

```powershell
.\venv\Scripts\python.exe src\train_models.py
```

Validate single prediction, batch prediction, and prioritization:

```powershell
.\venv\Scripts\python.exe src\validate_pipeline.py
```

Generate the technical report from current artifacts:

```powershell
.\venv\Scripts\python.exe src\generate_report.py
```

Start the dashboard:

```powershell
streamlit run app\app.py
```

## Limitations

- PC1 contains 1,109 modules from one older NASA software context.
- The 6.94% defect rate creates substantial class imbalance.
- Reported defects may not represent every defect that existed.
- Results may not generalize to modern languages, teams, or processes.
- Recall-oriented classification produces false positives that require review.
- PC1 does not include business criticality, test execution cost, or complete
	historical test-suite information.

## Future Improvements

Potential technical improvements include explainable AI, larger and newer
software datasets, CI/CD integration, automated test generation, probability
calibration, continuous model retraining, and MLOps monitoring.

## References

- Sayyad Shirabad, J. and Menzies, T. J. (2005). *The PROMISE Repository of
	Software Engineering Databases*. University of Ottawa.
- Shepperd, M., Song, Q., Sun, Z., and Mair, C. (2013). Data Quality: Some
	Comments on the NASA Software Defect Datasets. *IEEE Transactions on
	Software Engineering*, 39.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python.
	*Journal of Machine Learning Research*, 12, 2825-2830.
# AI-Based Software Defect Prediction and Test Prioritization

An ML-based system for predicting software defects from software metrics and
using the resulting risk scores to prioritize testing.

## Current Status

The project is implemented through model training, evaluation, prediction,
test prioritization, dashboard validation, and report generation. The selected
dataset is NASA MDP PC1, documented in `reports/dataset_selection.md`.

## Setup

Python 3.12 is the recommended project version for broad machine-learning
package compatibility. Python 3.14.7 is the verified fallback currently used
in this workspace because Python 3.12 is not installed on this machine.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src\verify_environment.py
```

If PowerShell blocks activation, run the project commands with the virtual
environment interpreter directly, for example:

```powershell
.\venv\Scripts\python.exe src\verify_environment.py
```

On Windows, the explicit interpreter form is the most reliable option:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe src\verify_environment.py
```

If Python 3.12 is unavailable, use the installed Python launcher to create
the environment, then continue with the same `venv` commands:

```powershell
py -3.14 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe src\verify_environment.py
```

Run notebooks with `jupyter notebook` or `jupyter lab`. Train the models and
generate evaluation artifacts with:

```powershell
.\venv\Scripts\python.exe src\train_models.py
.\venv\Scripts\python.exe src\validate_pipeline.py
.\venv\Scripts\python.exe src\generate_report.py
```

Launch the dashboard with:

```powershell
streamlit run app\app.py
```

## Project Structure

- `data/raw/` - Original, unmodified datasets
- `data/processed/` - Reproducibly transformed datasets
- `notebooks/` - Exploration and experimentation notebooks
- `src/` - Source code
- `models/` - Trained model artifacts
- `reports/` - Evaluation reports and visualizations
- `reports/figures/` - Generated figures
- `reports/dataset_selection.md` - Dataset provenance and measured statistics
- `reports/model_results.csv` - Cross-validation and held-out test metrics
- `reports/model_evaluation.md` - Model selection and evaluation record
- `reports/final_report.docx` - Generated technical report
- `app/` - Application and deployment code
- `requirements.txt` - Python dependencies

## Planned Workflow

The implementation uses duplicate removal, removal of the perfectly redundant
`T` metric, median imputation, standard scaling, balanced class weights where
supported, and no resampling. Five models are compared using stratified
cross-validation. Logistic Regression is selected by defective-class recall,
then average precision and F1, and is persisted as `models/best_model.joblib`.
The held-out test metrics are stored in `reports/model_results.csv`; all
dashboard predictions come from the persisted pipeline.
