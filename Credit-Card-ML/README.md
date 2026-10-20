# Credit Card Default Prediction Project

This project predicts whether a credit card customer will default on a monthly payment using the `Default.xlsx` dataset.

## Objective

Build an interpretable binary classification workflow for default-risk prediction using:

- Credit card balance
- Annual income
- Student status

## Project Structure

```text
credit_card_default_full_project/
  README.md
  requirements.txt
  src/
    credit_card_default_project.py
  outputs/
    generated after running the pipeline
```

## How To Run

From this project folder:

```powershell
python .\src\credit_card_default_project.py --data "C:\Users\User\Downloads\Default.xlsx"
```

If you want to use the bundled Codex Python runtime:

```powershell
& "C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\src\credit_card_default_project.py --data "C:\Users\User\Downloads\Default.xlsx"
```

## Outputs

The pipeline creates:

- `outputs/final_report.md` - project report with EDA, model results, and recommendations
- `outputs/metrics.csv` - model performance comparison
- `outputs/predictions.csv` - customer-level default probabilities and predictions
- `outputs/summary.json` - machine-readable project summary
- `outputs/*.svg` - charts for class balance, feature behavior, and ROC comparison

## Methods

The pipeline is intentionally self-contained:

- Data loading and cleaning use `pandas`
- Logistic regression is implemented with NumPy using Newton-Raphson optimization
- Linear Discriminant Analysis is implemented with NumPy
- Evaluation includes confusion matrix, accuracy, sensitivity, specificity, precision, recall, F1, and AUC
- Charts are written as SVG files without requiring external plotting packages

## Key Notes

The dataset is highly imbalanced: only about 3.33% of customers default. Because of that, accuracy alone is not enough to judge the model. The report emphasizes recall, precision, F1, AUC, and threshold behavior.
