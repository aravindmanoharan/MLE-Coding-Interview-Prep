# Question 4: Titanic Survival Prediction

## Problem

Build a model to predict whether a Titanic passenger survived.

Use the Titanic dataset from OpenML.

```python
from sklearn.datasets import fetch_openml

X, y = fetch_openml(
    "titanic",
    version=1,
    as_frame=True,
    return_X_y=True
)
```

## Expected Output

- End-to-end preprocessing and classification pipeline
- Predictions and probabilities on a held-out test set
- Accuracy, F1, ROC-AUC, and confusion matrix
- At least one engineered feature

## Tasks

1. Inspect columns, dtypes, missing values, duplicates, and target balance.
2. Identify numerical and categorical features.
3. Drop irrelevant or leakage-prone columns.
4. Build separate preprocessing pipelines for:
   - Numerical features: imputation + scaling
   - Categorical features: imputation + one-hot encoding
5. Combine them with `ColumnTransformer`.
6. Train a classifier using a single sklearn `Pipeline`.
7. Evaluate using accuracy, F1, ROC-AUC, and confusion matrix.
8. Engineer at least one useful feature from the raw columns.
9. Compare performance before and after feature engineering.
10. Explain how your pipeline handles unseen categories at inference time.

## Time Limit

**35–40 minutes**
