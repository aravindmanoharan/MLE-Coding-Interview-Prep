# Question 5: Adult Income Classification

## Problem

Build a classifier to predict whether a person's annual income exceeds **$50K**.

Use the Adult dataset from OpenML.

```python
from sklearn.datasets import fetch_openml

data = fetch_openml("adult", version=2, as_frame=True)
df = data.frame
```

## Expected Output

- End-to-end mixed-feature preprocessing pipeline
- Binary classifier
- Precision, recall, F1, and ROC-AUC
- At least two engineered features

## Tasks

1. Inspect dataset shape, dtypes, missing values, duplicates, and class distribution.
2. Identify numerical and categorical columns.
3. Handle missing values appropriately.
4. Build a `ColumnTransformer` with:
   - Numerical imputation + scaling
   - Categorical imputation + one-hot encoding
5. Train a baseline classifier in a `Pipeline`.
6. Evaluate precision, recall, F1, and ROC-AUC.
7. Engineer at least two useful features.
8. Retrain and compare against the baseline.
9. Inspect highly correlated or redundant features.
10. Explain how you would handle a categorical feature with very high cardinality.

## Time Limit

**40–45 minutes**
