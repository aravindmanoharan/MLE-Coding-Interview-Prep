# Question 1: Iris Species Classification

## Problem

Build a multiclass classification model to predict the species of an iris flower from its physical measurements.

Use the built-in **Iris dataset** from scikit-learn.

```python
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
df = data.frame
```

## Expected Output

Your solution should produce:

- A trained multiclass classification model
- Predictions on a held-out test set
- Accuracy score
- Macro F1 score
- Per-class precision and recall
- Confusion matrix
- Comparison against one alternative classifier

## Tasks

1. Load the dataset into a pandas DataFrame.
2. Inspect:
   - Shape
   - Column types
   - Missing values
   - Duplicate rows
   - Target-class distribution
3. Separate features and target.
4. Create a stratified train/test split.
5. Build a scikit-learn `Pipeline` with any required preprocessing.
6. Train a baseline multiclass classifier.
7. Evaluate the model using:
   - Accuracy
   - Precision
   - Recall
   - Macro F1
   - Confusion matrix
8. Train one alternative classifier and compare its performance.
9. Briefly explain:
   - Whether feature scaling is required
   - Why stratification is useful
   - Which model you would choose and why

## Time Limit

**25–30 minutes**
