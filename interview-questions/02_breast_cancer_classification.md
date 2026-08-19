# Question 2: Breast Cancer Classification

## Problem

Build a binary classifier to predict whether a tumor is **malignant or benign**.

Use the built-in Breast Cancer Wisconsin dataset from scikit-learn.

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)
df = data.frame
```

## Expected Output

- Trained binary classifier
- Predictions and probabilities on a held-out test set
- Precision, recall, F1, ROC-AUC, and confusion matrix
- A tuned classification threshold achieving high malignant recall

## Tasks

1. Load and inspect the dataset.
2. Check feature distributions, missing values, duplicates, and class balance.
3. Create a stratified train/test split.
4. Build a preprocessing + model `Pipeline`.
5. Train a baseline classifier.
6. Evaluate using precision, recall, F1, ROC-AUC, and confusion matrix.
7. Use predicted probabilities to adjust the decision threshold.
8. Find a threshold that achieves at least **95% recall for malignant tumors**.
9. Compare precision before and after threshold tuning.
10. Explain which metric matters most for this problem and why.

## Time Limit

**30 minutes**
