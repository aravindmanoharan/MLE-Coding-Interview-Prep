# Question 3: California Housing Price Prediction

## Problem

Build a regression model to predict median house value for California districts.

Use the California Housing dataset from scikit-learn.

```python
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame
```

## Expected Output

- Trained regression model
- Predictions on a held-out test set
- MAE, RMSE, and R²
- Comparison between a linear and nonlinear model

## Tasks

1. Load and inspect the dataset.
2. Check missing values, duplicates, distributions, correlations, and outliers.
3. Separate features and target.
4. Create a train/test split.
5. Build a preprocessing + regression `Pipeline`.
6. Train a simple baseline regression model.
7. Evaluate using MAE, RMSE, and R².
8. Train one nonlinear regression model and compare performance.
9. Inspect examples with the largest prediction errors.
10. Explain which regression metric you would report and why.

## Time Limit

**30–35 minutes**
