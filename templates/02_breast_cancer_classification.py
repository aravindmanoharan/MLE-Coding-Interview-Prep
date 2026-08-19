# Template for Question 2: Breast Cancer Classification
# See interview-questions/02_breast_cancer_classification.md for the full problem statement.

import pandas as pd
from sklearn.datasets import load_breast_cancer
# TODO: import whatever else you need, e.g.
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (
#     precision_score, recall_score, f1_score, roc_auc_score,
#     confusion_matrix, precision_recall_curve,
# )


def load_data() -> pd.DataFrame:
    """Load the Breast Cancer Wisconsin dataset into a pandas DataFrame."""
    data = load_breast_cancer(as_frame=True)
    return data.frame


def inspect_data(df: pd.DataFrame) -> None:
    """
    TODO: Load and inspect the dataset:
    - Feature distributions
    - Missing values
    - Duplicate rows
    - Class balance (malignant vs benign)
    """
    pass


def split_train_test(df: pd.DataFrame):
    """TODO: Separate features and target, then create a stratified train/test split."""
    pass


def build_pipeline():
    """
    TODO: Build a preprocessing + model Pipeline and train a baseline classifier.
    """
    pass


def evaluate_model(model, X_test, y_test) -> None:
    """
    TODO: Evaluate using:
    - Precision
    - Recall
    - F1
    - ROC-AUC
    - Confusion matrix
    """
    pass


def tune_decision_threshold(model, X_test, y_test):
    """
    TODO: Use predicted probabilities to adjust the decision threshold,
    finding one that achieves at least 95% recall for malignant tumors.
    """
    pass


def compare_threshold_precision(model, X_test, y_test, default_threshold, tuned_threshold) -> None:
    """TODO: Compare precision before and after threshold tuning."""
    pass


def main():
    df = load_data()
    print(f"Dataset size: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Sample rows:")
    print(df.head())

    # TODO: wire up the rest of the pipeline:
    # inspect_data(df)
    # X_train, X_test, y_train, y_test = split_train_test(df)
    # model = build_pipeline()
    # model.fit(X_train, y_train)
    # evaluate_model(model, X_test, y_test)
    # tuned_threshold = tune_decision_threshold(model, X_test, y_test)
    # compare_threshold_precision(model, X_test, y_test, 0.5, tuned_threshold)


if __name__ == "__main__":
    main()
