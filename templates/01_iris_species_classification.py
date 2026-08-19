# Template for Question 1: Iris Species Classification
# See interview-questions/01_iris_species_classification.md for the full problem statement.

import pandas as pd
from sklearn.datasets import load_iris
# TODO: import whatever else you need, e.g.
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def load_data() -> pd.DataFrame:
    """Load the Iris dataset into a pandas DataFrame."""
    data = load_iris(as_frame=True)
    return data.frame


def inspect_data(df: pd.DataFrame) -> None:
    """
    TODO: Inspect the dataset:
    - Shape
    - Column types
    - Missing values
    - Duplicate rows
    - Target-class distribution
    """
    pass


def split_features_target(df: pd.DataFrame):
    """TODO: Separate features (X) and target (y)."""
    pass


def split_train_test(X, y):
    """TODO: Create a stratified train/test split."""
    pass


def build_pipeline():
    """
    TODO: Build a scikit-learn Pipeline with any required preprocessing
    and a baseline multiclass classifier.
    """
    pass


def evaluate_model(model, X_test, y_test) -> None:
    """
    TODO: Evaluate the model using:
    - Accuracy
    - Precision
    - Recall
    - Macro F1
    - Confusion matrix
    """
    pass


def train_alternative_model(X_train, y_train):
    """TODO: Train an alternative classifier and compare it to the baseline."""
    pass


def main():
    df = load_data()
    print(f"Dataset size: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Sample rows:")
    print(df.head())

    # TODO: wire up the rest of the pipeline:
    # inspect_data(df)
    # X, y = split_features_target(df)
    # X_train, X_test, y_train, y_test = split_train_test(X, y)
    # model = build_pipeline()
    # model.fit(X_train, y_train)
    # evaluate_model(model, X_test, y_test)
    # alt_model = train_alternative_model(X_train, y_train)
    # evaluate_model(alt_model, X_test, y_test)


if __name__ == "__main__":
    main()
