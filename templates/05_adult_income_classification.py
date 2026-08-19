# Template for Question 5: Adult Income Classification
# See interview-questions/05_adult_income_classification.md for the full problem statement.

import pandas as pd
from sklearn.datasets import fetch_openml
# TODO: import whatever else you need, e.g.
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def load_data() -> pd.DataFrame:
    """Load the Adult Income dataset (OpenML) into a pandas DataFrame."""
    data = fetch_openml("adult", version=2, as_frame=True)
    return data.frame


def inspect_data(df: pd.DataFrame) -> None:
    """
    TODO: Inspect dataset shape, dtypes, missing values, duplicates,
    and class distribution.
    """
    pass


def identify_feature_types(df: pd.DataFrame):
    """TODO: Identify numerical and categorical columns."""
    pass


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """TODO: Handle missing values appropriately."""
    pass


def split_train_test(df: pd.DataFrame):
    """TODO: Separate features and target, then create a train/test split."""
    pass


def build_pipeline(numerical_cols, categorical_cols):
    """
    TODO: Build a ColumnTransformer with numerical imputation + scaling
    and categorical imputation + one-hot encoding, and train a baseline
    classifier in a single sklearn Pipeline.
    """
    pass


def evaluate_model(model, X_test, y_test) -> None:
    """
    TODO: Evaluate using:
    - Precision
    - Recall
    - F1
    - ROC-AUC
    """
    pass


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """TODO: Engineer at least two useful features."""
    pass


def compare_feature_engineering_impact(baseline_metrics, engineered_metrics) -> None:
    """TODO: Retrain and compare against the baseline."""
    pass


def inspect_correlated_features(df: pd.DataFrame) -> None:
    """TODO: Inspect highly correlated or redundant features."""
    pass


def main():
    df = load_data()
    print(f"Dataset size: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Sample rows:")
    print(df.head())

    # TODO: wire up the rest of the pipeline:
    # inspect_data(df)
    # df = handle_missing_values(df)
    # numerical_cols, categorical_cols = identify_feature_types(df)
    # X_train, X_test, y_train, y_test = split_train_test(df)
    # model = build_pipeline(numerical_cols, categorical_cols)
    # model.fit(X_train, y_train)
    # baseline_metrics = evaluate_model(model, X_test, y_test)
    # df_engineered = engineer_features(df)
    # (repeat split/build/train/evaluate on df_engineered)
    # engineered_metrics = evaluate_model(model_fe, X_test_fe, y_test_fe)
    # compare_feature_engineering_impact(baseline_metrics, engineered_metrics)
    # inspect_correlated_features(df_engineered)


if __name__ == "__main__":
    main()
