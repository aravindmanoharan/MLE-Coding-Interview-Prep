# Template for Question 4: Titanic Survival Prediction
# See interview-questions/04_titanic_survival_classification.md for the full problem statement.

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
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def load_data() -> pd.DataFrame:
    """Load the Titanic dataset (OpenML) into a pandas DataFrame."""
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    df = X.copy()
    df['survived'] = y
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    TODO: Inspect columns, dtypes, missing values, duplicates, and target balance.
    """
    pass


def identify_feature_types(df: pd.DataFrame):
    """TODO: Identify numerical and categorical features."""
    pass


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """TODO: Drop irrelevant or leakage-prone columns."""
    pass


def split_train_test(df: pd.DataFrame):
    """TODO: Separate features and target, then create a train/test split."""
    pass


def build_pipeline(numerical_cols, categorical_cols):
    """
    TODO: Build separate preprocessing pipelines for numerical features
    (imputation + scaling) and categorical features (imputation +
    one-hot encoding), combine them with a ColumnTransformer, and wrap
    everything with a classifier in a single sklearn Pipeline.
    """
    pass


def evaluate_model(model, X_test, y_test) -> None:
    """
    TODO: Evaluate using:
    - Accuracy
    - F1
    - ROC-AUC
    - Confusion matrix
    """
    pass


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """TODO: Engineer at least one useful feature from the raw columns."""
    pass


def compare_feature_engineering_impact(baseline_metrics, engineered_metrics) -> None:
    """TODO: Compare performance before and after feature engineering."""
    pass


def main():
    df = load_data()
    print(f"Dataset size: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Sample rows:")
    print(df.head())

    # TODO: wire up the rest of the pipeline:
    # inspect_data(df)
    # df = drop_leakage_columns(df)
    # numerical_cols, categorical_cols = identify_feature_types(df)
    # X_train, X_test, y_train, y_test = split_train_test(df)
    # model = build_pipeline(numerical_cols, categorical_cols)
    # model.fit(X_train, y_train)
    # evaluate_model(model, X_test, y_test)
    # df_engineered = engineer_features(df)
    # (repeat split/build/train/evaluate on df_engineered)
    # compare_feature_engineering_impact(baseline_metrics, engineered_metrics)


if __name__ == "__main__":
    main()
