# Template for Question 3: California Housing Price Prediction
# See interview-questions/03_california_housing_regression.md for the full problem statement.

import pandas as pd
from sklearn.datasets import fetch_california_housing
# TODO: import whatever else you need, e.g.
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data() -> pd.DataFrame:
    """Load the California Housing dataset into a pandas DataFrame."""
    data = fetch_california_housing(as_frame=True)
    return data.frame


def inspect_data(df: pd.DataFrame) -> None:
    """
    TODO: Load and inspect the dataset:
    - Missing values
    - Duplicates
    - Distributions
    - Correlations
    - Outliers
    """
    pass


def split_features_target(df: pd.DataFrame):
    """TODO: Separate features (X) and target (y)."""
    pass


def split_train_test(X, y):
    """TODO: Create a train/test split."""
    pass


def build_pipeline():
    """
    TODO: Build a preprocessing + regression Pipeline and train a
    simple baseline regression model.
    """
    pass


def evaluate_model(model, X_test, y_test) -> None:
    """
    TODO: Evaluate using:
    - MAE
    - RMSE
    - R²
    """
    pass


def train_alternative_model(X_train, y_train):
    """TODO: Train one nonlinear regression model and compare performance to the baseline."""
    pass


def inspect_prediction_errors(model, X_test, y_test) -> None:
    """TODO: Inspect examples with the largest prediction errors."""
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
    # inspect_prediction_errors(alt_model, X_test, y_test)


if __name__ == "__main__":
    main()
