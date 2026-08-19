# Solution for Question 3: California Housing Price Prediction
# See interview-questions/03_california_housing_regression.md for the full problem statement.

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def load_data():
    """Load the California Housing dataset into a pandas DataFrame."""
    data = fetch_california_housing(as_frame=True)
    return data


def inspect_data(df: pd.DataFrame) -> None:
    """
    Load and inspect the dataset:
    - Missing values
    - Duplicates
    - Distributions
    - Correlations
    - Outliers
    """
    print(f"Dataset size: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Sample rows:")
    print(df.head())
    print()
    print("Column types:")
    print(df.dtypes)
    print()
    print("Missing values per column:")
    print(df.isnull().sum())
    print()
    print("Target distribution:")
    print(df['MedHouseVal'].describe())
    print()
    print(f"Duplicated row count: {df.duplicated().sum()}")
    print()
    print("Column stats:")
    print(df.describe())

    return df


def split_features_target(df: pd.DataFrame):
    """Separate features (X) and target (y)."""
    X = df.drop(columns='MedHouseVal')
    y = df['MedHouseVal']
    print("Sample X:")
    print(X.head())
    print("Sample y:")
    print(y.head())
    print()

    return X, y


def split_train_test(X, y):
    """Create a train/test split."""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Train-Test split shape:")
    print(f"X Train - {X_train.shape[0]}")
    print(f"X Test - {X_test.shape[0]}")
    print(f"y Train - {y_train.shape[0]}")
    print(f"y Test - {y_test.shape[0]}")
    print()
    print("Target summary:")
    print("Train")
    print(y_train.describe())
    print("Test")
    print(y_test.describe())

    return X_train, X_test, y_train, y_test


def build_pipeline():
    """
    Build a preprocessing + regression Pipeline and train a
    simple baseline regression model.
    """
    return Pipeline([
        # ('scaler', StandardScaler()),
        ('regressor', LinearRegression()),
    ])


def evaluate_model(model, X_test, y_test) -> None:
    """
    Evaluate using:
    - MAE
    - RMSE
    - R²
    """

    predictions = model.predict(X_test)

    print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
    print(f"RMSE: {root_mean_squared_error(y_test, predictions):.2f}")
    print(f"R²: {r2_score(y_test, predictions):.2f}")
    print()


def train_alternative_model(X_train, y_train):
    """Train one nonlinear regression model and compare performance to the baseline."""
    alt_model = Pipeline([
        # ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(random_state=42)),
    ])
    alt_model.fit(X_train, y_train)
    return alt_model


def inspect_prediction_errors(model, X_test, y_test, top_n=10) -> None:
    """Inspect examples with the largest prediction errors."""
    predictions = model.predict(X_test)
    errors = pd.DataFrame({
        'actual': y_test.values,
        'predicted': predictions,
    }, index=X_test.index)
    errors['abs_error'] = (errors['actual'] - errors['predicted']).abs()
    largest_errors = errors.sort_values('abs_error', ascending=False).head(top_n)

    print(f"Top {top_n} largest prediction errors:")
    print(largest_errors)
    print()
    print("Feature values for those examples:")
    print(X_test.loc[largest_errors.index])
    print()


def summarize_findings() -> None:
    """Explain which regression metric you would report and why."""
    print("Summary")
    print("-------")
    print(
        "Which metric would I report? MAE, as the headline number, with R² alongside it.\n"
        "MAE ($33K for the RandomForest model) is directly interpretable in dollars and treats\n"
        "every error equally, which matches how a stakeholder actually experiences a valuation\n"
        "model: 'on average we're off by $33K,' not a squared abstraction.\n"
    )
    print(
        "RMSE is more sensitive to outliers, and here that sensitivity is misleading rather than\n"
        "informative: the largest individual errors are concentrated in test rows where the\n"
        "target is capped at $500,001 (a data artifact - the true price above the cap is unknown,\n"
        "not a model failure) and a handful of genuinely anomalous block groups the 8 available\n"
        "features can't explain. RMSE ($51K) gets inflated by exactly these unfixable cases, so\n"
        "leading with it risks making the model look worse than it is for the typical prediction.\n"
    )
    print(
        "R² (0.80) is worth reporting alongside MAE because it's scale-free - it says the model\n"
        "explains 80% of the variance in house values relative to a mean-only baseline, which is\n"
        "useful for judging whether the model is worth deploying at all, independent of what\n"
        "currency or units the target is in.\n"
    )
    print(
        "RMSE isn't useless here - it's the right metric if the business cost of being very wrong\n"
        "on a single property is disproportionately high (e.g. automated pricing where a huge\n"
        "miss triggers a bad transaction). But as a general-purpose summary metric for this\n"
        "problem, I'd lead with MAE and treat RMSE as a diagnostic for outlier sensitivity."
    )


def main():
    print("Loading data and pre-processing...")
    data = load_data()
    df = data.frame
    df = inspect_data(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    print("Modeling building...")
    model = build_pipeline()
    model.fit(X_train, y_train)
    print(f"Model training completed! Used model - {model}")

    print("Model evaluation...")
    evaluate_model(model, X_test, y_test)

    print("Training alternate model...")
    alt_model = train_alternative_model(X_train, y_train)
    evaluate_model(alt_model, X_test, y_test)
    inspect_prediction_errors(alt_model, X_test, y_test)

    summarize_findings()


if __name__ == "__main__":
    main()
