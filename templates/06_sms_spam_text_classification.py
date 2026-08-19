# Template for Question 6: SMS Spam Detection
# See interview-questions/06_sms_spam_text_classification.md for the full problem statement.

import pandas as pd
from datasets import load_dataset
# TODO: import whatever else you need, e.g.
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix


def load_data() -> pd.DataFrame:
    """Load the UCI SMS Spam dataset (Hugging Face) into a pandas DataFrame."""
    dataset = load_dataset("ucirvine/sms_spam")
    return dataset["train"].to_pandas()


def inspect_data(df: pd.DataFrame) -> None:
    """
    TODO: Inspect class balance, missing values, duplicates, and
    message lengths.
    """
    pass


def split_train_test(df: pd.DataFrame):
    """TODO: Separate features and target, then create a stratified train/test split."""
    pass


def build_pipeline():
    """
    TODO: Build a TF-IDF based text-classification pipeline and train
    a baseline classifier.
    """
    pass


def evaluate_model(model, X_test, y_test) -> None:
    """
    TODO: Evaluate using:
    - Precision
    - Recall
    - F1
    - PR-AUC
    - Confusion matrix
    """
    pass


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    TODO: Engineer at least three text-derived features, such as:
    - Message length
    - Number of digits
    - Number of uppercase characters
    - URL or currency indicators
    """
    pass


def build_combined_pipeline(numerical_cols):
    """TODO: Combine TF-IDF and engineered numerical features into a single Pipeline."""
    pass


def compare_feature_engineering_impact(baseline_metrics, engineered_metrics) -> None:
    """TODO: Compare against the text-only baseline."""
    pass


def inspect_misclassifications(model, X_test, y_test) -> None:
    """TODO: Inspect representative false positives and false negatives."""
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
    # baseline_metrics = evaluate_model(model, X_test, y_test)
    # df_engineered = engineer_features(df)
    # numerical_cols = [...]
    # (repeat split/build/train/evaluate using build_combined_pipeline on df_engineered)
    # engineered_metrics = evaluate_model(model_fe, X_test_fe, y_test_fe)
    # compare_feature_engineering_impact(baseline_metrics, engineered_metrics)
    # inspect_misclassifications(model_fe, X_test_fe, y_test_fe)


if __name__ == "__main__":
    main()
