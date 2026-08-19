import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix


def load_data() -> pd.DataFrame:
    """Load the UCI SMS Spam dataset (Hugging Face) into a pandas DataFrame."""
    dataset = load_dataset("ucirvine/sms_spam")
    return dataset["train"].to_pandas()


def inspect_data(df: pd.DataFrame) -> pd.DataFrame:
    """Inspect class balance, missing values, duplicates, and message lengths."""
    print(f"Dataset size: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Sample rows:")
    print(df.head())
    print()

    print("Missing values per column:")
    print(df.isnull().sum())
    print()

    print("Class balance (0=ham, 1=spam):")
    print(df['label'].value_counts())
    print(df['label'].value_counts(normalize=True).round(3))
    print()

    print(f"Duplicated row count: {df.duplicated().sum()}")
    df = df.drop_duplicates(keep='first', ignore_index=True)
    print(f"Dataset size after dropping duplicates: {df.shape[0]} rows")
    print()

    # local only - message_length becomes a real feature in engineer_features
    # (task 7), not here; this is just to inspect the pattern.
    message_length = df['sms'].str.len()
    print("Message length stats overall:")
    print(message_length.describe())
    print()
    print("Message length stats by class:")
    print(message_length.groupby(df['label']).describe())
    print()

    return df


def split_train_test(df: pd.DataFrame):
    """Separate features and target, then create a stratified train/test split."""
    X = df['sms']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print("Train-Test split shape:")
    print(f"X Train - {X_train.shape[0]}")
    print(f"X Test - {X_test.shape[0]}")
    print()
    print("Target distribution:")
    print("Train")
    print(y_train.value_counts(normalize=True).round(3))
    print("Test")
    print(y_test.value_counts(normalize=True).round(3))
    print()

    return X_train, X_test, y_train, y_test


def build_pipeline():
    """Build a TF-IDF based text-classification pipeline and train a baseline classifier."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', min_df=2)),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000)),
    ])


def evaluate_model(model, X_test, y_test, target_names=("ham", "spam")) -> dict:
    """
    Evaluate using:
    - Precision
    - Recall
    - F1
    - PR-AUC
    - Confusion matrix
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]  # P(spam)

    print("Sample predictions and probabilities:")
    print(pd.DataFrame({
        'actual': y_test.values[:5],
        'predicted': predictions[:5],
        'P(spam)': probabilities[:5].round(3),
    }))
    print()

    metrics = {
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1': f1_score(y_test, predictions),
        'pr_auc': average_precision_score(y_test, probabilities),
    }
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 score: {metrics['f1']:.3f}")
    print(f"PR-AUC: {metrics['pr_auc']:.3f}")
    print("Confusion matrix:")
    print(pd.DataFrame(
        confusion_matrix(y_test, predictions),
        index=[f"actual_{name}" for name in target_names],
        columns=[f"pred_{name}" for name in target_names],
    ))
    print()

    return metrics


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer at least three text-derived features, such as:
    - Message length
    - Number of digits
    - Number of uppercase characters
    - URL or currency indicators
    """
    df = df.copy()
    df['message_length'] = df['sms'].str.len()
    df['num_digits'] = df['sms'].str.count(r'\d')
    df['num_uppercase'] = df['sms'].apply(lambda s: sum(1 for c in s if c.isupper()))
    df['has_url'] = df['sms'].str.contains(r'www\.|http', case=False, regex=True).astype(int)
    df['has_currency'] = df['sms'].str.contains(r'£|\$|\bfree\b', case=False, regex=True).astype(int)

    feature_cols = ['message_length', 'num_digits', 'num_uppercase', 'has_url', 'has_currency']
    print("Engineered features - sample:")
    print(df[['sms'] + feature_cols].head())
    print()
    print("Feature means by class (0=ham, 1=spam):")
    print(df.groupby('label')[feature_cols].mean().round(2))
    print()

    return df


def build_combined_pipeline(numerical_cols):
    """Combine TF-IDF and engineered numerical features into a single Pipeline."""
    preprocessor = ColumnTransformer([
        ('tfidf', TfidfVectorizer(stop_words='english', min_df=2), 'sms'),
        ('numeric', StandardScaler(), numerical_cols),
    ])
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000)),
    ])


def compare_feature_engineering_impact(baseline_metrics, engineered_metrics) -> None:
    """Compare against the text-only baseline."""
    print("Text-only baseline vs. text + engineered features:")
    print(f"{'metric':<10}{'baseline':>10}{'engineered':>12}{'delta':>10}")
    for metric in ('precision', 'recall', 'f1', 'pr_auc'):
        before = baseline_metrics[metric]
        after = engineered_metrics[metric]
        print(f"{metric:<10}{before:>10.3f}{after:>12.3f}{after - before:>+10.3f}")
    print()


def inspect_misclassifications(model, X_test, y_test, n=5) -> None:
    """Inspect representative false positives and false negatives."""
    predictions = model.predict(X_test)
    text = X_test['sms'] if isinstance(X_test, pd.DataFrame) else X_test

    results = pd.DataFrame({
        'sms': text.values,
        'actual': y_test.values,
        'predicted': predictions,
    })

    false_positives = results[(results['actual'] == 0) & (results['predicted'] == 1)]
    false_negatives = results[(results['actual'] == 1) & (results['predicted'] == 0)]

    print(f"False positives - ham predicted as spam ({len(false_positives)} total):")
    for msg in false_positives['sms'].head(n):
        print(f"  - {msg[:100]}")
    print()
    print(f"False negatives - spam predicted as ham ({len(false_negatives)} total):")
    for msg in false_negatives['sms'].head(n):
        print(f"  - {msg[:100]}")
    print()


def main():
    print("Data loading and pre-processing...")
    df = load_data()
    df = inspect_data(df)
    X_train, X_test, y_train, y_test = split_train_test(df)

    print("Model building...")
    model = build_pipeline()
    model.fit(X_train, y_train)
    print(f"Model training completed! Used model - {model}")

    print("Model evaluation...")
    baseline_metrics = evaluate_model(model, X_test, y_test)

    print("Feature engineering...")
    df_engineered = engineer_features(df)
    numerical_cols = ['message_length', 'num_digits', 'num_uppercase', 'has_url', 'has_currency']

    X_fe = df_engineered.drop(columns=['label'])
    y_fe = df_engineered['label']
    X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(
        X_fe, y_fe, test_size=0.2, random_state=42, stratify=y_fe,
    )

    print("Model building (text + engineered features)...")
    model_fe = build_combined_pipeline(numerical_cols)
    model_fe.fit(X_train_fe, y_train_fe)

    print("Model evaluation (text + engineered features)...")
    engineered_metrics = evaluate_model(model_fe, X_test_fe, y_test_fe)

    compare_feature_engineering_impact(baseline_metrics, engineered_metrics)

    print("Misclassification inspection...")
    inspect_misclassifications(model_fe, X_test_fe, y_test_fe)


if __name__ == "__main__":
    main()
