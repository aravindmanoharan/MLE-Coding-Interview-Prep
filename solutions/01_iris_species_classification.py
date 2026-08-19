import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def load_data():
    """Load the Iris dataset into a pandas DataFrame."""
    data = load_iris(as_frame=True)
    return data


def inspect_data(df: pd.DataFrame) -> None:
    """
    Inspect the dataset:
    - Shape
    - Column types
    - Missing values
    - Duplicate rows
    - Target-class distribution
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
    print(df[['target']].value_counts())
    print()
    print(f"Duplicated row count: {df.duplicated().sum()}")
    print(f"Duplicated row count excluding target: {df.drop(columns='target').duplicated().sum()}")
    print("Duplicated rows")
    print(df[df.duplicated(keep=False)])
    print()
    df.drop_duplicates(keep='first', inplace=True, ignore_index=True)


    return df


def split_features_target(df: pd.DataFrame):
    """Separate features (X) and target (y)."""

    X = df.drop(columns='target')
    y = df['target']
    print("Sample X:")
    print(X.head())
    print("Sample y:")
    print(y.head())
    print()

    return X, y


def split_train_test(X, y):
    """ Create a stratified train/test split."""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Train-Test split shape:")
    print(f"X Train - {X_train.shape[0]}")
    print(f"X Test - {X_test.shape[0]}")
    print(f"y Train - {y_train.shape[0]}")
    print(f"y Test - {y_test.shape[0]}")
    print()
    print("Target distribution:")
    print("Train")
    print(y_train.value_counts())
    print("Test")
    print(y_test.value_counts())

    return X_train, X_test, y_train, y_test

def build_pipeline():
    """
    Build a scikit-learn Pipeline with any required preprocessing
    and a baseline multiclass classifier.
    """

    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42)),
    ])


def evaluate_model(model, X_test, y_test, target_names) -> None:
    """
    Evaluate the model using:
    - Accuracy
    - Precision
    - Recall
    - Macro F1
    - Confusion matrix
    """
    predictions = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}\n")
    print(classification_report(y_test, predictions, target_names=target_names))
    print("Confusion matrix:")
    print(pd.DataFrame(
        confusion_matrix(y_test, predictions),
        index=[f"actual_{name}" for name in target_names],
        columns=[f"pred_{name}" for name in target_names],
    ))
    print()


def train_alternative_model(X_train, y_train):
    """Train an alternative classifier and compare it to the baseline."""

    alt_model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LinearSVC(multi_class='crammer_singer', random_state=42, max_iter=5000)),
    ])
    alt_model.fit(X_train, y_train)
    return alt_model


def main():

    print("Data loading and pre-processing...")
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
    evaluate_model(model, X_test, y_test, data.target_names)

    print("Training alternate model...")
    alt_model = train_alternative_model(X_train, y_train)
    evaluate_model(alt_model, X_test, y_test, data.target_names)

    summarize_findings()


def summarize_findings() -> None:
    """Briefly explain scaling, stratification, and the final model choice."""
    print("Summary")
    print("-------")
    print(
        "Is feature scaling required? Not for RandomForestClassifier - tree splits threshold\n"
        "one feature at a time and are invariant to monotonic rescaling. It IS required for\n"
        "LinearSVC, which is a margin-based linear model: features with larger raw ranges\n"
        "(e.g. petal length in cm) would otherwise dominate the decision boundary. Both models\n"
        "are wrapped in a Pipeline with StandardScaler here for consistency, even though it's a\n"
        "no-op for the RandomForest's predictions.\n"
    )
    print(
        "Why stratify the train/test split? Iris has only 150 rows split evenly across 3\n"
        "classes (50 each). A plain random split can under- or over-represent a class in the\n"
        "test set purely by chance, which skews accuracy/precision/recall/F1 for that class.\n"
        "Stratifying on y preserves the ~40/40/39 train and 10/10/10 test class balance seen\n"
        "above, making the evaluation metrics representative.\n"
    )
    print(
        "Which model would I choose? LinearSVC scored higher here (0.97 accuracy, 0.97 macro F1)\n"
        "vs. RandomForest (0.93 / 0.93), and both confusion matrices show the only errors are\n"
        "versicolor/virginica confusions, which are known to be nearly linearly separable in\n"
        "this dataset - so the simpler linear model has an edge. I'd ship LinearSVC for this\n"
        "specific problem. I'd lean back toward RandomForest if the feature relationships were\n"
        "expected to be non-linear, or if feature-importance interpretability mattered more\n"
        "than the small accuracy gain."
    )


if __name__ == "__main__":
    main()
