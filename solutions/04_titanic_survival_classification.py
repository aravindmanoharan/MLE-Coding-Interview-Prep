import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def load_data() -> pd.DataFrame:
    """Load the Titanic dataset (OpenML) into a pandas DataFrame."""
    X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)
    df = X.copy()
    df['survived'] = y
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Inspect columns, dtypes, missing values, duplicates, and survived balance.
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
    print("survived distribution:")
    print(df[['survived']].value_counts())
    print()
    print(f"Duplicated row count: {df.duplicated().sum()}")
    print(f"Duplicated row count excluding survived: {df.drop(columns='survived').duplicated().sum()}")
    print()
    print(df.describe())
    print()
    df['survived'] = df['survived'].astype(int)

    return df


def split_features_survived(df: pd.DataFrame):
    """Separate features (X) and survived (y)."""

    X = df.drop(columns='survived')
    y = df['survived']
    print("Sample X:")
    print(X.head())
    print("Sample y:")
    print(y.head())
    print()

    return X, y


def identify_feature_types(df: pd.DataFrame):
    """Identify numerical and categorical features."""

    df_exclude_target = df.drop(columns=["survived"])
    numerical_cols = list(df_exclude_target.select_dtypes(include='number').columns)
    categorical_cols = list(df_exclude_target.select_dtypes(exclude='number').columns)

    # pclass is stored as int64, but it's a 3-level class label (1st/2nd/3rd),
    # not a continuous quantity - treat it as categorical, not numerical.
    numerical_cols.remove('pclass')
    categorical_cols.append('pclass')

    return numerical_cols, categorical_cols


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop irrelevant or leakage-prone columns."""

    # boat/body leak the outcome directly; name/ticket/home.dest are
    # near-unique identifiers (raw one-hot encoding would produce ~2,600
    # mostly-empty columns - more features than training rows).
    df.drop(columns=["boat", "body", "name", "ticket", "home.dest"], inplace=True)
    print("Missing values per column:")
    print(df.isnull().sum())
    print()
    return df


def split_train_test(X, y):
    """Separate features and survived, then create a train/test split."""
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


def build_pipeline(numerical_cols, categorical_cols):
    """
    Build separate preprocessing pipelines for numerical features
    (imputation + scaling) and categorical features (imputation +
    one-hot encoding), combine them with a ColumnTransformer, and wrap
    everything with a classifier in a single sklearn Pipeline.
    """
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scalar', StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols),
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42)),
    ])

    return model


def evaluate_model(model, X_test, y_test, target_names) -> dict:
    """
    Evaluate using:
    - Accuracy
    - F1
    - ROC-AUC
    - Confusion matrix
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]  # P(survived)

    print("Sample predictions and probabilities:")
    print(pd.DataFrame({
        'actual': y_test.values[:5],
        'predicted': predictions[:5],
        'P(survived)': probabilities[:5].round(3),
    }))
    print()

    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'f1': f1_score(y_test, predictions),
        'roc_auc': roc_auc_score(y_test, probabilities),
    }
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"F1 score: {metrics['f1']:.2f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.2f}")
    print("Confusion matrix:")
    print(pd.DataFrame(
        confusion_matrix(y_test, predictions),
        index=[f"actual_{name}" for name in target_names],
        columns=[f"pred_{name}" for name in target_names],
    ))
    print()

    return metrics


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer at least one useful feature from the raw columns."""
    df = df.copy()

    # family_size/is_alone: travelling alone vs. with family is a well-known
    # survival signal beyond sibsp/parch counted separately.
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    df['is_alone'] = (df['family_size'] == 1).astype(int)

    # deck: the raw cabin column is ~78% missing and has 186 near-unique
    # values (e.g. "C85"), which one-hot-encodes into mostly noise. The
    # deck letter (first character) is coarser and more likely to
    # generalize; missing cabins become their own "Unknown" deck rather
    # than being papered over by most-frequent imputation.
    df['deck'] = df['cabin'].str[0].fillna('U')
    df = df.drop(columns=['cabin'])

    print("Engineered features - sample:")
    print(df[['sibsp', 'parch', 'family_size', 'is_alone', 'deck']].head())
    print()
    print("Deck distribution:")
    print(df['deck'].value_counts())
    print()

    return df


def compare_feature_engineering_impact(baseline_metrics, engineered_metrics) -> None:
    """Compare performance before and after feature engineering."""
    print("Before vs. after feature engineering:")
    print(f"{'metric':<10}{'baseline':>10}{'engineered':>12}{'delta':>10}")
    for metric in ('accuracy', 'f1', 'roc_auc'):
        before = baseline_metrics[metric]
        after = engineered_metrics[metric]
        print(f"{metric:<10}{before:>10.3f}{after:>12.3f}{after - before:>+10.3f}")
    print()


def summarize_findings() -> None:
    """Explain how your pipeline handles unseen categories at inference time."""
    print("Summary")
    print("-------")
    print(
        "How does the pipeline handle unseen categories? The categorical branch uses\n"
        "OneHotEncoder(handle_unknown='ignore'), not the sklearn default ('error'). At fit\n"
        "time it learns the categories seen in training for each column (sex: male/female,\n"
        "embarked: C/Q/S, pclass: 1/2/3, deck: A-G/T/U). At inference, if a row has a value\n"
        "never seen during training, that row gets an all-zero vector across that feature's\n"
        "one-hot columns instead of raising a ValueError.\n"
    )
    print(
        "Practical effect: the pipeline never crashes on a novel category. The tradeoff is\n"
        "that the row loses all signal from that one feature for that one prediction - it's\n"
        "treated like 'unknown', not mapped to the nearest known category. The model still\n"
        "predicts using the row's other features (age, fare, family_size, etc.), so an unseen\n"
        "category degrades one input rather than failing the whole request.\n"
    )
    print(
        "This is distinct from missing values (NaN), which are handled earlier by\n"
        "SimpleImputer(strategy='most_frequent') and never reach the encoder as 'unknown' -\n"
        "they're filled with the most common training-time value for that column before\n"
        "one-hot encoding runs. Without handle_unknown='ignore', any single new category at\n"
        "serving time (e.g. a deck letter or embarkation port not present in training) would\n"
        "raise an exception and take down inference for that request entirely - a real risk\n"
        "for a model served over time as new data arrives, even if unlikely on this fixed\n"
        "historical dataset."
    )


def main():
    print("Data loading and pre-processing...")
    df = load_data()
    df = inspect_data(df)
    df = drop_leakage_columns(df)
    numerical_cols, categorical_cols = identify_feature_types(df)
    print(numerical_cols, categorical_cols)
    X, y = split_features_survived(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    print("Modeling building...")
    model = build_pipeline(numerical_cols, categorical_cols)
    model.fit(X_train, y_train)
    print(f"Model training completed! Used model - {model}")

    print("Model evaluation...")
    baseline_metrics = evaluate_model(model, X_test, y_test, ["died", "survived"])

    print("Feature engineering...")
    df_engineered = engineer_features(df)
    numerical_cols_fe, categorical_cols_fe = identify_feature_types(df_engineered)
    print(numerical_cols_fe, categorical_cols_fe)
    X_fe, y_fe = split_features_survived(df_engineered)
    X_train_fe, X_test_fe, y_train_fe, y_test_fe = split_train_test(X_fe, y_fe)

    print("Modeling building (engineered features)...")
    model_fe = build_pipeline(numerical_cols_fe, categorical_cols_fe)
    model_fe.fit(X_train_fe, y_train_fe)

    print("Model evaluation (engineered features)...")
    engineered_metrics = evaluate_model(model_fe, X_test_fe, y_test_fe, ["died", "survived"])

    compare_feature_engineering_impact(baseline_metrics, engineered_metrics)

    summarize_findings()


if __name__ == "__main__":
    main()
