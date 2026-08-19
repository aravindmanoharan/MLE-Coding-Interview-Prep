import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def load_data() -> pd.DataFrame:
    """Load the Adult Income dataset (OpenML) into a pandas DataFrame."""
    data = fetch_openml("adult", version=2, as_frame=True)
    return data


def inspect_data(df: pd.DataFrame) -> None:
    """
    Inspect dataset shape, dtypes, missing values, duplicates,
    and class distribution.
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
    print(df[['class']].value_counts())
    print()
    print(f"Duplicated row count: {df.duplicated().sum()}")
    print(f"Duplicated row count excluding survived: {df.drop(columns='class').duplicated().sum()}")
    print("Duplicated rows")
    print(df[df.duplicated(keep=False)])
    print()
    df.drop_duplicates(keep='first', inplace=True, ignore_index=True)
    print()
    print(df.describe())
    print()
    df['class'] = df['class'].map({'>50K': 1, '<=50K': 0})
    print(df[['class']].value_counts())
    print()

    return df


def identify_feature_types(df: pd.DataFrame):
    """Identify numerical and categorical columns."""
    df_exclude_target = df.drop(columns=["class"])
    numerical_cols = list(df_exclude_target.select_dtypes(include='number').columns)
    categorical_cols = list(df_exclude_target.select_dtypes(exclude='number').columns)

    print(f"Numerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")

    return numerical_cols, categorical_cols


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values appropriately."""
    df = df.copy()
    missing_cols = df.columns[df.isnull().any()].tolist()
    print(f"Columns with missing values: {missing_cols}")
    print(df[missing_cols].isnull().sum())
    print()

    # workclass/occupation/native-country are categorical, each only 2-6%
    # missing, and the missingness is not random: every row missing
    # workclass is also missing occupation (likely people who never worked /
    # are unemployed). That's itself predictive of income, so we encode
    # missingness as its own explicit category rather than papering over it
    # with most-frequent imputation, which would erase that signal.
    for col in missing_cols:
        df[col] = df[col].astype(object).fillna('Missing')

    print(f"Remaining missing values: {df.isnull().sum().sum()}")
    print()

    return df


def split_features_class(df: pd.DataFrame):
    """Separate features (X) and class (y)."""

    X = df.drop(columns='class')
    y = df['class']
    print("Sample X:")
    print(X.head())
    print("Sample y:")
    print(y.head())
    print()

    return X, y


def split_train_test(X, y):
    """Separate features and target, then create a train/test split."""
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
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1': f1_score(y_test, predictions),
        'roc_auc': roc_auc_score(y_test, probabilities),
    }
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"precision: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
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
    """Engineer at least two useful features."""
    df = df.copy()

    # capital-gain/capital-loss are heavily zero-inflated (~92%/~95% are 0)
    # with a long skewed tail up to 99999 - whether someone has any capital
    # activity at all is a cleaner signal than the raw skewed amount.
    df['has_capital_gain'] = (df['capital-gain'] > 0).astype(int)
    df['has_capital_loss'] = (df['capital-loss'] > 0).astype(int)

    # marital-status has 7 granular categories; collapsing to married-vs-not
    # captures most of the well-documented income signal (married-civ-spouse
    # households skew higher income) in one clean binary flag.
    df['is_married'] = df['marital-status'].isin(
        ['Married-civ-spouse', 'Married-AF-spouse']
    ).astype(int)

    # fnlwgt is a census sampling weight, not an attribute of the person
    # (near-zero correlation with the target - see inspect_correlated_features);
    # education duplicates education-num (perfectly 1:1 redundant). Drop both.
    df = df.drop(columns=['fnlwgt', 'education'])

    print("Engineered features - sample:")
    print(df[[
        'capital-gain', 'capital-loss', 'has_capital_gain', 'has_capital_loss',
        'marital-status', 'is_married',
    ]].head())
    print()
    print("has_capital_gain distribution:")
    print(df['has_capital_gain'].value_counts())
    print()
    print("is_married distribution:")
    print(df['is_married'].value_counts())
    print()

    return df


def compare_feature_engineering_impact(baseline_metrics, engineered_metrics) -> None:
    """Retrain and compare against the baseline."""
    print("Before vs. after feature engineering:")
    print(f"{'metric':<10}{'baseline':>10}{'engineered':>12}{'delta':>10}")
    for metric in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc'):
        before = baseline_metrics[metric]
        after = engineered_metrics[metric]
        print(f"{metric:<10}{before:>10.3f}{after:>12.3f}{after - before:>+10.3f}")
    print()


def inspect_correlated_features(df: pd.DataFrame) -> None:
    """Inspect highly correlated or redundant features."""
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    print("Correlation matrix (numerical features):")
    print(df[numeric_cols].corr().round(3))
    print()

    # fnlwgt is a census sampling weight (how many people this row
    # represents), not an attribute of the person - it should have ~no
    # relationship with the target. Confirm that, then treat it as noise.
    # (by this point in the pipeline, inspect_data() has already mapped
    # class to 0/1, so use it directly rather than re-mapping strings.)
    if 'class' in df.columns:
        fnlwgt_corr = df['fnlwgt'].corr(df['class'])
        print(f"fnlwgt correlation with target: {fnlwgt_corr:.3f} -> noise, not a real feature, drop it")
        print()

    # education and education-num encode the same information (categorical
    # label vs. its ordinal rank). If every education level maps to exactly
    # one education-num value, they're perfectly redundant - keep the
    # numeric one and drop the one-hot-encoded string version.
    mapping = df.groupby('education')['education-num'].nunique()
    is_redundant = (mapping == 1).all()
    print(f"education -> education-num is 1:1 redundant: {is_redundant} -> drop 'education', keep 'education-num'")
    print()


def summarize_findings() -> None:
    """Explain how you would handle a categorical feature with very high cardinality."""
    print("Summary")
    print("-------")
    print(
        "native-country is the high-cardinality column here: 41 unique values, but\n"
        "United-States alone is 89.7% of rows, and the remaining 10.3% is a long thin tail\n"
        "(Mexico ~2%, then a fast drop-off to countries with a few hundred or even single-digit\n"
        "rows). Plain OneHotEncoder(handle_unknown='ignore') technically works here (fit/predict\n"
        "won't crash), but most of those 41 columns are near-constant zero and get almost no\n"
        "training examples to learn a stable weight from - that's overfitting risk with very\n"
        "little signal to show for it, and any country absent from the training split gets\n"
        "silently zeroed out at inference anyway.\n"
    )
    print(
        "Options, roughly in order of how much engineering effort they cost:\n"
        "1) Frequency/count encoding - replace each category with its training-set frequency,\n"
        "   collapsing cardinality to one numeric column. Cheap, no leakage risk, works well for\n"
        "   tree models; weaker for linear models since frequency itself may not be linearly\n"
        "   related to the target.\n"
        "2) Bucket rare categories into 'Other' - keep the top-N most frequent categories\n"
        "   (e.g. top 10 countries) and fold everything else into one 'Other' bucket, the same\n"
        "   trick used for the Titanic deck feature's 'Unknown' category. Cuts cardinality a lot\n"
        "   while staying interpretable and leakage-free.\n"
        "3) Target encoding (mean target rate per category) - most powerful, but needs\n"
        "   out-of-fold computation to avoid leaking test-set target info into the category means,\n"
        "   and needs smoothing/regularization so rare categories don't just memorize a single\n"
        "   row's label.\n"
    )
    print(
        "For this specific column I'd skip all of the above in favor of a domain-specific\n"
        "collapse: a single is_us_native binary flag (or, if the non-US signal matters, a coarse\n"
        "continent/region grouping). Given 89.7% of rows are already one category, almost all the\n"
        "real predictive signal is US-vs-not; trying to distinguish among the ~30 countries with a\n"
        "handful of rows each is mostly fitting noise, not learning a genuine relationship. That\n"
        "also sidesteps target encoding's leakage risk entirely, at the cost of discarding\n"
        "whatever small amount of real country-level signal might exist beyond US-vs-not."
    )


def main():
    print("Data loading and pre-processing...")
    data = load_data()
    df = data.frame
    df = inspect_data(df)

    df = handle_missing_values(df)
    numerical_cols, categorical_cols = identify_feature_types(df)
    X, y = split_features_class(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    model = build_pipeline(numerical_cols, categorical_cols)
    model.fit(X_train, y_train)
    baseline_metrics = evaluate_model(model, X_test, y_test, ["teen", "adult"])

    # inspect fnlwgt/education before engineer_features drops them
    inspect_correlated_features(df)
    df_engineered = engineer_features(df)

    numerical_cols_fe, categorical_cols_fe = identify_feature_types(df_engineered)
    X_fe, y_fe = split_features_class(df_engineered)
    X_train_fe, X_test_fe, y_train_fe, y_test_fe = split_train_test(X_fe, y_fe)
    model_fe = build_pipeline(numerical_cols_fe, categorical_cols_fe)
    model_fe.fit(X_train_fe, y_train_fe)
    engineered_metrics = evaluate_model(model_fe, X_test_fe, y_test_fe, ["teen", "adult"])

    compare_feature_engineering_impact(baseline_metrics, engineered_metrics)

    summarize_findings()


if __name__ == "__main__":
    main()
