# Question 6: SMS Spam Detection

## Problem

Build a text classifier to predict whether an SMS message is **spam or ham**.

Use the UCI SMS Spam dataset from Hugging Face.

```bash
pip install datasets
```

```python
from datasets import load_dataset

dataset = load_dataset("ucirvine/sms_spam")
df = dataset["train"].to_pandas()
```

## Expected Output

- Text classification pipeline
- Predictions on a held-out test set
- Precision, recall, F1, PR-AUC, and confusion matrix
- Comparison of text-only vs text + engineered features

## Tasks

1. Load the dataset into pandas.
2. Inspect class balance, missing values, duplicates, and message lengths.
3. Create a stratified train/test split.
4. Build a TF-IDF based text-classification pipeline.
5. Train a baseline classifier.
6. Evaluate precision, recall, F1, PR-AUC, and confusion matrix.
7. Engineer at least three text-derived features, such as:
   - Message length
   - Number of digits
   - Number of uppercase characters
   - URL or currency indicators
8. Combine TF-IDF and engineered numerical features.
9. Compare against the text-only baseline.
10. Inspect representative false positives and false negatives.

## Time Limit

**40–45 minutes**
