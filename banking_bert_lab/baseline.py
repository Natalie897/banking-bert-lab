"""TF-IDF + Logistic Regression reference model; no neural network required."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .metrics import classification_metrics


def train_baseline(train_rows: list[dict], val_rows: list[dict], test_rows: list[dict], names: list[str]) -> dict:
    if not train_rows:
        raise ValueError("Training dataset cannot be empty")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20_000)),
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit([r["text"] for r in train_rows], [r["label"] for r in train_rows])
    v_pred = pipe.predict([r["text"] for r in val_rows]).tolist()
    t_pred = pipe.predict([r["text"] for r in test_rows]).tolist()
    return {
        "name": "tfidf_logistic_regression",
        "validation": classification_metrics([r["label"] for r in val_rows], v_pred, names),
        "test": classification_metrics([r["label"] for r in test_rows], t_pred, names),
        "test_predictions": t_pred,
    }
