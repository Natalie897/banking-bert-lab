"""Task-level metrics. Macro-F1 is a primary metric for multi-class intent classification."""

from __future__ import annotations

from collections import Counter


def classification_metrics(y_true: list[int], y_pred: list[int], names: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Predictions and labels must have equal nonzero lengths")
    label_ids = list(range(len(names)))
    if any(i not in label_ids for i in [*y_true, *y_pred]):
        raise ValueError("Unknown class id in labels or predictions")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)), 6),
        "classes": {
            name: {
                "precision": round(float(p), 6),
                "recall": round(float(r), 6),
                "f1": round(float(f), 6),
                "support": int(n),
            }
            for name, p, r, f, n in zip(names, precision, recall, f1, support)
        },
    }


def collect_errors(rows: list[dict], predictions: list[int], names: list[str], limit: int = 30) -> list[dict]:
    if len(rows) != len(predictions):
        raise ValueError("Rows and predictions must have equal lengths")
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    return [
        {"text": row["text"], "actual": names[row["label"]], "predicted": names[pred]}
        for row, pred in zip(rows, predictions)
        if pred != row["label"]
    ][:limit]


def class_distribution(rows: list[dict], names: list[str]) -> dict[str, int]:
    counts = Counter(row["label"] for row in rows)
    return {name: counts.get(i, 0) for i, name in enumerate(names)}
