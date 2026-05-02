"""Evaluate the persisted model on the held-out test split.

Usage:
    python -m src.models.evaluate
"""

from __future__ import annotations

import json
import sys

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import METRICS_PATH, MODEL_PATH
from src.data.preprocess import load_processed, split_xy


def evaluate() -> dict:
    if not MODEL_PATH.exists():
        print(f"  Model not found at {MODEL_PATH}. Run `python -m src.models.train` first.",
              file=sys.stderr)
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    _, test_df = load_processed()
    X, y = split_xy(test_df)

    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }

    print("  Classification report:")
    print(classification_report(y, pred, target_names=["no_disease", "disease"]))
    print(f"  Metrics: {json.dumps({k: v for k, v in metrics.items() if k != 'confusion_matrix'}, indent=2)}")
    print(f"  Confusion matrix: {metrics['confusion_matrix']}")

    # Append to metrics.json under "holdout_eval"
    summary = {}
    if METRICS_PATH.exists():
        try:
            summary = json.loads(METRICS_PATH.read_text())
        except json.JSONDecodeError:
            summary = {}
    summary["holdout_eval"] = metrics
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    print(f"  Updated -> {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    evaluate()
