"""Train Logistic Regression + Random Forest, track with MLflow, save best.

Behaviour:
    * Reads (or generates) processed train/test splits.
    * Builds a `Pipeline = preprocessor -> classifier` for each model.
    * Performs `GridSearchCV` (5-fold) tuned on ROC-AUC.
    * Logs params/metrics/artifacts (confusion matrix + ROC curves) to MLflow.
    * Persists the best overall pipeline as ``models/heart_disease_model.pkl``
      and writes a metrics summary to ``models/metrics.json``.

Usage:
    python -m src.models.train
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import (
    CV_FOLDS,
    FIGURE_DIR,
    METRICS_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    RANDOM_STATE,
)
from src.data.preprocess import build_preprocessor, load_processed, split_xy

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Model search space ───────────────────────────────────────────────
MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "logistic_regression": {
        "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "param_grid": {
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs"],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "param_grid": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 5, 10],
            "classifier__min_samples_split": [2, 5],
        },
    },
}


def _make_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[("preprocessor", build_preprocessor()), ("classifier", estimator)]
    )


def _evaluate(model, X, y) -> Dict[str, float]:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


def _log_plots(model, X_test, y_test, name: str, mlflow_mod) -> None:
    """Save & log confusion matrix + ROC curve under reports/figures/."""
    cm_path = FIGURE_DIR / f"confusion_matrix_{name}.png"
    roc_path = FIGURE_DIR / f"roc_curve_{name}.png"

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax,
                                          cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix — {name}")
    fig.tight_layout(); fig.savefig(cm_path, dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title(f"ROC curve — {name}")
    fig.tight_layout(); fig.savefig(roc_path, dpi=120); plt.close(fig)

    if mlflow_mod is not None:
        mlflow_mod.log_artifact(str(cm_path), artifact_path="plots")
        mlflow_mod.log_artifact(str(roc_path), artifact_path="plots")


def train() -> Dict[str, Any]:
    """Train all models, track in MLflow, save best, return summary dict."""
    train_df, test_df = load_processed()
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    # MLflow setup (degrades gracefully if not installed)
    mlflow = None
    try:
        import mlflow as _mlflow
        import mlflow.sklearn  # noqa: F401
        mlflow = _mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        print(f"  MLflow URI: {MLFLOW_TRACKING_URI}")
    except Exception as e:  # pragma: no cover
        print(f"  [WARN] MLflow disabled: {e}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results: Dict[str, Dict[str, Any]] = {}

    for name, spec in MODEL_SPECS.items():
        print(f"\n  == {name} ==")
        pipe = _make_pipeline(spec["estimator"])
        gs = GridSearchCV(pipe, spec["param_grid"], cv=cv, scoring="roc_auc",
                          n_jobs=-1, refit=True, return_train_score=False)

        run_ctx = mlflow.start_run(run_name=name) if mlflow else _NullCtx()
        with run_ctx:
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            cv_score = float(gs.best_score_)
            test_metrics = _evaluate(best, X_test, y_test)

            print(f"    best params: {gs.best_params_}")
            print(f"    cv_roc_auc:  {cv_score:.4f}")
            print(f"    test:        {test_metrics}")

            if mlflow:
                mlflow.log_param("model", name)
                for k, v in gs.best_params_.items():
                    mlflow.log_param(k, v)
                mlflow.log_metric("cv_roc_auc", cv_score)
                for k, v in test_metrics.items():
                    mlflow.log_metric(f"test_{k}", v)
                _log_plots(best, X_test, y_test, name, mlflow)
                try:
                    mlflow.sklearn.log_model(best, artifact_path="model")
                except Exception as e:  # pragma: no cover
                    print(f"    [WARN] MLflow log_model failed: {e}")

            results[name] = {
                "best_params": gs.best_params_,
                "cv_roc_auc": cv_score,
                "test_metrics": test_metrics,
                "model": best,
            }

    # Pick the best by test ROC-AUC
    best_name = max(results, key=lambda k: results[k]["test_metrics"]["roc_auc"])
    best_model = results[best_name]["model"]
    print(f"\n  >>> BEST MODEL: {best_name}")

    # Persist
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"  Saved model -> {MODEL_PATH}")

    summary = {
        "best_model": best_name,
        "models": {
            n: {
                "best_params": r["best_params"],
                "cv_roc_auc": r["cv_roc_auc"],
                "test_metrics": r["test_metrics"],
            }
            for n, r in results.items()
        },
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved metrics -> {METRICS_PATH}")

    print("\n  Final classification report (test set):")
    print(classification_report(y_test, best_model.predict(X_test),
                                target_names=["no_disease", "disease"]))
    return summary


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    train()
