"""Run multiple MLflow experiment variants for the assignment deliverable.

Each variant trains a different (model, hyperparameter) combination on the
same processed train/test split and logs params, CV + test metrics, the
confusion-matrix and ROC-curve plots, and the fitted sklearn pipeline as
artifacts to MLflow under the existing 'heart-disease-classification'
experiment.

Usage:
    python -m scripts.run_experiments              # all variants
    python -m scripts.run_experiments lr_baseline  # one variant
"""

from __future__ import annotations

import sys
import warnings
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import (
    CV_FOLDS,
    FIGURE_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
)
from src.data.preprocess import build_preprocessor, load_processed, split_xy

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Variant catalogue ────────────────────────────────────────────────
VARIANTS: Dict[str, Dict[str, Any]] = {
    "lr_baseline": {
        "estimator": LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs",
            max_iter=2000, random_state=RANDOM_STATE,
        ),
        "notes": "Logistic Regression, default C=1.0",
    },
    "lr_strong_reg": {
        "estimator": LogisticRegression(
            C=0.01, penalty="l2", solver="lbfgs",
            max_iter=2000, random_state=RANDOM_STATE,
        ),
        "notes": "Logistic Regression with strong L2 regularisation (C=0.01)",
    },
    "rf_shallow": {
        "estimator": RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_split=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "notes": "Random Forest, shallow trees (max_depth=5)",
    },
    "rf_deep": {
        "estimator": RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_split=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "notes": "Random Forest, deep unrestricted trees",
    },
    "gb_default": {
        "estimator": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "notes": "Gradient Boosting, n_estimators=200, lr=0.1",
    },
}


def _make_pipeline(estimator) -> Pipeline:
    return Pipeline(steps=[("preprocessor", build_preprocessor()),
                           ("classifier", estimator)])


def _evaluate(model, X, y) -> Dict[str, float]:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "test_accuracy": float(accuracy_score(y, pred)),
        "test_precision": float(precision_score(y, pred, zero_division=0)),
        "test_recall": float(recall_score(y, pred, zero_division=0)),
        "test_f1": float(f1_score(y, pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y, proba)),
    }


def _log_plots(model, X_test, y_test, name: str) -> None:
    cm_path = FIGURE_DIR / f"confusion_matrix_{name}.png"
    roc_path = FIGURE_DIR / f"roc_curve_{name}.png"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

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

    mlflow.log_artifact(str(cm_path), artifact_path="plots")
    mlflow.log_artifact(str(roc_path), artifact_path="plots")


def run_variant(name: str, X_train, y_train, X_test, y_test, cv) -> Dict[str, Any]:
    spec = VARIANTS[name]
    pipe = _make_pipeline(spec["estimator"])

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("variant", name)
        mlflow.set_tag("notes", spec["notes"])
        mlflow.log_param("model_class", type(spec["estimator"]).__name__)
        for k, v in spec["estimator"].get_params(deep=False).items():
            mlflow.log_param(k, v)

        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                    scoring="roc_auc", n_jobs=-1)
        mlflow.log_metric("cv_roc_auc_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_roc_auc_std", float(cv_scores.std()))

        pipe.fit(X_train, y_train)
        for k, v in _evaluate(pipe, X_test, y_test).items():
            mlflow.log_metric(k, v)

        _log_plots(pipe, X_test, y_test, name)
        mlflow.sklearn.log_model(pipe, artifact_path="model")
        print(f"  ✓ {name:<14}  cv_roc_auc={cv_scores.mean():.4f}  run_id={run.info.run_id[:12]}")
        return {"run_id": run.info.run_id, "cv_roc_auc": float(cv_scores.mean())}


def main(selected: List[str]) -> None:
    train_df, test_df = load_processed()
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"MLflow URI : {MLFLOW_TRACKING_URI}")
    print(f"Experiment : {MLFLOW_EXPERIMENT_NAME}")
    print(f"Variants   : {selected}\n")

    for name in selected:
        if name not in VARIANTS:
            print(f"  ! unknown variant '{name}' — skipping (known: {list(VARIANTS)})")
            continue
        run_variant(name, X_train, y_train, X_test, y_test, cv)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args if args else list(VARIANTS))
