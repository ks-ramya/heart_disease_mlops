"""Tests for the training pipeline and persisted model contract."""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline


def test_pipeline_is_sklearn_pipeline(trained_pipeline):
    assert isinstance(trained_pipeline, Pipeline)
    assert "preprocessor" in trained_pipeline.named_steps
    assert "classifier" in trained_pipeline.named_steps


def test_pipeline_predict_shape(trained_pipeline, synthetic_df):
    X = synthetic_df.drop(columns=["target"])
    preds = trained_pipeline.predict(X)
    assert preds.shape == (len(X),)
    assert set(np.unique(preds)).issubset({0, 1})


def test_pipeline_predict_proba_sums_to_one(trained_pipeline, synthetic_df):
    X = synthetic_df.drop(columns=["target"]).head(10)
    proba = trained_pipeline.predict_proba(X)
    assert proba.shape == (10, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_pipeline_minimum_accuracy(trained_pipeline, synthetic_df):
    """Sanity check: model should beat the trivial majority-class baseline."""
    X = synthetic_df.drop(columns=["target"])
    y = synthetic_df["target"]
    acc = trained_pipeline.score(X, y)
    baseline = max(y.mean(), 1 - y.mean())
    assert acc >= baseline, f"Model accuracy {acc:.3f} <= baseline {baseline:.3f}"


def test_pipeline_handles_single_record(trained_pipeline, synthetic_df):
    X = synthetic_df.drop(columns=["target"]).iloc[[0]]
    pred = trained_pipeline.predict(X)
    proba = trained_pipeline.predict_proba(X)
    assert pred.shape == (1,)
    assert proba.shape == (1, 2)
