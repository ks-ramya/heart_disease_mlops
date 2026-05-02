"""Shared pytest fixtures + path setup."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make `import src.*` work when running pytest from any directory
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """A small synthetic dataset that respects the UCI Heart Disease schema.

    We avoid hitting the network in unit tests by generating a deterministic
    sample with the same column names/types and a learnable target.
    """
    from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN

    rng = np.random.default_rng(42)
    n = 200
    data = {
        "age": rng.integers(29, 78, n),
        "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 564, n),
        "thalach": rng.integers(71, 202, n),
        "oldpeak": rng.uniform(0, 6.2, n).round(1),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(0, 4, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "exang": rng.integers(0, 2, n),
        "slope": rng.integers(0, 3, n),
        "ca": rng.integers(0, 5, n),
        "thal": rng.integers(0, 4, n),
    }
    df = pd.DataFrame(data)
    # Build a learnable target: high oldpeak / cp / exang -> disease
    score = (
        0.4 * (df["oldpeak"] > 1.5).astype(int)
        + 0.3 * (df["cp"] >= 2).astype(int)
        + 0.3 * df["exang"]
        + 0.2 * (df["thalach"] < 140).astype(int)
    )
    df[TARGET_COLUMN] = (score + rng.normal(0, 0.1, n) > 0.5).astype(int)
    # Sanity: keep only documented columns
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]]


@pytest.fixture(scope="session")
def trained_pipeline(synthetic_df):
    """A trained sklearn Pipeline used by API/model tests."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN
    from src.data.preprocess import build_preprocessor

    X = synthetic_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = synthetic_df[TARGET_COLUMN]
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit(X, y)
    return pipe


@pytest.fixture
def api_client(trained_pipeline, tmp_path, monkeypatch):
    """Spin up a FastAPI TestClient with the synthetic model loaded."""
    import joblib
    from fastapi.testclient import TestClient

    model_path = tmp_path / "model.pkl"
    joblib.dump(trained_pipeline, model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    # Force re-import so the new MODEL_PATH is honoured
    for mod_name in list(sys.modules):
        if mod_name.startswith("src."):
            sys.modules.pop(mod_name, None)

    from src.api.app import app, _load_model
    import src.api.app as app_module

    app_module._model = _load_model()
    with TestClient(app) as client:
        yield client
