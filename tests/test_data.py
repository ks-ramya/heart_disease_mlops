"""Tests for the data loading + preprocessing layer."""

from __future__ import annotations

import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data.preprocess import build_preprocessor, split_dataset, split_xy


def test_synthetic_dataset_schema(synthetic_df):
    """Synthetic data should match the documented schema exactly."""
    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN])
    assert set(synthetic_df.columns) == expected
    assert synthetic_df[TARGET_COLUMN].dtype.kind in "iu"
    assert set(synthetic_df[TARGET_COLUMN].unique()).issubset({0, 1})


def test_split_dataset_proportions(synthetic_df):
    train_df, test_df = split_dataset(synthetic_df)
    assert len(train_df) + len(test_df) == len(synthetic_df)
    expected_test = int(round(len(synthetic_df) * TEST_SIZE))
    assert abs(len(test_df) - expected_test) <= 1


def test_split_dataset_is_stratified(synthetic_df):
    train_df, test_df = split_dataset(synthetic_df)
    full_ratio = synthetic_df[TARGET_COLUMN].mean()
    train_ratio = train_df[TARGET_COLUMN].mean()
    test_ratio = test_df[TARGET_COLUMN].mean()
    assert abs(train_ratio - full_ratio) < 0.05
    assert abs(test_ratio - full_ratio) < 0.05


def test_split_xy_shapes(synthetic_df):
    X, y = split_xy(synthetic_df)
    assert X.shape[0] == y.shape[0] == len(synthetic_df)
    assert TARGET_COLUMN not in X.columns
    assert set(X.columns) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def test_build_preprocessor_output_shape(synthetic_df):
    """Preprocessor must produce a finite numeric matrix."""
    X, _ = split_xy(synthetic_df)
    pre = build_preprocessor()
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == len(X)
    assert Xt.shape[1] >= len(NUMERIC_FEATURES)
    assert pd.DataFrame(Xt).isna().sum().sum() == 0
