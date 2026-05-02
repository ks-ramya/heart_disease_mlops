"""Preprocessing utilities and reusable sklearn pipeline.

Exposes:
    * ``build_preprocessor()`` -> ``ColumnTransformer`` (scaling + one-hot)
    * ``split_dataset(df)``    -> stratified train/test DataFrames
    * ``load_processed()``     -> (train_df, test_df) from disk; (re)builds if absent
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    RAW_CSV,
    TARGET_COLUMN,
    TEST_CSV,
    TEST_SIZE,
    TRAIN_CSV,
)


def build_preprocessor() -> ColumnTransformer:
    """Return a ColumnTransformer that scales numerics + one-hot encodes categoricals.

    Compatible with both sklearn >= 1.2 (``sparse_output``) and older
    versions (``sparse``). Wrapped in try/except for portability.
    """
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - sklearn < 1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", ohe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split on TARGET_COLUMN."""
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_processed() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed train/test sets, generating them from raw if missing."""
    if not (TRAIN_CSV.exists() and TEST_CSV.exists()):
        if not RAW_CSV.exists():
            from src.data.download import download
            download()
        df = pd.read_csv(RAW_CSV)
        train_df, test_df = split_dataset(df)
        train_df.to_csv(TRAIN_CSV, index=False)
        test_df.to_csv(TEST_CSV, index=False)
        print(f"  Wrote processed splits -> {TRAIN_CSV}, {TEST_CSV}")
    return pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix X and target y."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COLUMN].astype(int).copy()
    return X, y
