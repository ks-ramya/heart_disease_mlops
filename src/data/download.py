"""Acquire the UCI Heart Disease (Cleveland) dataset.

Tries four sources in priority order:
    1. **Local file** at ``LOCAL_CLEVELAND_PATH`` (env: ``CLEVELAND_DATA_PATH``;
       default: ``../heart+disease/processed.cleveland.data`` relative to repo).
    2. ``ucimlrepo`` Python package (official UCI mirror).
    3. UCI raw ``processed.cleveland.data`` over HTTPS.
    4. Public GitHub mirror (last resort).

The dataset is normalised to 14 columns with a binary ``target`` column
(0 = no disease, 1 = disease) and saved to ``data/raw/heart_disease.csv``.

Usage:
    python -m src.data.download
    python -m src.data.download --force                       # refresh
    CLEVELAND_DATA_PATH=/abs/path python -m src.data.download # custom path
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import PROJECT_ROOT, RAW_CSV, FEATURE_COLUMNS, TARGET_COLUMN

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
GITHUB_FALLBACK_URL = (
    "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"
)
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

# Resolve the local Cleveland file from (in priority order):
#   1. CLEVELAND_DATA_PATH env var (explicit override)
#   2. <repo>/data/raw/processed.cleveland.data (bundled with the repo)
#   3. <repo>/../heart+disease/processed.cleveland.data (sibling dataset folder)
_BUNDLED_LOCAL_PATH = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"
_SIBLING_LOCAL_PATH = PROJECT_ROOT.parent / "heart+disease" / "processed.cleveland.data"
_env_path = os.getenv("CLEVELAND_DATA_PATH")
if _env_path:
    LOCAL_CLEVELAND_PATH = Path(_env_path)
elif _BUNDLED_LOCAL_PATH.exists():
    LOCAL_CLEVELAND_PATH = _BUNDLED_LOCAL_PATH
else:
    LOCAL_CLEVELAND_PATH = _SIBLING_LOCAL_PATH


def _from_local_file() -> Optional[pd.DataFrame]:
    """Read the canonical UCI ``processed.cleveland.data`` from disk."""
    if not LOCAL_CLEVELAND_PATH.exists():
        print(f"  [local] not found: {LOCAL_CLEVELAND_PATH}")
        return None
    try:
        df = pd.read_csv(LOCAL_CLEVELAND_PATH, header=None,
                         names=COLUMN_NAMES, na_values="?")
        df = df.rename(columns={"num": TARGET_COLUMN})
        df[TARGET_COLUMN] = (df[TARGET_COLUMN] > 0).astype(int)
        print(f"  [local] read {len(df)} rows from {LOCAL_CLEVELAND_PATH}")
        return df
    except Exception as e:
        print(f"  [local] failed: {e}")
        return None


def _from_ucimlrepo() -> Optional[pd.DataFrame]:
    """Fetch via the official ucimlrepo package."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        return None
    try:
        ds = fetch_ucirepo(id=45)  # 45 = Heart Disease
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        df.columns = [c.lower() for c in df.columns]
        if "num" in df.columns:
            df = df.rename(columns={"num": TARGET_COLUMN})
        df[TARGET_COLUMN] = (df[TARGET_COLUMN] > 0).astype(int)
        return df
    except Exception as e:  # pragma: no cover - network path
        print(f"  [ucimlrepo] failed: {e}")
        return None


def _from_uci_raw() -> Optional[pd.DataFrame]:
    """Fetch the raw .data file directly from UCI."""
    try:
        import requests
        r = requests.get(UCI_URL, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), header=None, names=COLUMN_NAMES,
                         na_values="?")
        df = df.rename(columns={"num": TARGET_COLUMN})
        df[TARGET_COLUMN] = (df[TARGET_COLUMN] > 0).astype(int)
        return df
    except Exception as e:  # pragma: no cover - network path
        print(f"  [UCI raw] failed: {e}")
        return None


def _from_github_fallback() -> Optional[pd.DataFrame]:
    """Fetch a clean copy from a public GitHub mirror."""
    try:
        df = pd.read_csv(GITHUB_FALLBACK_URL)
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:  # pragma: no cover - network path
        print(f"  [GitHub fallback] failed: {e}")
        return None


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply minimal cleaning: enforce schema, drop NA rows, cast types."""
    df = df.copy()
    keep = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c in df.columns]
    df = df[keep]
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df)} rows with missing values "
              f"({len(df)} remain).")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def download(force: bool = False) -> pd.DataFrame:
    """Fetch + clean + persist the dataset to ``RAW_CSV``."""
    if RAW_CSV.exists() and not force:
        print(f"  Dataset already present at {RAW_CSV} (use --force to refresh).")
        return pd.read_csv(RAW_CSV)

    print("  Fetching UCI Heart Disease dataset...")
    df: Optional[pd.DataFrame] = None
    for source_fn, name in (
        (_from_local_file, "local file"),
        (_from_ucimlrepo, "ucimlrepo"),
        (_from_uci_raw, "UCI raw"),
        (_from_github_fallback, "GitHub fallback"),
    ):
        print(f"  -> trying source: {name}")
        df = source_fn()
        if df is not None and not df.empty:
            print(f"  Loaded from {name}: {df.shape[0]} rows x {df.shape[1]} cols.")
            break

    if df is None or df.empty:
        raise RuntimeError("All download sources failed. Check your network.")

    df = _clean(df)
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_CSV, index=False)
    print(f"  Saved cleaned dataset -> {RAW_CSV}")
    print(f"  Class balance:\n{df[TARGET_COLUMN].value_counts().to_string()}")
    return df


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = p.parse_args()
    download(force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
