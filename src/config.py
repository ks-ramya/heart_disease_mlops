"""Central configuration for the Heart Disease MLOps project.

All paths, feature lists, and ML hyperparameters live here so every script,
notebook, and test pulls from a single source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODEL_DIR: Path = PROJECT_ROOT / "models"
REPORT_DIR: Path = PROJECT_ROOT / "reports"
FIGURE_DIR: Path = REPORT_DIR / "figures"
MLRUNS_DIR: Path = PROJECT_ROOT / "mlruns"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_CSV: Path = RAW_DATA_DIR / "heart_disease.csv"
TRAIN_CSV: Path = PROCESSED_DATA_DIR / "train.csv"
TEST_CSV: Path = PROCESSED_DATA_DIR / "test.csv"
MODEL_PATH: Path = MODEL_DIR / "heart_disease_model.pkl"
METRICS_PATH: Path = MODEL_DIR / "metrics.json"

# ─── Dataset schema (UCI Heart Disease - Cleveland, 14 attributes) ────
NUMERIC_FEATURES: list[str] = [
    "age", "trestbps", "chol", "thalach", "oldpeak",
]
CATEGORICAL_FEATURES: list[str] = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
]
FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN: str = "target"
TARGET_LABELS: dict[int, str] = {0: "no_disease", 1: "disease"}

# ─── Train/Test ───────────────────────────────────────────────────────
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

# ─── MLflow ───────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI", f"file://{MLRUNS_DIR.as_posix()}"
)
MLFLOW_EXPERIMENT_NAME: str = "heart-disease-classification"

# ─── API ──────────────────────────────────────────────────────────────
API_TITLE: str = "Heart Disease Prediction API"
API_VERSION: str = "1.0.0"
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8080"))
MODEL_PATH_ENV: str = os.getenv("MODEL_PATH", str(MODEL_PATH))
