"""CLI inference for the persisted Heart Disease pipeline.

Loads ``models/heart_disease_model.pkl`` and runs predictions from either
a CSV file (one row per patient) or a JSON payload (a single object or
a list of objects). Output is JSON to stdout so it composes with ``jq``.

Examples
--------
Single record from inline JSON::

    python scripts/predict.py --json '{
        "age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,
        "restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,
        "slope":0,"ca":0,"thal":1
    }'

Batch from a CSV (must contain the 13 feature columns)::

    python scripts/predict.py --csv data/processed/test.csv | jq '.[0:3]'

The script intentionally has zero hard dependencies beyond the project
requirements: it reuses ``src.config.FEATURE_COLUMNS`` so the input
schema stays in sync with training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

# Allow running as `python scripts/predict.py ...` from the repo root
# without needing `python -m scripts.predict`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import FEATURE_COLUMNS, MODEL_PATH, TARGET_LABELS  # noqa: E402


def _load_pipeline(model_path: Path):
    if not model_path.exists():
        sys.stderr.write(
            f"error: model file not found at {model_path}\n"
            "       run `make train` (or `python -m src.models.train`) first\n"
        )
        sys.exit(2)
    return joblib.load(model_path)


def _records_to_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        sys.stderr.write(
            f"error: input is missing required column(s): {missing}\n"
            f"       expected columns: {FEATURE_COLUMNS}\n"
        )
        sys.exit(2)
    return df[FEATURE_COLUMNS].copy()


def _predict(pipeline, df: pd.DataFrame) -> List[Dict[str, Any]]:
    preds = pipeline.predict(df).astype(int).tolist()
    proba = pipeline.predict_proba(df).tolist()
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(preds):
        probs = proba[i]
        out.append({
            "row": i,
            "prediction": p,
            "label": TARGET_LABELS[p],
            "confidence": round(float(max(probs)), 4),
            "probabilities": {
                TARGET_LABELS[0]: round(float(probs[0]), 4),
                TARGET_LABELS[1]: round(float(probs[1]), 4),
            },
        })
    return out


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run inference using the persisted Heart Disease pipeline.",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="Path to a CSV with the 13 feature columns.")
    src.add_argument("--json", type=str,
                     help="Inline JSON: single object or list of objects.")
    src.add_argument("--json-file", type=Path,
                     help="Path to a .json file (object or list).")
    p.add_argument("--model", type=Path, default=MODEL_PATH,
                   help=f"Path to the joblib model (default: {MODEL_PATH}).")
    p.add_argument("--limit", type=int, default=None,
                   help="Only predict on the first N rows (handy for sanity checks).")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    pipeline = _load_pipeline(args.model)

    if args.csv:
        df = pd.read_csv(args.csv)
        # Drop the target column if the user passed a labelled file by mistake.
        df = df[[c for c in df.columns if c in FEATURE_COLUMNS]]
        df = df.reindex(columns=FEATURE_COLUMNS)
    else:
        if args.json:
            payload = json.loads(args.json)
        else:
            payload = json.loads(args.json_file.read_text())
        if isinstance(payload, dict):
            payload = [payload]
        df = _records_to_df(payload)

    if args.limit is not None:
        df = df.head(args.limit)

    results = _predict(pipeline, df)
    json.dump(results, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
