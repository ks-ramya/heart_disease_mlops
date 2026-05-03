"""Render a static comparison figure of all MLflow runs as PNG.

Used because MLflow's React SPA doesn't render reliably in headless Chrome
for direct screenshot capture. This creates a publication-ready PNG showing
every run's params + metrics side-by-side.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

OUT = "reports/screenshots/05_mlflow_compare_runs.png"
METRIC_COLS = ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
runs = mlflow.search_runs([exp.experiment_id],
                          order_by=["attributes.start_time ASC"],
                          output_format="list")

names, vals = [], []
for r in runs:
    n = r.data.tags.get("mlflow.runName", r.info.run_id[:8])
    row = [r.data.metrics.get(c, np.nan) for c in METRIC_COLS]
    if all(v != v for v in row):
        continue
    names.append(n)
    vals.append(row)

vals = np.array(vals)
n_runs, n_metrics = vals.shape
x = np.arange(n_metrics)
bar_w = 0.8 / n_runs

fig, ax = plt.subplots(figsize=(13, 6.5))
cmap = plt.cm.tab10
for i, (name, row) in enumerate(zip(names, vals)):
    ax.bar(x + i * bar_w - 0.4 + bar_w / 2, row, bar_w,
           label=name, color=cmap(i % 10), edgecolor="white", linewidth=0.7)

ax.set_xticks(x)
ax.set_xticklabels([m.replace("test_", "") for m in METRIC_COLS],
                   fontsize=11, fontweight="bold")
ax.set_ylim(0.7, 1.0)
ax.set_ylabel("Score", fontsize=11)
ax.set_title(f"MLflow experiment '{MLFLOW_EXPERIMENT_NAME}' — {len(names)} runs compared",
             fontsize=13, fontweight="bold", pad=14)
ax.legend(title="Run", loc="upper left", bbox_to_anchor=(1.01, 1.0),
          fontsize=10, frameon=True)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

best_idx = int(np.argmax(vals[:, METRIC_COLS.index("test_roc_auc")]))
ax.text(0.01, 0.97, f"★ best test_roc_auc → {names[best_idx]} ({vals[best_idx, -1]:.4f})",
        transform=ax.transAxes, fontsize=10, fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff7e6",
                  edgecolor="#f59e0b", linewidth=1.2))

fig.tight_layout()
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"WROTE {OUT}  ({n_runs} runs × {n_metrics} metrics)")
