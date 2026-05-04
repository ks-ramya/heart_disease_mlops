"""Render reports/figures/architecture.png from a small box-and-arrow spec.

Used so the docx report can embed a real image instead of a Mermaid block.
Run from the repo root:

    python scripts/_render_architecture.py
"""

from pathlib import Path

import matplotlib.patches as mp
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "reports" / "figures" / "architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Three colour bands so the planes are visually distinct
COL_DATA = "#E8F1FA"   # data + training
COL_DEPLOY = "#EAF6EC"  # build + deploy
COL_OBS = "#FBEFE2"    # observability
EDGE = "#3A3A3A"

fig, ax = plt.subplots(figsize=(13, 7.5), dpi=150)
ax.set_xlim(0, 13)
ax.set_ylim(0, 7.5)
ax.axis("off")


def box(x, y, w, h, text, fill, *, bold=False):
    rect = mp.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.2, edgecolor=EDGE, facecolor=fill,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=9.5,
        fontweight="bold" if bold else "normal",
    )


def arrow(x1, y1, x2, y2, label="", style="-|>", curve=0.0):
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style, color=EDGE, lw=1.2,
            connectionstyle=f"arc3,rad={curve}",
        ),
    )
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label,
                fontsize=8, ha="center", style="italic", color="#333")


# ── Plane labels (left margin) ──────────────────────────────────────
ax.text(0.15, 6.7, "Train", fontsize=10, fontweight="bold", color="#1f4c7a")
ax.text(0.15, 4.0, "Build & deploy", fontsize=10, fontweight="bold", color="#2d6b3a")
ax.text(0.15, 1.2, "Observe", fontsize=10, fontweight="bold", color="#8a4a1a")

# ── Training plane (top) ────────────────────────────────────────────
box(1.4, 6.0, 2.0, 0.9, "UCI Cleveland\n303 rows × 14 cols", COL_DATA)
box(4.0, 6.0, 2.2, 0.9, "Preprocess\nscale + one-hot", COL_DATA)
box(6.8, 6.0, 2.2, 0.9, "Train + 5-fold CV\nLR / RF (+ GB)", COL_DATA)
box(9.6, 6.0, 2.2, 0.9, "MLflow\n7 runs", COL_DATA)
box(6.8, 4.7, 2.2, 0.7, "model.pkl\n(joblib)", COL_DATA, bold=True)

arrow(3.4, 6.45, 4.0, 6.45)
arrow(6.2, 6.45, 6.8, 6.45)
arrow(9.0, 6.45, 9.6, 6.45)
arrow(7.9, 6.0, 7.9, 5.4, label="best run")
arrow(9.0, 6.0, 9.6, 5.0, label="metrics+plots", curve=0.2)

# ── Deploy plane (middle) ───────────────────────────────────────────
box(1.4, 3.4, 2.0, 0.9, "GitHub repo", COL_DEPLOY)
box(4.0, 3.4, 2.2, 0.9, "GitHub Actions\nCI: lint+test+train\nCD: build+push", COL_DEPLOY)
box(6.8, 3.4, 2.2, 0.9, "Docker image\nghcr.io / latest", COL_DEPLOY, bold=True)
box(9.6, 3.4, 2.2, 0.9, "Kubernetes\nDeploy + HPA\nSvc + Ingress", COL_DEPLOY)

arrow(7.9, 4.7, 7.9, 4.3, label="baked into image")
arrow(3.4, 3.85, 4.0, 3.85)
arrow(6.2, 3.85, 6.8, 3.85, label="smoke /health")
arrow(9.0, 3.85, 9.6, 3.85, label="pull")

# ── Observability plane (bottom) ────────────────────────────────────
box(4.0, 0.6, 2.2, 0.9, "stdout JSON logs\nrequest_id, ms", COL_OBS)
box(6.8, 0.6, 2.2, 0.9, "Prometheus\n/metrics scrape", COL_OBS)
box(9.6, 0.6, 2.2, 0.9, "Grafana\n9 panels", COL_OBS)

# Pod box bridging deploy → observability
box(6.8, 2.0, 2.2, 0.9, "FastAPI pod\n/predict /health\n/metrics /ui", COL_DEPLOY)
arrow(7.9, 3.4, 7.9, 2.9, label="runs as")
arrow(7.9, 2.0, 5.1, 1.5, label="logs", curve=-0.15)
arrow(7.9, 2.0, 7.9, 1.5)
arrow(9.0, 1.05, 9.6, 1.05, label="dashboards")

# ── Title + caption ─────────────────────────────────────────────────
fig.text(0.5, 0.97, "Heart Disease MLOps — System Architecture",
         ha="center", fontsize=13, fontweight="bold")
fig.text(0.5, 0.03,
         "Three planes share one artefact: the trained pipeline (model.pkl). "
         "Same image runs in compose locally and in Minikube; metrics scrape lives in both.",
         ha="center", fontsize=8.5, style="italic", color="#444")

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT.relative_to(Path.cwd())} ({OUT.stat().st_size} bytes)")
