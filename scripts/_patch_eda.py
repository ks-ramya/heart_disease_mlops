"""One-shot helper: inject 4 new EDA sections into notebooks/01_eda.ipynb.

Gap 1: show raw (pre-clean) missing-value counts loaded from
       processed.cleveland.data with na_values='?'.
Gap 2: ranked |corr(feature, target)| horizontal bar chart.
Gap 3: per-class boxplots for numeric features.
Gap 4: outlier discussion appended to the key-takeaways markdown.

Idempotent: re-running it after the cells already exist is a no-op.
"""
from __future__ import annotations

from pathlib import Path
import nbformat

NB_PATH = Path("notebooks/01_eda.ipynb")
nb = nbformat.read(NB_PATH, as_version=4)


def code(src: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(src.strip("\n"))


def md(src: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(src.strip("\n"))


def has_md(prefix: str) -> bool:
    return any(c.cell_type == "markdown" and c.source.lstrip().startswith(prefix)
               for c in nb.cells)


# --- Gap 1: raw missing values, inserted right after the "## 2." heading ---
gap1_md = md("""
**Gap-1 evidence — raw missing values *before* cleaning.**
The cleaned `data/raw/heart_disease.csv` already has zero NAs, so we re-read the
canonical UCI file `processed.cleveland.data` with `na_values='?'` to show that
the `?` markers in `ca` and `thal` were correctly handled by `download.py::_clean()`.
""")
gap1_code = code("""
from src.data.download import LOCAL_CLEVELAND_PATH, COLUMN_NAMES

raw = pd.read_csv(LOCAL_CLEVELAND_PATH, header=None,
                  names=COLUMN_NAMES, na_values='?')
raw_missing = raw.isna().sum()
raw_missing = raw_missing[raw_missing > 0]
print(f'Raw rows : {len(raw)}')
print(f'Cleaned  : {len(df)}  (dropped {len(raw) - len(df)} rows with NA)')
print('\\nMissing values per column (raw, before cleaning):')
print(raw_missing.to_string() if len(raw_missing) else '  (none)')
""")

# --- Gap 3: per-class boxplots for numeric features ---
gap3_md = md("## 3b. Per-class boxplots (numeric features)")
gap3_code = code("""
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, col in zip(axes.flat, NUMERIC_FEATURES):
    sns.boxplot(data=df, x=TARGET_COLUMN, y=col,
                hue=TARGET_COLUMN, legend=False,
                palette={0: '#4C9F70', 1: '#D9534F'}, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([TARGET_LABELS[0], TARGET_LABELS[1]])
    ax.set_xlabel(''); ax.set_title(col)
for ax in axes.flat[len(NUMERIC_FEATURES):]:
    ax.axis('off')
plt.tight_layout(); plt.savefig(FIGURE_DIR / 'numeric_boxplots.png', dpi=120); plt.show()
""")

# --- Gap 2: ranked correlations with target ---
gap2_md = md("## 5b. Feature association with target (ranked |corr|)")
gap2_code = code("""
target_corr = (df.corr(numeric_only=True)[TARGET_COLUMN]
                 .drop(TARGET_COLUMN)
                 .abs()
                 .sort_values())
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(target_corr.index, target_corr.values, color='#3B7DB1')
ax.set_xlabel('|Pearson correlation with target|')
ax.set_title('Feature association with disease label')
for i, v in enumerate(target_corr.values):
    ax.text(v + 0.005, i, f'{v:.2f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(FIGURE_DIR / 'target_correlation_ranked.png', dpi=120); plt.show()
print('\\nTop 5 features by |corr| with target:')
print(target_corr.sort_values(ascending=False).head(5).round(3).to_string())
""")

# --- Gap 4: outlier discussion appended to takeaways (idempotent) ---
takeaways_idx = next(
    i for i, c in enumerate(nb.cells)
    if c.cell_type == "markdown" and c.source.lstrip().startswith("## 6.")
)
OUTLIER_NOTE = (
    "\n* **Outliers:** `chol` has a small number of values >400 (max ~564) "
    "and `trestbps` has a few >180 — kept in the dataset since they are "
    "physiologically plausible and tree-based models handle them gracefully; "
    "`StandardScaler` reduces their leverage on the linear model. The boxplots "
    "in section 3b make these points visible per class."
)
if "Outliers:" not in nb.cells[takeaways_idx].source:
    nb.cells[takeaways_idx].source = (
        nb.cells[takeaways_idx].source.rstrip() + OUTLIER_NOTE
    )


def find_md(prefix: str) -> int:
    return next(i for i, c in enumerate(nb.cells)
                if c.cell_type == "markdown" and c.source.lstrip().startswith(prefix))


# Insert from bottom to top so earlier indices stay valid.
if not has_md("## 5b."):
    idx_corr = find_md("## 5. Correlation")
    nb.cells[idx_corr + 2:idx_corr + 2] = [gap2_md, gap2_code]

if not has_md("## 3b."):
    idx_num = find_md("## 3. Numeric distributions")
    nb.cells[idx_num + 2:idx_num + 2] = [gap3_md, gap3_code]

if not has_md("**Gap-1 evidence"):
    idx_miss = find_md("## 2. Missing values")
    nb.cells[idx_miss + 1:idx_miss + 1] = [gap1_md, gap1_code]

nbformat.write(nb, NB_PATH)
print(f"Patched {NB_PATH} ({len(nb.cells)} cells total)")
