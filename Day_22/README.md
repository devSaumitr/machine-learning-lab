# Day_22

CampusX 100 Days of ML - Automated EDA with Pandas Profiling and Sweetviz.

## Day number

**22**

## Main topics

- Automated exploratory data analysis (EDA) with `ydata-profiling` (formerly `pandas-profiling`).
- Reading the report: overview, alerts, variables, interactions, correlations, missing values, and sample rows.
- Comparing `ydata-profiling` with Sweetviz.
- Separating generated report artifacts from the notebook claims.

## Learning objectives

1. Generate a profiling report from a pandas DataFrame.
2. Interpret dataset-level and column-level report sections.
3. Use alerts as investigation prompts, not automatic cleaning rules.
4. Compare profiling libraries without mixing their outputs.
5. Connect automated EDA to manual validation, cleaning, and leakage-safe preprocessing.

## Notebooks

| File | Role | Cells | Notes |
|---|---|---:|---|
| `my_notebook.ipynb` | Primary ydata-profiling workflow | 8 (4 code, 4 markdown) | Saved with `.venv312`, Python 3.12.10, ydata-profiling 4.18.4 |
| `DAY22_Pandas-Profiling-Sweetviz.ipynb` | Separate Sweetviz comparison | 10 (4 code, 6 markdown) | Saved with `.venv`, Python 3.14.2, Sweetviz 2.3.3 |
| `DAY22_Pandas-Profiling-empty-placeholder.ipynb` | Preserved placeholder | 3 markdown, 0 code | Kept separate; no code |
| `reference_notebooks/day22.ipynb` | Official CampusX legacy notebook | 6 code, 0 markdown | Python 3.8.3, pandas-profiling 2.11.0; do not edit |

The four notebooks are intentionally separate. Do not merge them or replace the personal implementations.

## Dataset

| File | Details |
|---|---|
| `datasets/train.csv` | Titanic training data: 891 rows x 12 columns; CampusX course dataset; byte-identical to the vendor copy |

## Generated report artifacts

| File | Generator | Status |
|---|---|---|
| `titanic_report.html` | ydata-profiling 4.18.4 | Present; 4,772,777 bytes |
| `../titanic_report.html` | ydata-profiling 4.18.4 | Older duplicate outside this folder |
| `vendor/100-days-of-machine-learning/day22-pandas-profiling/output.html` | pandas-profiling 2.11.0 | Official historical report; read-only reference |
| Sweetviz report | Sweetviz 2.3.3 | **Not present** in the saved artifacts |

Important: the local HTML report is a ydata-profiling report, not a Sweetviz report. The Sweetviz notebook says a report was generated, but no Sweetviz HTML file was saved. Do not quote Sweetviz-specific report contents from this folder.

Both local notebooks show kernel crashes after report generation. Rerun in a fresh kernel and verify the output file before relying on it. The saved ydata report predates the current notebook timestamp, so the exact run that produced it is uncertain.

## Dependencies

```text
python >= 3.12 (ydata-profiling environment used: 3.12.10)
pandas, numpy, jupyter
ydata-profiling 4.18.4
sweetviz 2.3.3
```

```bash
pip install pandas numpy jupyter ydata-profiling sweetviz
```

The official notebook uses the legacy `pandas_profiling` package and Python 3.8-era APIs. It is a historical reference; it is not the environment for the current notebooks.

## How to run

1. Open `Day_22/` as the working directory.
2. Activate the ydata environment (`.venv312`) or install the dependencies above.
3. Run `my_notebook.ipynb` from top to bottom and check that it writes `titanic_report.html`.
4. Open the report and read `notes.md` Part A alongside it.
5. Run `DAY22_Pandas-Profiling-Sweetviz.ipynb` in its own environment. Save its output under a distinct name such as `sweetviz_report.html` so it does not overwrite the ydata report.
6. Use `reference_notebooks/day22.ipynb` to compare the legacy workflow, not as a replacement for the personal notebooks.

## Study order

1. `notes.md` Part A: why automated EDA exists, what the notebooks do, how to read the report, and how to interpret the Titanic results.
2. `my_notebook.ipynb` and the ydata report.
3. `DAY22_Pandas-Profiling-Sweetviz.ipynb` and the Sweetviz comparison.
4. `reference_notebooks/day22.ipynb` for source-fidelity comparison.
5. `notes.md` Part B for report internals, correlation mathematics, source-fidelity details, and extended interview/exam material.

## Folder structure

```text
Day_22/
|-- my_notebook.ipynb
|-- DAY22_Pandas-Profiling-Sweetviz.ipynb
|-- DAY22_Pandas-Profiling-empty-placeholder.ipynb
|-- notes.md
|-- README.md
|-- titanic_report.html
|-- datasets/
|   `-- train.csv
`-- reference_notebooks/
    `-- day22.ipynb
```

## Privacy

The Titanic table and the HTML reports contain passenger names and other identifiers. Treat the reports as potentially sensitive; do not publish them without review.

## Status

`notes.md` has been rewritten to the approved Day 39 two-layer standard and validated (balanced fences, no raw LaTeX delimiters, KaTeX-clean math). The notebooks, dataset, reference notebook, and report artifacts were not modified.
