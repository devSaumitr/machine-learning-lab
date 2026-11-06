# Day 56 — Lasso Regression

CampusX 100 Days of Machine Learning · `Day_56/`

## Day number

**56**

## Main topics

- L1 regularisation, sparsity, and the lasso objective
- Geometry and soft-thresholding: why coefficients become exactly zero
- Coordinate descent and the Scikit-learn `Lasso` API
- Coefficient paths, alpha selection, and the bias-variance trade-off
- Both official notebooks: the demo and the 23-cell key-points lesson
- Limitations: correlated features, selection instability, and post-selection inference

## Learning objectives

1. Explain why OLS becomes unstable with many or correlated features.
2. Derive the lasso objective and the one-feature soft-threshold solution.
3. Explain geometrically and algebraically why L1 produces exact zeros.
4. Read coefficient bar charts, coefficient paths, and penalised-loss curves.
5. Fit and tune lasso with a leakage-safe Scikit-learn pipeline.
6. Compare OLS, ridge, lasso, and elastic net for prediction and interpretation.
7. Report a sparse model honestly, including stability and inference caveats.

## Reference notebooks

Both official notebooks are copied byte-identically from the master repository and kept separate.

| File | Role | Cells |
|---|---|---:|
| `reference_notebooks/lasso-regression-demo.ipynb` | Visual demonstrations of shrinkage and polynomial fits | 6 |
| `reference_notebooks/lasso-regression-key-points.ipynb` | Diabetes coefficients, paths, bias-variance, and loss curves | 23 |

Source: `vendor/100-days-of-machine-learning/day56-lasso-regression/` (read-only master repository).

## Datasets

No local dataset file is required. The notebooks use:

- `sklearn.datasets.make_regression` (synthetic one-feature regression);
- `sklearn.datasets.load_diabetes` (built-in diabetes data, 442 rows × 10 features).

## Dependencies

```text
numpy
pandas
matplotlib
scikit-learn
jupyter
```

`mlxtend` is additionally required to rerun cell 15 of the key-points notebook (`bias_variance_decomp`). It is not needed to read the saved output or study `notes.md`.

## How to run

1. Install the dependencies above.
2. Open either notebook from `reference_notebooks/` in Jupyter or VS Code.
3. Run the cells in order. The built-in datasets need no download.
4. Read `notes.md` for the full theory, worked examples, graph interpretation, interviews, and exam practice.

The demo notebook's train/test split and nonlinear data are not seeded, so a rerun can produce different plots. The key-points diabetes split uses `random_state=2`. Saved outputs are quoted as official; reproduced values can differ in the last digits across library versions.

## Status

`notes.md` complete. Both reference notebooks present and unmodified. No local dataset folder is required.
