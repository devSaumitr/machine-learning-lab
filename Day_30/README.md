# Day 30 — FunctionTransformer and the Log Transform

CampusX 100 Days of ML · folder `Day_30/`

## Main topics

- `FunctionTransformer` as a wrapper for ordinary functions.
- `log1p` for right-skewed, non-negative features.
- Histogram, Q-Q plot, and skewness checks.
- Selective routing with `ColumnTransformer`.
- Holdout versus cross-validation; preserved notebook bugs.

## Learning objectives

1. Explain what `FunctionTransformer` stores and when it executes.
2. Apply `log1p` and compare distributions before and after.
3. Route the transform to `Fare` only and track output column order.
4. Diagnose test-side refitting, pre-CV transformation, and duplicate-classifier bugs.
5. Explain why `np.sin` is a bad transform for `Fare`.

## Notebooks

| File | Role | Stored outputs |
|------|------|----------------|
| `my_notebook.ipynb` | Documented walkthrough (77 cells) | None |
| `DAY30_Function-Transformer-preview-OUT-OF-SCOPE.ipynb` | Executed preview (29 cells); source of saved scores | Yes |
| `reference_notebooks/day30.ipynb` | CampusX original — **do not edit** | — |

`my_notebook.ipynb` cell 70 is malformed and does not define `apply_transform`; the working helper is in the preview notebook.

## Dataset

| File | Source |
|------|--------|
| `datasets/train.csv` | CampusX Titanic dataset |

## Dependencies

```text
python >= 3.9
pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, jupyter
```

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn jupyter
```

## How to run

1. Open `Day_30/` as the working directory.
2. Run `my_notebook.ipynb` for the guided walkthrough.
3. Use the preview notebook for the executed outputs and the `apply_transform` helper.
4. Read `notes.md` for theory, math, preserved bugs, and interview material.

## Notes

The notebook uses `trf.fit_transform(X_test)` and transforms all data before CV. Both are preserved bugs; they are harmless here only because `log1p` is stateless. The notes explain the correct train/test and pipeline patterns.
