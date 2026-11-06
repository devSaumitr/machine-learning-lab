# Day 28 — ColumnTransformer

CampusX 100 Days of ML · folder `Day_28/`

## Main topics

- Manual preprocessing of a mixed-type table.
- `ColumnTransformer` with `(name, transformer, columns)` triples.
- Imputation, ordinal encoding, one-hot encoding, and passthrough columns.
- Output width, column order, and test-side refitting.

## Learning objectives

1. Build the manual impute → encode → concatenate workflow.
2. Explain what `fit`, `transform`, and `fit_transform` do for a composite transformer.
3. Predict output width from column selectors and transformer outputs.
4. Diagnose the notebook's `age`/`gender` selector bug and `fit_transform(X_test)` bug.
5. Explain why `remainder="passthrough"` can leave raw strings in the output.

## Notebooks

| File | Role | Cells |
|------|------|-------|
| `my_notebook.ipynb` | Primary practical notebook | 25 |
| `reference_notebooks/day28.ipynb` | CampusX original — **do not edit** | — |

The notebook ends after preprocessing; no model is trained.

## Dataset

| File | Source |
|------|--------|
| `datasets/covid_toy.csv` | CampusX course dataset (100 rows, 6 columns) |

`fever` has 10 missing values.

## Dependencies

```text
python >= 3.9
pandas, numpy, scikit-learn, jupyter
```

```bash
pip install pandas numpy scikit-learn jupyter
```

## How to run

1. Open `Day_28/` as the working directory.
2. Run `my_notebook.ipynb` in order.
3. Read `notes.md` for theory, the manual-versus-automated comparison, and preserved bugs.

## Notes

The automated path one-hot encodes `age` and `gender`, leaves `city` as a passthrough string, and calls `fit_transform(X_test)` instead of `transform(X_test)`. These are preserved bugs; the notes explain the correct column selection and train/test discipline.
