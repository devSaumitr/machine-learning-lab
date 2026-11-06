# Day 27 — One-Hot Encoding

CampusX 100 Days of ML · folder `Day_27/`

## Main topics

One-hot encoding for nominal features, the dummy-variable trap, pandas `get_dummies`, scikit-learn `OneHotEncoder`, and rare-category grouping.

## Learning objectives

1. Explain why unordered categories must not be integer-encoded.
2. Build one-hot columns with pandas and scikit-learn.
3. Apply $k-1$ encoding to avoid the dummy trap.
4. Fit on training data and transform test data.
5. Group rare categories in high-cardinality features.
6. Identify the preserved notebook bugs and explain their fixes.

## Notebooks

| File | Role | Cells |
|------|------|-------|
| `my_notebook.ipynb` | Primary practical notebook: pandas dummies, sklearn encoder, rare-brand grouping | 30 |

Reference original: `reference_notebooks/day27.ipynb` — **do not edit**.

## Dataset

| File | Source |
|------|--------|
| `datasets/cars.csv` | CampusX course dataset |

The notebook currently uses an absolute Windows path. For portability:

```python
import pandas as pd
cars = pd.read_csv("datasets/cars.csv")
```

## Dependencies

```text
python >= 3.9
pandas, numpy, scikit-learn, jupyter
```

```bash
pip install pandas numpy scikit-learn jupyter
```

## How to run

1. Open `Day_27/` as the working directory.
2. Run `my_notebook.ipynb` in order.
3. Read `notes.md` for the full theory, saved-output interpretation, and interview material.

## Notes on the notebook

- Saved outputs are from the user's environment, not regenerated here.
- The dataset has 8,128 rows and five columns.
- Full pandas dummies produce 12 columns; `drop_first=True` produces 10.
- The sklearn encoder block produces `(5689, 7)` after dropping one level per feature.
- The `hstack` result is `(5689, 9)`, but `brand` is still a string and the array is not model-ready.
- **Preserved bug:** `y = df.iloc[:, :-1]` overlaps `X` and is not `selling_price`. No model is trained; fix the target before adding one.
- **Preserved typo:** the rare-brand bucket is `"unommon"` instead of `"uncommon"`.
- No model, accuracy, or cross-validation result is present.

## Folder structure

```text
Day_27/
├── my_notebook.ipynb
├── notes.md
├── README.md
├── reference_notebooks/
│   └── day27.ipynb
└── datasets/
    └── cars.csv
```

## Status

`notes.md` rewritten to the Day 39 two-layer standard and validated for structure and math. Notebook implementation preserved.
