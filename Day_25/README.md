# Day 25 — Normalization and Min-Max Scaling

CampusX 100 Days of ML · folder `Day_25/`

## Main topics

Min-max normalization with `MinMaxScaler`, train/test leakage, custom feature ranges, and comparisons with standardization, max-absolute, robust, and row-wise normalization.

## Learning objectives

1. Explain what min-max scaling changes and what it preserves.
2. Derive and apply the min-max formula, including `feature_range`.
3. Use `fit_transform` on training data and `transform` on test data.
4. Interpret `data_min_`, `data_max_`, and `scale_`.
5. Compare min-max scaling with related scalers.
6. Avoid leakage in cross-validation and inference.

## Notebooks

| File | Role | Cells |
|------|------|-------|
| `my_notebook.ipynb` | Primary practical notebook: formula, scaler, split, pipeline, wine-data plots | 61 |
| `DAY25_Normalization-stub.ipynb` | Historical markdown stub (no code) | 4 |

Reference original: `reference_notebooks/day25.ipynb` — **do not edit**.

## Dataset

| File | Source |
|------|--------|
| `datasets/wine_data.csv` | CampusX course dataset |

The notebook currently loads the CSV from an absolute Windows path. Use the local relative path when re-running:

```python
import pandas as pd
df = pd.read_csv("datasets/wine_data.csv", header=None, usecols=[0, 1, 2])
```

## Dependencies

```text
python >= 3.9
pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter
```

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## How to run

1. Open `Day_25/` as the working directory.
2. Run `my_notebook.ipynb` in order.
3. Read `notes.md` for the full theory, saved-output interpretation, and interview material.
4. Compare with the Day 24 standardization notebook.

## Notes on the notebook

- Saved outputs are from the user's environment, not regenerated here.
- The notebook contains duplicate split/scale and plotting cells; they are preserved.
- Test-set values outside `[0, 1]` are expected because the scaler was fitted on the training range only.
- The toy pipeline accuracy is a correctness check on three test rows, not a tuned result.
- The trailing markdown cells sketch mean normalization, max-absolute scaling, and robust scaling; `notes.md` explains them.

## Folder structure

```text
Day_25/
├── my_notebook.ipynb
├── DAY25_Normalization-stub.ipynb
├── notes.md
├── README.md
├── reference_notebooks/
│   └── day25.ipynb
└── datasets/
    └── wine_data.csv
```

## Status

`notes.md` rewritten to the Day 39 two-layer standard and validated for structure and math. Notebook implementation preserved.
