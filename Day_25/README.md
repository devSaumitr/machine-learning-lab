# Day_25

CampusX 100 Days of ML · folder `Day_25/`

## Day number

**25**

## Main topics

Normalization with `MinMaxScaler`; comparison with standardization.

## Learning objectives

1. Explain what min-max normalization changes and what it preserves.
2. Apply `MinMaxScaler` with train-only fitting.
3. Compare normalization and standardization on the same data.
4. Decide which scaler fits a given model and feature distribution.

## Notebooks

| File | Role | Cells |
|------|------|-------|
| `my_notebook.ipynb` | Primary practical notebook (MinMaxScaler) | 61 |
| `DAY25_Normalization-stub.ipynb` | Original markdown stub (no code) | 4 |
| `reference_notebooks/reference_notebook.ipynb` | CampusX original — **do not edit** | — |

## Dataset

| File | Source |
|------|--------|
| `datasets/wine_data.csv` | CampusX course dataset |

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
2. Run `my_notebook.ipynb` and compare against the Day 24 scaler.
3. Read `notes.md` for the unified theory and interview material.

## Daily folder structure

```text
Day_25/
├── my_notebook.ipynb
├── DAY25_Normalization-stub.ipynb
├── notes.md
├── README.md
├── reference_notebooks/
└── datasets/
```

## Status

Notebook separation restored; the stub was recovered from git. `notes.md` covers both notebooks.
