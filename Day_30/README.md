# Day_30

CampusX 100 Days of ML · folder `Day_30/`

## Day number

**30**

## Main topics

`FunctionTransformer`; log transform; reciprocal and square-root transforms.

## Learning objectives

1. Explain what a `FunctionTransformer` wraps and when it executes.
2. Apply `np.log1p` to a right-skewed feature and compare distributions.
3. Choose a transform for the shape of the data, not by habit.
4. Place the transformer inside a `ColumnTransformer` / pipeline correctly.

## Notebooks

| File | Role | Cells |
|------|------|-------|
| `my_notebook.ipynb` | Primary practical notebook (log transform, documented) | 77 |
| `DAY30_Function-Transformer-preview-OUT-OF-SCOPE.ipynb` | Original preview notebook (kept distinct; spans Day 30–31) | 29 |
| `reference_notebooks/reference_notebook.ipynb` | CampusX original — **do not edit** | — |

## Dataset

| File | Source |
|------|--------|
| `datasets/train.csv` | CampusX course dataset (Titanic) |

## Dependencies

```text
python >= 3.9
pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter
```

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## How to run

1. Open `Day_30/` as the working directory.
2. Run `my_notebook.ipynb`, then the preview notebook for the Day 31 preview.
3. Read `notes.md` for the unified theory and interview material.

## Daily folder structure

```text
Day_30/
├── my_notebook.ipynb
├── DAY30_Function-Transformer-preview-OUT-OF-SCOPE.ipynb
├── notes.md
├── README.md
├── reference_notebooks/
└── datasets/
```

## Status

Preview notebook recovered from git and kept separate. `notes.md` covers both notebooks.
