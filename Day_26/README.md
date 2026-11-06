# Day 26 — Ordinal Encoding and Label Encoding

CampusX 100 Days of ML · folder `Day_26/`

## Main topics

Categorical variable types, `OrdinalEncoder` with declared category order, `LabelEncoder` for targets, and leakage-safe encoding.

## Learning objectives

1. Distinguish nominal, ordinal, and target variables.
2. Encode ordered input features with `OrdinalEncoder`.
3. Encode target labels with `LabelEncoder`.
4. Explain `categories_`, `classes_`, `fit`, `transform`, and `inverse_transform`.
5. Avoid alphabetical-order and label-encoding misuse.
6. Connect encoding to pipelines and train/test discipline.

## Notebooks

| File | Role | Cells |
|------|------|-------|
| `my_notebook.ipynb` | Primary practical notebook: split, ordinal encoding, target encoding | 27 |

Reference original: `reference_notebooks/day26.ipynb` — **do not edit**.

## Dataset

| File | Source |
|------|--------|
| `datasets/customer.csv` | CampusX course dataset |

The notebook currently uses an absolute Windows path. For portability:

```python
import pandas as pd
df = pd.read_csv("datasets/customer.csv")
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

1. Open `Day_26/` as the working directory.
2. Run `my_notebook.ipynb` in order.
3. Read `notes.md` for the full theory, saved-output interpretation, and interview material.

## Notes on the notebook

- Saved outputs are from the user's environment, not regenerated here.
- The split is 70/30 with `random_state=34`; `X_train.shape` is `(35, 2)`.
- `review` and `education` are encoded as ordinal inputs; `purchased` is encoded as the target.
- Reassigning `X_train = oe.transform(X_train)` converts the DataFrame to a NumPy array and loses column names.
- No model is trained in this notebook.
- The notebook comments identify `gender` as nominal but do not encode it; one-hot encoding is covered on Day 27.

## Folder structure

```text
Day_26/
├── my_notebook.ipynb
├── notes.md
├── README.md
├── reference_notebooks/
│   └── day26.ipynb
└── datasets/
    └── customer.csv
```

## Status

`notes.md` rewritten to the Day 39 two-layer standard and validated for structure and math. Notebook implementation preserved.
