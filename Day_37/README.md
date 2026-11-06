# Day 37 — Handling Missing Categorical Data

CampusX 100 Days of ML · folder `Day_37/`

---

## Day number

**37**

---

## Main topics

- Missing categorical values and their meaning
- Why mean/median imputation is invalid for categories
- Frequent-value (mode) imputation
- A separate `Missing` category
- `SimpleImputer(strategy='most_frequent')` and `statistics_`
- Missingness mechanisms, leakage, and preprocessing pipelines

---

## Learning objectives

1. Explain why the mean cannot be used for a categorical column.
2. Compare frequent-value imputation with an explicit missing category.
3. Interpret the mode learned from training data and the fitted `statistics_`.
4. Preserve the distinction between an observed category and a missing state.
5. Avoid fitting an imputer on test data or filling before the train/test split.
6. Choose a defensible method for low-missing, high-missing, and structural missingness.

---

## Reference notebooks (copied, read-only)

Both official notebooks are preserved separately in `reference_notebooks/`; they are byte-identical to the vendor copies.

| Local file | Official source | Role | Cells |
|---|---|---|---:|
| `reference_notebooks/frequent-value-imputation.ipynb` | `vendor/100-days-of-machine-learning/day37-handling-missing-categorical-data/frequent-value-imputation.ipynb` | Mode/frequent-value workflow, KDE diagnostics, pandas `fillna`, `SimpleImputer(strategy='most_frequent')`, `statistics_` | 24 |
| `reference_notebooks/missing-category-imputation.ipynb` | `vendor/100-days-of-machine-learning/day37-handling-missing-categorical-data/missing-category-imputation.ipynb` | Explicit `Missing` category, pandas `fillna`, `SimpleImputer(strategy='constant')`, `statistics_` | 13 |

> The reference notebooks are kept read-only. They contain the original code and saved outputs, including their pedagogical defects. Do not edit them; the corrected patterns and the cell-by-cell critique live in `notes.md`.

---

## Dataset

| Local file | Official source | Shape | Notes |
|---|---|---|---|
| `datasets/train.csv` | `day37-handling-missing-categorical-data/train.csv` | 1,460 rows × 3 columns | House Prices subset: `FireplaceQu`, `GarageQual`, `SalePrice` |

The local dataset is byte-identical to the vendor copy (SHA256 verified). `SalePrice` is the target; the two categorical columns contain 690 and 81 missing values, respectively.

---

## Dependencies

```text
pandas
numpy
matplotlib
scikit-learn
jupyter
```

The local verification used Python 3.14.2, pandas 3.0.5, NumPy 2.5.3, and scikit-learn 1.9.1.

---

## How to open the references

1. Open either notebook in `reference_notebooks/` in Jupyter and read the saved code/output cells. Do not edit or run over the original outputs.
2. The notebook code calls `pd.read_csv('train.csv')`. To rerun it without modifying the reference, provide `train.csv` in the notebook's working directory, or use a scratch copy outside `reference_notebooks/` that points to `datasets/train.csv`.
3. For the full theory, worked examples, notebook critique, interview preparation, and exam questions, read `notes.md`.

---

## Status

**NOTES COMPLETE.** `notes.md` covers both reference notebooks cell by cell, including the pre-split filling and `transform(X_train)` defects in the originals.
