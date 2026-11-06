# Day 36 — Imputing Numerical Data

CampusX 100 Days of ML · folder `DAY_36/`

---

## Day number

**36**

---

## Main topics

- Missing numerical values and complete-case bias
- Mean and median imputation
- Arbitrary-value and sentinel imputation
- Variance, covariance, distribution, and boxplot consequences
- `SimpleImputer`, `ColumnTransformer`, missingness indicators, and leakage-safe pipelines

---

## Learning objectives

1. Explain when mean, median, arbitrary-value, or indicator-based imputation is appropriate.
2. Distinguish what `fit` learns from what `transform` returns.
3. Quantify how imputation changes variance, covariance, correlation, and shape.
4. Build a leakage-safe preprocessing pipeline for numerical columns.
5. Recognise when a model-based imputer or a sensitivity analysis is preferable.

---

## Files

| File | Role | Cells |
|---|---|---:|
| `reference_notebooks/mean-median-imputation.ipynb` | Official notebook; pandas and scikit-learn mean/median audit | 30 |
| `reference_notebooks/arbitrary-value-imputation.ipynb` | Official notebook; constant/sentinel audit | 23 |
| `univariate_imputation.ipynb` | Existing personal stub; preserved separately and not merged | 3 |
| `notes.md` | Unified deep chapter covering both official notebooks | — |

The two reference notebooks are byte-identical local copies of the official files in `vendor/100-days-of-machine-learning/day36-imputing-numerical-data/`. The vendor repository remains the read-only master source.

---

## Dataset

| Local file | Official source | Shape |
|---|---|---:|
| `datasets/titanic_toy.csv` | `day36-imputing-numerical-data/titanic_toy.csv` | 891 × 4 |

Columns: `Age`, `Fare`, `Family`, `Survived`.

---

## Dependencies

```text
pandas, numpy, matplotlib, scikit-learn, jupyter
```

Verified locally with Python 3.14.2, pandas 3.0.5, NumPy 2.5.3, scikit-learn 1.9.1, and matplotlib 3.11.2.

---

## How to run

1. Open either notebook in `reference_notebooks/` and ensure the CSV path resolves to `datasets/titanic_toy.csv`. The official code reads `titanic_toy.csv`, so run with the dataset directory on the path or adjust the path in a copy.
2. Read `notes.md` for the derivations, notebook walkthroughs, numerical evidence, interview preparation, and exam practice.
3. `univariate_imputation.ipynb` is personal work and remains a separate notebook.

---

## Official source

Video: https://youtu.be/mCL2xLBDw8M

---

## Status

**NOTES COMPLETE.** `notes.md` covers all 30 cells of the mean/median notebook and all 23 cells of the arbitrary-value notebook, with source-status labels, validated mathematics, and supplementary comparisons clearly marked.
