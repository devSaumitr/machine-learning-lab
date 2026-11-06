# Day 38 — Missing Indicators, Random-Sample Imputation, and Automatic Imputer Selection

CampusX 100 Days of ML · folder `Day_38/`

## Day number

**38**

## Main topics

- Missingness as information
- `MissingIndicator` and `SimpleImputer(add_indicator=True)`
- Random-sample / hot-deck imputation
- Distribution and covariance effects of imputation
- `Pipeline`, `ColumnTransformer`, and `GridSearchCV`
- Leakage-safe imputer selection

## Learning objectives

1. Distinguish imputing a value from recording that a value was missing.
2. Build and interpret a missing-indicator feature.
3. Explain what random-sample imputation preserves and what it destroys.
4. Use `SimpleImputer(add_indicator=True)` and `MissingIndicator` correctly.
5. Tune imputation and regularisation parameters without touching the test set.
6. Recognise the source-fidelity issues in the official notebooks.

## Reference notebooks

The three official notebooks are copied here for convenient study. Treat them as read-only references; the vendor originals remain untouched.

| Local file | Vendor source | Role | Cells |
|---|---|---|---:|
| `reference_notebooks/missing-indicator.ipynb` | `vendor/100-days-of-machine-learning/day38-missing-indicator/missing-indicator.ipynb` | Missing indicators as features | 26 |
| `reference_notebooks/random-sample-imputation.ipynb` | `vendor/100-days-of-machine-learning/day38-missing-indicator/random-sample-imputation.ipynb` | Random-sample imputation | 32 |
| `reference_notebooks/automatically-select-imputer-parameters.ipynb` | `vendor/100-days-of-machine-learning/day38-missing-indicator/automatically-select-imputer-parameters.ipynb` | Automatic imputer configuration | 17 |

## Datasets

| Local file | Official source | Rows |
|---|---|---:|
| `datasets/train.csv` | `day38-missing-indicator/train.csv` | 891 |
| `datasets/house-train.csv` | `day38-missing-indicator/house-train.csv` | 1,460 |

Both local datasets are byte-identical to the vendor copies.

## Dependencies

```text
python, pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter
```

The notebooks were originally produced in a Python 3.8.3-era environment. The notes label outputs as **Official** or **Rerun**; the rerun environment is Python 3.14.2, pandas 3.0.5, NumPy 2.5.3, scikit-learn 1.9.1.

## How to run

1. Open a notebook from `reference_notebooks/` in Jupyter.
2. Point the notebook's CSV reads at `datasets/train.csv` or `datasets/house-train.csv` if the working directory is not this folder.
3. Read `notes.md` for the cell-by-cell walkthrough, theory, worked examples, debugging notes, and interview/exam material.
4. Treat the notebooks as references; no personal notebook is maintained in this folder.

## Status

**NOTES COMPLETE.** `notes.md` covers all 75 cells across the three notebooks, including the empty cells, source-fidelity bugs, official versus rerun results, mathematics, comparisons, interview Levels 1–7, and exam-style questions.
