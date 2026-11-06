# Day 49 — Regression Metrics

CampusX 100 Days of ML · folder `Day_49/`

---

## Day number

**49**

---

## Main topics

- Baseline prediction, residuals, and error decomposition
- MAE, MSE, RMSE, R-squared, and adjusted R-squared
- What each metric penalises and when it is appropriate
- Random-feature overfitting and target leakage
- Model comparison and metric selection

---

## Learning objectives

1. Compute and interpret MAE, MSE, RMSE, and R-squared.
2. Explain why RMSE penalises large errors more than MAE.
3. Use the mean baseline to interpret R-squared correctly.
4. Explain the degrees-of-freedom penalty in adjusted R-squared.
5. Distinguish a genuinely useful feature from leakage and chance.
6. Choose a metric that matches the real cost of prediction errors.

---

## Notebook

`reference_notebooks/Untitled.ipynb` is the 39-cell official notebook, copied
byte-for-byte from the vendor repository. It is preserved unchanged and is not
a personal working notebook.

The notebook reads `placement.csv` relative to its own working directory. To run
it, use a working copy with the dataset available as `placement.csv`, or change
the read path in that copy to `../datasets/placement.csv`. Do not modify the
preserved reference notebook.

---

## Dataset

| Local file | Official source |
|---|---|
| `datasets/placement.csv` | `day49-regression-metrics/placement.csv` |

200 rows, 2 columns (`cgpa`, `package`); byte-identical to the vendor dataset.

---

## Dependencies

```text
Python 3.14.2 (verified)
pandas 3.0.5
numpy 2.5.3
matplotlib 3.11.2
scikit-learn 1.9.1
jupyter
```

---

## How to run

1. Open `reference_notebooks/Untitled.ipynb` in Jupyter.
2. Make `placement.csv` available in the notebook's working directory, or adjust
   the read path in a working copy.
3. Run the cells from top to bottom.
4. Read `notes.md` for the full theory, worked examples, cell mapping, leakage
   analysis, and interview material.

---

## Files

```text
Day_49/
├── README.md
├── notes.md
├── datasets/
│   └── placement.csv
└── reference_notebooks/
    └── Untitled.ipynb
```

---

## Status

**NOTES COMPLETE.** `notes.md` is 4,214 lines with 742 validated math items
(0 raw LaTeX delimiters, 0 KaTeX errors). The reference notebook and dataset
remain unmodified.
