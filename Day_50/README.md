# Day 50 — Multiple Linear Regression

CampusX 100 Days of ML · folder `Day_50/`

---

## Day number

**50**

---

## Main topics

- From one feature to several features
- The multiple linear model in matrix form
- The normal equation and hat matrix
- Multiple regression with scikit-learn
- Fitting the same model from scratch

---

## Learning objectives

1. Interpret a coefficient while holding other features fixed.
2. Express the OLS solution as a matrix equation.
3. Fit and evaluate a multiple regression model.
4. Reproduce the core algorithm with NumPy.
5. Explain when the normal equation is computationally unattractive.

---

## Reference notebooks (official, read-only)

| File in `reference_notebooks/` | Official source | Cells |
|---|---|---:|
| `code-from-scratch.ipynb` | `reference_notebooks` | 29 |
| `multiple_linear_regression.ipynb` | `reference_notebooks` | 17 |

> The official notebooks are copied byte-for-byte into this folder. The vendor copies remain read-only.

---

## Datasets

No local dataset file is required for this day.

**Data note.** No external dataset is required: one notebook uses `make_regression` and the other uses scikit-learn's built-in diabetes dataset.

---

## Dependencies

```text
numpy, pandas, matplotlib, plotly, scikit-learn, jupyter
```

---

## How to run

1. Read `notes.md` for the complete theory, worked arithmetic, notebook walkthrough, comparisons, and interview material.
2. Open the notebooks in `reference_notebooks/` to inspect the original cells and saved outputs.
3. Use this folder's `datasets/` copies when rerunning a notebook.
4. No personal notebook is created during the course-preparation phase.

---

## Official source notes

Video: https://youtu.be/ashGekqstl8

---

## Daily folder structure

```text
Day_50/
├── notes.md
├── README.md
├── reference_notebooks/
│   └── code-from-scratch.ipynb
│   └── multiple_linear_regression.ipynb
└── datasets/
│   └── (not required)
```

---

## Status

**NOTES COMPLETE.** Deep `notes.md` written and validated; reference notebooks and datasets unchanged.
