# Day 43 — Outlier Removal Using the IQR Method

CampusX 100 Days of ML · folder `Day_43/`

---

## Day number

**43**

---

## Main topics

- Quartiles, IQR, and Tukey fences
- Lower and upper outlier boundaries
- Robustness compared with mean and standard deviation
- Trimming versus winsorisation/capping
- Visual diagnosis with box plots

---

## Learning objectives

1. Calculate Q1, Q3, IQR, and the outlier fences.
2. Identify observations outside the fences.
3. Explain why the IQR rule is more robust to skew than a z-score rule.
4. Choose between removing and capping extreme values.
5. Interpret a box plot correctly.

---

## Reference notebooks (official, read-only)

| File in `reference_notebooks/` | Official source | Cells |
|---|---|---:|
| `day43.ipynb` | `reference_notebooks` | 25 |

> The official notebooks are copied byte-for-byte into this folder. The vendor copies remain read-only.

---

## Datasets

| Local file | Official source |
|---|---|
| `datasets/placement.csv` | `day43-outlier-removal-using-iqr-method/placement.csv` |

---

## Dependencies

```text
pandas, numpy, matplotlib, seaborn, jupyter
```

---

## How to run

1. Read `notes.md` for the complete theory, worked arithmetic, notebook walkthrough, comparisons, and interview material.
2. Open the notebooks in `reference_notebooks/` to inspect the original cells and saved outputs.
3. Use this folder's `datasets/` copies when rerunning a notebook.
4. No personal notebook is created during the course-preparation phase.

---

## Official source notes

The official folder README contains no usable video link.

---

## Daily folder structure

```text
Day_43/
├── notes.md
├── README.md
├── reference_notebooks/
│   └── day43.ipynb
└── datasets/
│   └── placement.csv
```

---

## Status

**NOTES COMPLETE.** Deep `notes.md` written and validated; reference notebooks and datasets unchanged.
