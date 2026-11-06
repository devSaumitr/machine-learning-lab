# Day 44 — Outlier Detection Using Percentiles

CampusX 100 Days of ML · folder `Day_44/`

---

## Day number

**44**

---

## Main topics

- Percentiles and quantiles
- Winsorisation by percentile limits
- Trimmed distributions
- The effect of the chosen percentile cutoff
- Comparison with z-score and IQR rules

---

## Learning objectives

1. Interpret the 1st, 5th, 95th, and 99th percentiles.
2. Apply percentile-based capping and trimming.
3. Explain the trade-off between retained information and extreme-value control.
4. Recognise that percentile cutoffs are not automatic definitions of an outlier.
5. Compare the result with IQR and z-score treatment.

---

## Reference notebooks (official, read-only)

| File in `reference_notebooks/` | Official source | Cells |
|---|---|---:|
| `day44.ipynb` | `reference_notebooks` | 21 |

> The official notebooks are copied byte-for-byte into this folder. The vendor copies remain read-only.

---

## Datasets

| Local file | Official source |
|---|---|
| `datasets/weight-height.csv` | `day44-outlier-detection-using-percentiles/weight-height.csv` |

---

## Dependencies

```text
pandas, numpy, matplotlib, jupyter
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
Day_44/
├── notes.md
├── README.md
├── reference_notebooks/
│   └── day44.ipynb
└── datasets/
│   └── weight-height.csv
```

---

## Status

**NOTES COMPLETE.** Deep `notes.md` written and validated; reference notebooks and datasets unchanged.
