# Day 42 — Outlier Handling: Z-Score, Trimming, and Capping

CampusX 100 Days of ML · folder `Day_42/`

## Day number

**42**

## Main topics

- What an outlier is, what it is not, and why the definition is task-dependent
- Causes: data errors, sampling, mixtures, genuine rare events, context, and adversarial values
- Detection with histograms, box plots, the three-sigma rule, IQR, percentiles, and robust scores
- Trimming versus capping/winsorization, including the mathematics of both
- Effects on mean, variance, covariance, correlation, leverage, and regression
- Information loss, selection bias, and the choice between drop, cap, transform, and robust models
- Leakage-safe implementation in pandas and scikit-learn
- Comparison with IQR, modified z-score, Mahalanobis, isolation forests, and robust regression

## Learning objectives

1. Define an outlier precisely and distinguish it from an error, a rare event, and a high-leverage
   point.
2. Compute and interpret z-scores, three-sigma bounds, IQR fences, percentile bounds, and modified
   z-scores.
3. Explain why the mean and standard deviation are not robust and demonstrate masking.
4. Derive the effect of trimming and capping on moments, covariance, and regression.
5. Implement trimming and capping safely, with bounds fitted on training data only.
6. Choose an appropriate action using domain knowledge, model sensitivity, and deployment needs.

## Reference notebook

| File | Role | Cells |
|---|---|---:|
| `reference_notebooks/day42.ipynb` | Reference copy of the official Z-score notebook | 25 |
| `vendor/100-days-of-machine-learning/day42-outlier-removal-using-zscore/day42.ipynb` | Read-only official source | 25 |

The reference copy is byte-identical to the vendor notebook (SHA-256
`8A4393D335A7F0C9698FF040907E441E2A3EB3BE1D329BE611F6B5B31D294BEC`). The notebook has 23 code
cells and 2 Markdown cells; 15 code cells have saved outputs. Cell 24 is empty.

## Dataset

| Local file | Official source | Shape |
|---|---|---:|
| `datasets/placement.csv` | `vendor/100-days-of-machine-learning/day42-outlier-removal-using-zscore/placement.csv` | 1,000 × 3 |

Columns: `cgpa`, `placement_exam_marks`, `placed`. The local copy is byte-identical to the vendor
copy (SHA-256 `CA536F5F7C2C6EA1CE34D3B897319FBC6821293CD9F912031B764DB498EA49E9`).

## Dependencies

```text
pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, jupyter
```

The notebook was produced with Python 3.8.3. It uses seaborn's `distplot`, which is deprecated in
seaborn 0.13 and removed in later versions; use `histplot(kde=True)` if it is unavailable.

## How to open and run

1. Open `notes.md` first for the full theory, worked examples, and interview/exam material.
2. To run the reference notebook, note that it reads `placement.csv` relative to the working
   directory. Launch Jupyter from `Day_42/datasets` and open `../reference_notebooks/day42.ipynb`,
   or set the notebook working directory to `Day_42/datasets`.
3. Do not modify the reference notebook, the dataset, or `vendor/`.
4. No personal notebook is created for this documentation-only day.

## Folder structure

```text
Day_42/
├── notes.md
├── README.md
├── datasets/
│   └── placement.csv
└── reference_notebooks/
    └── day42.ipynb
```

## Official source

Video: https://youtu.be/OnPE-Z8jtqM

## Status

**NOTES COMPLETE.** Deep `notes.md` written and validated; reference notebook and dataset unchanged.
