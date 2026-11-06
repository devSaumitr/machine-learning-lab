# Day 39 — KNN Imputer

CampusX 100 Days of ML · folder `Day_39/`

---

## Day number

**39**

---

## Main topics

- Distance-based nearest-neighbour imputation
- Using correlated features to estimate a missing value
- `KNNImputer` and the `n_neighbors` parameter
- Feature scaling before distance calculations
- Computational and leakage considerations

---

## Learning objectives

1. Explain how a missing coordinate is excluded from a neighbour distance calculation.
2. Describe how the neighbours' observed values form the imputed estimate.
3. Choose and interpret `n_neighbors`.
4. Scale features when distances would otherwise be dominated by one variable.
5. Compare KNN imputation with simple statistical imputation.

---

## Reference notebooks (official, read-only)

| File in `reference_notebooks/` | Official source | Role | Cells |
|---|---|---|---:|
| `day39.ipynb` | `day39-knn-imputer/day39.ipynb` | KNN imputation workflow | 12 |

> The official notebook is copied byte-for-byte into this folder. The vendor copy remains read-only.

---

## Datasets

| Local file | Official source |
|---|---|
| `datasets/train.csv` | `day39-knn-imputer/train.csv` |

---

## Dependencies

```text
pandas, numpy, scikit-learn, jupyter
```

---

## How to run

1. Read **Part A** of `notes.md` first for the main learning chapter, worked arithmetic, notebook walkthrough, comparisons, and interview essentials.
2. Use **Part B** for the advanced reference: distance metrics, library internals, extended experiments, and the full interview/exam bank.
3. Open `reference_notebooks/day39.ipynb` to inspect the original 12-cell workflow and saved outputs.
4. Use this folder's `datasets/train.csv` copy when rerunning the notebook.
5. No personal notebook is created during the course-preparation phase.

---

## Official source notes

The official folder README contains no usable video link.

---

## Daily folder structure

```text
Day_39/
├── notes.md
├── README.md
├── datasets/
│   └── train.csv
└── reference_notebooks/
    └── day39.ipynb
```

---

## Status

**NOTES COMPLETE — PEDAGOGICAL PILOT REWRITTEN.** `notes.md` now has a 5,934-word main chapter (Part A) and a 3,203-word advanced reference (Part B); math, experiments, limitations, interview and exam coverage are preserved. Reference notebook and dataset unchanged.
