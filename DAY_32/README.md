# Day 32 — Binning and Binarization

CampusX 100 Days of ML · folder `DAY_32/`

---

## Day number

**Day 32**

---

## Main topics

- Why continuous variables may be discretized
- Equal-width, equal-frequency, and k-means binning
- `KBinsDiscretizer` and its `strategy` / `encode` options
- Binarization with a threshold
- Effect of binning on a decision-tree model
- Leakage-safe binning inside a pipeline

---

## Learning objectives

1. Explain what information binning keeps and what it discards.
2. Compare uniform, quantile, and k-means bin edges.
3. Interpret `bin_edges_` learned from training data.
4. Use `Binarizer` for a genuine binary split rather than arbitrary thresholding.
5. Put discretization inside a pipeline so cross-validation refits it per fold.
6. Compare model performance before and after binning without overclaiming causality.

---

## Notebooks

| Local file | Official source | Role | Cells |
|---|---|---|---:|
| `BINNING_BINERIZATION.ipynb` | `day32-binning-and-binarization/day32.ipynb` | Personal binning experiment, preserved | 23 |
| `BINNERZATION.ipynb` | `day32-binning-and-binarization/binarization.ipynb` | Personal binarization experiment, preserved | 20 |

Official notebooks stay read-only in `vendor/100-days-of-machine-learning/`; they are not copied here.

---

## Datasets

| Local file | Official source | Note |
|---|---|---|
| `datasets/train.csv` | `day32-binning-and-binarization/train.csv` | Titanic subset used by both notebooks; identical SHA-256 to the official file |

The existing personal notebooks currently read `E:\MACHINE_LEARNING\train.csv`. That file has the same SHA-256 as the official Day 32 dataset, so it is the same data, but future notebook maintenance should move to the relative path `datasets/train.csv`.

---

## Dependencies

```text
pandas, numpy, matplotlib, scikit-learn, jupyter
```

---

## How to run

1. Open this folder as the working directory.
2. Read `notes.md` Part A for the complete learning chapter; Part B is optional advanced reference.
3. Run the two personal notebooks in their existing order.
4. Use `datasets/train.csv` for a portable rerun; the old absolute path points to an identical copy.

---

## Study order

1. `notes.md` Part A — problem, binning strategies, API, both notebooks, results, workflow, essential practice
2. `BINNING_BINERIZATION.ipynb` — equal-width/quantile/k-means experiments and the leakage-safe pipeline
3. `BINNERZATION.ipynb` — family feature and `Binarizer` experiment
4. `notes.md` Part B — formal edge algorithms, implementation internals, source audit, extended practice
5. Revise from the revision summary at the end of Part A

---

## Daily folder structure

```text
DAY_32/
├── BINNING_BINERIZATION.ipynb
├── BINNERZATION.ipynb
├── notes.md
├── README.md
└── datasets/
    └── train.csv
```

---

## Status

Both official notebooks are mapped one-to-one to the two existing personal notebooks. Dataset intake and the deep `notes.md` are complete. The notes cover every official cell, explain the `pd.cut` boundary mismatch, the `Binarizer(copy=False)` mutation, the unseeded-tree variability, and the official cross-validation bug; reruns and supplementary comparisons are labelled separately.

- 2026-11-04: `notes.md` rewritten in place to the approved Day 39 two-layer standard; technical depth, source-fidelity audit, saved outputs, and supplementary checks preserved.
- Validation: 154 math items rendered by KaTeX with 0 errors; 0 raw LaTeX delimiters; balanced fences; 3 Mermaid diagrams.
