# Day 65 — Random Forest

CampusX 100 Days of ML · folder `Day_65/`

---

## Day number

**65**

---

## Main topics

- Bagging versus random forests
- Random feature selection at each split
- Out-of-bag evaluation
- Feature importance
- Random-forest hyperparameters and inspection tools

---

## Learning objectives

1. Explain how a random forest differs from plain bagging.
2. Describe why decorrelating trees reduces variance.
3. Interpret out-of-bag scores.
4. Compute and read impurity-based feature importance.
5. Tune `n_estimators`, `max_features`, and related controls.

---

## Reference notebooks (official, read-only)

| File in `vendor/100-days-of-machine-learning/` | Role | Cells |
|---|---|---:|
| `day65-random-forest/code-example-random-forest.ipynb` | Random forest with scikit-learn | 30 |
| `day65-random-forest/oob-score-demo.ipynb` | Out-of-bag score demonstration | 9 |
| `day65-random-forest/feature-importance-in-sklearn.ipynb` | Feature importance on digit features | 10 |
| `day65-random-forest/how-feature-importance-is-calculated.ipynb` | Feature importance calculation | 17 |
| `day65-random-forest/bagging_vs_random_forest.ipynb` | Bagging compared with random forests | 10 |
| `day65-random-forest/random_forest_demo.ipynb` | Interactive random-forest demonstration | 13 |
| `day65-random-forest/rf_learning_tool.ipynb` | Random-forest learning tool | 22 |

> The official notebooks are **not copied** into this folder. They stay in the read-only master repository and are mapped here by path.

---

## Datasets

| Local file | Official source |
|---|---|
| `datasets/heart.csv` | `day65-random-forest/heart.csv` |

**Data note.** `feature-importance-in-sklearn.ipynb` reads `train.csv` (a 28x28 digit-style dataset) that is not included in the repository. It is not the same file as the Titanic `train.csv`; do not substitute that file silently.

---

## Dependencies

```text
numpy, pandas, matplotlib, seaborn, scikit-learn, jupyter
```

---

## How to run

1. Open the official notebook from the master repository; it is read-only.
2. Use this folder's `datasets/` copy when a notebook expects a local file.
3. Read `notes.md` for the deep theory, worked examples, and interview material.
4. No personal notebook is created during the course-preparation phase.

---

## Official source notes

The official folder README says only: Nothing.

---

## Daily folder structure

```text
Day_65/
├── notes.md          # pending
├── README.md
└── datasets/         # only when data files are required
```

---

## Status

Folder scaffolded from the official repository; deep `notes.md` pending.
