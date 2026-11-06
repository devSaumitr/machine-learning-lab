# Day 60 — Polynomial Logistic Regression and Softmax

CampusX 100 Days of ML · folder `Day_60/`

---

## Day number

**60**

---

## Main topics

- Nonlinear decision boundaries with polynomial features
- Polynomial logistic regression
- Softmax regression for multiple classes
- Cross-entropy for multiclass classification
- Interactive visualisation

---

## Learning objectives

1. Explain why a linear logistic boundary can fail on a U-shaped dataset.
2. Expand features before fitting a logistic classifier.
3. Interpret softmax outputs as a probability distribution over classes.
4. Connect softmax regression to multinomial logistic regression.
5. Compare binary sigmoid and multiclass softmax.

---

## Reference notebooks (official, read-only)

| File in `vendor/100-days-of-machine-learning/` | Role | Cells |
|---|---|---:|
| `day60-logistic-regression-contd/polynomial-logistic-regression.ipynb` | Polynomial logistic decision boundaries | 21 |
| `day60-logistic-regression-contd/softmax-demo.ipynb` | Softmax regression demonstration | 18 |
| `day60-logistic-regression-contd/streamlit-viz-tool.py` | Optional Streamlit visualisation support, not a notebook | — |

> The official notebooks are **not copied** into this folder. They stay in the read-only master repository and are mapped here by path.

---

## Datasets

No local dataset file is required for this day.

**Data note.** `ushape.csv` is referenced by `polynomial-logistic-regression.ipynb` but is not included in the repository. The softmax notebook uses synthetic data. Do not fabricate the missing CSV.

---

## Dependencies

```text
numpy, pandas, matplotlib, scikit-learn, mlxtend, streamlit, jupyter
```

---

## How to run

1. Open the official notebook from the master repository; it is read-only.
2. Use this folder's `datasets/` copy when a notebook expects a local file.
3. Read `notes.md` for the deep theory, worked examples, and interview material.
4. No personal notebook is created during the course-preparation phase.

---

## Official source notes

The official folder README is blank.

---

## Daily folder structure

```text
Day_60/
├── notes.md          # pending
├── README.md
└── datasets/         # only when data files are required
```

---

## Status

Folder scaffolded from the official repository; deep `notes.md` pending.
