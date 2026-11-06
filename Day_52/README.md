# Day 52 — Types of Gradient Descent

CampusX 100 Days of ML · folder `Day_52/`

---

## Day number

**52**

---

## Main topics

- Batch gradient descent
- Stochastic gradient descent
- Mini-batch gradient descent
- Cost curves and convergence behaviour
- From-scratch implementations and animations

---

## Learning objectives

1. Compare batch, stochastic, and mini-batch updates.
2. Explain the trade-off between stable gradients and noisy updates.
3. Describe why mini-batch updates dominate practical training.
4. Implement each variant from scratch.
5. Interpret cost and parameter trajectories.

---

## Reference notebooks (official, read-only)

| File in `reference_notebooks/` | Official source | Cells |
|---|---|---:|
| `batch-gradient-descent.ipynb` | `reference_notebooks` | 14 |
| `mini-batch-gradient-descent-from-scratch.ipynb` | `reference_notebooks` | 21 |
| `stochastic-gradient-descent-animation.ipynb` | `reference_notebooks` | 18 |
| `stochastic-gradient-descent-from-scratch.ipynb` | `reference_notebooks` | 19 |

> The official notebooks are copied byte-for-byte into this folder. The vendor copies remain read-only.

---

## Datasets

No local dataset file is required for this day.

**Data note.** No external dataset is required; the notebooks use synthetic data and scikit-learn's built-in diabetes dataset.

---

## Dependencies

```text
numpy, pandas, matplotlib, scikit-learn, jupyter
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
Day_52/
├── notes.md
├── README.md
├── reference_notebooks/
│   └── batch-gradient-descent.ipynb
│   └── mini-batch-gradient-descent-from-scratch.ipynb
│   └── stochastic-gradient-descent-animation.ipynb
│   └── stochastic-gradient-descent-from-scratch.ipynb
└── datasets/
│   └── (not required)
```

---

## Status

**NOTES COMPLETE.** Deep `notes.md` written and validated; reference notebooks and datasets unchanged.
