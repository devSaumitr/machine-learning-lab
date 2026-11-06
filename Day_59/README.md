# Day 59 — Classification Metrics

CampusX 100 Days of ML · folder `Day_59/`

---

## Day number

**59**

---

## Main topics

- Confusion matrices for binary and multiclass problems
- Accuracy, precision, recall, and F1
- False positives, false negatives, and class imbalance
- Multiclass averaging
- Metric selection and practical trade-offs

---

## Learning objectives

1. Read a confusion matrix and identify each error type.
2. Compute precision, recall, and F1 from TP, FP, FN, and TN.
3. Explain why accuracy can be misleading under class imbalance.
4. Compare macro, micro, and weighted multiclass averaging.
5. Choose a metric based on the cost of errors.

---

## Reference notebooks (official, read-only)

| File in `vendor/100-days-of-machine-learning/` | Role | Cells |
|---|---|---:|
| `day59-classification-metrics/classification-metrics-binary.ipynb` | Binary classification metrics on heart data | 21 |
| `day59-classification-metrics/classification-metrics-multi-iris1.ipynb` | Multiclass metrics on Iris | 17 |
| `day59-classification-metrics/classification-metrics-multi-mnist1.ipynb` | Multiclass metrics on MNIST digits | 17 |

> The official notebooks are **not copied** into this folder. They stay in the read-only master repository and are mapped here by path.

---

## Datasets

No local dataset file is required for this day.

**Data note.** The official notebooks use Kaggle input paths: `/kaggle/input/heart-disease-uci/heart.csv`, `/kaggle/input/iris/Iris.csv`, and `/kaggle/input/digit-recognizer/train.csv`. These files are not bundled in the repository; obtain them from Kaggle rather than inventing replacements.

---

## Dependencies

```text
numpy, pandas, scikit-learn, jupyter
```

---

## How to run

1. Open the official notebook from the master repository; it is read-only.
2. Use this folder's `datasets/` copy when a notebook expects a local file.
3. Read `notes.md` for the deep theory, worked examples, and interview material.
4. No personal notebook is created during the course-preparation phase.

---

## Official source notes

The official folder README contains no usable video link.

---

## Daily folder structure

```text
Day_59/
├── notes.md          # pending
├── README.md
└── datasets/         # only when data files are required
```

---

## Status

Folder scaffolded from the official repository; deep `notes.md` pending.
