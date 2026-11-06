# Day 40 — Iterative Imputer

CampusX 100 Days of ML · folder `Day_40/`

---

## Day number

**40**

---

## Main topics

- Iterative imputation as repeated regression
- Predicting one missing feature from the others
- `IterativeImputer` and `max_iter`
- Round-robin updating of missing values
- Convergence and stopping behavior
- Multiple imputation and practical limitations
- Leakage-safe preprocessing with a `Pipeline`

---

## Learning objectives

1. Explain the iterative regression procedure in plain language.
2. Describe how each feature takes a turn as a regression target.
3. Interpret convergence and `max_iter`.
4. Recognise the additional computational and modelling assumptions.
5. Use `IterativeImputer` without leakage.
6. Compare univariate, KNN, and iterative imputation.

---

## Reference notebook (read-only)

| Local file | Official source | Role | Cells |
|---|---|---|---:|
| `reference_notebooks/step-by-step.ipynb` | `vendor/100-days-of-machine-learning/day40-iterative-imputer/step-by-step.ipynb` | Iterative imputer worked step by step | 46 |

The local copy is for study and traceability only. Do not edit it. The official notebook remains unchanged in the read-only course repository.

---

## Dataset

| Local file | Official source |
|---|---|
| `datasets/50_Startups.csv` | `vendor/100-days-of-machine-learning/day40-iterative-imputer/50_Startups.csv` |

The dataset has 50 rows and 5 columns: `R&D Spend`, `Administration`, `Marketing Spend`, `State`, and `Profit`.

The notebook loads `50_Startups.csv` by relative path. The local copy is in `datasets/`, so use an explicit path in any scratch copy you execute.

---

## Dependencies

```text
pandas
numpy
scikit-learn
jupyter
```

---

## How to open and run

1. Read `notes.md` for the complete theory, worked arithmetic, notebook walkthrough, comparisons, and interview material.
2. Open `reference_notebooks/step-by-step.ipynb` to inspect the original 46-cell workflow and saved outputs.
3. To execute the workflow, make a scratch copy beside a copy of `50_Startups.csv` (or change the CSV path in the scratch copy). Do not edit the reference notebook.
4. No `my_notebook.ipynb` is created; the course-preparation task requires notes and a read-only reference notebook only.

---

## Official source notes

- Video: https://youtu.be/a38ehxv3kyk
- Repository: `vendor/100-days-of-machine-learning/day40-iterative-imputer/`

---

## Folder structure

```text
Day_40/
├── README.md
├── notes.md
├── datasets/
│   └── 50_Startups.csv
└── reference_notebooks/
    └── step-by-step.ipynb
```

---

## Status

**NOTES COMPLETE**

`notes.md` covers all 46 notebook cells, the full iterative-imputation derivation, the official manual arithmetic, leakage-safe pipelines, comparisons, interview questions, and exam-style practice.

Validation: balanced code fences, no raw math delimiters, and 0 KaTeX rendering errors across all 417 extracted math expressions.
