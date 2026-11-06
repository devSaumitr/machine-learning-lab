# Day 31 — Power Transformer (Box-Cox & Yeo-Johnson)

CampusX 100 Days of ML · Day 31 · Folder: `DAY_31/`

---

## Day number

**Day 31**

---

## Main topics

- Right skew and why linear / distance models care
- Power transform idea; Box-Cox formula and λ dial
- Yeo-Johnson (zeros / negatives); sklearn default method
- `PowerTransformer`: `fit` / `transform` / `lambdas_` / `standardize=True`
- Leakage-aware use: fit on train only; Pipeline + CV; keep the holdout out of model selection
- Concrete-strength experiment: holdout R² 0.6276 → 0.8048 (Box-Cox) / 0.8162 (Yeo-Johnson) on the official split; CV is more modest and leakage-sensitive

---

## Learning objectives

After this day I can:

1. Explain **why** a power transform is needed before writing code.
2. Distinguish Box-Cox (X > 0) vs Yeo-Johnson (any real) vs Day 30 `log1p`.
3. Implement `PowerTransformer` correctly without train–test leakage.
4. Read `pt.lambdas_` and interpret near-0 / near-1 / negative λ.
5. Place the transformer inside a `Pipeline` and compare models with cross-validation.
6. Answer interview questions on domain limits, `standardize`, and honest interpretation of gains.

---

## Reference notebooks

| File | Role |
|------|------|
| `reference_notebooks/reference_notebook.ipynb` | **CampusX original** — read-only, never edit |
| `my_notebook.ipynb` | **My practical notebook** (working copy) |

---

## Dataset

| File | Source |
|------|--------|
| `datasets/concrete_data.csv` | Concrete compressive strength style data used with the Day 31 lecture / notebook demo |

- Prefer relative path `datasets/concrete_data.csv`.
- Do not invent replacements if the file is missing; report and wait.

---

## Dependencies

```text
python ≥ 3.9
pandas
numpy
matplotlib / seaborn   (plots in notebook)
scikit-learn           (PowerTransformer, LinearRegression, model_selection)
scipy                  (optional: scipy.stats.boxcox)
```

Install example:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

---

## How to run

1. Open this folder as the working directory (or open the notebook from `DAY_31/`).
2. Open **`my_notebook.ipynb`** side by side with **`reference_notebooks/reference_notebook.ipynb`**.
3. Read **`notes.md`** Part A for the complete learning chapter; Part B is optional advanced reference.
4. Run cells in order; dataset path: `datasets/concrete_data.csv`.

---

## Daily folder structure

```text
DAY_31/
├── my_notebook.ipynb
├── notes.md
├── README.md
├── reference_notebooks/
│   └── reference_notebook.ipynb      # CampusX original — DO NOT EDIT
└── datasets/
    └── concrete_data.csv
```

---

## Study order

1. `notes.md` Part A — problem, λ, math, API, experiment, essential practice
2. `my_notebook.ipynb` — run the concrete demo; print `lambdas_`
3. `notes.md` Part B — advanced derivations, internals, extended practice
4. Revise from the revision summary at the end of Part A

---

## Status

- Strict structure (Part 10) applied 2026-11-02: one notebook, one notes file, README, datasets, **references kept**.
- Legacy `topics.md` / `topic_notes.md` unique content merged into `notes.md` before removal.
- Simplified and duplicate learner notebook copies removed after merge.
- 2026-11-04: `notes.md` rewritten in place to the approved Day 39 two-layer standard; technical depth and source-fidelity audit preserved.

