# Day 24 — Standardization (z-Score Scaling)

CampusX 100 Days of Machine Learning · `Day_24/`

## Main topics

Feature scaling; standardization with `StandardScaler`; z-scores; scale-sensitive versus scale-invariant models; the effect of outliers; fitting preprocessing only on training data; leakage-safe pipelines.

## Learning objectives

1. Explain why features measured in different units distort distances, gradients, and coefficient penalties.
2. Apply `StandardScaler` correctly: split first, fit on training data, transform both splits.
3. Interpret `mean_`, `scale_`, and the standardized `describe()` output.
4. Explain why standardization does not remove outliers or make a distribution normal.
5. Compare scaled and unscaled LogisticRegression and DecisionTree results without overinterpreting one split.

## Notebooks

| File | Role | Cells |
|---|---|---:|
| `my_notebook.ipynb` | Primary practical notebook with saved outputs | 106 |
| `DAY24_Standardization-with-Outliers.ipynb` | Separate outlier experiment; code copy without outputs | 53 |
| `DAY24_Feature-Scaling-stub.ipynb` | Original markdown stub; no code | 5 |
| `reference_notebooks/day24.ipynb` | CampusX reference copy; do not edit | 42 |

Important: the saved outputs in `my_notebook.ipynb` come from the **outlier-inclusive** run (322 training rows, 81 test rows). The clean 320/80 split described earlier in the notebook gives different reproduced results. `notes.md` separates saved outputs from reproduced values.

## Dataset

| File | Source |
|---|---|
| `datasets/Social_Network_Ads.csv` | CampusX course dataset (400 rows, 5 columns) |

After `df.iloc[:, 2:]`, the modelling table contains `Age`, `EstimatedSalary`, and `Purchased`.

## Dependencies

```text
python >= 3.9
pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter
```

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## How to run

1. Open `Day_24/` as the working directory.
2. Run `my_notebook.ipynb` from top to bottom for the clean run; note that its saved outputs are from an earlier outlier-inclusive execution.
3. Run `DAY24_Standardization-with-Outliers.ipynb` for the separate outlier experiment.
4. Read `notes.md` for the full theory, worked examples, interview bank, and source-fidelity notes.

## Folder structure

```text
Day_24/
├── my_notebook.ipynb
├── DAY24_Standardization-with-Outliers.ipynb
├── DAY24_Feature-Scaling-stub.ipynb
├── notes.md
├── README.md
├── reference_notebooks/
│   └── day24.ipynb
└── datasets/
    └── Social_Network_Ads.csv
```

## Status

Validated against the Day 39 two-layer standard. `notes.md` covers all Day 24 notebooks, distinguishes saved from reproduced results, and documents the notebook artifacts and version caveats.
