# Day_24

CampusX 100 Days of ML · folder `Day_24/`

## Day number

**24**

## Main topics

Feature scaling; standardization; effect of outliers; scaling before vs after the split.

## Learning objectives

1. Explain why distance- and gradient-based models need comparable feature scales.
2. Apply `StandardScaler` correctly inside a train/test workflow.
3. Show how outliers distort the learned mean and standard deviation.
4. Interpret the before/after distributions and connect scaling to pipelines.

## Notebooks

| File | Role | Cells |
|------|------|-------|
| `my_notebook.ipynb` | Primary practical notebook (standardization, documented) | 106 |
| `DAY24_Standardization-with-Outliers.ipynb` | Separate outliers experiment (kept distinct) | 53 |
| `DAY24_Feature-Scaling-stub.ipynb` | Original markdown stub (no code) | 5 |
| `reference_notebooks/reference_notebook.ipynb` | CampusX original — **do not edit** | — |

## Dataset

| File | Source |
|------|--------|
| `datasets/Social_Network_Ads.csv` | CampusX course dataset |

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
2. Run `my_notebook.ipynb`, then the outliers experiment.
3. Read `notes.md` for the unified theory and interview material.

## Daily folder structure

```text
Day_24/
├── my_notebook.ipynb
├── DAY24_Standardization-with-Outliers.ipynb
├── DAY24_Feature-Scaling-stub.ipynb
├── notes.md
├── README.md
├── reference_notebooks/
└── datasets/
```

## Status

Notebook separation restored; the outliers experiment and stub were recovered from git. `notes.md` covers all notebooks.
