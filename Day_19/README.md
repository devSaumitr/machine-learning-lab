# Day 19 - Understanding Your Data: Descriptive Statistics

CampusX 100 Days of Machine Learning, Day 19.

## Main topics

- The seven first-look checks: `shape`, `sample`, `info`, `isnull`, `describe`, `duplicated`, `corr`.
- Titanic data: 891 rows and 12 columns.
- Missing-value counts and percentages.
- Mean, median, spread, quartiles, IQR, skewness, and binary proportions.
- Exact duplicates versus repeated values in other columns.
- Pearson correlation, the target correlation vector, and correlation limits.
- The local correlation heatmap and its saved figure.
- Pandas 3.0.5 differences from the official notebook's saved outputs.

This day performs **descriptive analysis only**. There is no train/test split and no model training.

## Learning objectives

1. Run the seven checks and explain what each result means.
2. Compute missing counts and percentages, and distinguish them from valid zero values.
3. Interpret mean versus median, skew, standard deviation, quartiles, and IQR.
4. Explain why the mean of a 0/1 target is a survival proportion.
5. Interpret the correlation matrix and explain why `Sex` is absent from it.
6. Identify data-quality decisions that must be made before cleaning or modeling.
7. Explain why preprocessing must later be fitted on training data only.
8. Reproduce the notebook outputs while accounting for pandas version changes.

## Files

| File | Role |
|---|---|
| `notes.md` | Unified deep chapter: Part A main learning chapter and Part B advanced reference. |
| `my_notebook.ipynb` | Personal practical notebook, 23 cells. |
| `reference_notebooks/day19.ipynb` | Official CampusX notebook, 17 cells. Read-only reference. |
| `datasets/train.csv` | Titanic dataset, 891 x 12, 61,194 bytes. |

The personal notebook currently reads `E:\MACHINE_LEARNING\train.csv` by absolute path. That root file exists and is byte-identical to `datasets/train.csv`. For portability, use `pd.read_csv("datasets/train.csv")` when running from `Day_19/`.

The official notebook calls `pd.read_csv("train.csv")`. That path is relative to the notebook's working directory. Since the dataset is under `datasets/`, either run from a directory containing `train.csv` or adjust the path.

## Dataset

`datasets/train.csv` is the Titanic training data used by the course:

- 891 rows, 12 columns, 10,692 cells.
- Missing values: Age 177, Cabin 687, Embarked 2; total 866 cells, 8.0995%.
- No exact duplicate rows.
- Target: 342 survived (38.3838%), 549 died (61.6162%).

The dataset is not generated or substituted. Keep the original file name and contents.

## Dependencies

```text
python >= 3.9
pandas
numpy
matplotlib
seaborn
jupyter
```

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

The saved official outputs were produced by an older pandas version. The local environment uses pandas 3.0.5, where:

- text dtypes display as `str` rather than `object`;
- `df.corr()` on the full frame raises `ValueError` because of string columns;
- `df.corr(numeric_only=True)` is required;
- `df.duplicated().sum()` may display as `np.int64(0)`.

## How to run

1. Open `Day_19/` as the working directory.
2. Open `my_notebook.ipynb` and run the cells in order.
3. Use `datasets/train.csv` for a portable path.
4. Open `reference_notebooks/day19.ipynb` only to compare the official sequence; do not edit it.
5. Read `notes.md` Part A for the main explanation, then Part B for advanced reference.

The local heatmap cell saves a 640 x 480 figure. Its markdown records that the kernel crashed after the heatmap rendered; the saved output is still usable. Re-running a read-only inspection is safe.

## Study order

1. `notes.md` Part A.
2. `my_notebook.ipynb`.
3. `reference_notebooks/day19.ipynb`.
4. `notes.md` Part B: statistical and correlation reference, pandas API, source-fidelity audit, extended interview and exam questions.
5. Revise the final checklist and the Titanic numbers table.

## Status

`notes.md` has been rewritten to the approved Day 39 two-layer standard and validated:

- 1,877 lines / 56,018 bytes / 8,906 checker words.
- Part A: 6,144 words; Part B: 2,762 words.
- 57 math items (40 display, 17 inline), all rendered by KaTeX with 0 errors.
- 82 balanced fences; 0 Mermaid diagrams; 0 raw LaTeX delimiters.

No notebook, dataset, `vendor/`, or `reference_notebooks/` file was modified.
