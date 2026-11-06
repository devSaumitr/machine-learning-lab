# Day 20 - Univariate Analysis

CampusX 100 Days of Machine Learning, Day 20.

## Main topics

- Univariate categorical analysis: counts, shares, countplot, bar chart, and pie chart.
- Univariate numerical analysis: histogram, bin choice, KDE/`distplot`, boxplot, and IQR fences.
- Titanic examples: `Embarked`, `Sex`, `Pclass`, `Survived`, `Age`, and `Fare`.
- Matplotlib figure/axes model and pandas/Seaborn plotting APIs.
- Deprecated `distplot` and legacy positional `countplot` calls.
- EDA findings that motivate later preprocessing decisions.

This day is **exploratory**: there is no train/test split, no model training, and no cleaning performed in the official notebook.

## Learning objectives

1. Choose a countplot, bar chart, or pie for a categorical variable.
2. Choose a histogram, KDE, or boxplot for a numerical variable.
3. Explain how bin count and KDE bandwidth change the picture.
4. Compute and interpret counts, proportions, quartiles, IQR, fences, and skewness.
5. Distinguish an outlier candidate from a data error.
6. Read the Age and Fare distributions and state conclusions without overclaiming.
7. Explain why `distplot` is deprecated and what replaces it.
8. Keep the official notebook, personal notebook, and cheat sheet distinct.

## Files

| File | Role |
|---|---|
| `notes.md` | Unified deep chapter: Part A main learning chapter and Part B advanced reference. |
| `my_notebook.ipynb` | Personal practical notebook, 22 cells (9 code, 13 markdown). |
| `reference_notebooks/day20.ipynb` | Official CampusX notebook, 21 cells (13 code, 8 markdown). Read-only reference. |
| `Matplotlib-Seaborn-Cheat-Sheet.ipynb` | Companion plotting reference, 74 cells (34 code, 40 markdown); synthetic data, no saved outputs. |
| `datasets/train.csv` | Titanic dataset, 891 x 12, 61,194 bytes. |

The personal notebook reads `E:\MACHINE_LEARNING\train.csv` by absolute path. That root file is byte-identical to `datasets/train.csv`; for portability, use `pd.read_csv("datasets/train.csv")` when running from `Day_20/`.

The official notebook calls `pd.read_csv("train.csv")`. That path is relative to its working directory; since the dataset is under `datasets/`, either run from a directory containing `train.csv` or adjust the path.

The cheat sheet uses a synthetic 12-row DataFrame, not the Titanic data. Its shapes, counts, and correlations are not Titanic facts.

## Notebook differences

| Question | Official notebook | Personal notebook |
|---|---|---|
| Categorical feature | `Embarked` | `Pclass` |
| Target plot | none | `Survived` pie |
| Age histogram bins | 5 | 300 |
| Age density | `sns.distplot` | `sns.distplot` |
| Boxplot variable | `Age` | `Fare` |
| Age summary | min, max, mean, skew | min only |
| Saved outputs | images and scalar results | none; final cell records a kernel crash |

The personal notebook's bare `plt.hist` cell references the function without calling it and draws nothing. The kernel-crash error in the final cell is an environment issue, not evidence that `df["Age"].min()` is invalid.

## Dataset

`datasets/train.csv` is the Titanic training data used by the course:

- 891 rows, 12 columns.
- Missing values: `Age` 177, `Cabin` 687, `Embarked` 2.
- `Sex`: male 577, female 314.
- `Survived`: died 549, survived 342.
- `Pclass`: 216, 184, 491.
- `Embarked`: S 644, C 168, Q 77.
- Age: mean 29.6991, median 28, standard deviation 14.5265, Q1 20.125, Q3 38, min 0.42, max 80, skew 0.3891.
- Fare: mean 32.2042, median 14.4542, standard deviation 49.6934, Q1 7.9104, Q3 31, min 0, max 512.3292, skew 4.7873.
- Fare has 116 values above the upper fence 65.6344; Age has 11 above 64.8125.

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

The saved official outputs were produced by an older environment. The local environment uses pandas 3.0.5, NumPy 2.5.3, Matplotlib 3.11.2, and Seaborn 0.13.2. Consequences:

- `sns.distplot` still exists in Seaborn 0.13.2 but is deprecated; use `histplot(kde=True)` or `kdeplot`.
- `sns.countplot(df["Embarked"])` still works but is a legacy positional call; prefer `data=` and `x=`.
- Old Matplotlib axes representations differ from current representations.
- Pandas 3.0 displays text columns as `str` rather than `object`.

## How to run

1. Open `Day_20/` as the working directory.
2. Open `my_notebook.ipynb` and run the cells in order; use `datasets/train.csv` for a portable path.
3. Rerun the final Age summary in a fresh kernel if the saved kernel crash occurs.
4. Open `reference_notebooks/day20.ipynb` only to compare the official sequence; do not edit it.
5. Use `Matplotlib-Seaborn-Cheat-Sheet.ipynb` for syntax lookup, not as Titanic evidence.
6. Read `notes.md` Part A for the main explanation, then Part B for advanced reference.

## Study order

1. `notes.md` Part A.
2. `my_notebook.ipynb`.
3. `reference_notebooks/day20.ipynb`.
4. `Matplotlib-Seaborn-Cheat-Sheet.ipynb`.
5. `notes.md` Part B: statistical reference, plotting internals, cheat-sheet audit, source-fidelity audit, extended interview and exam questions.
6. Revise the plot-choice table, Titanic numbers table, and final checklist.

## Status

`notes.md` has been rewritten to the approved Day 39 two-layer standard and validated:

- 2,659 lines / 84,993 bytes / 13,457 checker words.
- Part A: 8,988 words; Part B: 4,461 words.
- 135 math items (47 display, 88 inline), all rendered by KaTeX with 0 errors.
- 106 balanced fences; 0 Mermaid diagrams; 0 raw LaTeX delimiters.

No notebook, dataset, cheat-sheet, `vendor/`, or `reference_notebooks/` file was modified.
