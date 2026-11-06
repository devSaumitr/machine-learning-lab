# Day 21 - Bivariate and Multivariate Analysis

CampusX 100 Days of Machine Learning, Day 21.

## Main topics

- Bivariate and multivariate EDA: moving from one-column summaries to relationships.
- Numeric-numeric plots: `scatterplot` and the figure-level `relplot`.
- Categorical-numeric plots: `barplot`, `boxplot`, and `catplot`.
- Distribution comparisons: deprecated `distplot`, plus modern `histplot` and `kdeplot`.
- Categorical-categorical analysis: `crosstab`, count/rate heatmaps, and `clustermap`.
- Multivariate grids: `pairplot` for the iris measurements.
- Ordered numeric data: flights yearly lineplot and month-by-year clustermap.
- Pivot-table heatmaps and why exact-Age aggregation produces a sparse, weak matrix.
- Matplotlib/Seaborn object model, figure-level versus axes-level functions, and execution traps.
- Source-fidelity issues caused by old APIs, stale outputs, and version differences.

This day is **exploratory**: there is no train/test split, no model training, and no cleaning performed in the official notebook.

## Learning objectives

1. Choose a plot from the types of the two variables being compared.
2. Explain what `hue`, `style`, `size`, `row`, and `col` add to a plot.
3. Distinguish counts, rates, and means, and always report the denominator.
4. Read a crosstab with row and column percentages.
5. Compare a barplot of means with a boxplot of distributions.
6. Explain what a clustermap adds to a heatmap and what it does not prove.
7. Read a pairplot for separable and overlapping classes.
8. Read a lineplot for trend and seasonality.
9. Explain why the local pivot-table heatmap is sparse.
10. Identify the official and local source-fidelity bugs and distinguish a saved output from a reproducible run.

## Files

| File | Role |
|---|---|
| `notes.md` | Unified deep chapter: Part A main learning chapter and Part B advanced reference. |
| `my_notebook.ipynb` | Personal practical notebook, 26 cells (13 code, 13 markdown). |
| `reference_notebooks/day21.ipynb` | Official CampusX notebook, 33 cells (25 code, 8 markdown). Read-only reference. |
| `datasets/train.csv` | Titanic dataset, 891 x 12, 61,194 bytes. |

There is no cheat-sheet notebook in `Day_21/`. The folder contains only the files above.

The personal notebook reads `E:\MACHINE_LEARNING\train.csv` by absolute path. That root file is byte-identical to `datasets/train.csv`; for portability, use `pd.read_csv("datasets/train.csv")` when running from `Day_21/`.

The official notebook calls `pd.read_csv("train.csv")`. That path is relative to its working directory; since the dataset is under `datasets/`, either run from a directory containing `train.csv` or adjust the path.

## Notebook differences

| Question | Official notebook | Personal notebook |
|---|---|---|
| Numeric-numeric plot | `scatterplot` with `df` undefined | `relplot` with `row="sex"` |
| Categorical-numeric plot | positional `barplot` | `catplot(kind="bar")` |
| Boxplot | Sex x Age, split by Survived | Pclass x Age |
| Distribution comparison | two `distplot` calls | two `distplot` calls |
| Categorical-categorical | Pclass heatmap; Parch clustermap | Pclass heatmap; SibSp clustermap with typo |
| Pairplot | iris by species | iris by species |
| Flights | yearly lineplot and month-by-year clustermap | not included |
| Pivot heatmap | not included | exact-Age x SibSp heatmap |

The official notebook covers more plot types but several cells do not run in the current environment. The personal notebook is a working notebook with partial saved outputs.

## Dataset

`datasets/train.csv` is the Titanic training data used by the course:

- 891 rows, 12 columns.
- Missing values: `Age` 177, `Cabin` 687, `Embarked` 2.
- `Sex`: male 577, female 314.
- `Survived`: died 549, survived 342.
- `Pclass`: 216, 184, 491.
- `Embarked`: S 644, C 168, Q 77.
- Survival by class: 62.96%, 47.28%, 24.24%.
- Survival by sex: female 74.20%, male 18.89%.
- Survival by port: C 55.36%, Q 38.96%, S 33.70%.

The dataset is not generated or substituted. Keep the original file name and contents.

## Dependencies

```text
python >= 3.9
pandas
numpy
scipy
matplotlib
seaborn
jupyter
```

```bash
pip install pandas numpy scipy matplotlib seaborn jupyter
```

The saved official outputs were produced by an older environment. The local environment uses pandas 3.0.5, NumPy 2.5.3, SciPy 1.18.1, Matplotlib 3.11.2, and Seaborn 0.13.2. Consequences:

- `sns.distplot` still exists in Seaborn 0.13.2 but is deprecated; use `histplot(kde=True)` or `kdeplot`.
- Old positional `scatterplot`, `barplot`, `boxplot`, and `lineplot` calls raise `TypeError`.
- `titanic.groupby("Embarked").mean()` fails because string columns do not support `mean`; select `Survived` first.
- `flights.groupby("year").sum()` fails because categorical `month` does not support sum; select `passengers` first.
- The official saved flights month labels are full names; current Seaborn loads abbreviations.
- Pandas 3.0 displays text columns as `str` rather than `object`.

## How to run

1. Open `Day_21/` as the working directory.
2. Open `my_notebook.ipynb` and run the cells in order; use `datasets/train.csv` for a portable path.
3. Expect the local clustermap cell to raise `KeyError: 'Sibsp'` on its second call; the first call still draws a figure. The saved image does not prove the visible source completed cleanly.
4. Expect the local notebook to have no saved outputs for the relplot, catplot, boxplot, distplot, or first heatmap cells.
5. Open `reference_notebooks/day21.ipynb` only to compare the official sequence; do not edit it.

## Study order

1. `notes.md` Part A.
2. `my_notebook.ipynb`.
3. `reference_notebooks/day21.ipynb`.
4. `notes.md` Part B: statistical foundations, plotting internals, source-fidelity audit, extended comparisons, interview bank, and exam questions.
5. Revise the plot-choice table, Titanic numbers table, and final checklist.

## Status

`notes.md` has been rewritten to the approved Day 39 two-layer standard and validated:

- 3,303 lines / 106,709 bytes / 16,805 checker words.
- Part A: 11,932 words; Part B: 4,873 words.
- 105 math items (32 display, 73 inline), all rendered by KaTeX with 0 errors.
- 146 balanced fences; 0 Mermaid diagrams; 0 raw LaTeX delimiters.
- 3 structural H1 headings: title, Part A, Part B.

No notebook, dataset, `vendor/`, or `reference_notebooks/` file was modified.
