# Day 22 - Automated EDA with Pandas Profiling and Sweetviz

> **Course:** CampusX 100 Days of Machine Learning, Day 22  
> **Official notebook:** `reference_notebooks/day22.ipynb` (6 code cells, no markdown, Python 3.8.3)  
> **Personal notebooks:** `my_notebook.ipynb` (8 cells: 4 code, 4 markdown), `DAY22_Pandas-Profiling-Sweetviz.ipynb` (10 cells: 4 code, 6 markdown), `DAY22_Pandas-Profiling-empty-placeholder.ipynb` (3 markdown cells, no code)  
> **Dataset:** `datasets/train.csv` (Titanic: 891 rows, 12 columns)  
> **Main report artifact:** `titanic_report.html` (ydata-profiling 4.18.4; 4,772,777 bytes)  
> **Official report artifact:** `vendor/100-days-of-machine-learning/day22-pandas-profiling/output.html` (pandas-profiling 2.11.0; 3,228,626 bytes)

This chapter has two parts. **Part A** is the main learning chapter: it explains why automated EDA exists, what the notebooks actually do, how to read the generated report, and how to interpret the Titanic findings. **Part B** is the advanced reference: it covers report internals, correlation mathematics, type inference, performance, source-fidelity details, code patterns, and an extended interview and examination bank.

The day contains four distinct notebooks and two report generations. They are not merged:

1. the personal ydata-profiling notebook (`my_notebook.ipynb`),
2. the personal Sweetviz notebook (`DAY22_Pandas-Profiling-Sweetviz.ipynb`),
3. the personal empty placeholder,
4. the official legacy `pandas-profiling` notebook.

The old notes in this folder referred to `DAY22_Pandas-Profiling-ydata.ipynb`; that file is not present. The actual ydata notebook is now `my_notebook.ipynb`. The old notes also mentioned `topic_notes.md`, `eda_utils.py`, `eda_report.html`, and `eda_plots/`; those are not part of the current Day 22 folder and are not used as evidence here.

---

# Part A - Main Learning Chapter

## 1. The problem: manual EDA does not scale

Days 19-21 taught EDA one plot at a time:

- descriptive statistics and first checks,
- univariate distributions,
- bivariate and multivariate relationships.

That workflow is essential because it teaches judgment. It is also slow. A dataset with 200 columns would require hundreds of plots, tables, and checks before a model is trained. Many of those checks are repetitive:

- How many values are missing in each column?
- What is the distribution of each numeric column?
- Which columns are categorical, text, or identifiers?
- Are there duplicate rows?
- Which variables are correlated?
- Are there constant columns, high-cardinality columns, or columns with many zeros?
- Are there unusual values or impossible ranges?
- How do the variables interact?

**Automated EDA** runs these checks in one pass and produces a report. The goal is not to replace the analyst. The goal is to remove repetitive first-pass work so the analyst can spend time on interpretation, domain questions, and decisions.

The key sentence for this day is:

> Automated profiling gives you a map. It does not tell you which road to take.

The report can tell you that `Cabin` is 77.1% missing. It cannot tell you whether to drop the column, turn it into a deck feature, or keep the missingness as a signal. That decision needs domain knowledge and validation.

---

## 2. The dataset and the source map

### 2.1 Dataset

The dataset is the Titanic training table:

- 891 rows,
- 12 columns,
- 61,194 bytes,
- SHA-256 `7D118FEF8B6CCF7F81111877BC388536F7B1E498A655E3D649D19AAA010E9F6F`.

The Day 22 copy is byte-identical to the vendor copy. The columns are:

| Column | Meaning | Type in the data |
|---|---|---|
| `PassengerId` | Row identifier | Integer |
| `Survived` | Target: 0 = died, 1 = survived | Integer/binary |
| `Pclass` | Passenger class: 1, 2, 3 | Integer/categorical |
| `Name` | Passenger name | Text |
| `Sex` | Male/female | Categorical text |
| `Age` | Age in years | Float with missing values |
| `SibSp` | Siblings/spouses aboard | Integer count |
| `Parch` | Parents/children aboard | Integer count |
| `Ticket` | Ticket number | Text |
| `Fare` | Fare paid | Float |
| `Cabin` | Cabin number | Text with many missing values |
| `Embarked` | Port: S, C, Q | Categorical text with 2 missing |

The first five rows shown by both notebooks are the same. The `Cabin` value is missing in rows 0, 2, and 4. The first row is `Braund, Mr. Owen Harris`, male, age 22, fare 7.25, embarked S, survived 0.

### 2.2 Notebook map

| File | Role | Cells | Kernel | Library |
|---|---|---:|---|---|
| `my_notebook.ipynb` | Personal ydata-profiling workflow | 8 (4 code, 4 markdown) | `.venv312`, Python 3.12.10 | ydata-profiling 4.18.4 |
| `DAY22_Pandas-Profiling-Sweetviz.ipynb` | Personal Sweetviz comparison | 10 (4 code, 6 markdown) | `.venv`, Python 3.14.2 | Sweetviz 2.3.3 |
| `DAY22_Pandas-Profiling-empty-placeholder.ipynb` | Personal placeholder | 3 markdown, 0 code | `.venv`, Python 3.14.2 | none |
| `reference_notebooks/day22.ipynb` | Official CampusX notebook | 6 code, 0 markdown | Python 3.8.3 | pandas-profiling 2.11.0 |

The official notebook is the historical source. The personal notebooks are the current learning environment. They are not the same code and should not be merged.

### 2.3 Report map

| Artifact | Generator | Size | Date | Notes |
|---|---|---:|---|---|
| `Day_22/titanic_report.html` | ydata-profiling 4.18.4 | 4,772,777 bytes | 2026-10-22 | Current folder artifact; ydata, not Sweetviz |
| `titanic_report.html` at repo root | ydata-profiling 4.18.4 | 4,772,682 bytes | 2026-10-10 | Older duplicate outside the day folder |
| `vendor/.../day22-pandas-profiling/output.html` | pandas-profiling 2.11.0 | 3,228,626 bytes | 2021-04-05 | Official historical report |
| Sweetviz report | Sweetviz 2.3.3 | not found | — | Notebook says generated, but no Sweetviz HTML artifact is present |

This is an important source-fidelity finding. The notebook output says `Report titanic_report.html was generated!`, but the two HTML files that exist are ydata-profiling reports. The Sweetviz output was either written elsewhere, overwritten, or lost. We cannot quote Sweetviz-specific report contents from the saved artifacts.

---

## 3. What automated profiling actually does

A profiler performs a sequence of summaries:

1. **Type inference:** decide whether each column is numeric, categorical, text, date, or another type.
2. **Per-column description:** count, missing values, distinct values, minimum, maximum, mean, median, quartiles, standard deviation, skewness, kurtosis, zeros, and frequent values.
3. **Dataset-level summary:** number of rows, columns, missing cells, duplicate rows, memory size, and variable types.
4. **Alerts:** heuristic warnings such as high correlation, high cardinality, many missing values, uniform distribution, unique values, or many zeros.
5. **Interactions:** pairwise relationships and joint distributions.
6. **Correlations:** several correlation measures, depending on the library and version.
7. **Missing-value analysis:** counts, matrix, heatmap, and sometimes a dendrogram.
8. **Sample rows:** first and last rows for a quick view.
9. **Rendering:** one self-contained HTML file with tables, charts, and metadata.

The profiler does not train a model. It does not decide which features to use. It does not prove causation. It is a description of the data at a particular moment.

### 3.1 What it does well

- finds missing values and duplicate rows quickly,
- summarizes distributions and extreme values,
- exposes high-cardinality and identifier-like columns,
- gives a first look at correlations and interactions,
- makes a dataset understandable to another person,
- creates a reproducible starting point for deeper analysis.

### 3.2 What it cannot do

- decide whether a value is an error or a genuine rare event,
- know the business cost of a false positive,
- detect target leakage without domain knowledge,
- prove that a feature will improve a model,
- replace train/test validation,
- guarantee that type inference is correct,
- preserve privacy if the report contains raw samples.

---

## 4. The ydata-profiling notebook

### 4.1 The code

The main report cell in `my_notebook.ipynb` is:

```python
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv(r"E:\MACHINE_LEARNING\train.csv")

profile = ProfileReport(
    df,
    title="Titanic EDA Report",
    explorative=True
)

profile.to_file("titanic_report.html")
```

The notebook also has an earlier cell:

```python
import numpy as np
import pandas as pd
```

and a cell that runs:

```python
! pip install ydata-profiling
```

The dataset is read twice: once in the `df.head()` cell and again inside the report cell. That is redundant but harmless for a small dataset.

### 4.2 Syntax breakdown

| Piece | What it does |
|---|---|
| `from ydata_profiling import ProfileReport` | Imports the report class from the installed package |
| `pd.read_csv(...)` | Loads the CSV into a DataFrame |
| `ProfileReport(df, ...)` | Creates a report object; it does not write a file yet |
| `title="Titanic EDA Report"` | Sets the HTML title |
| `explorative=True` | Requests a richer report with more interaction and correlation work |
| `profile.to_file(...)` | Renders the report and writes the self-contained HTML |

`ProfileReport` is not an estimator in the scikit-learn sense. There is no `fit`/`predict` workflow. It is a reporting object. The expensive work happens when the object is constructed or when `to_file` renders the report, depending on the internal lazy configuration. The output is a file, not a model.

### 4.3 Execution flow

The saved output shows:

1. imports run,
2. `! pip install ydata-profiling` reports that 4.18.4 is already installed,
3. `df.head()` displays the first five rows,
4. `ProfileReport(...)` runs the describe/summarize/render/export stages,
5. progress bars reach 100%,
6. the kernel crashes after the export.

The report cell emits a `TqdmWarning` about `IProgress` and a deprecation warning saying that `import ydata_profiling` is deprecated in favor of `fg-data-profiling`. These warnings are not the cause of the crash. The kernel crash is an environment/runtime problem that must be investigated separately.

The important lesson: **a 100% progress bar is not proof that the report is complete.** Open the HTML, check its size, check its metadata, and regenerate in a clean kernel if necessary.

### 4.4 `explorative=True`

The `explorative` flag asks for a more thorough report. It increases the amount of interaction and correlation analysis. It can be slower and can produce a larger file. For a small dataset like Titanic, the cost is manageable. For a wide dataset, the default or a lighter configuration may be preferable.

The notebook's experiment cell suggests comparing `explorative=True` and `explorative=False` under different filenames. That is a good exercise:

- predict which report is larger,
- predict which is slower,
- compare which sections change,
- check whether the extra detail changes any decision.

Do not overwrite the original report while experimenting. Use names such as `titanic_report_explorative.html` and `titanic_report_minimal.html`.

### 4.5 `to_file` and the output

`to_file("titanic_report.html")` writes a self-contained HTML file. The file contains the tables, charts, CSS, and JavaScript needed to view the report. It does not need the Python environment to open, but it may need a modern browser.

Other useful output methods in the ydata-profiling family include:

- `to_notebook_iframe()` for inline display in a notebook,
- `to_json()` for a machine-readable description,
- `to_widgets()` for an interactive widget in some versions.

The exact API can change between versions. The stable idea is: the report object can be rendered to HTML, JSON, or an inline view.

---

## 5. Reading the actual ydata report

The current `Day_22/titanic_report.html` is a ydata-profiling 4.18.4 report titled `Titanic EDA Report`. Its reproduction metadata says:

- analysis started: 2026-10-22 17:06:02.039959,
- analysis finished: 2026-10-22 17:06:03.909109,
- duration: 1.87 seconds,
- software: ydata-profiling v4.18.4.

The report has six main tabs: **Overview**, **Variables**, **Interactions**, **Correlations**, **Missing values**, and **Sample**.

### 5.1 Overview

The overview reports:

| Statistic | Value |
|---|---:|
| Number of variables | 12 |
| Number of observations | 891 |
| Missing cells | 866 |
| Missing cells (%) | 8.1% |
| Duplicate rows | 0 |
| Duplicate rows (%) | 0.0% |
| Total size in memory | 285.6 KiB |
| Average record size | 328.2 B |
| Numeric variables | 5 |
| Categorical variables | 4 |
| Text variables | 3 |

The type counts correspond to:

- numeric: `PassengerId`, `Age`, `SibSp`, `Parch`, `Fare`,
- categorical: `Survived`, `Pclass`, `Sex`, `Embarked`,
- text: `Name`, `Ticket`, `Cabin`.

This is a useful reminder that a profiler's type is not the same as a database type. `Pclass` is stored as an integer, but it is categorical in meaning. `Survived` is stored as 0/1, but it is the binary target. `PassengerId` is numeric in storage, but it is an identifier, not a measurement.

### 5.2 Alerts

The report lists 10 alerts:

1. `Sex` is highly overall correlated with `Survived`.
2. `Survived` is highly overall correlated with `Sex`.
3. `Age` has 177 (19.9%) missing values.
4. `Cabin` has 687 (77.1%) missing values.
5. `PassengerId` is uniformly distributed.
6. `PassengerId` has unique values.
7. `Name` has unique values.
8. `SibSp` has 608 (68.2%) zeros.
9. `Parch` has 678 (76.1%) zeros.
10. `Fare` has 15 (1.7%) zeros.

These alerts are **heuristics**, not errors. Each one needs interpretation.

- The `Sex`/`Survived` correlation is expected in the Titanic data. It is an association, not a data-quality bug. It does not prove causation.
- `Age` missingness is substantial and is related to passenger class. It is not missing completely at random.
- `Cabin` missingness is so high that the column may be unusable without a deliberate strategy.
- `PassengerId` is an identifier. Its uniform distribution and uniqueness are expected, and it should not be used as a predictive feature.
- `Name` uniqueness is expected for passenger names. It is not a useful numeric feature, but it can be parsed for titles or surnames.
- `SibSp` and `Parch` zeros are expected because most passengers travelled alone. A zero is a meaningful count, not a missing value.
- `Fare` zeros are unusual but may be valid: the 15 zero-fare rows include `LINE` tickets and special bookings. Do not automatically treat them as missing.

The old official report has a different alert set because it uses an older library and different type inference. That comparison is covered in Section 12.

### 5.3 Variables tab

The Variables tab contains one section per column. A typical section includes:

- type and flags,
- distinct count,
- missing count and percentage,
- infinite count,
- mean, minimum, maximum, zeros,
- memory size,
- quantiles and descriptive statistics,
- histogram,
- common values,
- extreme values,
- text statistics for text columns.

The profiler is not just showing a `describe()` table. It is also showing distribution shape and frequent values. That is why the report is useful for a first pass.

#### Age

The report gives:

| Statistic | Value |
|---|---:|
| Distinct | 88 |
| Missing | 177 |
| Missing (%) | 19.9% |
| Mean | 29.699118 |
| Minimum | 0.42 |
| Maximum | 80 |
| Q1 | 20.125 |
| Median | 28 |
| Q3 | 38 |
| Standard deviation | 14.526497 |
| Skewness | 0.3891 |
| Kurtosis | 0.1783 |

The distribution is mildly right-skewed. The mean is slightly above the median. The minimum of 0.42 years is a real infant, not an error. The maximum of 80 is plausible.

The missingness is not evenly distributed by class:

| Pclass | Missing Age | Total | Missing rate |
|---:|---:|---:|---:|
| 1 | 30 | 216 | 13.89% |
| 2 | 11 | 184 | 5.98% |
| 3 | 136 | 491 | 27.70% |

This is a strong warning that simple mean imputation would distort the class-specific age distribution. It also previews later imputation topics.

#### Fare

The report gives:

| Statistic | Value |
|---|---:|
| Distinct | 248 |
| Missing | 0 |
| Mean | 32.204208 |
| Minimum | 0 |
| Maximum | 512.3292 |
| Q1 | 7.9104 |
| Median | 14.4542 |
| Q3 | 31 |
| Standard deviation | 49.693429 |
| Skewness | 4.7873 |
| Kurtosis | 33.3981 |
| Zeros | 15 (1.7%) |

The mean is more than twice the median. The maximum is far above the upper quartile. This is a strongly right-skewed distribution with a heavy tail. The report's histogram and extreme-value table make the skew visible.

The 15 zero fares are not missing values. They are real records with a fare of zero, including `LINE` tickets and special bookings. Whether they should be treated as unusual depends on the question. A model that predicts fare may need a separate treatment; a model that uses fare as a predictor should not silently replace zeros with the mean.

#### Sex

- male: 577 (64.8%),
- female: 314 (35.2%).

The survival rates are:

- female: 233/314 = 74.20%,
- male: 109/577 = 18.89%.

The risk difference is about 55.3 percentage points. The odds ratio is about 12.35. This is a strong association and explains the correlation alert. It is also a reminder that a high correlation with the target can be legitimate; it is not automatically leakage.

#### Pclass

- class 3: 491 (55.1%),
- class 1: 216 (24.2%),
- class 2: 184 (20.7%).

Survival rates:

- class 1: 62.96%,
- class 2: 47.28%,
- class 3: 24.24%.

`Pclass` is stored as an integer, but the values are categories with an order. The profiler's "numeric" or "categorical" label does not decide how the feature should be encoded.

#### Embarked

- S: 644 (72.3% of non-missing),
- C: 168 (18.9%),
- Q: 77 (8.6%),
- missing: 2 (0.2%).

The two missing values are easy to handle, but the choice of mode imputation should still be recorded.

#### Cabin

- missing: 687 (77.1%),
- non-missing: 204,
- distinct: 147,
- unique among non-missing: 101 (49.5%).

The high missing rate and high cardinality make `Cabin` a feature-engineering problem, not a simple numeric column. A first letter or deck extraction may be more useful than the raw cabin string.

#### Name and Ticket

- `Name`: 891 distinct (100% unique), text.
- `Ticket`: 681 distinct (76.4%), 547 unique (61.4%), text.

These are high-cardinality text columns. They are not directly usable as numeric features. `Name` can yield titles, surnames, or family information; `Ticket` can yield prefixes or group information. The profiler flags their cardinality, but the decision to engineer them is a modeling choice.

### 5.4 Interactions tab

The Interactions tab shows pairwise relationships. For numeric-numeric pairs it may show scatterplots or heatmaps; for categorical pairs it may show grouped summaries. The purpose is to reveal patterns that are invisible in one-dimensional summaries.

A useful Titanic example is the interaction between `Sex` and `Pclass`:

| Sex | Pclass | Survived rate | Count |
|---|---:|---:|---:|
| female | 1 | 96.8% | 94 |
| female | 2 | 92.1% | 76 |
| female | 3 | 50.0% | 144 |
| male | 1 | 36.9% | 122 |
| male | 2 | 15.7% | 108 |
| male | 3 | 13.5% | 347 |

The `Sex` effect is not constant across classes. A female third-class passenger has a very different survival rate from a female first-class passenger. A model that uses only `Sex` misses this interaction. The report's interaction view helps you notice it, but you should verify the pattern with a crosstab and a validation split.

### 5.5 Correlations tab

The report can show several correlation measures. In the ydata-profiling family these include:

- Pearson correlation for numeric-numeric relationships,
- Spearman rank correlation for monotone relationships,
- Phik correlation for mixed-type relationships,
- Cramér's V for categorical-categorical associations.

The exact tabs and defaults depend on the version and configuration. Do not compare a Pearson coefficient with a Phik coefficient as if they were the same scale.

For the Titanic data, supplementary Pearson correlations computed from the dataset are:

| Pair | Pearson $r$ |
|---|---:|
| `Survived` - `Pclass` | -0.3385 |
| `Survived` - `Fare` | 0.2573 |
| `Survived` - `Age` | -0.0772 |
| `Pclass` - `Fare` | -0.5495 |
| `Age` - `Pclass` | -0.3692 |
| `SibSp` - `Parch` | 0.4148 |

These are descriptive associations. They do not adjust for confounding, and they do not establish causation. The strong `Pclass`-`Fare` correlation is expected: first-class passengers paid more. The `Sex`-`Survived` association is stronger than any of these Pearson values because `Sex` is categorical; a Pearson coefficient on a binary encoding is possible but not the most informative summary.

### 5.6 Missing values tab

The Missing values tab shows:

- the count and percentage missing per column,
- a missing-value matrix,
- a heatmap of co-missingness,
- sometimes a dendrogram grouping columns with similar missingness patterns.

The Titanic missingness is concentrated:

- `Cabin`: 687 missing (77.1%),
- `Age`: 177 missing (19.9%),
- `Embarked`: 2 missing (0.2%),
- all other columns: 0 missing.

The total is 866 missing cells. The missing-cell percentage is

$$
\frac{866}{891\times12}\times100
=\frac{866}{10{,}692}\times100
\approx8.10\%.
$$

Missingness is not automatically a problem to erase. `Cabin` missingness may itself be informative: passengers without a recorded cabin may differ from those with one. `Age` missingness is related to class. `Embarked` missingness is small enough that mode imputation is plausible, but the decision should still be recorded.

### 5.7 Sample tab

The Sample tab shows first and last rows. It is useful for checking that the file was read correctly and for seeing raw values, but it is not a substitute for a random sample or a stratified sample. The first rows may be ordered by passenger ID and may not represent the population.

### 5.8 Reproduction section

The reproduction section records:

- start time,
- finish time,
- duration,
- software version,
- a configuration download link.

This is important for reproducibility. A report without a version and configuration is hard to reproduce. The current report says 1.87 seconds and ydata-profiling 4.18.4. The official report says 8.02 seconds and pandas-profiling 2.11.0. The times are not directly comparable because the hardware, data, library, and configuration differ.

---

## 6. The Sweetviz notebook

### 6.1 The code

The Sweetviz notebook reads the same data:

```python
import pandas as pd

titanic = pd.read_csv(r"E:\MACHINE_LEARNING\train.csv")
```

It installs the package inside the notebook:

```python
%pip install sweetviz
```

Then it builds and saves a report:

```python
import pandas as pd
import sweetviz as sv

report = sv.analyze(titanic)
report.show_html("titanic_report.html")
```

### 6.2 Syntax breakdown

| Piece | What it does |
|---|---|
| `%pip install sweetviz` | Installs into the current kernel environment |
| `import sweetviz as sv` | Imports the library |
| `sv.analyze(titanic)` | Builds a Sweetviz report object |
| `report.show_html(...)` | Writes the HTML report and normally opens a browser |

`sv.analyze` is the analysis step. `show_html` is the rendering step. The object is not a model; it is a report builder.

Sweetviz can also take a target:

```python
report = sv.analyze(titanic, target_feature="Survived")
```

This adds target-comparison panels. The local notebook's experiment cell suggests trying this under a new filename. There is no saved output for that experiment, so we cannot claim what the target report looked like.

### 6.3 `%pip` versus `!pip`

This difference matters:

- `!pip install ...` runs a shell command. It may use a different Python interpreter from the one running the notebook.
- `%pip install ...` is a notebook magic that installs into the current kernel environment.

The ydata notebook uses `! pip install`; the Sweetviz notebook uses `%pip install`. In a well-managed environment, neither is necessary because packages should be installed outside the notebook. In a notebook, `%pip` is usually safer for kernel consistency.

### 6.4 What happened in the saved output

The saved Sweetviz output shows:

1. `%pip install sweetviz` downloads/uses Sweetviz 2.3.3 and installs `importlib_resources`,
2. `sv.analyze(titanic)` shows a progress widget,
3. `report.show_html("titanic_report.html")` prints `Report titanic_report.html was generated!`,
4. the cell emits many `findfont: Failed to find font weight normal for Roboto` warnings,
5. the kernel crashes.

The font warnings are cosmetic. The kernel crash is not cosmetic. The report may have been written before the crash, but the saved artifact does not prove it.

### 6.5 The missing Sweetviz artifact

The current Day 22 folder contains only a ydata report. The root contains another ydata report. No Sweetviz HTML is present. This means:

- we can describe the Sweetviz code,
- we can describe what the notebook claims,
- we cannot quote Sweetviz-specific report panels from this run,
- we cannot compare the visual layout from saved artifacts.

This is a source-fidelity limitation, not a reason to invent a result.

### 6.6 Sweetviz versus ydata-profiling

Both libraries produce a one-call HTML report. They differ in emphasis:

| Aspect | ydata-profiling | Sweetviz |
|---|---|---|
| Main call | `ProfileReport(df)` | `sv.analyze(df)` |
| Output | `to_file(path)` | `show_html(path)` |
| Type detail | Extensive per-column statistics | Compact visual associations |
| Correlations | Multiple measures | Association-focused |
| Target comparison | Available through configuration | `target_feature` comparison |
| Report style | Detailed tables and tabs | Visual, dashboard-like |
| Best use | Deep first-pass audit | Quick visual comparison |

The best choice depends on the task. For a detailed audit of every column, ydata-profiling is often more informative. For a fast visual comparison, Sweetviz is convenient. Neither replaces manual validation.

---

## 7. The official legacy notebook

The official notebook is `reference_notebooks/day22.ipynb`:

- 6 code cells,
- 0 markdown cells,
- Python 3.8.3,
- `pandas-profiling` 2.11.0.

Its code is:

```python
import pandas as pd

df = pd.read_csv('train.csv')

df.head()

!pip install pandas-profiling

from pandas_profiling import ProfileReport
prof = ProfileReport(df)
prof.to_file(output_file='output.html')
```

Important differences from the personal ydata notebook:

- it uses the old package name `pandas_profiling`,
- it reads a relative `train.csv`,
- it does not set a title,
- it does not use `explorative=True`,
- it writes `output.html`,
- it has no markdown explanations,
- its saved outputs show progress widgets, not the report contents.

The official report artifact is the vendor `output.html`:

- 3,228,626 bytes,
- pandas-profiling v2.11.0,
- generated 2021-04-05,
- duration 8.02 seconds,
- 12 variables,
- 891 observations,
- 866 missing cells (8.1%),
- 0 duplicate rows,
- 83.7 KiB memory,
- 96.1 B average record,
- 5 numeric, 7 categorical,
- 14 warnings.

The old report classifies seven columns as categorical, including `Name`, `Ticket`, and `Cabin`; that is why it warns about their high cardinality. The new ydata report treats those three as text, leaving four categorical and three text variables. This is a type-inference difference, not a change in the underlying data.

The old warning list is:

1. `Name` has high cardinality: 891 distinct values.
2. `Ticket` has high cardinality: 681 distinct values.
3. `Cabin` has high cardinality: 147 distinct values.
4. `Age` has 177 (19.9%) missing values.
5. `Cabin` has 687 (77.1%) missing values.
6. `PassengerId` is uniformly distributed.
7. `Name` is uniformly distributed.
8. `Ticket` is uniformly distributed.
9. `Cabin` is uniformly distributed.
10. `PassengerId` has unique values.
11. `Name` has unique values.
12. `SibSp` has 608 (68.2%) zeros.
13. `Parch` has 678 (76.1%) zeros.
14. `Fare` has 15 (1.7%) zeros.

Some of these warnings are awkward. Calling `Name` "uniformly distributed" because every value is unique is not the same as a uniform numeric distribution. The warning is a heuristic based on uniqueness/cardinality, not a formal test. This is a useful lesson: **read the alert, not just the word "warning."**

The official notebook is historically important, but it is not the current API. The package was renamed from `pandas-profiling` to `ydata-profiling`, and the import warning in the personal notebook mentions a further rename toward `fg-data-profiling`. For learning, use the current package and label the version difference. Do not silently pretend the old and new reports are identical.

---

## 8. The empty placeholder notebook

`DAY22_Pandas-Profiling-empty-placeholder.ipynb` has:

- 3 markdown cells,
- 0 code cells,
- kernel `.venv` Python 3.14.2,
- no outputs.

It is a personal bookkeeping file, not an official notebook. It says to do the Day 22 work in the two sibling notebooks. It is intentionally empty and should not be reported as "unfinished analysis." Its only value is to keep the file inventory explicit.

The old notes said the placeholder pointed to `topic_notes.md`. That file is not present. The current notes do not depend on it.

---

## 9. Actual results and source fidelity

### 9.1 What is verified

Verified from the notebooks and files:

- the dataset is 891 x 12 and byte-identical to the vendor copy,
- the ydata notebook has 8 cells (4 code, 4 markdown) and uses ydata-profiling 4.18.4,
- the Sweetviz notebook has 10 cells (4 code, 6 markdown) and uses Sweetviz 2.3.3,
- the placeholder has 3 markdown cells and no code,
- the official notebook has 6 code cells and no markdown,
- the official report is pandas-profiling 2.11.0 and 3,228,626 bytes,
- the current Day 22 report is ydata-profiling 4.18.4 and 4,772,777 bytes,
- the root report is also ydata-profiling 4.18.4,
- no Sweetviz HTML artifact is present.

### 9.2 What is not verified

- The exact saved ydata report may not come from the exact saved notebook run. The report timestamp is 2026-10-22; the notebook file timestamp is 2026-11-02. The report could be from an earlier run, and the notebook could have been edited later.
- The Sweetviz report content is unknown because no Sweetviz artifact is present.
- The target-comparison Sweetviz experiment has no saved output.
- The kernel crash cause is unknown. It may be an environment issue, a memory issue, or a Jupyter/VSCode issue.
- The old official report's exact environment is not reproducible from the current environment without installing the old package.

These limitations are part of the record. They should not be hidden behind a claim that "the notebook ran successfully."

### 9.3 Kernel crashes

Both local report notebooks show a kernel crash after the report step. The ydata output reaches 100% and then crashes. The Sweetviz output says the report was generated and then crashes.

What to do:

1. Check whether the HTML file exists.
2. Open it in a browser.
3. Check its size and metadata.
4. Re-run in a clean kernel with a smaller configuration if needed.
5. Save the report under a new name.
6. Do not assume the report is complete just because progress reached 100%.

A kernel crash after export is still a failed end-to-end run. The artifact may be usable, but the process is not clean.

### 9.4 File-name conflict

Both local notebooks write `titanic_report.html`. The last writer wins. This is a real workflow bug:

- the ydata report can overwrite the Sweetviz report,
- the Sweetviz report can overwrite the ydata report,
- the root and Day 22 files can be confused.

Use distinct names:

```text
titanic_report_ydata.html
titanic_report_sweetviz.html
```

and keep the report metadata with each file.

---

## 10. Mathematical foundations

### 10.1 Missing-cell rate

For a table with $n$ rows and $p$ columns, the total number of cells is $np$. If $M$ cells are missing, the missing-cell rate is

$$
\text{missing rate}=\frac{M}{np}.
$$

For Titanic:

$$
M=177+687+2=866,
\qquad
n=891,
\qquad
p=12,
$$

so

$$
\frac{866}{891\cdot12}
=\frac{866}{10{,}692}
\approx0.0810
=8.1\%.
$$

The per-column missing rate is

$$
m_j=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}[x_{ij}\text{ is missing}].
$$

Examples:

- `Age`: $177/891=0.1987$ (19.9%),
- `Cabin`: $687/891=0.7710$ (77.1%),
- `Embarked`: $2/891=0.0022$ (0.2%).

The denominator matters. A report may show a "distinct (%)" relative to non-missing values, not to all rows. For `Cabin`, 147 distinct values among 204 non-missing values is 72.1%, but 147 among all 891 rows is 16.5%. Always check which denominator the report is using.

### 10.2 Duplicate-row rate

If $D$ rows are exact duplicates of an earlier row, the duplicate-row rate is

$$
\frac{D}{n}.
$$

For Titanic, $D=0$. This only checks exact row equality. It does not detect near-duplicates, the same person entered twice with a different ID, or inconsistent representations of the same entity.

### 10.3 Numeric summaries

For a numeric column $x_1,\dots,x_n$:

$$
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i,
\qquad
s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2,
\qquad
s=\sqrt{s^2}.
$$

The median is the middle value after sorting. The quartiles $Q_1,Q_2,Q_3$ divide the sorted data into four parts. The interquartile range is

$$
\operatorname{IQR}=Q_3-Q_1.
$$

The coefficient of variation is

$$
\operatorname{CV}=\frac{s}{\bar{x}},
$$

which is meaningful when the values are positive and measured on a ratio scale.

Skewness measures asymmetry. For sample skewness, a common definition is

$$
g_1=\frac{\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^3}
{\left(\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2\right)^{3/2}}.
$$

Kurtosis measures tail heaviness relative to a normal distribution. A common excess kurtosis definition is

$$
g_2=
\frac{\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^4}
{\left(\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2\right)^2}-3.
$$

The exact bias correction differs by library. The report's numbers should be read as descriptive, not as a formal test.

For `Fare`:

$$
\bar{x}=32.204208,
\qquad
\operatorname{median}=14.4542,
\qquad
s=49.693429.
$$

The mean is much larger than the median, and the skewness is 4.7873. This is a right-skewed distribution with a long upper tail. The maximum 512.3292 is far above the third quartile 31.

For `Age`:

$$
\bar{x}=29.699118,
\qquad
\operatorname{median}=28,
\qquad
s=14.526497,
\qquad
\text{skewness}=0.3891.
$$

This is a mild right skew. The mean and median are close relative to the spread.

### 10.4 Cardinality and uniqueness

For a column with $d$ distinct values among $n$ rows:

- **distinct ratio:** $d/n$,
- **unique ratio:** the fraction of rows whose value occurs exactly once.

For `Name`, $d=891$ and every value is unique, so both ratios are 100%. For `Ticket`, $d=681$ (76.4% distinct), but only 547 values are unique (61.4%). For `Cabin`, among 204 non-missing values, $d=147$ (72.1% distinct) and 101 are unique (49.5%).

High cardinality is not automatically bad. A high-cardinality column can be:

- an identifier (drop or use only for joins),
- a text field with extractable structure (name titles, ticket prefixes),
- a genuine categorical feature with many levels (needs encoding or grouping),
- a data-quality problem (inconsistent spellings).

The profiler can flag it; the analyst must classify it.

### 10.5 Correlation measures

**Pearson correlation** measures linear association between two numeric variables:

$$
r=
\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}
\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}}.
$$

It ranges from -1 to 1. It is sensitive to outliers and only captures linear relationships.

**Spearman correlation** is Pearson correlation applied to ranks. It captures monotone association and is more robust to extreme values.

**Cramér's V** measures association between two categorical variables:

$$
V=\sqrt{\frac{\chi^2}{n\min(r-1,c-1)}},
$$

where $\chi^2$ is the chi-square statistic and $r,c$ are the numbers of rows and columns in the contingency table. It ranges from 0 to 1, but its interpretation depends on the table dimensions.

**Phik correlation** is designed for mixed-type data. It is based on the chi-square statistic over discretized intervals and can handle numeric, categorical, and mixed pairs. It is not on the same scale as Pearson $r$, so do not compare the two directly.

The profiler may show more than one matrix. Choose the measure that matches the variable types and the question. A high Phik value between `Sex` and `Survived` is an association, not a causal effect.

### 10.6 Why the report's memory size differs between versions

The old report says 83.7 KiB; the new report says 285.6 KiB. The data is the same. The difference comes from type inference and object representation:

- the old report treats `Name`, `Ticket`, and `Cabin` as categorical objects,
- the new report treats them as text with additional metadata,
- pandas versions and internal dtypes differ,
- the profiler may store additional summaries.

Memory size is a useful rough indicator, but it is not a data-quality metric and it is not comparable across library versions.

---

## 11. Workflow connections and leakage

### 11.1 Where automated EDA belongs

Automated EDA belongs near the start of the workflow:

```text
raw data
   |
   v
data understanding  <-- automated profiling
   |
   v
cleaning and feature engineering
   |
   v
train/test split
   |
   v
preprocessing fit on training data
   |
   v
model training and validation
```

The report is a description of the raw data. It should be used to form hypotheses, not to make final modeling decisions on the test set.

### 11.2 Train/test discipline

Profiling the training set is normal. Profiling the test set can be useful for checking distribution shift, but it must not drive feature choices or threshold tuning. If you choose a cap, an imputation value, or a feature based on the test set, you have leaked information.

The correct pattern is:

1. split the data,
2. profile the training set,
3. build preprocessing on the training set,
4. apply the fitted preprocessing to validation and test,
5. profile the test set only for monitoring, not for tuning.

### 11.3 Target leakage

A report can reveal suspiciously high associations with the target. Ask:

- Is the feature available at prediction time?
- Was it recorded after the outcome?
- Does it encode the target indirectly?
- Is it an identifier that is unique to each row?

`PassengerId` and `Name` are identifiers, not predictive features. A high-cardinality ID can memorize rows and produce misleading training performance. The report's uniqueness alerts are useful here.

### 11.4 Privacy

The report contains sample values. The Titanic data includes passenger names, ticket numbers, and cabin numbers. A profiling report can expose personally identifiable information. Before sharing a report:

- remove or mask sensitive columns,
- avoid sharing raw sample rows,
- check the organization's data-handling policy,
- use synthetic or aggregated examples when possible.

### 11.5 Reproducibility

Record:

- the dataset version or hash,
- the library version,
- the configuration,
- the random seed if sampling is used,
- the report generation time,
- the code that produced it.

The report's reproduction section helps, but it is not a substitute for version-controlled code and data.

---

## 12. Common mistakes

1. **Treating alerts as errors.**  
   Many alerts are expected: unique IDs, zeros in count columns, high correlation with the target.

2. **Using the report as proof of causation.**  
   Correlation and association are not causation.

3. **Profiling the test set and tuning on it.**  
   This leaks information and inflates performance.

4. **Ignoring type inference.**  
   `Pclass` is stored as an integer but is categorical. `PassengerId` is numeric but is an identifier.

5. **Assuming the report is complete after a kernel crash.**  
   Check the file, its size, and its metadata. Re-run if necessary.

6. **Writing both reports to the same filename.**  
   The last writer wins. Use distinct names.

7. **Installing packages inside notebooks without checking the kernel.**  
   `!pip` and `%pip` can target different environments.

8. **Quoting the report's memory size or duration as if it were a data property.**  
   These values depend on the library version, configuration, and hardware.

9. **Treating missing values as a single problem.**  
   `Cabin` missingness, `Age` missingness, and `Embarked` missingness have different causes and different consequences.

10. **Treating zeros as missing.**  
    `SibSp=0` and `Parch=0` are meaningful counts. `Fare=0` may be a special booking.

11. **Using a high-cardinality text column directly.**  
    `Name`, `Ticket`, and `Cabin` need feature engineering or exclusion.

12. **Sharing the HTML without checking for PII.**  
    The report includes raw sample values.

---

## 13. Interview essentials

### Level 1 - definitions

**Q1. What is automated EDA?**  
The use of a library to compute and render a broad set of descriptive statistics, distributions, missingness checks, correlations, and data-quality alerts in one report.

**Q2. What is the difference between ydata-profiling and Sweetviz?**  
Both generate HTML EDA reports. ydata-profiling emphasizes detailed per-column statistics and multiple correlation measures; Sweetviz emphasizes visual associations and target comparison.

**Q3. Does profiling train a model?**  
No. It describes the data. It does not fit a predictive model or validate feature usefulness.

### Level 2 - mechanics

**Q4. What does `ProfileReport(df)` do?**  
It creates a report object that summarizes the DataFrame. `to_file` renders it to HTML.

**Q5. What does `explorative=True` change?**  
It requests a richer report with more interaction and correlation analysis, usually at higher cost.

**Q6. What does `sv.analyze(df)` do?**  
It builds a Sweetviz report object; `show_html` writes the HTML.

**Q7. Why do the two notebooks overwrite each other?**  
Both write `titanic_report.html`. The last writer wins. Use distinct filenames.

### Level 3 - mathematics

**Q8. How is the missing-cell percentage computed?**  
Total missing cells divided by total cells: $866/(891\cdot12)\approx8.1\%$.

**Q9. Why can the report's distinct percentage be misleading?**  
It may use the non-missing count as the denominator. For `Cabin`, 147/204 is 72.1%, but 147/891 is only 16.5%.

**Q10. What is the difference between Pearson and Spearman correlation?**  
Pearson measures linear association on the original values; Spearman measures monotone association on ranks and is more robust to outliers.

### Level 4 - diagnosis

**Q11. The report says `PassengerId` is unique and uniformly distributed. Is that a problem?**  
It is expected for an identifier. It means `PassengerId` should not be used as a predictive feature; it is not a data-quality error.

**Q12. The report says `Sex` is highly correlated with `Survived`. Is that leakage?**  
Not necessarily. `Sex` is a legitimate passenger attribute available at prediction time. It is a strong association, not proof of leakage.

**Q13. `Cabin` is 77.1% missing. What are the options?**  
Drop it, create a missing indicator, extract deck/letter information, or treat missingness as a category. The choice depends on validation and domain knowledge.

**Q14. `Age` missingness differs by class. Why does that matter?**  
It suggests the data is not missing completely at random. A global mean imputation can distort class-specific age distributions. A group-aware or model-based imputation may be better.

### Level 5 - design

**Q15. How would you use a profiling report in a real project?**  
Profile the training data, record data-quality issues, form hypotheses, verify key findings manually, then build leakage-safe preprocessing and compare models on validation data.

**Q16. How would you compare ydata-profiling and Sweetviz fairly?**  
Use the same data snapshot, the same target, distinct output names, and a checklist of sections. Compare usefulness for the decisions you need to make, not just file size.

**Q17. What would you check before sharing a report?**  
PII, sample rows, sensitive columns, file size, version metadata, and whether the report contains test-set information.

---

## 14. Exam practice

### Short answers

1. What is the missing-cell rate for 866 missing cells in an 891 x 12 table?
2. Why does the ydata report call `Pclass` categorical even though it is stored as an integer?
3. What is the difference between a numeric type and a semantic type?
4. Why are `SibSp=0` and `Parch=0` not missing values?
5. What does a high-cardinality alert mean?
6. Why is `PassengerId` not a useful feature?
7. What is the risk of writing both reports to the same path?
8. Why is the report's duration not a fair comparison between versions?
9. What is the difference between exact duplicate rows and near-duplicates?
10. Why should profiling the test set not drive feature selection?

### Worked questions

**Q1.** The report says 12 variables, 891 observations, 866 missing cells. Compute the missing-cell percentage.

**Answer.** Total cells $=891\times12=10{,}692$. $866/10{,}692\approx0.0810=8.1\%$.

**Q2.** `Cabin` has 687 missing values out of 891. What is the missing rate? How many non-missing values are there?

**Answer.** $687/891\approx0.7710=77.1\%$. Non-missing $=891-687=204$.

**Q3.** `Cabin` has 147 distinct values among 204 non-missing values. What is the distinct percentage relative to non-missing? Relative to all rows?

**Answer.** $147/204\approx72.1\%$; $147/891\approx16.5\%$. The report's 72.1% uses the non-missing denominator.

**Q4.** Why can a high correlation with the target be legitimate?

**Answer.** The feature may be a genuine predictor available at prediction time, such as `Sex` in the Titanic data. High correlation is a reason to investigate, not automatic proof of leakage.

**Q5.** Why is `Fare` skewed, and what does that imply for a report reader?

**Answer.** A few passengers paid very high fares, producing a long right tail. The mean is pulled above the median. The report's histogram and extreme values show this; a mean-only summary would hide it.

**Q6.** Why is the old report's "Name is uniformly distributed" warning misleading?

**Answer.** The warning is based on uniqueness/cardinality, not a formal uniform distribution test. Every name is unique, so the heuristic fires. It does not mean the names are uniformly distributed in a statistical sense.

**Q7.** What does a kernel crash after report generation imply?

**Answer.** The report may exist, but the run did not finish cleanly. Check the file and its metadata, and re-run in a clean kernel before trusting the result.

---

## 15. One-page revision

- Automated EDA summarizes a dataset quickly; it does not replace judgment.
- ydata-profiling: `ProfileReport(df, title=..., explorative=True)` then `to_file(...)`.
- Sweetviz: `sv.analyze(df)` then `show_html(...)`.
- The official notebook uses the legacy `pandas_profiling` package and writes `output.html`.
- The current report has 12 variables, 891 rows, 866 missing cells (8.1%), 0 duplicate rows, 5 numeric / 4 categorical / 3 text.
- Alerts are heuristics. `Sex`-`Survived` correlation is expected; `PassengerId` uniqueness is expected; zeros in `SibSp`/`Parch` are meaningful.
- `Age` is 19.9% missing and its missingness varies by class.
- `Cabin` is 77.1% missing and high-cardinality.
- `Fare` is strongly right-skewed; 15 zero fares are real records.
- `Name` and `Ticket` are high-cardinality text fields.
- The report contains PII; do not share it blindly.
- Both local notebooks write the same filename; use distinct names.
- Kernel crashes mean the run is not clean even if the HTML exists.
- Profile training data; do not tune on the test set.

---

# Part B - Advanced Reading and Reference

> Part B extends beyond the video and the basic report reading. It is intended for interviews, implementation, and deeper source analysis.

## B1. Report internals

### B1.1 Type inference

ydata-profiling uses a type-inference layer (historically the `visions` library) to classify columns. It distinguishes numeric, categorical, text, date, boolean, and other types. The classification affects:

- which statistics are computed,
- which plots are rendered,
- which alerts fire,
- how cardinality is reported.

Type inference is a convenience, not a guarantee. A numeric code such as `Pclass` can be treated as numeric even though it is categorical. A binary target can be treated as categorical. An identifier can be treated as numeric. Always check the type assignment against the meaning of the column.

### B1.2 Describe, summarize, render

The report pipeline has three broad stages:

1. **Describe:** compute per-column statistics.
2. **Summarize:** compute dataset-level summaries, correlations, interactions, and missingness.
3. **Render:** build the HTML structure and write the file.

The progress bars in the notebook reflect these stages. The exact stage names and counts differ between library versions.

### B1.3 Lazy evaluation and configuration

The report object may be lazy: some work happens at construction, and some at rendering. The configuration controls which sections are computed. A lighter configuration can be much faster on wide data.

Useful configuration ideas:

- disable interactions for a quick first pass,
- reduce the number of samples,
- disable expensive correlation measures,
- set a smaller HTML report,
- use a sample of rows for exploration, then profile the full data separately.

The exact parameter names depend on the version. Check the installed version's documentation before relying on a specific argument.

### B1.4 Report size

The HTML report embeds CSS, JavaScript, tables, and charts. A 4.7 MB report for 891 rows is normal. On a dataset with thousands of columns, the report can become very large. Strategies:

- profile a sample of rows,
- profile a subset of columns,
- disable interactions,
- use a lighter configuration,
- write JSON and render selectively.

The report size is not a measure of data quality.

---

## B2. Correlation mathematics

### B2.1 Pearson

Pearson $r$ measures linear association. It is the covariance divided by the product of standard deviations:

$$
r_{xy}=\frac{\operatorname{cov}(x,y)}{s_x s_y}.
$$

Properties:

- range $[-1,1]$,
- invariant to positive linear transformations,
- sensitive to outliers,
- zero does not imply independence, only no linear association,
- affected by range restriction.

### B2.2 Spearman

Spearman $\rho$ is Pearson correlation on ranks:

$$
\rho=1-\frac{6\sum d_i^2}{n(n^2-1)}
$$

for data without ties, where $d_i$ is the difference between the ranks. With ties, the general Pearson-on-ranks formula is used.

Properties:

- captures monotone relationships,
- more robust to extreme values,
- less sensitive to the exact spacing of values,
- still not a causal measure.

### B2.3 Cramér's V

For a contingency table with $r$ rows and $c$ columns:

$$
V=\sqrt{\frac{\chi^2}{n\min(r-1,c-1)}}.
$$

Properties:

- range $[0,1]$ for many tables,
- based on the chi-square statistic,
- depends on the table dimensions,
- not directly comparable across tables with different dimensions,
- sensitive to small expected counts.

### B2.4 Phik

Phik is a correlation coefficient for mixed-type data. It is based on the chi-square statistic computed over discretized intervals. It is useful when one variable is numeric and the other is categorical, or when both are categorical. It is not on the same scale as Pearson $r$. The report may show it alongside other measures; read the label.

### B2.5 Correlation is not causation

A correlation can arise from:

- a direct causal effect,
- reverse causation,
- a common cause (confounding),
- selection bias,
- a mathematical artifact,
- a spurious pattern in a small sample.

The Titanic `Sex`-`Survived` association is strong, but it is not a causal estimate. It reflects the historical evacuation policy and many other factors. A model can use the association for prediction without claiming causation.

---

## B3. Missingness and its structure

### B3.1 Missingness mechanisms

- **MCAR (missing completely at random):** the probability of missingness is independent of all variables.
- **MAR (missing at random):** the probability depends on observed variables but not on the missing value itself, given the observed data.
- **MNAR (missing not at random):** the probability depends on the unobserved value.

The `Age` missingness by class suggests it is not MCAR. A model-based imputation or a missing indicator may be more appropriate than a global mean.

### B3.2 Missingness as information

A missing value can be informative. For `Cabin`, the fact that a cabin was not recorded may correlate with class, fare, or survival. A missing indicator can preserve this information. The report's missingness tab helps identify the pattern; it does not decide the modeling strategy.

### B3.3 Co-missingness

The missingness matrix and heatmap show whether columns are missing together. If two columns are always missing in the same rows, they may represent the same underlying data-collection issue. If missingness clusters by group, a group-aware strategy may be needed.

---

## B4. Alerts and heuristics

### B4.1 What alerts are

Alerts are rule-based warnings generated from thresholds. They are not statistical tests with error rates. Examples:

- high correlation above a threshold,
- high cardinality above a threshold,
- missing rate above a threshold,
- zero rate above a threshold,
- unique rate above a threshold,
- constant or near-constant columns.

The thresholds can be configured. The alert set therefore depends on the configuration and the library version.

### B4.2 Why the alert sets differ

The old report has 14 warnings; the new report has 10 alerts. The difference is not a change in the Titanic data. It comes from:

- different type inference,
- different default thresholds,
- different alert definitions,
- different library versions.

This is a good example of why the report metadata matters. An alert is not a universal fact.

### B4.3 Reading alerts correctly

For each alert, ask:

1. Is the alert expected for this column?
2. Is it a data-quality issue or a semantic property?
3. Does it affect the modeling question?
4. What would happen if I ignored it?
5. What would happen if I treated it automatically?

An alert is a prompt for investigation, not an instruction.

---

## B5. Performance and scalability

### B5.1 Cost drivers

The cost of profiling grows with:

- number of rows,
- number of columns,
- number of distinct values,
- correlation and interaction work,
- text analysis,
- HTML rendering.

`explorative=True` increases the cost. Text columns can be expensive because the profiler computes word and character statistics. High-cardinality categorical columns can also be expensive.

### B5.2 Sampling

For a first pass on a very large dataset, profile a random or stratified sample. This gives a quick overview without the full cost. Then profile the full data if needed. Record the sample size and seed.

### B5.3 Column selection

Profile only the columns relevant to the current question. A 500-column report is hard to read and expensive to render. Split the audit into logical groups of columns.

### B5.4 Report size management

- use a lighter configuration,
- reduce samples,
- disable interactions,
- write JSON and query it,
- render only selected sections.

The report is a tool, not a deliverable that must contain everything.

---

## B6. Sweetviz internals and comparison

### B6.1 Sweetviz analysis

Sweetviz computes per-feature summaries and associations, then renders a dashboard. With `target_feature`, it adds target comparisons. It is designed for quick visual inspection.

### B6.2 Sweetviz output

`show_html` writes a self-contained HTML report and normally opens a browser. In a headless environment, set `open_browser=False` if the version supports it. The exact signature depends on the version.

### B6.3 Fair comparison

To compare ydata-profiling and Sweetviz:

1. use the same data snapshot,
2. use the same target,
3. write to distinct files,
4. compare the same questions: missingness, distributions, associations, outliers, cardinality,
5. record the library versions,
6. do not compare file size as a quality measure.

### B6.4 What the saved artifacts support

Because no Sweetviz artifact is present, this chapter cannot compare the rendered Sweetviz panels. It can only compare the code and the stated behavior. That limitation is documented rather than filled with invented observations.

---

## B7. Official legacy report details

### B7.1 Version

The official report is pandas-profiling v2.11.0, generated 2021-04-05. The package name later changed to ydata-profiling, and the current notebook warns about a further rename toward fg-data-profiling.

### B7.2 Type differences

The old report treats `Name`, `Ticket`, and `Cabin` as categorical. The new report treats them as text. This changes:

- the variable-type counts (5 numeric + 7 categorical versus 5 numeric + 4 categorical + 3 text),
- the warnings (high cardinality versus text summaries),
- the memory estimate,
- the report sections.

The underlying data is the same. The difference is the inference and the library version.

### B7.3 Warning interpretation

The old "uniformly distributed" warnings for `Name`, `Ticket`, and `Cabin` are heuristic artifacts of uniqueness/cardinality. They should not be read as formal distribution tests. The new report does not produce the same warnings because it classifies those columns as text.

### B7.4 Reproducibility

The old report is historically useful, but it is not the current API. To reproduce it exactly, you would need the old package, the old Python environment, and the old configuration. For learning, use the current package and document the version difference.

---

## B8. Code patterns

### B8.1 ydata-profiling with distinct output names

```python
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv("datasets/train.csv")

profile = ProfileReport(
    df,
    title="Titanic EDA Report",
    explorative=True,
)
profile.to_file("titanic_report_ydata.html")
```

### B8.2 Sweetviz with a target and no browser

```python
import pandas as pd
import sweetviz as sv

df = pd.read_csv("datasets/train.csv")

report = sv.analyze(df, target_feature="Survived")
report.show_html("titanic_report_sweetviz.html", open_browser=False)
```

Check the installed Sweetviz version for the exact `show_html` arguments.

### B8.3 Profile a training split only

```python
from sklearn.model_selection import train_test_split
import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv("datasets/train.csv")
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Survived"],
)

train_profile = ProfileReport(
    train_df,
    title="Titanic Training EDA",
    explorative=True,
)
train_profile.to_file("titanic_train_report.html")
```

This pattern prevents the test set from influencing exploratory choices. Profiling the test set separately is acceptable for monitoring, but not for tuning.

### B8.4 Minimal versus explorative

```python
from ydata_profiling import ProfileReport

minimal = ProfileReport(df, title="Minimal", minimal=True)
minimal.to_file("titanic_minimal.html")

explorative = ProfileReport(df, title="Explorative", explorative=True)
explorative.to_file("titanic_explorative.html")
```

Compare file size, runtime, and which sections change. Do not overwrite the original report.

### B8.5 A simple post-profile check

```python
import pandas as pd

df = pd.read_csv("datasets/train.csv")

print(df.shape)
print(df.isna().sum().sort_values(ascending=False))
print(df.duplicated().sum())
print(df.dtypes)
```

The report should agree with these direct checks. If it does not, investigate the type inference or the report configuration.

---

## B9. Leakage, privacy, and governance

### B9.1 Leakage

Automated EDA is descriptive, but it can still cause leakage if:

- it is run on the test set and used to choose features,
- it reveals target-related information that is not available at prediction time,
- it leads to imputation or capping parameters learned from the full data,
- it encourages dropping rows based on test outcomes.

### B9.2 Privacy

Reports can contain:

- names,
- addresses,
- ticket or account numbers,
- free-text fields,
- sample rows with sensitive values.

Before sharing:

- remove or mask sensitive columns,
- use a report configuration that hides samples if available,
- share aggregate findings instead of raw HTML,
- follow the organization's data policy.

### B9.3 Governance

For a team, define:

- who may generate reports,
- where reports may be stored,
- how long they are retained,
- which datasets may be profiled,
- how PII is handled,
- how report versions are linked to data versions.

A report is a data artifact and should be governed like one.

---

## B10. Extended interview bank

### Level 1 - definitions

1. What is automated EDA?
2. What is a profiling report?
3. What is the difference between a report and a model?
4. What is a high-cardinality column?
5. What is a missing indicator?
6. What is the difference between a numeric type and a semantic type?
7. What is an identifier column?
8. What is a duplicate row?
9. What is a correlation matrix?
10. What is a self-contained HTML report?

### Level 2 - mechanics

11. What does `ProfileReport` do?
12. What does `to_file` do?
13. What does `sv.analyze` do?
14. What does `show_html` do?
15. What is the difference between `!pip` and `%pip`?
16. Why do both notebooks write the same file?
17. What is `explorative=True`?
18. What is `minimal=True`?
19. How does the profiler infer types?
20. How does the profiler compute missing percentages?

### Level 3 - mathematics

21. Write the formula for the missing-cell rate.
22. Write the formula for Pearson correlation.
23. Write the formula for Spearman correlation.
24. Write the formula for Cramér's V.
25. What is the difference between Pearson and Phik?
26. Why can the mean be misleading for `Fare`?
27. What does skewness measure?
28. What does kurtosis measure?
29. Why is the distinct percentage denominator important?
30. Why is the memory estimate version-dependent?

### Level 4 - diagnosis

31. The report says `PassengerId` is unique. What do you do?
32. The report says `Sex` is highly correlated with `Survived`. Is that leakage?
33. `Cabin` is 77.1% missing. What are the options?
34. `Age` missingness differs by class. What does that imply?
35. `Fare` has 15 zeros. Are they missing?
36. `SibSp` and `Parch` have many zeros. Are they missing?
37. The report has 10 alerts, but the old report has 14. Why?
38. The kernel crashed after 100%. Is the report trustworthy?
39. No Sweetviz HTML exists. What can you conclude?
40. The report's duration is 1.87 seconds. Can you compare it with the old 8.02 seconds?

### Level 5 - design

41. How would you use automated EDA in a real project?
42. How would you avoid leakage?
43. How would you compare two profiling libraries?
44. How would you handle PII in a report?
45. How would you profile a very large dataset?
46. How would you choose between ydata-profiling and Sweetviz?
47. How would you document a report for reproducibility?
48. How would you integrate profiling into a team workflow?

### Level 6 - advanced

49. How does type inference affect the alert set?
50. What are the limitations of correlation matrices for mixed data?
51. What is the difference between exact duplicates and near-duplicates?
52. How can a report reveal target leakage?
53. What are the risks of profiling the test set?
54. How does sampling affect the report's conclusions?
55. What are the failure modes of a self-contained HTML report?
56. How would you test a profiling pipeline?

### Level 7 - synthesis

57. A stakeholder wants to drop every column with a missing value after seeing the report. How do you respond?
58. A model improves after removing `Cabin`, but the report says it is 77.1% missing. What else should you check?
59. A report shows a high correlation between a feature and the target. How do you decide whether to keep the feature?
60. A report is 200 MB and takes 20 minutes. How would you redesign the profiling workflow?
61. A team shares reports containing customer names. What governance would you propose?

---

## B11. Extended examination questions

### Question 1

The report says 12 variables, 891 observations, 866 missing cells, and 0 duplicate rows. Compute the missing-cell percentage and explain why the duplicate count does not detect near-duplicates.

**Answer.** $866/(891\cdot12)=8.1\%$. The duplicate count checks exact row equality. Two records for the same person with different IDs or slightly different spellings are not exact duplicates, so entity resolution may still be needed.

### Question 2

`Cabin` is 77.1% missing and has 147 distinct values among 204 non-missing rows. Explain why the report might say "Distinct 72.1%" while a naive calculation gives 16.5%.

**Answer.** The report uses the non-missing count as the denominator for the distinct percentage: $147/204=72.1\%$. The naive calculation uses all rows: $147/891=16.5\%$. Both are correct for different questions.

### Question 3

Why does the old report have 14 warnings while the new report has 10 alerts?

**Answer.** The libraries use different type inference, thresholds, and alert definitions. The old report treats `Name`, `Ticket`, and `Cabin` as categorical and warns about high cardinality; the new report treats them as text. The data is the same; the metadata and rules differ.

### Question 4

A `Fare` value of 512.3292 is far above the rest. Should it be removed?

**Answer.** Not automatically. It may be a genuine first-class fare. Check the passenger's class, ticket, and survival; compare models with and without the value; consider a log transform or a robust model. Removing it changes the population.

### Question 5

Why is `PassengerId` not a useful predictor even though it is numeric?

**Answer.** It is an identifier assigned to each row. It has no meaningful numeric relationship with survival. A model can memorize it, but it will not generalize. It should be excluded from features.

### Question 6

What is the danger of profiling the test set before choosing preprocessing?

**Answer.** The test distribution can influence the choice of features, caps, imputation values, or thresholds. The test set then stops being an unbiased estimate of unseen performance. Profile the training set for decisions; profile the test set only for monitoring.

### Question 7

Why can a kernel crash after export still leave a usable HTML file?

**Answer.** The file may have been written before the crash. The crash could occur during cleanup, rendering, or a later step. The file should be opened and checked; the run should still be repeated in a clean kernel before being trusted.

### Question 8

What is the difference between a high-cardinality categorical column and a text column?

**Answer.** A high-cardinality categorical column has many repeated categories; a text column contains free-form strings. The distinction affects encoding, feature extraction, and the profiler's type-specific statistics. `Name` is text; a `Country` column with 200 categories is high-cardinality categorical.

### Question 9

Why is `Sex`-`Survived` correlation not proof of causation?

**Answer.** The association may reflect evacuation policies, class, age, and other factors. Correlation measures association, not the effect of changing `Sex`. Causal claims require a causal design and assumptions.

### Question 10

How would you make the report reproducible?

**Answer.** Record the data hash, library version, configuration, random seed, code, and generation time. Store the report with the code and data version. Re-run in a clean environment to verify.

---

## B12. Coverage and source fidelity

### B12.1 Coverage

This chapter covers:

- the ydata-profiling personal notebook,
- the Sweetviz personal notebook,
- the empty placeholder,
- the official legacy notebook,
- the current ydata report,
- the official pandas-profiling report,
- the root ydata report,
- the dataset facts,
- the report sections and alerts,
- the mathematical foundations,
- the workflow, leakage, privacy, and reproducibility issues,
- interview and examination preparation.

### B12.2 Source-fidelity issues

- The old notes referenced `DAY22_Pandas-Profiling-ydata.ipynb`, which is not present. The actual ydata notebook is `my_notebook.ipynb`.
- The old notes referenced `topic_notes.md`, `eda_utils.py`, `eda_report.html`, and `eda_plots/`; these are not part of the current Day 22 folder.
- The current `titanic_report.html` is a ydata report, not a Sweetviz report.
- No Sweetviz HTML artifact is present.
- The report timestamp predates the notebook timestamp, so the exact saved run is uncertain.
- Both local notebooks show kernel crashes after report generation.
- The official notebook is legacy and uses `pandas_profiling` 2.11.0, not the current `ydata_profiling` API.
- The old and new reports have different type counts, memory estimates, durations, and alert sets because of version and configuration differences.
- The report contains PII and should not be shared without review.

### B12.3 What is not claimed

- No Sweetviz report content is quoted.
- No target-comparison Sweetviz result is claimed.
- No kernel-crash cause is asserted.
- No causal effect is claimed from the correlations.
- No exact reproduction of the old report is claimed.

---

## B13. References

- CampusX 100 Days of Machine Learning, Day 22: Pandas Profiling.
- ydata-profiling documentation: <https://docs.profiling.ydata.ai/>.
- Sweetviz documentation: <https://github.com/fbdesignpro/sweetviz>.
- pandas documentation: <https://pandas.pydata.org/docs/>.
- scikit-learn documentation: <https://scikit-learn.org/stable/>.
- Titanic dataset: the course `train.csv`; the Day 22 copy is byte-identical to the vendor copy.
- Official pandas-profiling report artifact: `vendor/100-days-of-machine-learning/day22-pandas-profiling/output.html`.
