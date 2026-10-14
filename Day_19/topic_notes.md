# Understanding Your Data — Study Notes (Day 19)

> Reference: `day19-understanding-your-data-descriptive-stats` in the CampusX repo. Your work: `DAY19_Understanding-Your-Data.ipynb` on Titanic (`E:/MACHINE_LEARNING/train.csv`, 891 rows × 12 columns).

## 1. Basic Idea

Before any cleaning or modeling, interrogate a new table with seven quick checks: how big it is, what it looks like, column types, missing values, numeric summaries, duplicates, and correlations.

## 2. Intuition and Mental Model

A doctor's first appointment: measure (shape), glance (sample), blood-type check (dtypes), find symptoms (missing), vitals (describe), rule out chart mix-ups (duplicates), and see which symptoms move together (correlation). Treatment (Days 23–29) starts only after diagnosis.

## 3. Why Do We Need It?

Every preprocessing choice depends on this diagnosis: Cabin (687 missing) must be dropped or engineered, Age (177 missing) imputed, Embarked (2 missing) mode-filled. Skipping inspection means imputing blindly and encoding wrongly.

## 4. Detailed Explanation

- **Size — `df.shape`.** Tuple metadata `(891, 12)`: 891 passengers to learn from, 12 columns to consider. No scan needed.
- **Glimpse — `head` vs `sample(5)`.** `head` shows the top rows (possibly one sorted segment); `sample(5)` shows random rows and reveals variety `head` hides.
- **Types — `df.info()`.** Per-column dtype plus non-null counts: Age 714, Cabin 204, Embarked 889 present. IDs stored as numbers (`PassengerId`) are still not predictive numbers.
- **Missing — `df.isnull().sum()`.** Boolean mask summed per column: Age 177, Cabin 687, Embarked 2. Percentages decide treatment (Cabin ~77% → drop/engineer).
- **Math — `df.describe()`.** Count/mean/std/min/quartiles/max per numeric column — the fingerprint used later to spot skew and scale needs.
- **Duplicates — `df.duplicated().sum()`.** Hashed-row equality: 0 here, so no exact-duplicate rows inflate counts.
- **Links — `df.corr(numeric_only=True)` + `sns.heatmap(annot=True)`.** Pearson r (−1…1) per numeric pair; annotations give exact numbers behind the colors.

## 5. Examples

```python
df.shape            # (891, 12)
df.sample(5)        # random glimpse
df.info()           # dtypes + non-null counts
df.isnull().sum()   # Age 177, Cabin 687, Embarked 2
df.describe()       # numeric fingerprint
df.duplicated().sum()  # 0
sns.heatmap(df.corr(numeric_only=True), annot=True)
```

## 6. Important Formulas

Correlation `r` ranges −1 (perfect opposite) to +1 (perfect together); near 0 means no *linear* link (non-linear links can still exist).

## 7. Code Connection

All seven are one-liners on the DataFrame — `shape/info/isnull/describe/duplicated/corr` plus one seaborn call. `numeric_only=True` keeps string columns out of `corr`.

## 8. Interpretation

- Cabin mostly missing → drop or engineer (e.g. deck letter), never blind-impute 77%.
- Correlation near 0 = weak linear relation — read the annotated numbers, not just colors; correlation is not causation.
- 0 duplicates = counts (e.g. survival rates) are not inflated by repeated rows.

## 9. Common Mistakes

1. Stopping after `head` — ordered files hide variety.
2. Treating ID columns as predictive numbers.
3. Reading heatmap colors without the annotated values.
4. Confusing "no correlation" with "no relationship" (non-linearity survives).
5. Jumping to modeling before the missing-value table exists.

## 10. Machine Learning Connection

Day 19 output is the work order for Days 23–29: fill Age, handle Cabin, encode Sex/Embarked/Pclass, scale Fare/Age. Models never see this step, but every model depends on it.

## 11. Interview and Conceptual Questions

1. What does each check (`shape/info/isnull/describe/duplicated/corr`) tell you? — Size, types+gaps, exact missing counts, numeric fingerprint, repeated rows, linear links.
2. Cabin is 77% missing — treatment? — Drop, or engineer a simpler signal (has-cabin/deck); imputing is noise.
3. Why `sample` alongside `head`? — Random rows expose variety and ordering artifacts.
4. Correlation 0.05 — conclusion? — Essentially no linear relationship.

## 12. Quick Revision

Size → glimpse → types → missing → math → duplicates → correlations. Titanic: 891×12, Age/Cabin/Embarked gaps, 0 duplicates, weak numeric links. Diagnose first, treat later.
