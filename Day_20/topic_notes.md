# Univariate Analysis — Study Notes (Day 20)

> Reference: `day20-univariate-analysis` in the CampusX repo. Your work: `DAY20_Univariate-Analysis.ipynb` (Titanic counts/pie/hist/KDE/box) + `Matplotlib-Seaborn-Cheat-Sheet.ipynb` (self-made plot reference on synthetic data — not from the course, kept as your companion).

## 1. Basic Idea

Univariate analysis studies **one column at a time**: which values occur and how often. Categories get counts (bars) and shares (pie); numbers get distributions (histogram, KDE) and spread summaries (boxplot).

## 2. Intuition and Mental Model

One musician auditioning alone: for categories, count how many play each instrument; for numbers, hear the range — where the notes cluster (peak), how far they stretch (spread), whether high notes screech off (skew), and who plays absurdly loud (outliers).

## 3. Why Do We Need It?

Single-column shapes drive treatment: Fare's right skew argues for log/robust scaling, Pclass sizes affect encoding choices, Age gaps confirm imputation needs. You cannot fix a distribution you have never looked at.

## 4. Detailed Explanation

- **Counts — `sns.countplot`.** Bar per level (Pclass 1/2/3 sizes). Prefer `x="Pclass", data=df` form for clean labels.
- **Shares — `value_counts().plot.pie`.** Dead/alive proportions at a glance.
- **Shape — `hist(bins=...)`.** Values binned into ranges; ~20–40 bins read well. Your `bins=300` is over-binned noise — a deliberate lesson in the notebook.
- **Smooth shape — KDE (`sns.histplot(..., kde=True)`).** A smoothed density over the histogram. Note: `sns.distplot` used in the notebook is deprecated/removed — `histplot`/`kdeplot` replace it.
- **Spread + outliers — `sns.boxplot`.** Box = middle 50% (Q1–Q3), line = median, whiskers = Q1−1.5·IQR / Q3+1.5·IQR, dots beyond = outlier candidates (your attached diagram). Fare stretches far right: right-skewed with genuine high-fare outliers.
- **Anchors — `min`/`max`.** `Age.min()` grounds the scale the plots suggest.

## 5. Examples

```python
sns.countplot(x="Pclass", data=df)
df["Survived"].value_counts().plot.pie(autopct="%1.1f%%")
df["Age"].hist(bins=30)
sns.histplot(df["Age"], kde=True)   # modern distplot replacement
sns.boxplot(x=df["Fare"])
```

## 6. Important Formulas

IQR = Q3 − Q1. Lower fence = Q1 − 1.5·IQR; upper fence = Q3 + 1.5·IQR. Points outside are outlier *candidates* (domain decides).

## 7. Code Connection

Seaborn picks the statistic, Matplotlib draws it: `countplot`/`histplot`/`kdeplot`/`boxplot` each take a column and return the shape. The cheat-sheet notebook maps every data-type pair to its plot — use its *Which Plot* table before guessing.

## 8. Interpretation

- Tall bars = common levels; thin pie slices = rare outcomes (watch for imbalance).
- Symmetric hump = well-behaved; long tail = skew needing transforms later.
- Box dots = candidates, not verdicts — a first-class fare of 500 is real, not an error.

## 9. Common Mistakes

1. `bins=300` noise; bare `plt.hist` (no call) draws nothing.
2. `sns.distplot` no longer exists — use `histplot`/`kdeplot`.
3. Reading only means (bars) without spread (boxes), or vice versa.
4. Deleting every boxplot dot as "error" — outliers can be the signal.
5. Treating the cheat sheet's synthetic shapes as Titanic facts.

## 10. Machine Learning Connection

Univariate shapes choose treatments: skew → log/RobustScaler (Days 24–25), levels → encoding strategy (Days 26–27), gaps → imputation. Features are screened here, engineered later.

## 11. Interview and Conceptual Questions

1. Categorical vs numerical — which plots? — Counts/pie vs hist/KDE/box.
2. Why is `bins=300` bad? — Too fine: spiky noise hides peak/skew.
3. Box anatomy? — Q1–Q3 box, median line, 1.5·IQR whiskers, outlier dots.
4. `distplot` replacement? — `sns.histplot(..., kde=True)` or `kdeplot`.
5. Right-skewed Fare — modeling consequence? — Consider log transform or robust scaling.

## 12. Quick Revision

One column: categories → count/pie; numbers → hist+KDE/box. IQR fences flag candidates; bins ~30; `distplot` is dead. Shape first, treatment second.
