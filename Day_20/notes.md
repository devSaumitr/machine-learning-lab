# Day 20 - Univariate Analysis

> CampusX 100 Days of Machine Learning, Day 20.
> One unified chapter covering the official `day20.ipynb`, the personal `my_notebook.ipynb`, and the separate `Matplotlib-Seaborn-Cheat-Sheet.ipynb`.

### How to read this chapter

**Part A** teaches univariate analysis: how to inspect one column at a time, choose a plot for its type, read the result, and connect the finding to later preprocessing. It is sufficient to understand and implement the day without reading Part B.

**Part B** is optional advanced reading: histogram and KDE mathematics, plotting internals, a cell-level source audit, the cheat-sheet inventory, extended comparisons, an extended interview bank, and extra examination practice.

The day is still exploratory. There is no model training, no train/test split, and no cleaning performed in the official notebook. The goal is to turn the Day 19 numbers into pictures and decisions.

---

# Part A - Main Learning Chapter

## 1. The problem: a table is not yet a set of understood variables

Day 19 answered structural questions: how many rows, which columns are numeric, what is missing, and which columns correlate. Those checks produce numbers, but a table of statistics can hide the shape of a variable.

Consider Fare in the Titanic data:

- mean Fare is 32.2042;
- median Fare is 14.4542;
- maximum Fare is 512.3292;
- minimum Fare is 0.

The mean is more than twice the median. A reader who looks only at the mean might imagine a typical fare near 32. A histogram or boxplot shows why that picture is misleading: most passengers paid much less, while a small number paid very large fares. The distribution has a long right tail.

Univariate analysis studies **one variable at a time**. For each column, it asks:

1. What type of variable is this?
2. Which values occur, and how often?
3. Where is the center?
4. How spread out are the values?
5. Is the distribution symmetric, skewed, or multi-peaked?
6. Are there rare categories, impossible values, or outlier candidates?
7. What treatment might this variable need later?

The official notebook performs this in two blocks:

```text
Categorical data
  countplot(Embarked)
  pie chart(Sex)

Numerical data
  histogram(Age, bins=5)
  distplot(Age)
  boxplot(Age)
  min / max / mean / skew of Age
```

The personal notebook follows the same idea with different columns and settings:

```text
Categorical data
  countplot(Pclass)
  pie chart(Survived)

Numerical data
  histogram(Age, bins=300)
  distplot(Age)
  boxplot(Fare)
  Age.min()
```

The differences are not cosmetic. They change which question each plot answers. A countplot of Embarked describes a feature; a pie chart of Survived describes the target. A histogram of Age with five bins shows a coarse shape; a histogram with 300 bins shows noise. A boxplot of Age and a boxplot of Fare describe different variables with different scales and different outlier behavior.

### 1.1 Why univariate analysis comes before bivariate analysis

A relationship between two variables is only interpretable after each variable is understood separately.

If Fare is heavily skewed and Age is mildly skewed, a scatter plot of Age versus Fare can be dominated by a few high-fare points. If Survived is imbalanced, a count of survivors can look large while the survival rate is still low. If Embarked has a missing category, a groupby by Embarked can silently drop two rows.

Univariate analysis prevents these mistakes by establishing:

- the scale of each numeric variable;
- the frequency of each category;
- the amount and location of missingness;
- the shape and tail behavior;
- the candidate outliers;
- the target balance.

Day 21 then adds a second variable. Day 20 is the prerequisite, not a detour.

### 1.2 What goes wrong when this step is skipped

| Skipped check | Consequence |
|---|---|
| Category counts | rare levels are treated as representative; encoders create unstable columns |
| Numeric shape | a skewed feature is scaled or modeled as if it were symmetric |
| Outlier inspection | genuine high fares are deleted, or extreme values distort a model |
| Missingness | `NaN` is plotted as zero or silently dropped without explanation |
| Target balance | accuracy looks acceptable while the minority class is ignored |
| Plot choice | a countplot on a continuous variable creates hundreds of useless bars |

Univariate analysis is not decoration. It is the evidence behind preprocessing decisions.

### 1.3 The workflow position

```text
raw CSV
  |
  v
Day 19: structure, types, missing counts, duplicates, correlations
  |
  v
Day 20: one-column distributions and frequencies
  |
  v
Day 21: relationships between two or more columns
  |
  v
cleaning, imputation, encoding, scaling, transformation
  |
  v
train/test split and modeling
```

It is acceptable to inspect the full dataset for exploratory understanding. It is not acceptable to learn imputation values, scaling parameters, or feature-selection rules from the test set later. Day 20 does not fit a model, but it informs decisions that must eventually be fitted on training data only.

---

## 2. Foundations: variables, distributions, and plots

### 2.1 Variable types

A variable's type determines which summary and which plot are appropriate.

| Type | Titanic examples | Useful univariate tools |
|---|---|---|
| nominal categorical | `Sex`, `Embarked` | counts, proportions, countplot, bar chart |
| ordinal categorical | `Pclass` | counts, ordered bars, median category |
| binary target | `Survived` | counts, proportion, pie or bar |
| discrete numeric | `SibSp`, `Parch` | counts, bar chart, histogram with integer bins |
| continuous numeric | `Age`, `Fare` | histogram, KDE, boxplot, summary statistics |
| identifier | `PassengerId` | uniqueness check, not a distribution to interpret |
| high-cardinality text | `Name`, `Ticket`, `Cabin` | uniqueness, prefix/title extraction, not a simple pie |

`Pclass` is stored as an integer, but it is an ordinal category: 1st, 2nd, and 3rd class. A mean of `Pclass` (2.3086) is arithmetically computable but not a meaningful "average class" in the same way that mean Age is meaningful. The countplot is usually more informative than a histogram for `Pclass`.

`Survived` is stored as 0/1. It is a binary categorical target, not a continuous measurement. Its mean is a proportion, not a rating.

`PassengerId` is numeric but is an identifier. Plotting its distribution answers no useful modeling question.

### 2.2 What a distribution is

A distribution describes which values a variable takes and how often.

For a categorical variable, the distribution is a frequency table:

| Embarked | Count | Share of all 891 rows |
|---|---:|---:|
| S | 644 | 72.28% |
| C | 168 | 18.86% |
| Q | 77 | 8.64% |
| missing | 2 | 0.22% |

For a numeric variable, the distribution is the pattern of values across their range. It can be summarized by center, spread, shape, and unusual points, and displayed by a histogram, KDE, or boxplot.

The observed distribution is **empirical**: it describes this dataset. It is not proof of the population distribution. A histogram of 891 passengers does not tell us the exact age distribution of all Titanic passengers or of future passengers.

### 2.3 Center, spread, and shape

These are different questions.

**Center** asks where the typical value lies:

- mean: arithmetic average;
- median: 50th percentile;
- mode: most frequent value or category.

**Spread** asks how far values extend:

- range: maximum minus minimum;
- variance and standard deviation: squared and root-mean-squared deviation from the mean;
- IQR: distance between the 25th and 75th percentiles.

**Shape** asks how the values are arranged:

- symmetric or skewed;
- one peak or several peaks;
- light or heavy tails;
- bounded or unbounded;
- outliers or no outliers.

A mean without spread can be misleading. A spread without shape can be misleading. A shape without counts can hide a tiny sample. Use numbers and pictures together.

### 2.4 Matplotlib, pandas plotting, and Seaborn

The notebook uses three layers.

**Matplotlib** is the low-level plotting library. It provides figures, axes, lines, patches, ticks, labels, and legends.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.hist(df["Age"].dropna(), bins=5)
ax.set_xlabel("Age")
ax.set_ylabel("Count")
plt.show()
```

**pandas plotting** is a convenience wrapper around Matplotlib:

```python
df["Sex"].value_counts().plot(kind="pie", autopct="%.2f")
```

The Series or DataFrame chooses the data; pandas creates a Matplotlib axes and draws the plot.

**Seaborn** builds on Matplotlib and provides statistical plot types with convenient defaults:

```python
import seaborn as sns

sns.countplot(data=df, x="Embarked")
sns.histplot(data=df, x="Age", bins=5, kde=True)
sns.boxplot(data=df, x="Fare")
```

Seaborn is not a replacement for Matplotlib. It usually creates a Matplotlib axes and draws onto it. You can still use Matplotlib to set titles, labels, limits, and layouts.

| Layer | Strength | Typical use |
|---|---|---|
| Matplotlib | full control, explicit artists | custom plots, subplots, annotations |
| pandas `.plot` | shortest syntax for common plots | quick Series/DataFrame plots |
| Seaborn | statistical defaults, DataFrame-aware API | distributions, categorical comparisons, heatmaps |

### 2.5 The grammar of a plot

Every plot in this day can be read with the same questions:

1. **Data:** which column or columns are being plotted?
2. **Mapping:** which variable is on the x-axis, which on the y-axis, which is color or hue?
3. **Geometry:** bars, boxes, points, lines, or areas?
4. **Statistic:** raw counts, proportions, density, median, quartiles, or a fitted curve?
5. **Scale:** linear, log, or a fixed range?
6. **Labels:** do the axis titles and legend explain the encoding?
7. **Conclusion:** what does the pattern imply, and what does it not imply?

A countplot and a bar chart can look similar but answer different questions. A countplot counts rows per category. A bar chart can display any already-computed number per category, such as a mean, a rate, or a total.

---

## 3. Categorical univariate analysis

### 3.1 Counts and proportions

The basic categorical summary is a frequency table.

```python
df["Embarked"].value_counts()
df["Embarked"].value_counts(normalize=True)
df["Embarked"].value_counts(dropna=False)
```

- `value_counts()` counts occurrences and sorts by descending count by default.
- `normalize=True` returns proportions rather than counts.
- `dropna=False` includes missing values as a separate row.

For Embarked:

```text
S    644
C    168
Q     77
NaN    2
```

The countplot drawn by the official notebook shows three bars because Seaborn drops missing values by default. The two missing rows are absent from the picture unless they are explicitly handled.

The share of all 891 rows is:

$$
\text{share}_S=\frac{644}{891}\times100\%=72.28\%,
$$

$$
\text{share}_C=\frac{168}{891}\times100\%=18.86\%,
$$

$$
\text{share}_Q=\frac{77}{891}\times100\%=8.64\%,
$$

$$
\text{share}_{\mathrm{missing}}=\frac{2}{891}\times100\%=0.22\%.
$$

If the denominator is the 889 non-missing rows, the shares are 72.44%, 18.90%, and 8.66%. Always state the denominator.

### 3.2 Countplot

A countplot counts observations in each category.

```python
sns.countplot(data=df, x="Embarked")
```

Important arguments:

| Argument | Meaning |
|---|---|
| `data` | DataFrame containing the column |
| `x` | categorical column name or values |
| `hue` | split each category by another categorical variable |
| `order` | explicit category order |
| `hue_order` | explicit hue order |
| `stat` | statistic to compute; default `"count"` |
| `ax` | Matplotlib axes to draw on |

The official notebook uses:

```python
sns.countplot(df['Embarked'])
```

This is a legacy positional call. The modern, clearer form is:

```python
sns.countplot(data=df, x="Embarked")
```

The local notebook uses:

```python
sns.countplot(data=df, x=df['Pclass'])
```

Passing `data=df` and also `x=df['Pclass']` is redundant: the x values are already a Series, so the DataFrame is not needed for that mapping. A cleaner form is:

```python
sns.countplot(data=df, x="Pclass")
```

Both calls produce the same counts when they run, but the cleaner form keeps the data and the column name separate.

Counts for `Pclass`:

| Pclass | Count | Share of 891 |
|---:|---:|---:|
| 1 | 216 | 24.24% |
| 2 | 184 | 20.65% |
| 3 | 491 | 55.11% |

Third class is the largest group. This is a feature-distribution fact, not a target fact.

### 3.3 Bar chart

A bar chart displays a numeric value for each category. The value can be a count, a proportion, a mean, or any other summary.

```python
df["Survived"].value_counts().plot(kind="bar")
```

This is a pandas plotting call. It creates a Matplotlib axes and draws one bar per index label.

The official notebook comments out this alternative:

```python
# df['Survived'].value_counts().plot(kind='bar')
```

The bar chart is often better than a pie chart when:

- there are more than a few categories;
- the categories have similar shares;
- exact comparisons matter;
- labels are long;
- you want a horizontal layout.

### 3.4 Pie chart

A pie chart displays each category's share of a whole.

```python
df["Sex"].value_counts().plot(kind="pie", autopct="%.2f")
```

The official notebook uses Sex:

| Sex | Count | Share |
|---|---:|---:|
| male | 577 | 64.76% |
| female | 314 | 35.24% |

The local notebook uses Survived:

| Survived | Count | Share |
|---:|---:|---:|
| 0 (died) | 549 | 61.62% |
| 1 (survived) | 342 | 38.38% |

The `autopct="%.2f"` format displays two decimal places but does **not** add a percent sign. To display a percent sign, use a format such as `autopct="%1.1f%%"`.

Pie charts have limitations:

- human perception is better at comparing lengths than angles;
- many slices become unreadable;
- similar slices are hard to distinguish;
- a 3D or exploded pie can distort area;
- the chart does not show the denominator directly.

For two categories, a pie is acceptable as a quick share display. For many categories, a bar chart is usually clearer.

### 3.5 Interpreting categorical plots

When reading a countplot or bar chart, ask:

- Which category is the mode?
- Are any categories rare?
- Is the target balanced or imbalanced?
- Are missing values shown or silently dropped?
- Is the order meaningful (ordinal) or arbitrary (nominal)?
- Are the counts large enough to support conclusions?

When reading a pie chart, ask:

- How many slices are there?
- Are the slices close in size?
- Is the whole clearly defined?
- Is the percentage label a share of all rows or of non-missing rows?

For Titanic:

- Embarked is dominated by S (644 of 891); C and Q are smaller.
- Pclass is dominated by 3rd class (491 of 891).
- Sex is imbalanced: 577 male, 314 female.
- Survived is imbalanced: 549 died, 342 survived. A majority-class baseline is 61.62%.

These are descriptive facts. They do not establish causation and they do not describe a population beyond this dataset.

### 3.6 Worked comparison: countplot, bar chart, and pie

| Question | Best default | Why |
|---|---|---|
| How many passengers in each Pclass? | countplot or bar chart | counts per category |
| What share of passengers survived? | bar chart or pie for two categories | proportion of a binary target |
| How many passengers embarked at each port? | countplot | counts per nominal category |
| Which sex group is larger? | bar chart or pie for two categories | simple two-level share |
| How does survival rate vary by Pclass? | bar chart of rates, or groupby | this is a rate, not a raw count |
| How many categories are rare? | countplot with sorted counts | frequency ranking |

A countplot of raw counts and a bar chart of survival rates are not interchangeable. The first answers "how many rows?" The second answers "what fraction survived within each group?"

---

## 4. Numerical univariate analysis

### 4.1 Histogram

A histogram groups a numeric variable into bins and counts how many observations fall into each bin.

```python
plt.hist(df["Age"], bins=5)
```

The official notebook uses five bins. Its saved output is:

```text
(array([100., 346., 188.,  69.,  11.]),
 array([ 0.42 , 16.336, 32.252, 48.168, 64.084, 80.   ]),
 <a list of 5 Patch objects>)
```

The first array contains counts. The second contains bin edges. There are five bins and six edges.

| Bin | Age range | Count |
|---:|---|---:|
| 1 | 0.420 to 16.336 | 100 |
| 2 | 16.336 to 32.252 | 346 |
| 3 | 32.252 to 48.168 | 188 |
| 4 | 48.168 to 64.084 | 69 |
| 5 | 64.084 to 80.000 | 11 |

The counts sum to:

$$
100+346+188+69+11=714.
$$

There are 714 observed ages and 177 missing ages. The histogram displays the observed ages, not all 891 rows.

The bin width is approximately:

$$
\frac{80-0.42}{5}=15.916.
$$

The tallest bin is 16.336 to 32.252, with 346 passengers. The distribution has a large central mass and a thin right tail toward age 80.

### 4.2 Bin choice changes the picture

A histogram is not a unique object. Different bin counts or bin edges produce different pictures of the same data.

For Age with 714 observed values, using NumPy's equal-width histogram gives:

| Bins | Non-empty bins | Largest bin count |
|---:|---:|---:|
| 5 | 5 | 346 |
| 20 | 20 | 98 |
| 100 | 70 | 31 |
| 300 | 85 | 30 |

The local notebook uses `bins=300`. With 714 observations and 300 bins, many bins are empty or contain only a few values. The histogram looks spiky and noisy. It does not reveal the main shape more accurately; it hides it in fine-grained sampling noise.

A reasonable starting range is often 20 to 40 bins for a few hundred observations, but there is no universal rule. Try several bin counts and check that the main conclusions are stable.

The bin count is a modeling choice, not a property of the data. If a conclusion appears only at one bin width, it is fragile.

### 4.3 Histogram versus bar chart

A histogram and a bar chart both use rectangular bars, but they are different.

| Property | Histogram | Bar chart |
|---|---|---|
| x-axis | numeric bins | categories |
| bar order | ordered by numeric range | category order |
| bar adjacency | usually touching | usually separated |
| y-axis | count or density | any numeric summary |
| purpose | distribution of a numeric variable | comparison across categories |

A bar chart of `Pclass` counts is categorical, even though the labels are numbers. A histogram of Age is numeric because the x-axis represents a continuous range.

### 4.4 KDE and distplot

A kernel density estimate (KDE) is a smoothed estimate of a probability density.

```python
sns.kdeplot(data=df, x="Age")
sns.histplot(data=df, x="Age", kde=True)
```

The official and local notebooks use:

```python
sns.distplot(df["Age"])
```

`distplot` is deprecated. It combines a histogram and a KDE in one call. The modern replacements are:

- `sns.histplot(..., kde=True)` for a histogram with a KDE overlay;
- `sns.kdeplot(...)` for a KDE alone.

The KDE is a smooth curve, but it is not the data. Its shape depends on the kernel and the bandwidth. A Gaussian kernel can extend beyond the observed range, so a KDE of a positive variable can show a small density below zero. That does not mean negative ages or negative fares exist.

For Age, the KDE shows a main mass around the late 20s and a right tail toward older ages. For Fare, a KDE would show an extreme right tail; a linear-scale plot is dominated by the bulk and the high-fare tail.

### 4.5 Boxplot

A boxplot summarizes a numeric distribution with five key values:

- minimum (or lower whisker);
- first quartile $Q_1$ (25th percentile);
- median $Q_2$ (50th percentile);
- third quartile $Q_3$ (75th percentile);
- maximum (or upper whisker).

The box spans the interquartile range:

$$
\mathrm{IQR}=Q_3-Q_1.
$$

Common whiskers extend to the most extreme observations within:

$$
Q_1-1.5\,\mathrm{IQR}
\quad\text{and}\quad
Q_3+1.5\,\mathrm{IQR}.
$$

Points outside those fences are drawn individually as outlier candidates.

The official notebook plots Age:

```python
sns.boxplot(df["Age"])
```

The local notebook plots Fare:

```python
sns.boxplot(data=df, x=df["Fare"])
```

These are different variables and should not be compared as if they were the same plot.

For Age:

- $Q_1=20.125$;
- median $=28$;
- $Q_3=38$;
- $\mathrm{IQR}=17.875$;
- lower fence $=-6.6875$;
- upper fence $=64.8125$;
- 11 observed ages are above the upper fence;
- no observed ages are below the lower fence;
- minimum is 0.42 and maximum is 80.

For Fare:

- $Q_1=7.9104$;
- median $=14.4542$;
- $Q_3=31$;
- $\mathrm{IQR}=23.0896$;
- lower fence $=-26.724$;
- upper fence $=65.6344$;
- 116 fares are above the upper fence;
- no fares are below the lower fence;
- minimum is 0 and maximum is 512.3292.

The Fare box is compressed near the low end, with a long upper whisker and many outlier points. The Age box is wider relative to its range and has fewer high outliers.

A boxplot does not show multimodality. Two very different groups can produce the same box. Use a histogram or KDE when the shape matters.

### 4.6 The IQR rule is a flag, not a deletion rule

An outlier candidate is not automatically an error.

A Fare of 512.3292 may be a genuine first-class ticket. A newborn age of 0.42 is unusual but possible. The IQR fence is a descriptive convention; it does not know the domain.

Before removing an outlier, ask:

1. Is the value possible in the real world?
2. Is it a data-entry error?
3. Is it a different population or unit?
4. Does it carry signal?
5. Will the model be sensitive to it?

For Fare, the high values are likely genuine and informative. For a sensor reading of $-999$, the value may be a missing-value code. The rule is the same; the decision is domain-specific.

### 4.7 Summary statistics as anchors

The official notebook computes:

```python
df["Age"].min()    # 0.42
df["Age"].max()    # 80.0
df["Age"].mean()   # 29.69911764705882
df["Age"].skew()   # 0.38910778230082704
```

These numbers anchor the plots:

- min and max define the observed range;
- mean and median locate the center;
- skew quantifies the asymmetry.

The local notebook only reaches `df["Age"].min()` and the kernel crashes. No local max, mean, or skew output is saved. The local notebook also has a bare `plt.hist` cell that references the function without calling it, so it draws nothing.

### 4.8 Skewness and the mean-median relationship

For a right-skewed distribution, a long right tail pulls the mean above the median:

$$
\text{mean}>\text{median}.
$$

For a left-skewed distribution, a long left tail pulls the mean below the median:

$$
\text{mean}<\text{median}.
$$

For a roughly symmetric distribution, mean and median are close.

Titanic examples:

| Variable | Mean | Median | Shape |
|---|---:|---:|---|
| Age | 29.6991 | 28.0000 | mild right skew |
| Fare | 32.2042 | 14.4542 | strong right skew |

Fare's mean is more than twice its median:

$$
\frac{32.2042}{14.4542}\approx2.23.
$$

Age's mean is only 1.6991 above its median.

Skewness is not a complete description. A distribution can have zero skew and still be bimodal. Use plots with the statistic.

### 4.9 Worked numerical example: a small boxplot

Consider the sorted values:

$$
1,\ 2,\ 2,\ 3,\ 4,\ 5,\ 7,\ 10,\ 12,\ 30.
$$

There are $n=10$ observations. Using linear interpolation:

- $Q_1=2.25$;
- median $=4.5$;
- $Q_3=9.25$;
- $\mathrm{IQR}=9.25-2.25=7$;
- lower fence $=2.25-1.5(7)=-8.25$;
- upper fence $=9.25+1.5(7)=19.75$.

The value 30 lies above 19.75, so it is an outlier candidate. The mean is 7.6, while the median is 4.5. The single value 30 pulls the mean upward but does not move the median much. This is the robustness difference between mean and median.

A boxplot of these values would show a box from 2.25 to 9.25, a median line at 4.5, whiskers extending to 1 and 12, and a separate point at 30.

---

## 5. Mathematics of univariate summaries

### 5.1 Counts and proportions

For category $i$ with count $n_i$ out of $n$ observations:

$$
p_i=\frac{n_i}{n}.
$$

The percentage is:

$$
100p_i.
$$

The shares of all categories sum to 1 if every observation belongs to exactly one category. Missing values complicate the denominator: decide whether missing is a category or an excluded value.

### 5.2 Mean

For observations $x_1,\dots,x_n$:

$$
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i.
$$

The mean uses every value. A single extreme value can move it substantially.

For a binary variable coded 0/1:

$$
\bar{x}=\frac{\text{number of 1s}}{n}=p.
$$

This is why the mean of `Survived` is the survival proportion.

### 5.3 Median

Sort the observations:

$$
x_{(1)}\le x_{(2)}\le\cdots\le x_{(n)}.
$$

For odd $n$, the median is the middle value. For even $n$, it is the average of the two middle values. The median depends on rank, not magnitude, so it is robust to extreme values.

### 5.4 Variance and standard deviation

Sample variance:

$$
s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2.
$$

Sample standard deviation:

$$
s=\sqrt{s^2}.
$$

Pandas `std()` uses `ddof=1` by default. The population standard deviation uses `ddof=0`. The sample version has a larger denominator adjustment when $n$ is small.

Standard deviation is in the same units as the data. Variance is in squared units.

### 5.5 Quantiles and the IQR

For a probability $p\in[0,1]$, the $p$-quantile is a value below which approximately a proportion $p$ of the data lies.

Pandas uses linear interpolation by default. For sorted values $x_0,\dots,x_{n-1}$ (0-based), let:

$$
h=(n-1)p,\qquad j=\lfloor h\rfloor,\qquad \gamma=h-j.
$$

Then:

$$
Q_p=(1-\gamma)x_j+\gamma x_{j+1}.
$$

The interquartile range is:

$$
\mathrm{IQR}=Q_3-Q_1.
$$

The IQR contains the middle 50% of the observations and is less sensitive to extreme tails than the range.

### 5.6 Outlier fences

A common boxplot rule flags values outside:

$$
Q_1-1.5\,\mathrm{IQR}
\quad\text{and}\quad
Q_3+1.5\,\mathrm{IQR}.
$$

The multiplier 1.5 is a convention, not a law. Some analyses use 3.0 for extreme outliers. The fence identifies candidates for inspection; it does not prove that a value is wrong.

### 5.7 Skewness

A moment-based skewness coefficient is:

$$
g_1=
\frac{
\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^3
}{
\left(
\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2
\right)^{3/2}
}.
$$

- $g_1>0$: right tail is longer;
- $g_1<0$: left tail is longer;
- $g_1\approx0$: roughly symmetric.

Pandas uses a sample-adjusted version of this statistic, so the exact value can differ slightly from the population formula. The sign and rough magnitude are the useful part.

Fare's skewness is 4.7873, indicating a strong right tail. Age's skewness is 0.3891, indicating a mild right tail.

### 5.8 Histogram density

A histogram can display counts or density.

For a bin of width $h$ containing $c$ observations out of $n$:

$$
\text{density}=\frac{c}{n h}.
$$

The area of each bar is approximately the fraction of observations in that bin. The total area is approximately 1 when the bins cover the full range.

Count histograms are easier to read for a single dataset. Density histograms are useful when comparing datasets of different sizes or when overlaying a KDE.

---

## 6. Implementation and syntax

### 6.1 Setup

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("datasets/train.csv")
```

The official notebook reads `train.csv` relative to its working directory. The local notebook reads the absolute path `E:\MACHINE_LEARNING\train.csv`. The root file is byte-identical to `Day_20/datasets/train.csv`, but the absolute path is not portable.

### 6.2 Categorical code

```python
# Frequency table
counts = df["Embarked"].value_counts()
shares = df["Embarked"].value_counts(normalize=True)

# Countplot
sns.countplot(data=df, x="Embarked")
plt.title("Embarked counts")
plt.show()

# Bar chart of counts
df["Survived"].value_counts().plot(kind="bar")
plt.title("Survival counts")
plt.show()

# Pie chart of shares
df["Sex"].value_counts().plot(kind="pie", autopct="%.2f")
plt.title("Sex share")
plt.show()
```

Important syntax:

- `value_counts()` returns a Series indexed by category.
- `normalize=True` returns proportions.
- `plot(kind="bar")` draws vertical bars.
- `plot(kind="pie", autopct="%.2f")` draws a pie with two-decimal labels.
- `sns.countplot(data=df, x="Embarked")` keeps the column name as a string.

### 6.3 Numerical code

```python
# Histogram
plt.hist(df["Age"].dropna(), bins=20)
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age histogram")
plt.show()

# Histogram with KDE
sns.histplot(data=df, x="Age", bins=20, kde=True)
plt.title("Age distribution")
plt.show()

# KDE only
sns.kdeplot(data=df, x="Age")
plt.title("Age density")
plt.show()

# Boxplot
sns.boxplot(data=df, x="Fare")
plt.title("Fare boxplot")
plt.show()

# Summary statistics
print(df["Age"].describe())
print("skew:", df["Age"].skew())
```

The `.dropna()` call is optional for Seaborn, which usually handles missing values, but it makes the input to Matplotlib explicit. The official `plt.hist(df["Age"], bins=5)` saved output contains 714 observations, so the missing ages are not counted.

### 6.4 Figure, axes, and subplots

Matplotlib separates the figure from the axes.

- **Figure:** the entire canvas.
- **Axes:** one plotting area inside the figure.
- **Axis:** the x-axis or y-axis of an axes.
- **Artist:** any drawn object, such as a line, bar, text, or patch.

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(data=df, x="Age", bins=20, ax=axes[0])
axes[0].set_title("Age")

sns.boxplot(data=df, x="Fare", ax=axes[1])
axes[1].set_title("Fare")

plt.tight_layout()
plt.show()
```

`plt.subplots(1, 2)` returns one figure and an array of two axes. Passing `ax=axes[0]` tells Seaborn where to draw. Without `ax`, Seaborn draws on the current axes.

### 6.5 Return values and object types

| Call | Typical return |
|---|---|
| `plt.hist(...)` | tuple `(counts, bin_edges, patches)` |
| `sns.histplot(...)` | Matplotlib `Axes` |
| `sns.kdeplot(...)` | Matplotlib `Axes` |
| `sns.boxplot(...)` | Matplotlib `Axes` |
| `sns.countplot(...)` | Matplotlib `Axes` |
| `df["Sex"].value_counts().plot(kind="pie")` | Matplotlib `Axes` |
| `df["Age"].describe()` | pandas `Series` |
| `df["Age"].skew()` | scalar float |

The official saved outputs show old Matplotlib representations such as `<matplotlib.axes._subplots.AxesSubplot at 0x...>`. Current Matplotlib uses a different `Axes` representation. The plot is the same kind of object; the display string changed.

### 6.6 Common execution traps

**Bare function reference**

```python
plt.hist
```

This evaluates the function object and draws nothing. You must call it:

```python
plt.hist(df["Age"].dropna(), bins=20)
```

**Missing `plt.show()`**

In a notebook, the last expression may display automatically. In a script, call `plt.show()` when you need the window to appear.

**Missing values**

`plt.hist` may ignore or mishandle `NaN` depending on the input and version. Drop or impute explicitly when the behavior matters.

**Deprecated `distplot`**

```python
sns.distplot(df["Age"])
```

This is deprecated. Use:

```python
sns.histplot(data=df, x="Age", kde=True)
```

or:

```python
sns.kdeplot(data=df, x="Age")
```

**Legacy positional countplot**

```python
sns.countplot(df["Embarked"])
```

Prefer:

```python
sns.countplot(data=df, x="Embarked")
```

**Kernel crash**

The local notebook's `df["Age"].min()` cell has a saved error output saying the kernel crashed. The code itself is trivial; the crash is an environment/kernel problem, not evidence that `min()` is invalid. The official notebook's same call returns 0.42.

### 6.7 A small reusable summary function

```python
def summarize_numeric(series):
    return {
        "count": series.count(),
        "missing": series.isna().sum(),
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "min": series.min(),
        "q1": series.quantile(0.25),
        "q3": series.quantile(0.75),
        "max": series.max(),
        "skew": series.skew(),
    }

def summarize_categorical(series):
    counts = series.value_counts(dropna=False)
    shares = series.value_counts(normalize=True, dropna=False)
    return counts.to_frame("count").assign(share=shares)
```

These functions do not replace plots. They provide the numeric anchors that the plots should agree with.

---

## 7. Actual notebook results

### 7.1 Official notebook: 21 cells

| Cell | Type | Content | Saved result |
|---:|---|---|---|
| 0 | code | import pandas, seaborn | none |
| 1 | code | `pd.read_csv('train.csv')` | none |
| 2 | code | `df.head()` | first five rows |
| 3 | markdown | `# 1. Categorical Data` | heading |
| 4 | markdown | `## a. Countplot` | heading |
| 5 | code | `sns.countplot(df['Embarked'])` | axes + image |
| 6 | markdown | `## b. PieChart` | heading |
| 7 | code | `df['Sex'].value_counts().plot(kind='pie',autopct='%.2f')` | axes + image |
| 8 | markdown | divider | none |
| 9 | markdown | `# 2. Numerical Data` | heading |
| 10 | markdown | `## a. Histogram` | heading |
| 11 | code | `plt.hist(df['Age'],bins=5)` | counts, edges, patches, image |
| 12 | markdown | `## b. Distplot` | heading |
| 13 | code | `sns.distplot(df['Age'])` | axes + image |
| 14 | markdown | `## c. Boxplot` | heading |
| 15 | code | `sns.boxplot(df['Age'])` | axes + image |
| 16 | code | `df['Age'].min()` | 0.42 |
| 17 | code | `df['Age'].max()` | 80.0 |
| 18 | code | `df['Age'].mean()` | 29.69911764705882 |
| 19 | code | `df['Age'].skew()` | 0.38910778230082704 |
| 20 | code | empty | none |

The official notebook is a sequence of separate univariate questions:

1. How many passengers embarked at each port?
2. What is the sex split?
3. What is the Age histogram with five bins?
4. What does a smooth Age density look like?
5. What does the Age boxplot show?
6. What are the minimum, maximum, mean, and skewness of Age?

It does not plot Fare. It does not plot Pclass or Survived. The local notebook makes different choices.

### 7.2 Local notebook: 22 cells

The local notebook has 9 code cells and 13 markdown cells.

| Cell | Type | Content | Saved result |
|---:|---|---|---|
| 0 | markdown | document banner | none |
| 1 | markdown | univariate introduction | none |
| 2 | code | imports pandas, matplotlib, seaborn | none |
| 3 | code | read root `train.csv`; `df.sample(5)` | no saved output |
| 4 | markdown | categorical columns | none |
| 5 | markdown | countplot / pie explanation | none |
| 6 | code | `sns.countplot(data=df, x=df['Pclass'])` | no saved output |
| 7 | code | `df["Survived"].value_counts().plot(kind='pie', autopct="%.2f")` | no saved output |
| 8 | markdown | numerical columns | none |
| 9 | markdown | histogram explanation | none |
| 10 | markdown | hist / distplot explanation | none |
| 11 | code | bare `plt.hist` | no saved output |
| 12 | code | `plt.hist(df["Age"], bins=300)` | no saved output |
| 13 | code | `sns.distplot(df["Age"])` | no saved output |
| 14 | markdown | boxplot explanation | none |
| 15 | markdown | IQR fences | none |
| 16 | markdown | boxplot skew | none |
| 17 | markdown | outliers and crash note | none |
| 18 | code | `sns.boxplot(data=df, x=df["Fare"])` | no saved output |
| 19 | code | `df['Age'].min()` | kernel crash error |
| 20 | markdown | bins experiment | none |
| 21 | markdown | key takeaways | none |

The local notebook has no saved plot images. It is a working notebook whose code cells were not executed successfully, or whose outputs were cleared. The only saved code output is the kernel-crash error in cell 19.

The local notebook's bare `plt.hist` cell is not a plot. It references the function and returns nothing visible.

The local notebook's `df.sample(5)` has no `random_state`, so even if it were run, the sampled rows would change from run to run.

### 7.3 Cheat sheet: 74 cells

`Matplotlib-Seaborn-Cheat-Sheet.ipynb` is a separate plotting reference, not the Day 20 official notebook.

- 74 cells total;
- 34 code cells;
- 40 markdown cells;
- all code cells have no execution count;
- no saved outputs;
- synthetic 12-row DataFrame for most examples;
- `plotly.express` is imported but never used.

The cheat sheet covers:

| Section | Content |
|---|---|
| 1 | imports and setup |
| 2 | Matplotlib line, scatter, bar, histogram, box, pie |
| 3 | figure, axes, subplots, customization |
| 4 | Seaborn histplot, kdeplot, countplot, barplot, boxplot, violinplot, scatterplot, lineplot, regplot |
| 5 | correlation heatmap, pairplot, jointplot, general heatmap |
| 6 | which plot to choose |
| 7 | Matplotlib versus Seaborn |
| 8 | ML-focused EDA workflow |
| 9 | common mistakes |
| 10 | quick syntax revision |
| 11 | self-test |
| 12 | final mental model |

The synthetic DataFrame has columns `Age`, `Salary`, `Gender`, `Purchased`, `Year`, and `Sales`. Its values are not Titanic data. Do not treat the cheat sheet's shapes, counts, or correlations as Titanic facts.

The cheat sheet is useful for syntax lookup and plot selection. The primary univariate analysis is in `my_notebook.ipynb` and the official `day20.ipynb`.

### 7.4 Official versus local choices

| Question | Official notebook | Local notebook |
|---|---|---|
| Categorical feature plotted | Embarked | Pclass |
| Categorical target plotted | none | Survived |
| Pie variable | Sex | Survived |
| Age histogram bins | 5 | 300 |
| Age density | `sns.distplot` | `sns.distplot` |
| Boxplot variable | Age | Fare |
| Age summary | min, max, mean, skew | min only |
| Saved outputs | images and text | none; kernel crash |
| Data path | `train.csv` relative | absolute root path |

The official notebook answers "what does Age look like?" and "what does Embarked look like?" The local notebook answers "what does Pclass look like?", "what does Survived look like?", and "what does Fare look like?" Both are valid univariate questions; they are not the same analysis.

### 7.5 Version and environment differences

Current environment:

- pandas 3.0.5;
- NumPy 2.5.3;
- Matplotlib 3.11.2;
- Seaborn 0.13.2.

Consequences:

- `sns.distplot` still exists in Seaborn 0.13.2 but is deprecated; future versions may remove it.
- `sns.countplot(df['Embarked'])` is a legacy positional call; prefer keyword arguments.
- Old Matplotlib axes representations in saved outputs differ from current representations.
- `df['Age'].min()` is valid; the local kernel crash is not a pandas error.
- `sns.boxplot(data=df, x=df['Fare'])` works, but passing the column name is cleaner.

The official saved outputs are valid for the environment that produced them. They should not be treated as current-version output without checking.

---

## 8. Interpreting graphs

### 8.1 Countplot

Axes:

- x-axis: categories;
- y-axis: count of rows.

Pattern:

- the tallest bar is the mode;
- short bars are rare categories;
- missing values may be absent.

Conclusion:

- Embarked is dominated by S;
- Pclass is dominated by 3rd class;
- Sex is imbalanced toward male;
- Survived is imbalanced toward death.

What not to conclude:

- that the largest category is more predictive;
- that a rare category is unimportant;
- that the counts represent a population;
- that missing values are zero.

### 8.2 Pie chart

Axes:

- angle and area encode share;
- labels identify categories.

Pattern:

- the whole is 100%;
- each slice is a share.

Conclusion:

- male passengers are 64.76% of the dataset;
- female passengers are 35.24%;
- survivors are 38.38%;
- deaths are 61.62%.

What not to conclude:

- that a 61.62% share means the model will be 61.62% accurate on unseen data;
- that a two-category pie is better than a bar chart;
- that a small slice is negligible;
- that the pie shows the number of rows.

### 8.3 Histogram

Axes:

- x-axis: numeric bins;
- y-axis: count or density.

Pattern:

- peaks show common ranges;
- gaps show empty ranges;
- long tails show skew;
- multiple peaks suggest subgroups.

Conclusion:

- Age has a large mass between 16.336 and 32.252;
- Age has a mild right tail;
- Fare has a strong right tail;
- bins=300 is too fine for 714 observations.

What not to conclude:

- that the distribution is normal from one histogram;
- that a gap is a data error;
- that the bin boundaries are meaningful cut points;
- that the histogram shows causality.

### 8.4 KDE

Axes:

- x-axis: numeric value;
- y-axis: estimated density.

Pattern:

- smooth peaks and tails;
- bandwidth controls smoothness.

Conclusion:

- the main Age mass is in the 20s;
- the density is an estimate, not the data.

What not to conclude:

- that the KDE is exact;
- that density values are probabilities;
- that a density below zero means negative observations;
- that the bandwidth is objectively correct.

### 8.5 Boxplot

Axes:

- x-axis: variable or category;
- y-axis: numeric value.

Pattern:

- box: Q1 to Q3;
- line: median;
- whiskers: non-outlier range;
- dots: outlier candidates.

Conclusion:

- Age has a median of 28 and a moderate spread;
- Fare has a median of 14.4542 and a long upper tail;
- Fare has 116 high outlier candidates;
- Age has 11 high outlier candidates.

What not to conclude:

- that every dot is an error;
- that the box contains 50% of the data in a symmetric way;
- that the distribution is unimodal;
- that the whiskers are the min and max.

---

## 9. Workflow connections

### 9.1 Day 19 to Day 20

Day 19 produced:

- shape: 891 x 12;
- missing counts: Age 177, Cabin 687, Embarked 2;
- summary statistics: mean, median, std, quartiles;
- target balance: 342 survived, 549 died;
- correlations.

Day 20 turns those numbers into pictures:

- countplots for categorical frequencies;
- pie/bar charts for shares;
- histograms for shape and bins;
- KDE for smooth density;
- boxplots for quartiles and outlier candidates.

### 9.2 Day 20 to Day 21

Day 21 adds a second variable:

- countplot with `hue`;
- boxplot of a numeric variable by category;
- scatter plot of two numeric variables;
- crosstab of two categorical variables;
- correlation heatmap.

A univariate finding often motivates a bivariate question:

- Age is mildly skewed and has missing values, so compare Age by survival;
- Fare is strongly skewed, so compare Fare by Pclass;
- Sex is imbalanced, so compare survival rate by Sex;
- Embarked has a small missing group, so be careful when grouping.

### 9.3 Day 20 to preprocessing

| Univariate finding | Possible later action |
|---|---|
| Fare right-skewed with high outliers | log or power transform; robust scaling; tree models may not need it |
| Age missing 177 | impute; consider missing indicator |
| Cabin missing 687 | drop or engineer `HasCabin`/deck |
| Embarked missing 2 | mode impute or drop two rows |
| Pclass ordinal | ordinal encoding or one-hot |
| Sex categorical | binary encoding |
| Survived imbalanced | choose metrics beyond accuracy; consider class weights |
| Fare zeros | check whether valid; do not treat as missing automatically |

These are candidates, not automatic commands. The model, validation strategy, and domain knowledge decide.

### 9.4 Leakage discipline

Exploratory plots may use the full dataset to understand it. But:

- imputation values must be learned from training data only;
- scaling parameters must be learned from training data only;
- encoding categories must be based on training data only;
- feature-selection decisions must not use the test target;
- target-rate plots on the test set can leak information if used to choose features.

Day 20 itself does not fit preprocessing. It identifies what might need to be fitted later.

---

## 10. Common mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Using 300 bins for 714 observations | spiky, noisy histogram | try 20 to 40 bins; compare several widths |
| Using `sns.distplot` | deprecation warning; future removal | use `histplot(kde=True)` or `kdeplot` |
| Countplot on a continuous variable | hundreds of bars | use a histogram |
| Pie chart with many categories | unreadable slices | use a sorted bar chart |
| Deleting every boxplot dot | loses genuine high fares | inspect domain meaning first |
| Treating `NaN` as zero | wrong mean and histogram | drop, impute, or report missingness |
| Bare `plt.hist` | no plot appears | call the function with data |
| Forgetting `plt.show()` in a script | no window appears | call `plt.show()` |
| Reading counts without denominators | misleading shares | report count and share |
| Treating `Pclass` mean as continuous | meaningless "average class" | treat as ordinal/category |
| Treating `Survived` mean as a rating | wrong interpretation | it is a proportion |
| Comparing official and local plots as the same | wrong variable or bins | check the cell source |
| Treating cheat-sheet synthetic data as Titanic | wrong facts | use `datasets/train.csv` |
| Ignoring the kernel crash | missing local results | document it; rerun in a stable kernel |
| Concluding causation from a univariate plot | unsupported claim | use experiments/domain knowledge |

---

## 11. Essential interview and examination preparation

### 11.1 Level 1 - definitions

**Q1. What is univariate analysis?**

The analysis of one variable at a time: its frequency distribution, center, spread, shape, and unusual values.

**Q2. What is a categorical variable?**

A variable whose values are categories or labels, such as Sex, Embarked, or Pclass.

**Q3. What is a numerical variable?**

A variable whose values are numbers on which arithmetic or ordering is meaningful, such as Age or Fare.

**Q4. What is a histogram?**

A plot that groups a numeric variable into bins and shows the count or density in each bin.

**Q5. What is a boxplot?**

A plot of the five-number summary: minimum/whisker, Q1, median, Q3, maximum/whisker, with outlier candidates beyond the fences.

**Q6. What is the IQR?**

The interquartile range, $Q_3-Q_1$, containing the middle 50% of the data.

**Q7. What is skewness?**

A measure of asymmetry. Positive skew means a longer right tail; negative skew means a longer left tail.

**Q8. What is a KDE?**

A kernel density estimate: a smoothed estimate of a probability density.

### 11.2 Level 2 - mechanics

**Q9. Which plot for a categorical variable?**

A countplot or bar chart; a pie chart for a small number of categories when shares are the focus.

**Q10. Which plot for a numerical variable?**

A histogram or KDE for shape, and a boxplot for quartiles and outlier candidates.

**Q11. Countplot versus barplot?**

A countplot counts rows per category. A barplot displays an aggregated numeric value per category, such as a mean or median, often with uncertainty intervals.

**Q12. Histogram versus bar chart?**

A histogram bins a numeric variable; a bar chart compares categories.

**Q13. Why does `bins=300` look noisy?**

With 714 observations, 300 bins leave many bins empty or nearly empty. Sampling noise dominates the shape.

**Q14. What does `autopct="%.2f"` display?**

Two-decimal percentage values without a percent sign.

**Q15. What does `value_counts(normalize=True)` return?**

Proportions rather than counts.

**Q16. What does `dropna=False` do in `value_counts`?**

Includes missing values as a separate category.

**Q17. What does a boxplot whisker represent?**

The most extreme observation within 1.5 IQR of the box, not necessarily the minimum or maximum.

**Q18. What does `df["Age"].skew()` return?**

A sample-adjusted skewness statistic; 0.3891 for Age in this dataset.

**Q19. What does `plt.hist` return?**

A tuple of counts, bin edges, and patches.

**Q20. What does `sns.boxplot` return?**

A Matplotlib `Axes` object.

### 11.3 Level 3 - applied and numerical

**Q21. Embarked has 644 S, 168 C, 77 Q, and 2 missing. What share is S of all rows?**

$$
\frac{644}{891}\times100\%=72.28\%.
$$

**Q22. What share of rows is missing Embarked?**

$$
\frac{2}{891}\times100\%=0.22\%.
$$

**Q23. Pclass has 216, 184, and 491 rows. What share is third class?**

$$
\frac{491}{891}\times100\%=55.11\%.
$$

**Q24. Sex has 577 male and 314 female. What share is male?**

$$
\frac{577}{891}\times100\%=64.76\%.
$$

**Q25. Survived has 342 positives and 549 negatives. What is the positive rate?**

$$
\frac{342}{891}=0.383838=38.38\%.
$$

**Q26. What is the majority-class baseline?**

$$
\frac{549}{891}=0.616162=61.62\%.
$$

**Q27. The official Age histogram has counts 100, 346, 188, 69, 11. What is the total?**

$$
100+346+188+69+11=714.
$$

**Q28. Why is the histogram total 714, not 891?**

Because 177 ages are missing and are not counted.

**Q29. What is the width of each of the five Age bins?**

$$
\frac{80-0.42}{5}=15.916.
$$

**Q30. What is the Fare IQR?**

$$
Q_3-Q_1=31.0000-7.9104=23.0896.
$$

**Q31. What is the Fare upper fence?**

$$
31.0000+1.5(23.0896)=65.6344.
$$

**Q32. How many fares are above the upper fence?**

116.

**Q33. What is the Age upper fence?**

$$
38.0000+1.5(17.875)=64.8125.
$$

**Q34. How many ages are above the upper fence?**

11.

**Q35. Fare mean is 32.2042 and median is 14.4542. What does that indicate?**

A strong right skew: high fares pull the mean above the median.

### 11.4 Level 4 - debugging and scenarios

**Q36. A countplot shows only three Embarked bars, but the data has two missing values. Why?**

Seaborn drops missing values by default. Use `dropna=False` in `value_counts` or handle missing explicitly.

**Q37. A histogram with 300 bins looks like noise. What should you do?**

Try 20 to 40 bins and compare; check that conclusions are stable across bin widths.

**Q38. `sns.distplot` warns. What is the replacement?**

Use `sns.histplot(..., kde=True)` or `sns.kdeplot(...)`.

**Q39. A boxplot shows many Fare points above the whisker. Should you delete them?**

No. They are outlier candidates. Check whether they are genuine high fares before deciding.

**Q40. A pie chart of Ticket has hundreds of slices. What is the problem?**

Pie charts are unreadable with many categories. Use a bar chart of the most frequent tickets or aggregate categories.

**Q41. `plt.hist` alone produces no plot. Why?**

It references the function without calling it. Pass data and bin arguments.

**Q42. The local `df['Age'].min()` cell shows a kernel crash. Is `min()` invalid?**

No. The official notebook returns 0.42. The crash is an environment/kernel issue.

**Q43. The cheat sheet shows a different distribution from Titanic. Why?**

It uses synthetic 12-row data, not the Titanic dataset.

**Q44. A colleague says "Fare has outliers, so remove all fares above 65.6344." What is wrong?**

The fence is a descriptive flag. High fares may be genuine and informative. Removal needs a domain and modeling justification.

**Q45. A colleague says "Age is symmetric because the mean is close to the median." What should you check?**

Plot the histogram/KDE and inspect skewness and tails. Mean-median closeness is only one clue.

### 11.5 Worked code-reading questions

**Q46. What is the output?**

```python
df["Embarked"].value_counts()
```

S 644, C 168, Q 77, with 2 missing values excluded by default.

**Q47. What is the output?**

```python
df["Pclass"].value_counts().sort_index()
```

1: 216, 2: 184, 3: 491.

**Q48. What is the output?**

```python
df["Sex"].value_counts(normalize=True).round(4)
```

male 0.6476, female 0.3524.

**Q49. What is the output?**

```python
df["Age"].isnull().sum()
```

177.

**Q50. What is wrong with this code?**

```python
sns.countplot(df["Pclass"])
```

It is a legacy positional call. Prefer `sns.countplot(data=df, x="Pclass")`.

**Q51. What is wrong with this code?**

```python
plt.hist
```

It references the function without calling it. Pass the data and bins.

**Q52. What is wrong with this conclusion?**

"Fare has 116 outliers, so the data is dirty."

An IQR fence flags candidates, not errors. High fares can be genuine.

### 11.6 Exam-style theory with answers

**Question 1. Why is univariate analysis necessary before bivariate analysis?**

Because each variable's scale, shape, missingness, and category frequencies affect how a relationship can be interpreted. A skewed variable or imbalanced target can dominate or distort a two-variable plot.

**Question 2. Why can a histogram be misleading with too many bins?**

Too many bins create many empty or low-count bins. Sampling noise dominates, and the underlying shape becomes harder to see.

**Question 3. Why is a boxplot useful even when a histogram is available?**

It gives a compact, robust summary of center, spread, quartiles, and outlier candidates, and it is easy to compare across groups.

**Question 4. Why is a boxplot insufficient on its own?**

It hides multimodality and the detailed shape. Two different distributions can share the same five-number summary.

**Question 5. Why is a pie chart often worse than a bar chart?**

People compare lengths more accurately than angles, and pies become unreadable with many or similar slices.

**Question 6. Why is the mean of Survived a proportion?**

Because Survived is coded 0/1. The sum is the number of survivors, so the mean is survivors divided by the number of rows.

**Question 7. Why should missing values be reported with the denominator?**

Because a count without a total is hard to interpret, and missing values change the denominator for proportions.

**Question 8. Why are high Fare values not automatically errors?**

They may be genuine first-class fares. The IQR rule is a flag; domain knowledge decides whether a value is possible.

**Question 9. Why does `distplot` need replacement?**

It is deprecated and will eventually be removed. `histplot(kde=True)` and `kdeplot` provide the modern API.

**Question 10. Why does the local notebook have no plot outputs?**

Its code cells have no saved outputs, and the final code cell records a kernel crash. The notebook is preserved as the learner's work, not as a source of saved results.

---

## 12. Revision summary

### 12.1 Plot choice

| Variable / question | Default plot |
|---|---|
| one categorical variable, counts | countplot or bar chart |
| one categorical variable, shares of a few categories | bar chart; pie only for two or three |
| one numeric variable, shape | histogram or KDE |
| one numeric variable, quartiles and outliers | boxplot |
| one numeric variable, exact summary | `describe`, min, max, mean, median, skew |
| two numeric variables | scatter plot (Day 21) |
| numeric by category | boxplot or violin plot (Day 21) |
| two categorical variables | crosstab or grouped countplot (Day 21) |

### 12.2 Titanic numbers

| Fact | Value |
|---|---:|
| Shape | 891 x 12 |
| Age missing | 177 |
| Fare missing | 0 |
| Embarked missing | 2 |
| Sex | male 577, female 314 |
| Survived | died 549, survived 342 |
| Pclass | 1: 216, 2: 184, 3: 491 |
| Embarked | S: 644, C: 168, Q: 77 |
| Age mean / median / std | 29.6991 / 28 / 14.5265 |
| Age Q1 / Q3 / IQR | 20.125 / 38 / 17.875 |
| Age min / max / skew | 0.42 / 80 / 0.3891 |
| Fare mean / median / std | 32.2042 / 14.4542 / 49.6934 |
| Fare Q1 / Q3 / IQR | 7.9104 / 31 / 23.0896 |
| Fare min / max / skew | 0 / 512.3292 / 4.7873 |
| Fare above upper fence | 116 |
| Age above upper fence | 11 |

### 12.3 The three warnings

1. A plot is not the data; bin width, bandwidth, and chart type change the picture.
2. An outlier candidate is not automatically an error.
3. Univariate analysis does not establish relationships or causation.

### 12.4 The three decisions this day enables

1. Fare's skew and high tail motivate a later transformation or robust treatment.
2. Age's missingness motivates imputation and possibly a missing indicator.
3. The target imbalance motivates metrics beyond accuracy and a careful validation design.

### 12.5 The one-sentence summary

Inspect one variable at a time with the plot that matches its type, read center, spread, shape, and unusual values together, and use those findings to choose later preprocessing rather than guessing.

---

# Part B - Advanced Reading and Reference

## 13. Statistical reference

### 13.1 The histogram as an estimator

A histogram partitions the observed range into bins and counts observations.

For equal-width bins with width $h$, a bin containing $c$ observations out of $n$ has density estimate:

$$
\hat{f}_{\text{bin}}=\frac{c}{n h}.
$$

The bar height in a count histogram is $c$. The bar height in a density histogram is $c/(nh)$.

The histogram is a biased estimator of the underlying density:

- too few bins oversmooth the shape and hide peaks;
- too many bins undersmooth and show sampling noise;
- bin origin and width change the picture;
- the estimate is discontinuous at bin edges.

The bias-variance trade-off is controlled by the number of bins. There is no universally correct number.

### 13.2 Common bin-count rules

Three heuristics are often mentioned.

**Sturges' rule:**

$$
B=\lceil \log_2 n\rceil+1.
$$

For $n=714$, this gives about 11 bins.

**Rice's rule:**

$$
B=\lceil 2n^{1/3}\rceil.
$$

For $n=714$, this gives about 18 bins.

**Freedman-Diaconis rule:**

$$
h=2\,\mathrm{IQR}\,n^{-1/3},
\qquad
B=\left\lceil\frac{\max(x)-\min(x)}{h}\right\rceil.
$$

For Age:

- $n=714$;
- $\mathrm{IQR}=17.875$;
- $h=2(17.875)(714)^{-1/3}\approx3.9998$;
- $B\approx(80-0.42)/3.9998\approx19.90$;
- so about 20 bins.

The Freedman-Diaconis result agrees with the practical observation that 20 bins show the Age shape better than 300. These rules are starting points, not laws. Always inspect several bin counts.

### 13.3 Kernel density estimation

A KDE estimates a density by placing a kernel around every observation and averaging.

For observations $x_1,\dots,x_n$, bandwidth $h$, and kernel $K$:

$$
\hat{f}_h(x)=
\frac{1}{nh}
\sum_{i=1}^{n}
K\left(\frac{x-x_i}{h}\right).
$$

The Gaussian kernel is:

$$
K(u)=\frac{1}{\sqrt{2\pi}}e^{-u^2/2}.
$$

The bandwidth $h$ controls smoothness:

- small $h$: spiky, high variance;
- large $h$: oversmooth, high bias;
- moderate $h$: compromise.

Seaborn's `bw_adjust` multiplies the automatically chosen bandwidth. Larger values smooth more.

A Gaussian KDE can assign positive density outside the observed range. For a nonnegative variable such as Age or Fare, a small density below zero is an artifact of the kernel, not evidence of negative observations. Boundary correction is a more advanced topic.

A KDE is a density estimate, not a probability at a point. To get a probability over an interval, integrate the density over that interval.

### 13.4 Quantile interpolation

For sorted values $x_0,\dots,x_{n-1}$ and probability $p$:

$$
h=(n-1)p,\qquad
j=\lfloor h\rfloor,\qquad
\gamma=h-j.
$$

Pandas' default linear interpolation is:

$$
Q_p=(1-\gamma)x_j+\gamma x_{j+1}.
$$

Different software and different quantile methods can produce slightly different quartiles. This is why two tools may disagree at the third decimal place. The IQR and fence conclusions are usually stable, but exact boundary values can differ.

### 13.5 Skewness

The moment coefficient is:

$$
g_1=
\frac{
\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^3
}{
\left(
\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2
\right)^{3/2}
}.
$$

Pandas uses a sample-adjusted version. For large $n$, the adjustment is small. For small $n$, it can matter.

Interpretation:

- positive: right tail longer;
- negative: left tail longer;
- near zero: roughly symmetric.

Skewness is sensitive to extreme values. A single very large Fare can produce a large positive skew. That does not mean every high fare is an error.

### 13.6 Robust statistics

The median and IQR are robust to a minority of extreme values. The mean and standard deviation are not.

A rough notion is the **breakdown point**: the fraction of observations that can be made arbitrarily large before the statistic becomes arbitrarily large.

- median: breakdown point near 50%;
- mean: breakdown point 0%;
- IQR: breakdown point near 25%;
- standard deviation: breakdown point 0%.

This is why the boxplot is often a better first look at a skewed variable than a mean-plus-standard-deviation summary.

### 13.7 Binary variables

For a binary variable coded 0/1 with $P(X=1)=p$:

$$
E[X]=p,
\qquad
\mathrm{Var}(X)=p(1-p).
$$

The sample mean is the observed proportion. The sample standard deviation is not exactly $\sqrt{p(1-p)}$ because pandas uses the $n-1$ denominator, but it is close for large $n$.

A proportion is descriptive. A confidence interval for a population proportion is inferential and is not part of Day 20.

### 13.8 Counts, rates, and denominators

A count is an integer. A rate is a count divided by a denominator.

For a group $g$:

$$
\text{rate}_g=\frac{\text{events in }g}{\text{rows in }g}.
$$

A countplot shows counts. A bar chart of rates shows a different quantity. A grouped countplot with `hue` shows counts split by group, but the visual comparison of counts can be misleading when group sizes differ. A rate plot or a normalized table is often better for comparing groups.

---

## 14. Plotting internals and API reference

### 14.1 Matplotlib object model

Matplotlib has a hierarchy:

```text
Figure
  └── Axes
        ├── Axis (x)
        ├── Axis (y)
        └── Artists (lines, patches, text, images)
```

- **Figure:** the full canvas.
- **Axes:** one plotting region with data coordinates.
- **Axis:** the tick marks, labels, and scale for one dimension.
- **Artist:** anything drawn.

There are two common styles.

**Pyplot state-machine style:**

```python
plt.hist(df["Age"].dropna(), bins=20)
plt.title("Age")
plt.show()
```

**Object-oriented style:**

```python
fig, ax = plt.subplots()
ax.hist(df["Age"].dropna(), bins=20)
ax.set_title("Age")
plt.show()
```

The object-oriented style is clearer when there are multiple axes or reusable functions.

### 14.2 pandas plotting wrapper

`Series.plot` and `DataFrame.plot` create Matplotlib axes and dispatch to a plot kind.

```python
ax = df["Survived"].value_counts().plot(kind="bar")
```

Useful `kind` values:

- `"bar"`: vertical bars;
- `"barh"`: horizontal bars;
- `"pie"`: pie chart;
- `"hist"`: histogram;
- `"box"`: boxplot;
- `"kde"`: KDE.

The pandas wrapper is convenient, but it exposes fewer statistical options than Seaborn.

### 14.3 Seaborn axes-level and figure-level functions

**Axes-level functions** draw on one Matplotlib axes and usually return that axes:

- `histplot`;
- `kdeplot`;
- `boxplot`;
- `violinplot`;
- `countplot`;
- `barplot`;
- `scatterplot`;
- `lineplot`;
- `regplot`.

They accept `ax=`.

**Figure-level functions** create their own figure and manage multiple axes:

- `displot`;
- `catplot`;
- `relplot`;
- `pairplot`;
- `jointplot`.

They do not accept `ax=` in the same way. They use parameters such as `height`, `aspect`, and `facet_kws`.

The cheat sheet uses `pairplot` and `jointplot`, which are figure-level. In a notebook they display automatically; in a script, call `plt.show()` if needed.

### 14.4 API details

**`countplot`**

```python
sns.countplot(data=df, x="Embarked", order=["S", "C", "Q"])
```

- `stat="count"` by default;
- missing categories are dropped by default;
- `hue` splits counts by another category;
- `order` controls category order.

**`barplot`**

```python
sns.barplot(data=df, x="Pclass", y="Fare", estimator="median")
```

- default estimator is the mean;
- default error bar is a 95% confidence interval;
- `estimator` can be `"mean"`, `"median"`, or a callable;
- `errorbar=None` removes the interval.

**`histplot`**

```python
sns.histplot(data=df, x="Age", bins=20, kde=True, stat="count")
```

- `stat` can be `"count"`, `"density"`, `"probability"`, or `"percent"`;
- `kde=True` overlays a KDE;
- `hue` splits the distribution by category;
- `multiple` controls how hue levels are drawn.

**`kdeplot`**

```python
sns.kdeplot(data=df, x="Age", bw_adjust=1.0, fill=True)
```

- `bw_adjust` changes smoothing;
- `cut` controls how far the curve extends beyond the data;
- `fill=True` fills under the curve.

**`boxplot`**

```python
sns.boxplot(data=df, x="Fare", showfliers=False)
```

- `showfliers=False` hides outlier points but does not remove observations;
- `whis` changes the fence multiplier;
- `hue` splits by category.

**`pie`**

```python
df["Sex"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=90,
)
```

- `autopct` formats the percentage label;
- `startangle` rotates the first slice;
- `explode` offsets slices;
- `labels` can override index labels.

### 14.5 Deprecations and version differences

Current environment:

- pandas 3.0.5;
- NumPy 2.5.3;
- Matplotlib 3.11.2;
- Seaborn 0.13.2.

Important differences:

| Item | Older notebooks | Current recommendation |
|---|---|---|
| `sns.distplot` | common | use `histplot(kde=True)` or `kdeplot` |
| positional `sns.countplot(series)` | common | use `data=` and `x=` |
| old axes repr | `matplotlib.axes._subplots.AxesSubplot` | current `Axes` repr |
| pandas `object` dtype | common | pandas 3.0 uses `str` for text |
| `df.corr()` on mixed dtypes | older behavior | use `numeric_only=True` |

The official notebook's saved outputs are not wrong; they reflect an older environment. Reproduce the analysis, not the exact repr string.

### 14.6 Execution and display issues

**Bare function reference**

```python
plt.hist
```

This is an expression that evaluates to a function object. It does not call the function.

**Missing `plt.show()`**

In a notebook, the last expression may display automatically. In a script, call `plt.show()`.

**NaN in histograms**

The official histogram's counts sum to 714, matching the non-missing ages. The missing ages are not represented. If the behavior matters, drop or impute explicitly.

**Kernel crash**

The local notebook's final code cell records a kernel crash. A kernel crash can be caused by environment, memory, or extension issues. It is not evidence that the operation is invalid. Rerun in a fresh kernel and verify with the official result.

**Figure size and layout**

Large figures can be slow or clipped. Use `figsize`, `tight_layout`, and `constrained_layout` deliberately.

---

## 15. Cheat sheet audit

### 15.1 Inventory

`Matplotlib-Seaborn-Cheat-Sheet.ipynb`:

- 74 cells total;
- 34 code cells;
- 40 markdown cells;
- all code cells have `execution_count: null`;
- no saved outputs;
- synthetic 12-row DataFrame;
- `plotly.express` imported but unused.

### 15.2 Sections

| Section | Content | Cells |
|---|---|---:|
| 1 | imports and setup | 0-3 |
| 2 | Matplotlib basics | 4-15 |
| 3 | figure, axes, subplots | 16-19 |
| 4 | Seaborn basics | 20-41 |
| 5 | EDA plots | 42-49 |
| 6 | which plot to choose | 50 |
| 7 | Matplotlib versus Seaborn | 51 |
| 8 | ML-focused EDA workflow | 52-67 |
| 9 | common mistakes | 68 |
| 10 | quick syntax revision | 69 |
| 11 | self-test | 70 |
| 12 | final mental model | 71 |
| extras | experiment and usage banner | 72-73 |

### 15.3 Synthetic data

The cheat sheet creates:

```python
df = pd.DataFrame({
    "Age": [19, 22, 25, 28, 32, 35, 21, 27, 30, 40, 24, 33],
    "Salary": [22000, 28000, 35000, 42000, 52000, 60000, 26000, 39000, 48000, 75000, 32000, 58000],
    "Gender": ["F", "M", "F", "M", "F", "M", "M", "F", "M", "F", "F", "M"],
    "Purchased": ["No", "No", "Yes", "Yes", "Yes", "Yes", "No", "Yes", "Yes", "Yes", "No", "Yes"],
    "Year": [2019, 2020, 2021, 2022, 2023, 2024, 2019, 2020, 2021, 2022, 2023, 2024],
    "Sales": [100, 120, 150, 180, 210, 250, 90, 130, 160, 190, 220, 270],
})
```

This is not Titanic data. The distributions, counts, and correlations in the cheat sheet are synthetic examples.

### 15.4 Code observations

- `plotly.express` is imported but never used. It is unnecessary for the notebook's examples.
- `df.shape, df.info()` evaluates `df.info()`, which prints and returns `None`; the tuple contains the shape and `None`.
- `sns.pairplot` and `sns.jointplot` are figure-level and do not use `ax=`.
- `sns.barplot(..., estimator=np.median)` is valid in Seaborn 0.13.2; it displays the median with a default uncertainty interval.
- The cheat sheet's EDA workflow uses the synthetic `df`, not the Titanic dataset.
- No code cell has a saved output, so the cheat sheet is a syntax reference, not an executed analysis.

### 15.5 Role in the learning system

The cheat sheet is useful for:

- looking up plot syntax;
- choosing a plot for a data-type pair;
- recalling the Matplotlib versus Seaborn distinction;
- reviewing the leakage rule;
- practicing plot customization.

It is not a substitute for the Day 20 analysis. The official notebook and the personal notebook are the primary evidence; the cheat sheet is a companion reference.

---

## 16. Source-fidelity audit

### 16.1 Official notebook: 21 cells

The official notebook has 13 code cells and 8 markdown cells.

Key source facts:

- imports: `pandas`, `seaborn`;
- reads `train.csv` by relative path;
- `df.head()` shows the first five rows;
- `sns.countplot(df['Embarked'])` uses a legacy positional call;
- `df['Sex'].value_counts().plot(kind='pie',autopct='%.2f')` plots the sex split;
- `plt.hist(df['Age'],bins=5)` returns counts, edges, and patches;
- `sns.distplot(df['Age'])` uses the deprecated API;
- `sns.boxplot(df['Age'])` plots Age, not Fare;
- `df['Age'].min()`, `max()`, `mean()`, and `skew()` return 0.42, 80.0, 29.69911764705882, and 0.38910778230082704;
- the final code cell is empty.

Saved outputs:

- countplot and pie have axes and image outputs;
- histogram has counts `[100, 346, 188, 69, 11]` and edges `[0.42, 16.336, 32.252, 48.168, 64.084, 80]`;
- distplot and boxplot have axes and image outputs;
- min, max, mean, and skew have scalar outputs;
- old Matplotlib axes representations appear in the saved output.

### 16.2 Local notebook: 22 cells

The local notebook has 9 code cells and 13 markdown cells.

Key source facts:

- imports: pandas, matplotlib, seaborn;
- reads `E:\MACHINE_LEARNING\train.csv` by absolute path;
- `df.sample(5)` has no `random_state`;
- countplot uses Pclass, not Embarked;
- pie uses Survived, not Sex;
- bare `plt.hist` cell draws nothing;
- histogram uses `bins=300`;
- distplot uses the deprecated API;
- boxplot uses Fare, not Age;
- `df['Age'].min()` has a saved kernel-crash error;
- there are no saved plot images or scalar outputs.

The local notebook is preserved as the learner's work. Its choices are valid univariate questions, but they are not the same as the official notebook's choices.

### 16.3 Dataset and paths

`datasets/train.csv`:

- 891 rows x 12 columns;
- 61,194 bytes;
- SHA-256 `7d118fef8b6ccf7f81111877bc388536f7b1e498a655e3d649d19aaa010e9f6f`;
- root `E:\MACHINE_LEARNING\train.csv` is byte-identical.

The official notebook's `pd.read_csv('train.csv')` is relative to the working directory. The dataset is under `datasets/`, so the official notebook needs an adjusted path or a working directory containing `train.csv`.

### 16.4 Version differences

| Aspect | Official saved output | Current environment |
|---|---|---|
| `sns.distplot` | works in saved output | exists in Seaborn 0.13.2 but deprecated |
| `sns.countplot(df['Embarked'])` | works in saved output | legacy positional call |
| Matplotlib axes repr | old `_subplots.AxesSubplot` | current `Axes` repr |
| pandas text dtype | older `object` | pandas 3.0 `str` |
| `df.corr()` on mixed dtypes | older behavior | use `numeric_only=True` |

### 16.5 README discrepancies

The pre-rewrite README referred to:

- `reference_notebooks/reference_notebook.ipynb`, but the actual file is `reference_notebooks/day20.ipynb`;
- a status that did not reflect the completed Day 39-standard rewrite.

The rewrite corrects the README to the files that actually exist.

### 16.6 What was not changed

- `my_notebook.ipynb` was not modified.
- `reference_notebooks/day20.ipynb` was not modified.
- `Matplotlib-Seaborn-Cheat-Sheet.ipynb` was not modified.
- `datasets/train.csv` was not modified.
- `vendor/` was not modified.

---

## 17. Extended comparisons

### 17.1 Countplot, barplot, and pie

| Property | Countplot | Barplot | Pie |
|---|---|---|---|
| Statistic | count of rows | aggregated numeric value | share of whole |
| y variable | not required | required | not applicable |
| Hue support | yes | yes | no direct equivalent |
| Missing categories | dropped by default | depends on data | depends on input |
| Best for | category frequencies | group means/medians/rates | two or three shares |
| Weakness | no aggregation | can hide raw counts | angle perception, many slices |

### 17.2 Histogram and KDE

| Property | Histogram | KDE |
|---|---|---|
| Output | discrete bins | smooth curve |
| Bin/bandwidth choice | explicit bins | bandwidth |
| Shows exact counts | yes | no |
| Smooth shape | no | yes |
| Boundary behavior | bounded by bins | can extend beyond data |
| Best for | count distribution | smooth shape estimate |
| Weakness | bin-dependent | bandwidth-dependent |

### 17.3 Boxplot and violin plot

| Property | Boxplot | Violin plot |
|---|---|---|
| Shows quartiles | yes | usually with inner box |
| Shows full shape | no | yes, via density |
| Outlier points | yes | optional |
| Compact comparison | excellent | good but busier |
| Multimodality | hidden | visible |
| Best for | robust summary | shape across groups |

### 17.4 Histogram and bar chart

| Property | Histogram | Bar chart |
|---|---|---|
| Variable type | numeric | categorical |
| x-axis | numeric bins | categories |
| Bar order | ordered range | category order |
| Gaps between bars | usually none | usually present |
| Statistic | count/density | any summary |
| Example | Age distribution | Pclass counts |

### 17.5 Matplotlib and Seaborn

| Property | Matplotlib | Seaborn |
|---|---|---|
| Level | low-level | higher-level statistical API |
| Control | maximum | convenient defaults |
| Data input | arrays/Series | DataFrame-aware |
| Figure/axes | explicit | often managed |
| Categorical plots | manual | built-in |
| Best for | custom layouts | quick statistical plots |

### 17.6 Mean and median

| Property | Mean | Median |
|---|---|---|
| Uses all values | yes | rank-based |
| Sensitive to outliers | yes | no |
| Units | same as data | same as data |
| Right-skew behavior | pulled above median | stays near center |
| Best for | symmetric data | skewed/robust summaries |

### 17.7 IQR and standard deviation

| Property | IQR | Standard deviation |
|---|---|---|
| Measures | middle 50% spread | root-mean-square spread |
| Robust to outliers | yes | no |
| Units | same as data | same as data |
| Used by boxplot | yes | no |
| Best for | skewed data | roughly symmetric data |

### 17.8 Univariate and bivariate analysis

| Property | Univariate | Bivariate |
|---|---|---|
| Variables | one | two or more |
| Questions | distribution, frequency, shape | relationship, association, group differences |
| Plots | countplot, histogram, KDE, boxplot | scatter, grouped box, heatmap, crosstab |
| Day | 20 | 21 |
| Risk | misreading a single variable | confounding, Simpson's paradox |

---

## 18. Extended interview bank

### Level 1 - Foundations

1. **What is univariate analysis?**
   Analysis of one variable at a time: frequencies, center, spread, shape, and unusual values.

2. **What is the difference between a categorical and a numerical variable?**
   Categorical values are labels or groups; numerical values have meaningful magnitude and arithmetic.

3. **What is a distribution?**
   The pattern of values and their frequencies.

4. **What is an outlier?**
   A value far from the bulk of the data; it may be genuine, an error, or a different population.

5. **What is a quartile?**
   A quantile at 25%, 50%, or 75%; Q1, Q2 (median), and Q3.

### Level 2 - Mechanics

6. **Countplot versus barplot?**
   Countplot counts rows per category; barplot aggregates a numeric y per category.

7. **What does `value_counts(normalize=True)` return?**
   Proportions rather than counts.

8. **What does `bins` control?**
   The number and width of histogram intervals.

9. **What does `autopct` control?**
   The format of percentage labels in a pie chart.

10. **How does Seaborn handle missing categories in a countplot?**
    It drops them by default; use explicit missing handling if they matter.

### Level 3 - Mathematics

11. **How is the IQR computed?**
    $Q_3-Q_1$.

12. **What do the boxplot fences represent?**
    $Q_1-1.5\mathrm{IQR}$ and $Q_3+1.5\mathrm{IQR}$; points outside are candidates.

13. **What does positive skew mean?**
    A longer right tail; the mean is usually greater than the median.

14. **How is histogram density computed?**
    Count divided by $n$ times bin width.

15. **What does KDE bandwidth control?**
    Smoothness: small bandwidth is spiky, large bandwidth is oversmooth.

### Level 4 - Debugging

16. **Why does `sns.distplot` warn?**
    It is deprecated; use `histplot(kde=True)` or `kdeplot`.

17. **Why does a countplot show fewer bars than expected?**
    Missing values are dropped by default.

18. **Why does a bare `plt.hist` draw nothing?**
    It references the function without calling it.

19. **Why did the local notebook's final cell crash?**
    The saved error says the kernel crashed; this is an environment issue, not a pandas error.

20. **Why is a 300-bin histogram hard to read?**
    Too many bins create empty or low-count bins and sampling noise.

### Level 5 - Internals

21. **What is the difference between a Figure and an Axes?**
    A Figure is the canvas; an Axes is one plotting region inside it.

22. **What does `plt.hist` return?**
    Counts, bin edges, and patches.

23. **What is an axes-level Seaborn function?**
    A function that draws on one Axes and usually returns it, such as `histplot` or `boxplot`.

24. **What is a figure-level Seaborn function?**
    A function that creates its own figure and axes, such as `displot`, `catplot`, `pairplot`, or `jointplot`.

25. **What does pandas `.plot` return?**
    A Matplotlib Axes for most plot kinds.

### Level 6 - Design

26. **How do you choose the number of histogram bins?**
    Try several counts; use heuristics such as Freedman-Diaconis; check stability of conclusions.

27. **When do you use a pie chart?**
    Only for a small number of categories and when shares are the main message.

28. **How do you decide whether an outlier is an error?**
    Check domain plausibility, data provenance, units, and model sensitivity.

29. **How do you report missingness?**
    Report counts, percentages, and the denominator; state how missing values are handled in plots.

30. **How can univariate EDA leak information?**
    If plots on the test set are used to choose features or preprocessing, the test set influences the model.

### Level 7 - Advanced

31. **What is KDE boundary bias?**
    A Gaussian KDE can place density outside the data range; boundary correction reduces this.

32. **What is the bias-variance trade-off in a histogram?**
    Few bins oversmooth (bias); many bins undersmooth (variance).

33. **Why are median and IQR robust?**
    They depend on ranks/quantiles, so a minority of extreme values cannot move them arbitrarily.

34. **Why can a low univariate correlation still matter?**
    The relationship may be nonlinear, conditional, or interaction-driven; univariate summaries cannot see that.

35. **Why is target imbalance a univariate finding with modeling consequences?**
    It changes the baseline accuracy and the cost of false positives versus false negatives.

---

## 19. Extended examination questions

### 19.1 Theory

1. Explain why univariate analysis precedes bivariate analysis.
2. Explain the difference between a count and a proportion.
3. Explain why a histogram is not a bar chart.
4. Explain why a boxplot can hide multimodality.
5. Explain why `distplot` is deprecated.
6. Explain why missing values change the denominator.
7. Explain why an outlier fence is not a deletion rule.
8. Explain why a KDE can extend beyond the observed range.

### 19.2 Numerical

9. Compute the share of S in Embarked: 644 out of 891.
10. Compute the share of missing Embarked: 2 out of 891.
11. Compute the share of third class: 491 out of 891.
12. Compute the share of male passengers: 577 out of 891.
13. Compute the survival rate: 342 out of 891.
14. Compute the majority-class baseline: 549 out of 891.
15. Compute the width of the five official Age bins.
16. Compute the Fare IQR.
17. Compute the Fare upper fence.
18. Compute the Age upper fence.
19. Compute the density of the Age bin with 346 observations, $n=714$, width 15.916.
20. Compute the Freedman-Diaconis bin count for Age using IQR 17.875 and $n=714$.

### 19.3 Analytical

21. A colleague uses 300 bins and says Age is "noisy." Correct the statement.
22. A colleague deletes all 116 high fares. What questions should they answer first?
23. A colleague uses a pie chart for Ticket. What is the problem?
24. A colleague says the mean Age is 29.7, so the typical passenger is 29.7 years old. What is missing?
25. A colleague compares a countplot of Pclass with a barplot of survival rate by Pclass. Are they showing the same thing?
26. A colleague plots only the official notebook's Age boxplot and concludes Fare has no outliers. What is wrong?
27. A colleague uses the cheat sheet's synthetic data to describe Titanic passengers. What is wrong?
28. A colleague uses the full dataset to choose an imputation value, then evaluates on a test set. What is the leakage risk?
29. A colleague says "the histogram has a gap, so the data is invalid." What should they check?
30. A colleague says "the KDE goes below zero, so there are negative ages." Correct the statement.

### 19.4 Answer outlines

9. $644/891\times100\%=72.28\%$.
10. $2/891\times100\%=0.22\%$.
11. $491/891\times100\%=55.11\%$.
12. $577/891\times100\%=64.76\%$.
13. $342/891=0.383838=38.38\%$.
14. $549/891=0.616162=61.62\%$.
15. $(80-0.42)/5=15.916$.
16. $31-7.9104=23.0896$.
17. $31+1.5(23.0896)=65.6344$.
18. $38+1.5(17.875)=64.8125$.
19. $346/(714\times15.916)\approx0.03045$.
20. $h=2(17.875)(714)^{-1/3}\approx3.9998$, $B\approx19.90$, so about 20 bins.

---

## 20. References and glossary

### 20.1 Course references

- CampusX Day 20 official notebook: `vendor/100-days-of-machine-learning/day20-univariate-analysis/day20.ipynb`.
- Local reference copy: `reference_notebooks/day20.ipynb`.
- Local practical notebook: `my_notebook.ipynb`.
- Companion plotting reference: `Matplotlib-Seaborn-Cheat-Sheet.ipynb`.
- Dataset: `datasets/train.csv` (Titanic, 891 x 12).

### 20.2 Technical references

- pandas documentation: `value_counts`, `plot`, `describe`, `quantile`, `skew`.
- Matplotlib documentation: `Figure`, `Axes`, `hist`, `subplots`, `show`.
- Seaborn documentation: `countplot`, `barplot`, `histplot`, `kdeplot`, `boxplot`, `violinplot`, `distplot` deprecation.
- Histogram bin rules: Sturges, Rice, Freedman-Diaconis.
- Kernel density estimation and bandwidth selection.
- Titanic dataset documentation.

### 20.3 Glossary

| Term | Meaning |
|---|---|
| axes | one plotting region inside a figure |
| bar chart | rectangular bars comparing categories |
| bin | one interval of a histogram |
| boxplot | five-number summary with outlier candidates |
| countplot | bar chart of row counts per category |
| density | count divided by sample size and bin width |
| distribution | values and their frequencies |
| figure | the complete plotting canvas |
| histogram | binned numeric distribution |
| IQR | interquartile range, $Q_3-Q_1$ |
| KDE | kernel density estimate |
| mean | arithmetic average |
| median | 50th percentile |
| outlier | value far from the bulk of the data |
| pie chart | shares of a whole encoded by angles |
| proportion | count divided by total |
| quartile | 25th, 50th, or 75th percentile |
| skewness | asymmetry of a distribution |
| standard deviation | square root of variance |
| univariate | one variable at a time |
| variance | average squared deviation from the mean |
| whisker | boxplot line extending to non-outlier extremes |

### 20.4 Final checklist

- [ ] I can choose a countplot, bar chart, or pie for a categorical variable.
- [ ] I can choose a histogram, KDE, or boxplot for a numeric variable.
- [ ] I can compute and interpret counts, proportions, and denominators.
- [ ] I can explain how bin width changes a histogram.
- [ ] I can read the five-number summary and IQR fences.
- [ ] I can explain why Fare is right-skewed and Age is mildly right-skewed.
- [ ] I can distinguish an outlier candidate from an error.
- [ ] I can explain why `distplot` is deprecated.
- [ ] I can read a Matplotlib figure with multiple axes.
- [ ] I can explain why univariate analysis must precede bivariate analysis.
- [ ] I can identify the official/local notebook differences without treating them as errors.
- [ ] I can explain why preprocessing must be fitted on training data only.

