# Day 21 - Bivariate and Multivariate Analysis

> CampusX 100 Days of Machine Learning, Day 21.

This chapter is one unified document for Day 21. **Part A** is the main learning chapter and is sufficient on its own. **Part B** is optional advanced reading: statistical details, plotting internals, a source-fidelity audit, extended comparisons, and additional interview and examination material.

The day is still exploratory. There is no train/test split, no model training, and no cleaning performed in the official notebook. The goal is to move from one-column summaries to relationships between columns, and then to connect those relationships to later feature and preprocessing decisions.

---

# Part A - Main Learning Chapter

## 1. The problem: one column is not a relationship

Day 19 described the structure of the Titanic data: shape, types, missing values, duplicates, summary statistics, and correlations. Day 20 looked at one column at a time: how many passengers were in each class, what the Age distribution looked like, and how skewed Fare was.

Those questions are necessary, but they cannot answer the questions that matter for prediction:

- Did survival differ by sex?
- Did survival differ by passenger class?
- Was Age related to survival?
- Did the relationship between class and survival change when sex was considered?
- Which pairs of measurements separate the iris species?
- How did airline passengers change over time?

A model learns from relationships. A feature that has a large spread but no relationship with the target may be useless. A feature with a small spread can still be highly informative if it separates the classes. Bivariate and multivariate analysis is how we look for those relationships before modeling.

### 1.1 Bivariate and multivariate

**Bivariate analysis** studies two variables together. Examples:

- `Age` versus `Survived`;
- `total_bill` versus `tip`;
- `Pclass` versus `Survived`;
- `petal_length` versus `petal_width`.

**Multivariate analysis** studies three or more variables together. Examples:

- `Age` versus `Survived`, split by `Sex`;
- `tip` versus `total_bill`, colored by `sex`, with marker style by `smoker`;
- a `pairplot` of all numeric iris measurements, colored by species;
- a heatmap of `Pclass` by `Survived`, with counts or rates.

The boundary is not rigid. A scatterplot with `hue` already contains a third variable. A faceted plot with `row` and `col` can contain several. The important idea is that we are no longer looking at one marginal distribution in isolation.

### 1.2 What this day does not establish

A bivariate plot can show **association**, not causation. If survival is higher in first class, that does not prove that paying for first class caused survival. Passenger class is associated with sex, age, cabin location, and other factors. Later chapters will discuss confounding and experimental design; for now, the correct language is:

- "survival was higher among first-class passengers";
- not "first class caused survival."

The same caution applies to correlation. A correlation coefficient measures a particular kind of linear association. It does not prove that one variable drives the other.

### 1.3 The questions this day answers

By the end of Day 21, you should be able to answer:

1. Which plot matches a given pair of variable types?
2. What does `hue`, `style`, `size`, `row`, and `col` add to a plot?
3. What is the difference between a count, a rate, and a mean?
4. What does a crosstab contain, and how do row and column percentages differ?
5. What does a barplot show that a boxplot hides, and vice versa?
6. What does a clustermap add to a heatmap?
7. What does a pairplot show, and when is it too expensive?
8. How do you read a lineplot for trend and seasonality?
9. Why is a pivot-table heatmap not automatically a good visualization?
10. Which notebook outputs are real, which are stale, and which code is not reproducible in the current environment?

---

## 2. Foundations: variable types and the plot-choice map

### 2.1 Variable types

The first decision in bivariate analysis is the type of each variable.

| Type | Meaning | Titanic examples |
|---|---|---|
| Nominal categorical | unordered labels | `Sex`, `Embarked` |
| Ordinal categorical | ordered labels | `Pclass` (1st, 2nd, 3rd) |
| Binary | two categories, often 0/1 | `Survived`, `Sex` |
| Continuous numeric | meaningful magnitude and arithmetic | `Age`, `Fare` |
| Discrete numeric | counts or integers | `SibSp`, `Parch`, `size` |
| Time | ordered numeric or date | `year`, `month` |

`Pclass` is stored as integers 1, 2, 3, but it is not a continuous measurement. The average of `Pclass` is not a meaningful "average class" in the same way that average Age is meaningful. It can still be used as an ordinal feature or encoded later.

`Survived` is stored as 0/1. Its mean is the survival proportion, which is meaningful. Its "average" is not a rating.

### 2.2 The plot-choice map

| X | Y / grouping | Default plot | Main question |
|---|---|---|---|
| numeric | numeric | scatterplot, `relplot` | association, clusters, outliers |
| categorical | numeric | boxplot, barplot, violin | group differences in center and spread |
| categorical | categorical | crosstab + heatmap, grouped countplot | joint frequencies and rates |
| numeric | numeric, with categories | `relplot` with `hue`/`style`/`size`/facets | does the relationship change by group? |
| many numeric columns | — | `pairplot` | which pairs separate classes? |
| time | numeric | lineplot | trend, seasonality, change |
| two grouping columns | aggregate | pivot-table heatmap | a matrix of summaries |

The table is a starting point, not a law. A scatterplot can show a categorical variable through color; a boxplot can show two categorical variables through `hue`; a heatmap can show rates instead of counts.

### 2.3 Tidy data and the meaning of `x`, `y`, `hue`

Seaborn works most naturally with **long-form** or **tidy** data: one row per observation, one column per variable.

For the tips data:

```text
total_bill | tip | sex    | smoker | day | time   | size
16.99      | 1.01| Female | No     | Sun | Dinner | 2
10.34      | 1.66| Male   | No     | Sun | Dinner | 3
...
```

Then:

- `x="total_bill"` maps the total bill column to the horizontal axis;
- `y="tip"` maps the tip column to the vertical axis;
- `hue="sex"` maps sex to color;
- `style="smoker"` maps smoker status to marker shape;
- `size="size"` maps party size to marker size;
- `row="sex"` creates one row of subplots per sex;
- `col="day"` creates one column of subplots per day.

This is a declarative mapping. You describe which column plays which role, and Seaborn draws the marks. You are not manually looping over groups.

### 2.4 Axes-level and figure-level functions

Seaborn has two broad families.

**Axes-level functions** draw on one Matplotlib `Axes` and usually return that `Axes`:

- `scatterplot`;
- `barplot`;
- `boxplot`;
- `histplot`;
- `kdeplot`;
- `heatmap`;
- `lineplot`.

They accept `ax=`.

**Figure-level functions** create and manage their own figure and axes:

- `relplot`;
- `catplot`;
- `displot`;
- `pairplot`;
- `jointplot`;
- `clustermap` (a matrix grid).

They are convenient for faceting, but their return type is not a simple `Axes`. For example, `relplot` returns a `FacetGrid`, `pairplot` returns a `PairGrid`, and `clustermap` returns a `ClusterGrid`.

This distinction matters when you want to customize a specific axis or add a Matplotlib call. With a figure-level function, you often need to access the grid's axes rather than calling `plt.title` on the current axis.

### 2.5 Missing values and denominators

Seaborn usually drops rows with missing values in the mapped columns. That means different plots can use different numbers of rows.

For Titanic:

- `Pclass` and `Survived` are complete: 891 rows.
- `Age` has 177 missing values: Age plots use 714 rows.
- `Cabin` has 687 missing values.
- `Embarked` has 2 missing values.

A crosstab of `Pclass` by `Survived` uses all 891 rows. An Age-by-survival density uses 714 rows. A countplot of `Embarked` shows only the three non-missing ports by default.

Always report the denominator. "62.96% survived in first class" is based on 216 first-class passengers. "74.20% of females survived" is based on 314 females. A percentage without a denominator can hide a very small group.

---

## 3. Numerical-numerical: scatterplot and relplot

### 3.1 The problem

Suppose we want to know whether a larger restaurant bill is associated with a larger tip. We have two numeric columns: `total_bill` and `tip`. A histogram of each column separately cannot show whether the two move together.

A scatterplot places one point per observation at `(total_bill, tip)`. The pattern of points shows the association.

### 3.2 The official notebook code

The official notebook uses:

```python
sns.scatterplot(
    tips["total_bill"],
    tips["tip"],
    hue=df["sex"],
    style=df["smoker"],
    size=df["size"],
)
```

This is an older Seaborn style. In the current environment it has two problems:

1. `df` is not defined anywhere in the notebook. The saved output exists, so it was produced in a kernel where a variable named `df` happened to exist.
2. Current Seaborn expects `x` and `y` as keyword arguments, not as two positional arguments.

A reproducible modern version is:

```python
sns.scatterplot(
    data=tips,
    x="total_bill",
    y="tip",
    hue="sex",
    style="smoker",
    size="size",
)
```

The local notebook uses the figure-level version:

```python
sns.relplot(
    data=tips,
    y="tip",
    x="total_bill",
    kind="scatter",
    hue="sex",
    row="sex",
    style="smoker",
    size="size",
)
```

`relplot` is the figure-level interface. `kind="scatter"` selects a scatter plot. `row="sex"` creates one facet row per sex. `hue="sex"` also colors by sex, which is redundant with the row split, but not wrong. `style="smoker"` changes marker shape. `size="size"` changes marker size.

### 3.3 Data flow

```text
tips DataFrame
     |
     v
x = total_bill
y = tip
hue = sex
style = smoker
size = size
row = sex
     |
     v
Seaborn maps columns to visual channels
     |
     v
one scatter panel per sex row
     |
     v
points, colors, marker shapes, marker sizes
     |
     v
read association, group differences, outliers
```

### 3.4 Syntax breakdown

**`sns.scatterplot`**

```python
sns.scatterplot(
    data=tips,
    x="total_bill",
    y="tip",
    hue="sex",
    style="smoker",
    size="size",
    alpha=0.7,
)
```

- `data`: the DataFrame.
- `x`, `y`: column names or arrays.
- `hue`: color grouping.
- `style`: marker grouping.
- `size`: marker-size grouping.
- `alpha`: transparency, useful for overplotting.
- Returns a Matplotlib `Axes`.

**`sns.relplot`**

```python
sns.relplot(
    data=tips,
    x="total_bill",
    y="tip",
    kind="scatter",
    hue="sex",
    style="smoker",
    size="size",
    row="sex",
    col=None,
    height=4,
    aspect=1.2,
)
```

- `kind="scatter"` or `"line"`.
- `row`, `col`: facet variables.
- `height`, `aspect`: facet size.
- Returns a `FacetGrid`, not an `Axes`.
- `facet_kws` can pass additional facet options.

**Common additions**

```python
sns.scatterplot(data=tips, x="total_bill", y="tip", alpha=0.5)
plt.title("Tip versus total bill")
plt.xlabel("Total bill")
plt.ylabel("Tip")
plt.show()
```

### 3.5 What the tips data shows

The tips dataset has 244 rows and 7 columns:

- `total_bill`, `tip`: numeric;
- `sex`, `smoker`, `day`, `time`: categorical;
- `size`: integer party size.

Basic facts:

| Fact | Value |
|---|---:|
| Rows | 244 |
| Male / female | 157 / 87 |
| Non-smoker / smoker | 151 / 93 |
| Mean total bill | 19.786 |
| Mean tip | 2.998 |
| Correlation between total bill and tip | 0.6757 |
| Coefficient of determination $R^2$ | 0.4566 |
| Least-squares slope | 0.1050 |
| Least-squares intercept | 0.9203 |

The correlation is positive and moderately strong. The slope says that a one-unit increase in the bill is associated with about 0.105 units more tip on average in a simple linear fit. The intercept is the fitted tip when the bill is zero; it is not a meaningful causal statement.

Group means:

| Group | Mean total bill | Mean tip |
|---|---:|---:|
| Male | 20.744 | 3.090 |
| Female | 18.057 | 2.833 |
| Smoker | — | 3.009 |
| Non-smoker | — | 2.992 |

The male-female difference in mean tip is small compared with the spread of tips. The smoker and non-smoker means are almost identical. A colored scatterplot can suggest a group difference, but the numeric summary shows that the difference is modest.

The correlation is similar within each sex:

- male: about 0.670;
- female: about 0.683.

This is a useful check: an overall correlation can hide different within-group patterns.

### 3.6 What a scatterplot can and cannot show

A scatterplot can show:

- direction: positive, negative, or no clear association;
- strength: how tightly points follow a pattern;
- shape: linear, curved, clustered;
- outliers;
- group differences through color, shape, size, or facets;
- overplotting and repeated values.

A scatterplot cannot, by itself, show:

- causation;
- the exact correlation without calculation;
- a relationship that is hidden by a third variable;
- whether a pattern is statistically stable;
- whether the sample is representative.

A scatterplot with many points can also hide density. Use `alpha`, smaller markers, jitter, hexbin, or a two-dimensional density estimate when overplotting is severe.

### 3.7 Correlation is not the whole story

The correlation coefficient is a summary of linear association. Two variables can have a strong nonlinear relationship and a correlation near zero. A single outlier can move the correlation substantially. A restricted range can make a real relationship look weak.

For the tips data, the positive correlation is consistent with the scatterplot, but it does not prove that increasing the bill causes a larger tip. Bill size, party size, day, time, and service quality are entangled.

### 3.8 Official and local differences

| Aspect | Official notebook | Local notebook |
|---|---|---|
| Function | `sns.scatterplot` | `sns.relplot` |
| x/y style | two positional Series | keyword `x`, `y` |
| Grouping | `hue=df["sex"]`, `style=df["smoker"]`, `size=df["size"]` | `hue="sex"`, `style="smoker"`, `size="size"` |
| Facets | none | `row="sex"` |
| Saved output | image exists | no saved output |
| Reproducibility | `df` undefined; old positional API | modern API; output not saved |

The official saved image is evidence that the notebook was run in some environment, but it is not evidence that the visible source runs from a clean kernel.

---

## 4. Categorical-numerical: barplot, boxplot, and catplot

### 4.1 The problem

Suppose we want to compare Age across passenger classes, or Age across sex groups. Age is numeric; `Pclass` and `Sex` are categorical. A scatterplot is possible but not ideal because the x-axis has only a few distinct values. A barplot or boxplot summarizes the numeric distribution within each category.

The key question is whether you want:

- a **summary** of the group, such as the mean;
- the **whole distribution**, including spread and outliers;
- both.

A barplot and a boxplot answer different questions.

### 4.2 Barplot: mean plus uncertainty

The official notebook uses:

```python
sns.barplot(
    titanic["Pclass"],
    titanic["Age"],
    hue=titanic["Sex"],
)
```

The local notebook uses:

```python
sns.catplot(
    data=titanic,
    x="Pclass",
    y="Age",
    kind="bar",
    hue="Sex",
)
```

`catplot(kind="bar")` is the figure-level categorical plot. It draws a bar for the mean Age in each `(Pclass, Sex)` group. The vertical lines are uncertainty intervals, by default a 95% bootstrap confidence interval.

A modern axes-level equivalent is:

```python
sns.barplot(data=titanic, x="Pclass", y="Age", hue="Sex")
```

The bar height is the mean. It is not a count. If you want counts, use `countplot`.

### 4.3 Barplot data facts

Mean Age by passenger class and sex:

| Pclass | Sex | Count | Mean Age | Median Age |
|---:|---|---:|---:|---:|
| 1 | female | 85 | 34.612 | 35.0 |
| 1 | male | 101 | 41.281 | 40.0 |
| 2 | female | 74 | 28.723 | 28.0 |
| 2 | male | 99 | 30.741 | 30.0 |
| 3 | female | 102 | 21.750 | 21.5 |
| 3 | male | 253 | 26.508 | 25.0 |

The pattern is clear:

- first-class passengers were older on average than second-class, who were older than third-class;
- within each class, males were older on average than females;
- the first-class male mean is based on 101 observed ages, not all 179 first-class males, because Age is missing for some passengers.

The barplot's uncertainty intervals are wider for smaller groups. A short bar with a wide interval is less precise than a similar bar with a narrow interval. The interval is not a guarantee that the true mean lies inside it; it is a procedure with a stated confidence level under assumptions.

### 4.4 Boxplot: distribution, quartiles, and outliers

The local notebook uses:

```python
sns.boxplot(data=titanic, x="Pclass", y="Age")
```

The official notebook uses a different comparison:

```python
sns.boxplot(
    titanic["Sex"],
    titanic["Age"],
    hue=titanic["Survived"],
)
```

These are not the same plot. The local plot compares Age across Pclass. The official plot compares Age across Sex, split by survival.

A modern version of the official plot is:

```python
sns.boxplot(data=titanic, x="Sex", y="Age", hue="Survived")
```

A boxplot shows:

- the box from $Q_1$ to $Q_3$;
- the median line;
- whiskers extending to the most extreme observations within the fences;
- individual points beyond the fences as outlier candidates.

The IQR is:

$$
\mathrm{IQR}=Q_3-Q_1.
$$

The common fences are:

$$
Q_1-1.5\,\mathrm{IQR}
\quad\text{and}\quad
Q_3+1.5\,\mathrm{IQR}.
$$

### 4.5 Boxplot data facts

Age by Pclass:

| Pclass | Count | Q1 | Median | Q3 | Min | Max |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 186 | 27.0 | 37.0 | 49.0 | 0.92 | 80.0 |
| 2 | 173 | 23.0 | 29.0 | 36.0 | 0.67 | 70.0 |
| 3 | 355 | 18.0 | 24.0 | 32.0 | 0.42 | 74.0 |

The medians fall from 37 to 29 to 24. The boxes also shift downward. The first-class box is wider and higher; the third-class box is lower and narrower.

Age by sex and survival:

| Sex | Survived | Count | Mean | Median | Std |
|---|---|---:|---:|---:|---:|
| female | 0 | 64 | 25.047 | 24.5 | 13.619 |
| female | 1 | 197 | 28.848 | 28.0 | 14.175 |
| male | 0 | 360 | 31.618 | 29.0 | 14.056 |
| male | 1 | 93 | 27.276 | 28.0 | 16.505 |

Within females, survivors were slightly older on average. Within males, survivors were slightly younger on average. The medians are close, and the distributions overlap heavily. Age alone is not a strong separator.

### 4.6 Barplot versus boxplot

| Question | Barplot | Boxplot |
|---|---|---|
| What is the group mean? | yes | not directly |
| What is the group median? | not by default | yes |
| How spread out is the group? | no | yes, through the box and whiskers |
| Are there outliers? | no | yes, as points |
| Is the shape visible? | no | partially; multimodality is hidden |
| Is uncertainty shown? | yes, by default | no |
| Is it good for many groups? | yes, compact | yes, but can become busy |

A barplot of means can hide a bimodal distribution. A boxplot can hide the exact mean and can hide multimodality. Use both when the decision depends on both center and spread.

### 4.7 `catplot` and faceting

`catplot` is the figure-level interface for categorical plots:

```python
sns.catplot(
    data=titanic,
    x="Pclass",
    y="Age",
    kind="bar",
    hue="Sex",
    height=4,
    aspect=1.2,
)
```

Useful `kind` values:

- `"bar"`: mean plus interval;
- `"box"`: boxplot;
- `"violin"`: violin plot;
- `"count"`: countplot;
- `"point"`: point estimate with interval;
- `"strip"`: strip plot;
- `"swarm"`: swarm plot.

`catplot` returns a `FacetGrid`. It is convenient when you want to split by `row` or `col`, but it is not a drop-in replacement for an axes-level function when you need to customize one axis.

### 4.8 Missing Age and group counts

Age is missing for 177 passengers. The group counts above are counts of observed ages, not counts of all passengers in the group. For example, Pclass 1 has 216 passengers, but only 186 observed ages. The barplot and boxplot silently drop the missing ages.

This matters when comparing groups:

- the group means are based on different denominators;
- missingness may not be random;
- a group with many missing ages has a less precise estimate;
- the visual sample size is not the number of passengers in the dataset.

Always check `count`, not just the bar height.

---

## 5. Numerical-numerical by category: distplot, histplot, and kdeplot

### 5.1 The problem

A barplot compares group means. A boxplot compares quartiles. Sometimes we want to compare the full shape of a numeric distribution across two groups.

The official and local notebooks both use:

```python
sns.distplot(titanic[titanic["Survived"] == 0]["Age"], hist=False)
sns.distplot(titanic[titanic["Survived"] == 1]["Age"], hist=False)
```

This draws two kernel density estimates on the same axes: one for passengers who died, one for passengers who survived. `hist=False` suppresses the histogram and leaves the smooth density curve.

`distplot` is deprecated. The modern replacements are:

```python
sns.kdeplot(data=titanic, x="Age", hue="Survived", common_norm=False)
```

or:

```python
sns.histplot(data=titanic, x="Age", hue="Survived", kde=True)
```

The first is a density comparison. The second is a histogram with a KDE overlay. Both handle the grouping through `hue`.

### 5.2 Data facts

Age by survival, using observed ages only:

| Survived | Count | Mean | Median | Std | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| 0 (died) | 424 | 30.626 | 28.0 | 14.172 | 1.00 | 74.0 |
| 1 (survived) | 290 | 28.344 | 28.0 | 14.951 | 0.42 | 80.0 |

There are 177 missing ages: 125 among passengers who died and 52 among passengers who survived. The density plot uses 714 rows, not 891.

The means differ by about 2.28 years, but the medians are both 28 and the standard deviations are similar. The distributions overlap heavily. A density plot may show a small shift, but it does not show a strong separation.

### 5.3 KDE intuition

A kernel density estimate places a small smooth bump around every observation and averages the bumps. The result is a smooth curve that approximates the distribution.

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

The bandwidth controls smoothness:

- small $h$: spiky, high variance;
- large $h$: oversmooth, high bias;
- moderate $h$: a compromise.

A KDE is an estimate, not the data. A Gaussian KDE can place a small positive density below zero even for a nonnegative variable such as Age. That is a boundary artifact, not evidence of negative ages.

### 5.4 `distplot` versus `kdeplot` versus `histplot`

| Function | Level | What it draws | Status |
|---|---|---|---|
| `sns.distplot` | axes-level | histogram and/or KDE | deprecated |
| `sns.histplot` | axes-level | histogram, optionally with KDE | current |
| `sns.kdeplot` | axes-level | KDE only | current |
| `sns.displot` | figure-level | histogram/KDE with facets | current |

The local notebook uses `distplot` without saving an output. The official notebook uses it with a saved image. In the current environment, `distplot` still works but emits a deprecation warning that says it will be removed in Seaborn v0.14.0.

### 5.5 Reading the Age-by-survival curves

When reading two density curves, ask:

- Where are the peaks?
- How much do the curves overlap?
- Is one curve shifted relative to the other?
- Are the tails different?
- Are the sample sizes similar?
- Are missing values different between groups?

For Titanic Age, the curves overlap substantially. The survived group has a slightly lower mean, but the median is the same. The visual difference is not a strong basis for a rule such as "younger passengers survived." It is a clue to investigate with other variables, not a conclusion.

### 5.6 Modern grouped density code

```python
sns.kdeplot(
    data=titanic,
    x="Age",
    hue="Survived",
    common_norm=False,
    fill=True,
    alpha=0.4,
)
plt.title("Age density by survival")
plt.show()
```

`common_norm=False` normalizes each group separately, so each curve integrates to 1. With `common_norm=True`, the curves are scaled by group size, which can make the larger group dominate visually. Neither choice is universally correct; state which one you used.

---

## 6. Categorical-categorical: crosstab, heatmap, and clustermap

### 6.1 The problem

Suppose we want to know whether survival is associated with passenger class. Both variables are categorical. A scatterplot is meaningless, and a barplot of means is not natural. We need a joint frequency table and a visualization of that table.

`pd.crosstab` counts every combination of two categorical variables. A heatmap colors the cells. A clustermap additionally reorders rows and columns by similarity.

### 6.2 Crosstab counts

```python
grid = pd.crosstab(titanic["Pclass"], titanic["Survived"])
print(grid)
```

The result is:

| Pclass | Died (0) | Survived (1) |
|---:|---:|---:|
| 1 | 80 | 136 |
| 2 | 97 | 87 |
| 3 | 372 | 119 |

The rows are Pclass; the columns are Survived. Each cell is a count. The row totals are 216, 184, and 491. The column totals are 549 and 342. The grand total is 891.

A crosstab is a contingency table. It contains joint frequencies:

- 80 passengers were first class and died;
- 136 were first class and survived;
- 372 were third class and died;
- 119 were third class and survived.

### 6.3 Counts versus rates

Raw counts can be misleading because the row totals differ. First class has 216 passengers; third class has 491. A larger count in third class may simply reflect a larger group.

Row-normalized crosstab:

```python
pd.crosstab(
    titanic["Pclass"],
    titanic["Survived"],
    normalize="index",
)
```

The row percentages are:

| Pclass | Died | Survived |
|---:|---:|---:|
| 1 | 37.04% | 62.96% |
| 2 | 52.72% | 47.28% |
| 3 | 75.76% | 24.24% |

This is the table to read for survival rates by class. The gradient is strong: first class 62.96% survived, second 47.28%, third 24.24%.

Column-normalized crosstab:

| Pclass | Died | Survived |
|---:|---:|---:|
| 1 | 14.57% | 39.77% |
| 2 | 17.67% | 25.44% |
| 3 | 67.76% | 34.80% |

This answers a different question: among those who died, what share were in each class? Among those who survived, what share were in each class? It is not the same as the survival rate within a class.

Always state whether a percentage is row-wise or column-wise.

### 6.4 Heatmap of counts

The official notebook uses:

```python
sns.heatmap(pd.crosstab(titanic["Pclass"], titanic["Survived"]))
```

The local notebook uses:

```python
grid = pd.crosstab(titanic["Pclass"], titanic["Survived"])
sns.heatmap(grid)
```

A heatmap maps each cell value to a color. The axes are the row and column categories. The colorbar shows the scale.

A better version annotates the counts:

```python
sns.heatmap(grid, annot=True, fmt="d", cmap="Blues")
```

- `annot=True` prints the value in each cell.
- `fmt="d"` formats as an integer.
- `cmap` selects the color map.

Without annotation, the reader sees relative color but not exact counts. With annotation, the heatmap becomes a readable table plus a visual gradient.

### 6.5 Heatmap of rates

A count heatmap is dominated by the largest count, which is often the largest group. For Pclass, the third-class died count is 372, much larger than the others. The color scale may make the first-class cells look pale even though first class has the highest survival rate.

A rate heatmap is often better for comparing groups:

```python
rates = pd.crosstab(
    titanic["Pclass"],
    titanic["Survived"],
    normalize="index",
)
sns.heatmap(rates, annot=True, fmt=".2%", cmap="Blues")
```

Now each row sums to 1, and the color represents the within-class proportion. The first-class survival cell is the darkest in the survived column.

### 6.6 Clustermap

A clustermap is a heatmap plus hierarchical clustering of rows and columns:

```python
sns.clustermap(pd.crosstab(titanic["SibSp"], titanic["Survived"]))
```

The official notebook clusters `Parch` by `Survived`:

```python
sns.clustermap(pd.crosstab(titanic["Parch"], titanic["Survived"]))
```

The local notebook clusters `SibSp` by `Survived`:

```python
sns.clustermap(pd.crosstab(titanic["SibSp"], titanic["Survived"]))
```

The dendrograms show which rows and columns are similar according to a distance metric and linkage rule. Seaborn's default is Euclidean distance with average linkage. The rows and columns are reordered to place similar patterns together.

A clustermap does not create new information. It changes the order to make similarity easier to see. With a small table, the clustering can be unstable: a single count can change the dendrogram. Do not overinterpret the branch order.

### 6.7 Crosstab data facts

`SibSp` by `Survived`:

| SibSp | Died | Survived | Group size | Survival rate |
|---:|---:|---:|---:|---:|
| 0 | 398 | 210 | 608 | 34.54% |
| 1 | 97 | 112 | 209 | 53.59% |
| 2 | 15 | 13 | 28 | 46.43% |
| 3 | 12 | 4 | 16 | 25.00% |
| 4 | 15 | 3 | 18 | 16.67% |
| 5 | 5 | 0 | 5 | 0.00% |
| 8 | 7 | 0 | 7 | 0.00% |

`Parch` by `Survived`:

| Parch | Died | Survived | Group size | Survival rate |
|---:|---:|---:|---:|---:|
| 0 | 445 | 233 | 678 | 34.37% |
| 1 | 53 | 65 | 118 | 55.08% |
| 2 | 40 | 40 | 80 | 50.00% |
| 3 | 2 | 3 | 5 | 60.00% |
| 4 | 4 | 0 | 4 | 0.00% |
| 5 | 4 | 1 | 5 | 20.00% |
| 6 | 1 | 0 | 1 | 0.00% |

The largest groups are `SibSp=0` and `Parch=0`. Small groups such as `Parch=3` (5 passengers) and `Parch=6` (1 passenger) have unstable rates. A 60% survival rate based on 5 people is not comparable to a 34% rate based on 678 people.

### 6.8 Syntax breakdown

**`pd.crosstab`**

```python
ct = pd.crosstab(
    titanic["Pclass"],
    titanic["Survived"],
    normalize="index",
    margins=True,
)
```

- First argument: row variable.
- Second argument: column variable.
- `normalize="index"`: row proportions.
- `normalize="columns"`: column proportions.
- `normalize="all"`: grand-total proportions.
- `margins=True`: add row and column totals.
- `dropna=False`: include missing categories.

**`sns.heatmap`**

```python
sns.heatmap(
    ct,
    annot=True,
    fmt=".2%",
    cmap="Blues",
    cbar=True,
    linewidths=0.5,
)
```

- `annot`: print values.
- `fmt`: format string.
- `cmap`: color map.
- `cbar`: show colorbar.
- Returns a Matplotlib `Axes`.

**`sns.clustermap`**

```python
sns.clustermap(
    ct,
    annot=True,
    fmt="d",
    cmap="Blues",
    metric="euclidean",
    method="average",
)
```

- `metric`: distance metric.
- `method`: linkage method.
- Returns a `ClusterGrid`.
- The dendrogram axes are part of the grid, not ordinary data axes.

### 6.9 What a crosstab cannot tell you

A crosstab shows association, not causation. It also does not control for confounding. For example, `Embarked` is associated with survival, but `Embarked` is also associated with `Pclass`. The higher survival rate in Cherbourg may partly reflect the larger share of first-class passengers there.

A crosstab also does not show whether the association is statistically significant. A chi-square test can help, but it has assumptions and does not measure the size of the effect. Part B covers this in more detail.

---

## 7. Multivariate grids: pairplot

### 7.1 The problem

When there are several numeric columns, a single scatterplot is not enough. We want a quick overview of every pair. A `pairplot` creates a grid of scatterplots for every pair of numeric variables, with the same hue applied to all panels.

The official and local notebooks use the iris dataset:

```python
sns.pairplot(iris, hue="species")
```

The iris dataset has 150 rows and 5 columns:

- `sepal_length`;
- `sepal_width`;
- `petal_length`;
- `petal_width`;
- `species`.

There are 50 observations for each of three species: setosa, versicolor, virginica.

### 7.2 What the grid shows

A pairplot with four numeric variables creates a $4\times4$ grid of panels:

- off-diagonal panels: scatterplots for each pair;
- diagonal panels: univariate distributions, usually KDE or histogram;
- hue: species color;
- shared axes by default.

The diagonal is important: it shows the marginal distribution of each variable by species. The off-diagonal panels show pairwise relationships.

### 7.3 Iris data facts

Species means:

| Species | Sepal length | Sepal width | Petal length | Petal width |
|---|---:|---:|---:|---:|
| setosa | 5.006 | 3.428 | 1.462 | 0.246 |
| versicolor | 5.936 | 2.770 | 4.260 | 1.326 |
| virginica | 6.588 | 2.974 | 5.552 | 2.026 |

Ranges:

| Species | Petal length min–max | Petal width min–max |
|---|---|---|
| setosa | 1.0–1.9 | 0.1–0.6 |
| versicolor | 3.0–5.1 | 1.0–1.8 |
| virginica | 4.5–6.9 | 1.4–2.5 |

Setosa is linearly separable from the other two species using petal length or petal width alone. Versicolor and virginica overlap, especially in petal width. Sepal width overlaps heavily across all three species.

This is why the pairplot is useful: it shows that some pairs separate classes strongly and others do not. It also shows that a single feature may not be enough for a multiclass problem.

### 7.4 Syntax breakdown

```python
sns.pairplot(
    data=iris,
    hue="species",
    vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    diag_kind="kde",
    corner=False,
    plot_kws={"alpha": 0.6},
)
```

- `data`: DataFrame.
- `hue`: color grouping.
- `vars`: subset of columns to plot; important for speed.
- `diag_kind`: `"kde"`, `"hist"`, or `None`.
- `corner=True`: plot only the lower triangle; useful for many variables.
- `plot_kws`: pass options to the underlying scatterplot.
- Returns a `PairGrid`.

### 7.5 Cost and limits

A pairplot with $p$ numeric variables creates about $p^2$ panels. For 4 variables, 16 panels. For 15 variables, 225 panels. That is slow and visually overwhelming. Use `vars` to select a subset, or use a correlation matrix and targeted scatterplots.

A pairplot is exploratory. It does not fit a model, does not select features automatically, and does not prove that the classes are separable. It is a way to generate hypotheses.

### 7.6 Pairplot and the target

For a classification problem, a pairplot with `hue=target` can suggest which features separate the classes. But it can also be misleading:

- overlapping classes may still be separable in higher dimensions;
- a clear separation in one pair may disappear after conditioning on another variable;
- the plot does not account for missing values;
- the plot does not account for class imbalance.

Use it as a first look, then confirm with quantitative checks.

---

## 8. Time and ordered numeric: lineplot and flights

### 8.1 The problem

Some numeric relationships have a natural order. Time is the most common example. A scatterplot of year versus passengers is valid, but a lineplot emphasizes the sequence and makes trends easier to see.

The official notebook uses the flights dataset:

```python
flights = sns.load_dataset("flights")
new = flights.groupby("year").sum().reset_index()
sns.lineplot(new["year"], new["passengers"])
```

The flights dataset has 144 rows: 12 years (1949–1960) and 12 months per year. The columns are `year`, `month`, and `passengers`.

### 8.2 Data facts

| Fact | Value |
|---|---:|
| Rows | 144 |
| Years | 1949–1960 |
| Months per year | 12 |
| Total passengers | 40,363 |
| 1949 total | 1,520 |
| 1960 total | 5,714 |
| Growth ratio | 3.759 |
| Minimum monthly value | 104 (1949 November) |
| Maximum monthly value | 622 (1960 July) |

Yearly totals rise every year in this dataset, from 1,520 to 5,714. That is a strong upward trend.

Monthly seasonality:

| Month | Total passengers |
|---|---:|
| Jan | 2,901 |
| Feb | 2,820 |
| Mar | 3,242 |
| Apr | 3,205 |
| May | 3,262 |
| Jun | 3,740 |
| Jul | 4,216 |
| Aug | 4,213 |
| Sep | 3,629 |
| Oct | 3,199 |
| Nov | 2,794 |
| Dec | 3,142 |

July and August are the peak months; February and November are the lowest. The lineplot of yearly totals hides this seasonality because it aggregates over months. The pivot heatmap of month by year reveals both the long-term growth and the recurring seasonal pattern.

### 8.3 The official code is not reproducible in the current environment

`flights.groupby("year").sum()` fails in pandas 3.0 because `month` is a categorical column and category types do not support sum operations. The saved official output came from an older pandas version.

A modern version selects the numeric column explicitly:

```python
new = (
    flights.groupby("year", as_index=False)["passengers"]
    .sum()
    .reset_index(drop=True)
)
```

or:

```python
new = flights.groupby("year", as_index=False).agg(
    passengers=("passengers", "sum")
)
```

The lineplot call is also old-style:

```python
sns.lineplot(new["year"], new["passengers"])
```

Current Seaborn expects:

```python
sns.lineplot(data=new, x="year", y="passengers", marker="o")
```

### 8.4 Lineplot interpretation

When reading a lineplot, ask:

- Is the x-axis ordered?
- Is there a trend?
- Is there seasonality?
- Are there level shifts or breaks?
- Are the points equally spaced?
- Is the line connecting observations that should be connected?
- Is there a third variable that explains the pattern?

A lineplot should not be used for unordered categories. Connecting points across `Embarked` categories, for example, implies an order that does not exist. Use a bar chart or point plot for unordered categories.

### 8.5 Flights clustermap

The official notebook also uses:

```python
sns.clustermap(
    flights.pivot_table(
        values="passengers",
        index="month",
        columns="year",
    )
)
```

The pivot table has 12 rows (months) and 12 columns (years). The clustermap reorders months and years by similarity. The strong year trend dominates the color scale: later years are darker across all months. The dendrogram may group summer months together and winter months together.

This is a useful multivariate view: it shows trend, seasonality, and similarity at once. It is still exploratory, not a model.

---

## 9. Pivot-table heatmap: a matrix, not a line plot

### 9.1 The local notebook code

The local notebook uses:

```python
sns.heatmap(
    titanic.pivot_table(
        values="Pclass",
        index="Age",
        columns="SibSp",
    ),
    cmap="viridis",
)
```

The markdown header calls this a "Pivot-table heatmap (a numeric grid, not a line plot)" and notes that the original header called it a line plot. The code is a heatmap, not a line plot.

### 9.2 What the pivot table contains

`pivot_table` groups by `Age` and `SibSp`, then aggregates `Pclass` with the default `mean`. The result is a matrix:

- rows: exact Age values;
- columns: SibSp values;
- cells: mean Pclass for passengers with that exact Age and SibSp combination.

The local result has shape **88 × 6**, not 80 × 7 or 891 × 7. There are 88 distinct observed ages. The columns are SibSp 0 through 5. SibSp 8 is absent because all seven passengers with SibSp 8 have missing Age, so that column is entirely NaN and is dropped by the default `dropna=True`.

Only **185 of 528 cells** are non-null: about 35.0%. The matrix is mostly empty. Exact Age is a very fine index; most Age–SibSp combinations contain no passengers.

The non-null count by SibSp column:

| SibSp | Non-null cells |
|---:|---:|
| 0 | 81 |
| 1 | 57 |
| 2 | 18 |
| 3 | 11 |
| 4 | 13 |
| 5 | 5 |

The cell values range from 1 to 3, because Pclass is coded 1, 2, 3. A cell value of 1.67 is a mean of Pclass codes, not a meaningful "one and two-thirds class" passenger.

### 9.3 Why this is a weak visualization

There are three problems:

1. **Sparsity.** Most cells are empty. The heatmap is mostly blank, and the non-empty cells are scattered.
2. **Exact Age as an index.** Age is continuous. Using every distinct age creates a huge, sparse grid. Binning Age would produce a more interpretable matrix.
3. **Averaging an ordinal category.** Pclass is ordinal. A mean of 1, 2, and 3 is arithmetically possible but conceptually weak. A survival rate or a count would be more meaningful.

A better question is: "How does survival rate vary by Age group and SibSp?" That suggests binning Age and aggregating `Survived`:

```python
titanic["AgeGroup"] = pd.cut(
    titanic["Age"],
    bins=[0, 12, 18, 30, 45, 60, 100],
    labels=["0-12", "13-18", "19-30", "31-45", "46-60", "60+"],
)

age_sibsp = titanic.pivot_table(
    values="Survived",
    index="AgeGroup",
    columns="SibSp",
    aggfunc="mean",
)

sns.heatmap(age_sibsp, annot=True, fmt=".2f", cmap="Blues")
```

This matrix is smaller, each cell has a clearer denominator, and the values are survival rates rather than mean class codes.

### 9.4 `pivot_table` syntax

```python
table = df.pivot_table(
    values="Survived",
    index="AgeGroup",
    columns="SibSp",
    aggfunc="mean",
    fill_value=0,
    margins=True,
)
```

- `values`: the column to aggregate.
- `index`: row grouping.
- `columns`: column grouping.
- `aggfunc`: `"mean"`, `"sum"`, `"count"`, `"median"`, or a callable.
- `fill_value`: value for missing cells.
- `margins=True`: add totals.
- `dropna=True` by default: drop columns whose entries are all NaN.

`pivot_table` is powerful, but it does not choose a meaningful aggregation for you. The analyst must decide whether a mean, count, sum, or rate is appropriate.

### 9.5 The lesson

The original header said "LinePlot," but the code drew a heatmap. The current local header documents that correction. The lesson is not to trust labels. Read the code:

- `pivot_table` creates a matrix;
- `heatmap` draws a matrix;
- `lineplot` would draw a line.

The notebook intentionally records the original mislabel as a debugging lesson. In your own work, keep the corrected header so it matches the code.

---

## 10. Matplotlib and Seaborn internals

### 10.1 Figure, Axes, and Artists

Matplotlib separates the canvas from the plotting regions:

```text
Figure
  └── Axes
        ├── Axis (x)
        ├── Axis (y)
        └── Artists (points, lines, patches, text)
```

- **Figure:** the entire canvas.
- **Axes:** one plotting region with data coordinates.
- **Axis:** the ticks, labels, and scale for one dimension.
- **Artist:** anything drawn.

A figure can contain multiple axes. A figure-level Seaborn function creates and manages those axes for you.

### 10.2 Pyplot style versus object-oriented style

Pyplot style:

```python
plt.scatter(tips["total_bill"], tips["tip"])
plt.title("Tip versus total bill")
plt.show()
```

Object-oriented style:

```python
fig, ax = plt.subplots()
ax.scatter(tips["total_bill"], tips["tip"])
ax.set_title("Tip versus total bill")
plt.show()
```

The object-oriented style is clearer when there are multiple axes or reusable functions. Seaborn axes-level functions accept `ax=` so they can draw on a specific axis.

### 10.3 Data flow through Seaborn

```text
DataFrame
   |
   v
Seaborn function
   |
   +--> validate and drop missing rows
   |
   +--> map columns to x, y, hue, style, size, row, col
   |
   +--> create FacetGrid / PairGrid / Axes / ClusterGrid
   |
   +--> draw marks
   |
   v
Figure with axes, legends, colorbar, dendrograms
```

The mapping is declarative. You do not write the loop; Seaborn groups the data and draws each group.

### 10.4 Return types

| Call | Typical return |
|---|---|
| `sns.scatterplot` | `Axes` |
| `sns.barplot` | `Axes` |
| `sns.boxplot` | `Axes` |
| `sns.histplot` | `Axes` |
| `sns.kdeplot` | `Axes` |
| `sns.heatmap` | `Axes` |
| `sns.lineplot` | `Axes` |
| `sns.relplot` | `FacetGrid` |
| `sns.catplot` | `FacetGrid` |
| `sns.pairplot` | `PairGrid` |
| `sns.clustermap` | `ClusterGrid` |
| `pd.crosstab` | DataFrame |
| `df.pivot_table` | DataFrame |

This table matters when you want to customize a plot. A `FacetGrid` is not an `Axes`; it contains axes.

### 10.5 Categorical order and missing values

Seaborn respects pandas categorical order when the column is categorical. Otherwise, it often uses the order of appearance or an internal sort. Use `order=` to make the intended order explicit:

```python
sns.boxplot(
    data=titanic,
    x="Pclass",
    y="Age",
    order=[1, 2, 3],
)
```

For missing values, Seaborn generally drops rows with missing values in the mapped columns. A crosstab excludes missing categories by default; use `dropna=False` to include them. A countplot excludes missing categories by default.

### 10.6 Common execution traps

**Bare function reference**

```python
sns.scatterplot
```

This evaluates the function object and draws nothing. You must call it with data.

**Old positional API**

```python
sns.barplot(titanic["Pclass"], titanic["Age"])
```

Current Seaborn expects keyword-only `x` and `y`. The old call raises `TypeError`.

**Undefined variable**

```python
hue=df["sex"]
```

If `df` is not defined, the cell raises `NameError`. The official scatterplot cell has this bug even though a saved image exists.

**Typo in a column name**

```python
titanic["Sibsp"]
```

The column is `SibSp`, not `Sibsp`. Pandas raises `KeyError`. The local clustermap cell contains this typo in its second call.

**Duplicate call**

The local clustermap cell calls `sns.clustermap` twice. The first call draws a figure; the second call is a duplicate with a typo. Even after fixing the typo, the duplicate is unnecessary.

**Stale outputs**

A saved output is not proof that the current source runs. The official scatterplot has a saved image but an undefined `df`. The local clustermap has a saved image but a typo in the visible source. Treat outputs as evidence about a past run, not a guarantee about the current notebook.

---

## 11. Actual notebook results

### 11.1 Official notebook: 33 cells

The official `reference_notebooks/day21.ipynb` has 33 cells: 25 code cells and 8 markdown cells.

| Cell | Type | Content | Saved result |
|---:|---|---|---|
| 0 | code | imports pandas, seaborn | none |
| 1 | code | load tips | none |
| 2 | code | `tips.head()` | table |
| 3 | code | read `train.csv` | none |
| 4 | code | load flights | none |
| 5 | code | `flights.head()` | table |
| 6 | code | load iris | none |
| 7 | code | `iris` | table |
| 8 | markdown | Scatterplot heading | none |
| 9 | code | `sns.scatterplot(...)` with `df` | axes + image |
| 10 | markdown | Bar Plot heading | none |
| 11 | code | `titanic.head()` | table |
| 12 | code | `sns.barplot(...)` positional | axes + image |
| 13 | markdown | Box Plot heading | none |
| 14 | code | `sns.boxplot(...)` positional | axes + image |
| 15 | markdown | Distplot heading | none |
| 16 | code | two `sns.distplot` calls | axes + image |
| 17 | markdown | HeatMap heading | none |
| 18 | code | `titanic.head(3)` | table |
| 19 | code | heatmap of Pclass × Survived | axes + image |
| 20 | code | groupby Embarked mean survival | percentages |
| 21 | markdown | ClusterMap heading | none |
| 22 | code | clustermap Parch × Survived | grid + image |
| 23 | markdown | Pairplot heading | none |
| 24 | code | `iris.head()` | table |
| 25 | code | `sns.pairplot(iris, hue="species")` | grid + image |
| 26 | markdown | Lineplot heading | none |
| 27 | code | `flights.head()` | table |
| 28 | code | `new = flights.groupby("year").sum()` | none |
| 29 | code | `sns.lineplot(new["year"], new["passengers"])` | axes + image |
| 30 | code | `flights` | table |
| 31 | code | clustermap flights pivot | grid + image |
| 32 | code | empty | none |

The official notebook is a sequence of plot-type demonstrations:

1. scatterplot: tips;
2. barplot: Pclass × Age, split by Sex;
3. boxplot: Sex × Age, split by Survived;
4. distplot: Age by Survived;
5. heatmap: Pclass × Survived;
6. groupby: survival rate by Embarked;
7. clustermap: Parch × Survived;
8. pairplot: iris by species;
9. lineplot: flights by year;
10. clustermap: flights month × year.

It covers more plot types than the local notebook, but several cells use old APIs that do not run in the current environment.

### 11.2 Official saved outputs

| Cell | Output |
|---:|---|
| 2 | tips head: 244 rows, 7 columns |
| 5 | flights head: 144 rows, 3 columns; saved month names are full names |
| 7 | iris: 150 rows, 5 columns |
| 9 | scatterplot image, 432 × 288 |
| 11 | Titanic head |
| 12 | barplot image, 432 × 288 |
| 14 | boxplot image, 432 × 288 |
| 16 | distplot image, 432 × 288 |
| 18 | Titanic head(3) |
| 19 | heatmap image, 432 × 288, 2 axes |
| 20 | Embarked survival percentages |
| 22 | clustermap Parch image, 720 × 720 |
| 24 | iris head |
| 25 | pairplot image, 804.75 × 720 |
| 27 | flights head |
| 29 | lineplot image, 432 × 288 |
| 30 | flights table |
| 31 | clustermap flights image, 720 × 720 |

The groupby Embarked output is:

```text
C    55.357143
Q    38.961039
S    33.695652
Name: Survived, dtype: float64
```

The saved outputs use old Matplotlib axes representations such as `<matplotlib.axes._subplots.AxesSubplot at 0x...>`. Current Matplotlib uses a different display string; the object is still an `Axes`.

### 11.3 Official source-fidelity issues

The official notebook is not fully reproducible from a clean kernel in the current environment.

**Undefined `df`**

Cell 9 uses `df["sex"]`, `df["smoker"]`, and `df["size"]`, but no cell defines `df`. The saved image must have come from a kernel where `df` existed. A clean run raises `NameError`.

**Old positional APIs**

- `sns.scatterplot(tips["total_bill"], tips["tip"], ...)` uses two positional arguments. Current Seaborn allows at most one positional `data` argument.
- `sns.barplot(titanic["Pclass"], titanic["Age"], ...)` uses two positional arguments. Current Seaborn raises `TypeError`.
- `sns.boxplot(titanic["Sex"], titanic["Age"], ...)` uses two positional arguments. Current Seaborn raises `TypeError`.
- `sns.lineplot(new["year"], new["passengers"])` uses two positional arguments. Current Seaborn raises `TypeError`.

**Pandas 3.0 differences**

- `titanic.groupby("Embarked").mean()` fails because string columns do not support `mean`. Use `groupby("Embarked", observed=True)["Survived"].mean()`.
- `flights.groupby("year").sum()` fails because `month` is a categorical column and category types do not support sum. Select `passengers` explicitly.

**Deprecated `distplot`**

`sns.distplot` still exists in Seaborn 0.13.2 but is deprecated and will be removed in v0.14.0.

**Dataset path**

`pd.read_csv("train.csv")` is relative to the working directory. The dataset is under `datasets/`, so the official notebook needs an adjusted path or a working directory containing `train.csv`.

**Month labels**

The saved flights output uses full month names (`January`, `February`, ...). Current Seaborn loads abbreviations (`Jan`, `Feb`, ...). This is a version difference, not a data change.

### 11.4 Local notebook: 26 cells

The local `my_notebook.ipynb` has 26 cells: 13 code cells and 13 markdown cells.

| Cell | Type | Content | Saved result |
|---:|---|---|---|
| 0 | markdown | document banner | none |
| 1 | code | imports pandas, matplotlib, seaborn | none |
| 2 | code | load iris | none |
| 3 | markdown | plot-choice explanation | none |
| 4 | code | load tips; display tips | none |
| 5 | markdown | relplot explanation | none |
| 6 | code | tips relplot | no saved output |
| 7 | code | read Titanic; head | none |
| 8 | markdown | catplot/boxplot explanation | none |
| 9 | code | catplot Pclass × Age, hue Sex | no saved output |
| 10 | markdown | boxplot explanation | none |
| 11 | code | boxplot Pclass × Age | no saved output |
| 12 | code | two distplot calls | no saved output |
| 13 | markdown | heatmap explanation | none |
| 14 | markdown | cat-cat explanation; notes typos | none |
| 15 | code | crosstab heatmap | no saved output |
| 16 | markdown | clustermap explanation | none |
| 17 | code | clustermap SibSp; duplicate with `Sibsp` typo | image only |
| 18 | markdown | pairplot explanation | none |
| 19 | code | iris head | table |
| 20 | code | pairplot iris | grid + image |
| 21 | markdown | pivot heatmap explanation | none |
| 22 | code | pivot-table heatmap | image |
| 23 | markdown | drop duplicate note | none |
| 24 | markdown | experiment prompt | none |
| 25 | markdown | key takeaways | none |

The local notebook has no saved outputs for the relplot, catplot, boxplot, distplot, or first heatmap cells. It is a working notebook whose outputs are partial. The only saved outputs are:

- cell 17: clustermap image;
- cell 19: iris head table;
- cell 20: pairplot image;
- cell 22: pivot heatmap image.

### 11.5 Local source-fidelity issues

**Absolute Titanic path**

The local notebook reads:

```python
titanic = pd.read_csv(r"E:/MACHINE_LEARNING/train.csv")
```

That root file exists and is byte-identical to `Day_21/datasets/train.csv`, but the path is not portable. Use `pd.read_csv("datasets/train.csv")` when running from `Day_21/`.

**Clustermap typo and duplicate**

The local cell 17 contains:

```python
sns.clustermap(pd.crosstab(titanic["SibSp"], titanic["Survived"]))
sns.clustermap(pd.crosstab(titanic["Sibsp"], titanic["Survived"]))
plt.show()
```

`Sibsp` is not a column; pandas raises `KeyError`. The cell has a saved image but no saved error, so the visible output is incomplete or stale. The second call is also a duplicate.

**Pivot heatmap is sparse**

The pivot table uses exact Age as the row index. It produces an 88 × 6 matrix with only 185 non-null cells. Most cells are empty. The mean of Pclass is not a meaningful continuous summary. The header originally called this a line plot, but the code is a heatmap.

**No saved outputs for most plots**

The local notebook does not save the relplot, catplot, boxplot, distplot, or heatmap outputs. Do not claim to have seen patterns that are not saved; use the data to compute the facts and label them as computed.

### 11.6 Local saved outputs

| Cell | Output | Size |
|---:|---|---|
| 17 | clustermap SibSp image | 1000 × 1000, 4 axes |
| 19 | iris head table | 5 rows × 5 columns |
| 20 | pairplot iris image | 1120 × 1000, saved text reports 20 axes |
| 22 | pivot heatmap image | 640 × 480, 2 axes |

The pairplot saved text says 20 axes; current Seaborn creates a 4 × 4 grid of 16 axes for the four numeric iris columns. The difference is version/display metadata, not a different dataset.

### 11.7 What can and cannot be concluded from the notebooks

Can conclude:

- the official notebook demonstrates the plot types;
- the local notebook practices a subset and adds the pivot heatmap;
- the Titanic crosstabs and groupby rates are computable from the dataset;
- the iris petal measurements separate setosa;
- the flights data has a strong upward trend and seasonality;
- several official cells are not reproducible in the current environment.

Cannot conclude:

- that the official saved images were produced by the visible source in a clean kernel;
- that the local clustermap typo is harmless;
- that the pivot heatmap is a good visualization;
- that any of these associations are causal;
- that a model trained on these features would achieve a particular accuracy.

---

## 12. Workflow connections

### 12.1 Day 20 to Day 21

Day 20 produced:

- category counts and shares;
- histograms, KDEs, and boxplots for Age and Fare;
- skewness and outlier candidates;
- missingness and denominators.

Day 21 adds a second variable:

- survival by sex and class;
- Age by survival;
- tip versus total bill;
- iris species separation;
- flights trend and seasonality;
- crosstab and pivot-table matrices.

The movement is from "what does one column look like?" to "how do two or more columns move together?"

### 12.2 Day 21 to Day 22

Day 22 introduces automated profiling. The plots and tables from Day 21 are the manual version of what a profiling report summarizes. Understanding the manual version is what lets you read the automated report critically.

### 12.3 Day 21 to preprocessing

| Finding | Possible later action |
|---|---|
| Sex strongly associated with survival | encode as binary feature |
| Pclass ordinal and associated with survival | ordinal encoding or one-hot |
| Embarked associated with survival but confounded | encode, then validate with CV |
| Age missing 177 | impute; consider missing indicator |
| Age weak marginal association | interactions with Sex/Pclass may matter |
| Fare right-skewed | log or power transform |
| SibSp/Parch small groups | combine rare categories |
| Iris petal features separate setosa | feature selection and scaling |
| Flights trend/seasonality | time-aware validation, not random split |

These are hypotheses, not automatic commands. The model, validation strategy, and domain knowledge decide.

### 12.4 Leakage discipline

Exploratory plots may use the full dataset to understand it. But:

- imputation values must be learned from training data only;
- scaling parameters must be learned from training data only;
- encoding categories must be based on training data only;
- feature-selection decisions must not use the test target;
- target-rate plots on the test set can leak information if used to choose features.

Day 21 itself does not fit preprocessing. It identifies what might need to be fitted later.

### 12.5 Association is not causation

The Titanic data is observational. Passengers were not randomly assigned to class, sex, or port. A strong association between class and survival may reflect other variables:

- first-class passengers may have had better access to lifeboats;
- women and children may have been prioritized;
- port of embarkation is associated with class;
- family size is associated with class and age.

A model can exploit associations without understanding causation. That is useful for prediction but dangerous for policy. State the distinction clearly.

---

## 13. Common mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Using a barplot for counts | bar heights are means, not counts | use `countplot` |
| Reading a count heatmap as a rate heatmap | large groups dominate color | normalize rows |
| Forgetting the denominator | a 60% rate based on 5 people looks strong | report group size |
| Treating Pclass as continuous | "average class" is meaningless | treat as ordinal/category |
| Using a lineplot for unordered categories | implies false order | use bar/point plot |
| Using a scatterplot for categorical x | overplotting at a few x values | use boxplot/barplot |
| Overloading hue/style/size/row/col | unreadable plot | use fewer channels or facets |
| Using a pairplot on all columns | slow and cluttered | pass `vars` |
| Ignoring missing values | group counts differ from row counts | report observed n |
| Trusting a saved output | source may not run | rerun in a clean kernel |
| Copying old positional Seaborn calls | `TypeError` | use `data=`, `x=`, `y=` |
| Using `distplot` | deprecation warning | use `histplot`/`kdeplot` |
| Typo in a column name | `KeyError` | check exact column spelling |
| Duplicate plot calls | multiple figures | remove duplicates |
| Interpreting a clustermap order as causal | dendrogram is similarity, not causation | treat as exploratory |
| Averaging an ordinal code | 1.67 "class" is meaningless | use rates/counts |
| Using exact Age as a pivot index | sparse, mostly empty matrix | bin Age |
| Concluding causation from a crosstab | observational association | use causal design |

---

## 14. Essential interview and examination preparation

### 14.1 Level 1 - definitions

**Q1. What is bivariate analysis?**

Analysis of two variables together, looking for association, group differences, or joint frequencies.

**Q2. What is multivariate analysis?**

Analysis of three or more variables together, often through color, shape, size, faceting, pairplots, or matrix summaries.

**Q3. What is a scatterplot?**

A plot with one point per observation at `(x, y)`, used to show the relationship between two numeric variables.

**Q4. What is a barplot?**

A categorical plot whose bar height is an aggregated numeric value, usually the mean, often with an uncertainty interval.

**Q5. What is a boxplot?**

A plot of the five-number summary: lower whisker, Q1, median, Q3, upper whisker, with outlier candidates beyond the fences.

**Q6. What is a crosstab?**

A frequency table of counts for every combination of two categorical variables.

**Q7. What is a heatmap?**

A plot that maps matrix values to colors.

**Q8. What is a clustermap?**

A heatmap with hierarchical clustering and dendrograms that reorder rows and columns by similarity.

**Q9. What is a pairplot?**

A grid of scatterplots and univariate distributions for every pair of numeric variables.

**Q10. What is a lineplot?**

A plot that connects ordered observations, usually over time, to show trend and seasonality.

### 14.2 Level 2 - mechanics

**Q11. Which plot for numeric versus numeric?**

Scatterplot or `relplot`.

**Q12. Which plot for categorical versus numeric?**

Boxplot for distribution, barplot for means, violin for shape.

**Q13. Which plot for categorical versus categorical?**

Crosstab plus heatmap, grouped countplot, or mosaic plot.

**Q14. What does `hue` do?**

Maps a variable to color.

**Q15. What does `style` do?**

Maps a variable to marker shape.

**Q16. What does `size` do?**

Maps a variable to marker size.

**Q17. What does `row` do?**

Creates facet rows, one subplot row per category.

**Q18. What does `col` do?**

Creates facet columns, one subplot column per category.

**Q19. What is the difference between `relplot` and `scatterplot`?**

`relplot` is figure-level and supports facets; `scatterplot` is axes-level and draws on one `Axes`.

**Q20. What is the difference between `catplot` and `boxplot`?**

`catplot` is figure-level and supports facets; `boxplot` is axes-level.

**Q21. What does `normalize="index"` do in `crosstab`?**

Divides each row by its row total, producing row proportions.

**Q22. What does `annot=True` do in `heatmap`?**

Prints the value in each cell.

**Q23. What does `pairplot(hue=...)` do?**

Colors every panel by the hue variable and adds a legend.

**Q24. What does `lineplot` do with multiple y values per x?**

By default, it aggregates them and shows a mean with a confidence interval.

### 14.3 Level 3 - applied and numerical

**Q25. Pclass 1 has 136 survivors and 80 deaths. What is the survival rate?**

$$
\frac{136}{216}=0.62963=62.96\%.
$$

**Q26. Pclass 3 has 119 survivors and 372 deaths. What is the survival rate?**

$$
\frac{119}{491}=0.24236=24.24\%.
$$

**Q27. What is the survival rate for females?**

$$
\frac{233}{314}=0.74204=74.20\%.
$$

**Q28. What is the survival rate for males?**

$$
\frac{109}{577}=0.18891=18.89\%.
$$

**Q29. What is the odds ratio for female versus male survival?**

$$
\frac{233\times468}{81\times109}\approx12.35.
$$

**Q30. What is the risk ratio for female versus male survival?**

$$
\frac{233/314}{109/577}\approx3.93.
$$

**Q31. What is the Embarked survival rate for Cherbourg?**

$$
\frac{93}{168}=0.55357=55.36\%.
$$

**Q32. What is the Embarked survival rate for Southampton?**

$$
\frac{217}{644}=0.33696=33.70\%.
$$

**Q33. What is the correlation between total bill and tip?**

Approximately 0.6757. The coefficient of determination is about 0.4566.

**Q34. What is the slope of the simple linear fit for tip versus total bill?**

Approximately 0.1050. A one-unit increase in bill is associated with about 0.105 units more tip on average.

**Q35. What is the flights growth ratio from 1949 to 1960?**

$$
\frac{5714}{1520}\approx3.759.
$$

**Q36. What are the peak and low months in flights?**

July and August are the peak months; February and November are the lowest.

**Q37. What is the shape of the local pivot table?**

88 rows × 6 columns, with 185 non-null cells out of 528.

**Q38. Why is SibSp 8 missing from the local pivot table?**

All seven passengers with SibSp 8 have missing Age, so the column is entirely NaN and is dropped by default.

**Q39. How many non-null cells are in the local pivot table?**

185 of 528, about 35.0%.

**Q40. Why is the local pivot table sparse?**

It uses exact Age as the row index, creating 88 distinct rows, and only some Age–SibSp combinations contain passengers.

### 14.4 Level 4 - debugging and scenarios

**Q41. Why does `sns.barplot(titanic["Pclass"], titanic["Age"])` fail?**

Current Seaborn expects `data=`, `x=`, and `y=` keyword arguments; two positional arguments are not allowed.

**Q42. Why does `titanic.groupby("Embarked").mean()` fail in pandas 3.0?**

The string columns do not support `mean`. Select the numeric column first.

**Q43. Why does `flights.groupby("year").sum()` fail in pandas 3.0?**

`month` is a categorical column and category types do not support sum. Select `passengers` explicitly.

**Q44. Why does the official scatterplot cell fail even though it has a saved image?**

It uses an undefined `df` and old positional arguments. The saved image came from a different kernel state.

**Q45. Why does the local clustermap cell have an image but a typo?**

The visible source contains `Sibsp`, but the saved output does not include the `KeyError`. The output is stale or incomplete.

**Q46. Why is a count heatmap misleading for Pclass survival?**

The third-class death count is much larger, so it dominates the color scale. Normalize rows to compare rates.

**Q47. A barplot shows a group mean of 60 with a wide confidence interval. What does that mean?**

The estimate is imprecise; the group may be small or variable. Do not compare it naively with a narrow-interval group.

**Q48. A scatterplot shows a strong positive association. Can you conclude causation?**

No. Association can be confounded, reversed, or spurious.

**Q49. A pairplot shows overlapping classes. Does that mean the features are useless?**

No. The classes may separate in higher dimensions or with a nonlinear model. The pairplot is only a pairwise view.

**Q50. A clustermap reorders the rows. What does the order mean?**

It groups similar rows according to the chosen distance and linkage. It is not a causal or temporal order.

### 14.5 Worked code-reading questions

**Q51. What is the output?**

```python
pd.crosstab(titanic["Pclass"], titanic["Survived"])
```

A 3 × 2 table with counts: `[[80, 136], [97, 87], [372, 119]]`.

**Q52. What is the output?**

```python
pd.crosstab(
    titanic["Pclass"],
    titanic["Survived"],
    normalize="index",
).round(3)
```

Row proportions: first class `[0.370, 0.630]`, second `[0.527, 0.473]`, third `[0.758, 0.242]`.

**Q53. What is wrong with this code?**

```python
sns.boxplot(titanic["Sex"], titanic["Age"])
```

It uses old positional arguments. Use `data=titanic, x="Sex", y="Age"`.

**Q54. What is wrong with this code?**

```python
sns.distplot(titanic["Age"])
```

`distplot` is deprecated. Use `histplot(kde=True)` or `kdeplot`.

**Q55. What is wrong with this code?**

```python
titanic.pivot_table(
    values="Pclass",
    index="Age",
    columns="SibSp",
)
```

It averages an ordinal code over exact Age, producing a sparse and conceptually weak matrix. Bin Age and use a rate or count.

**Q56. What is wrong with this conclusion?**

"First-class passengers survived because they were first class."

It is an associational statement. Class may be confounded with sex, age, cabin location, and access to lifeboats.

**Q57. What is wrong with this conclusion?**

"The pairplot shows petal length separates all three species."

It separates setosa from the other two, but versicolor and virginica overlap.

**Q58. What is wrong with this conclusion?**

"The lineplot shows passengers increased every month."

The lineplot aggregates by year. The monthly pattern has seasonality; the yearly trend is upward, but not every month increases monotonically.

### 14.6 Exam-style theory with answers

**Question 1. Why is bivariate analysis necessary after univariate analysis?**

Because univariate summaries cannot show relationships. A feature may look unremarkable alone but separate classes when paired with another variable. Bivariate analysis generates feature hypotheses and reveals group differences.

**Question 2. Why can a barplot and a boxplot disagree?**

They summarize different aspects. A barplot shows the mean and uncertainty; a boxplot shows quartiles, spread, and outliers. A skewed group can have a mean far from its median, and a bimodal group can have a box that hides the two modes.

**Question 3. Why is a count heatmap often a poor choice for comparing rates?**

Color is dominated by the largest counts, which often come from the largest groups. Normalizing rows or columns converts counts to proportions and makes comparisons fairer.

**Question 4. Why is a crosstab not enough to establish independence?**

A crosstab shows observed counts. Independence is a model-based statement about expected counts. A chi-square test compares observed and expected counts, but it has assumptions and does not measure effect size.

**Question 5. Why is a clustermap not a clustering model?**

A clustermap performs hierarchical clustering for display. It does not produce a predictive model, does not choose the number of clusters, and does not validate cluster stability.

**Question 6. Why is a pairplot exploratory rather than confirmatory?**

It shows pairwise relationships without controlling for other variables, without inference, and without a model. It can generate hypotheses but cannot confirm them.

**Question 7. Why is a lineplot inappropriate for unordered categories?**

A line implies continuity and order. Connecting unordered categories suggests a sequence that does not exist.

**Question 8. Why is averaging Pclass questionable?**

Pclass is ordinal. The arithmetic mean of codes 1, 2, and 3 is computable, but it does not correspond to a meaningful passenger class. Rates or counts are more interpretable.

**Question 9. Why does missing Age change the interpretation of Age plots?**

The plots use 714 observed ages, not 891 passengers. If missingness is related to survival or class, the observed distribution may be biased. Report the denominator and consider a missing indicator.

**Question 10. Why can a saved notebook output be misleading?**

The output may come from a different kernel state, a different library version, or a different code revision. The visible source may not reproduce it. Always rerun in a clean kernel when the result matters.

---

## 15. Revision summary

### 15.1 Plot choice

| Data types / question | Default plot |
|---|---|
| numeric × numeric | scatterplot, `relplot` |
| categorical × numeric, center | barplot |
| categorical × numeric, distribution | boxplot or violin |
| categorical × categorical, counts | crosstab + heatmap |
| categorical × categorical, rates | row-normalized crosstab + heatmap |
| many numeric columns | pairplot with `vars` |
| time × numeric | lineplot |
| two grouping columns + aggregate | pivot-table heatmap |
| numeric × numeric + categories | `relplot` with hue/style/size/facets |

### 15.2 Titanic numbers

| Fact | Value |
|---|---:|
| Shape | 891 × 12 |
| Age missing | 177 |
| Fare missing | 0 |
| Embarked missing | 2 |
| Sex | male 577, female 314 |
| Survived | died 549, survived 342 |
| Pclass | 1: 216, 2: 184, 3: 491 |
| Embarked | S: 644, C: 168, Q: 77 |
| Survival by Pclass | 62.96%, 47.28%, 24.24% |
| Survival by Sex | female 74.20%, male 18.89% |
| Survival by Embarked | C 55.36%, Q 38.96%, S 33.70% |
| Age by survival | died mean 30.63, survived mean 28.34; medians 28 |
| Pclass 1 age median | 37 |
| Pclass 2 age median | 29 |
| Pclass 3 age median | 24 |

### 15.3 Other dataset facts

| Dataset | Fact | Value |
|---|---|---:|
| tips | rows | 244 |
| tips | male / female | 157 / 87 |
| tips | non-smoker / smoker | 151 / 93 |
| tips | correlation total_bill–tip | 0.6757 |
| tips | $R^2$ | 0.4566 |
| tips | slope | 0.1050 |
| iris | rows | 150 |
| iris | species counts | 50 each |
| iris | setosa petal length | 1.0–1.9 |
| iris | versicolor petal length | 3.0–5.1 |
| iris | virginica petal length | 4.5–6.9 |
| flights | rows | 144 |
| flights | years | 1949–1960 |
| flights | total passengers | 40,363 |
| flights | 1949 / 1960 totals | 1,520 / 5,714 |
| flights | growth ratio | 3.759 |
| flights | min / max month | 104 / 622 |

### 15.4 The three warnings

1. Association is not causation.
2. A plot type is not a substitute for a denominator; counts, rates, and means answer different questions.
3. A saved output is not proof that the visible source runs in the current environment.

### 15.5 The three decisions this day enables

1. Sex, Pclass, and Embarked are strong survival correlates and should be encoded carefully later.
2. Age and Fare need missing-value and skew treatment before modeling.
3. Pairplots and crosstabs generate feature hypotheses, but validation must decide which features actually help.

### 15.6 The one-sentence summary

Choose the plot by the types of the variables you are comparing, read the counts and denominators behind the picture, and treat every pattern as an associational clue rather than a causal conclusion.

---

# Part B - Advanced Reading and Reference

## 16. Statistical foundations

### 16.1 Covariance

For paired observations $(x_1,y_1),\dots,(x_n,y_n)$, the sample covariance is:

$$
s_{xy}=
\frac{1}{n-1}
\sum_{i=1}^{n}
(x_i-\bar{x})(y_i-\bar{y}).
$$

Covariance measures the direction of linear co-movement. Its sign is useful, but its magnitude depends on the units of $x$ and $y$. A covariance of 10 means something different for dollars and for years.

### 16.2 Pearson correlation

The Pearson correlation is the covariance divided by the product of the standard deviations:

$$
r=
\frac{s_{xy}}{s_x s_y}
=
\frac{
\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})
}{
\sqrt{
\sum_{i=1}^{n}(x_i-\bar{x})^2
\sum_{i=1}^{n}(y_i-\bar{y})^2
}
}.
$$

Properties:

- $-1\le r\le 1$;
- $r=1$: perfect positive linear relationship;
- $r=-1$: perfect negative linear relationship;
- $r=0$: no linear association;
- $r$ is unitless;
- $r$ is sensitive to outliers;
- $r$ measures linear association, not any association.

For tips, $r\approx0.6757$, so $R^2\approx0.4566$. About 45.7% of the sample variance in tip is linearly associated with total bill in a simple linear fit. This is not a causal decomposition.

### 16.3 Spearman correlation

Spearman's rank correlation replaces the values with ranks and computes Pearson correlation on the ranks:

$$
r_s=
\frac{
\sum_{i=1}^{n}(R(x_i)-\bar{R}_x)(R(y_i)-\bar{R}_y)
}{
\sqrt{
\sum_{i=1}^{n}(R(x_i)-\bar{R}_x)^2
\sum_{i=1}^{n}(R(y_i)-\bar{R}_y)^2
}
}.
$$

It measures monotonic association and is less sensitive to extreme values. Use it when the relationship is monotonic but not linear, or when outliers dominate.

### 16.4 Correlation limits

Correlation can be misleading when:

- the relationship is nonlinear, such as a U-shape;
- there are outliers;
- the range is restricted;
- the data contain subgroups with different patterns;
- the variables are ordinal or categorical;
- the sample is not representative.

Always plot the data. A correlation coefficient is a summary, not a substitute for a scatterplot.

### 16.5 Group means and standard errors

For a group with sample size $n$, sample mean $\bar{x}$, and sample standard deviation $s$, the standard error of the mean is:

$$
\mathrm{SE}=\frac{s}{\sqrt{n}}.
$$

A rough 95% confidence interval under normality is:

$$
\bar{x}\pm1.96\,\mathrm{SE}.
$$

Seaborn's barplot uses a bootstrap confidence interval by default, not the normal approximation. The bootstrap resamples the observed data many times and takes percentiles of the resampled means. The interval reflects uncertainty about the mean under the observed distribution.

For Pclass 1 females:

- $n=85$;
- mean $=34.612$;
- standard deviation $=13.612$;
- $\mathrm{SE}=13.612/\sqrt{85}\approx1.476$;
- normal-approximation 95% CI $\approx[31.72,37.51]$.

The bootstrap interval will be similar but not identical. The important point is that small groups have wider intervals.

### 16.6 Confidence intervals are not guarantees

A 95% confidence interval is a procedure: if we repeated the sampling process many times, about 95% of the intervals would contain the true mean under the assumptions. It is not a 95% probability that the true mean is inside this particular interval. The interpretation is subtle; avoid overclaiming.

### 16.7 Contingency tables and probabilities

For a crosstab with counts $n_{ij}$:

- row total: $n_{i\cdot}=\sum_j n_{ij}$;
- column total: $n_{\cdot j}=\sum_i n_{ij}$;
- grand total: $n=\sum_i\sum_j n_{ij}$.

Joint probability:

$$
P(R=i,C=j)=\frac{n_{ij}}{n}.
$$

Row-conditional probability:

$$
P(C=j\mid R=i)=\frac{n_{ij}}{n_{i\cdot}}.
$$

Column-conditional probability:

$$
P(R=i\mid C=j)=\frac{n_{ij}}{n_{\cdot j}}.
$$

Row normalization answers "given this row, what is the column share?" Column normalization answers "given this column, what is the row share?" They are different.

### 16.8 Independence

Two categorical variables are independent if:

$$
P(R=i,C=j)=P(R=i)P(C=j)
$$

for all $i,j$.

The expected count under independence is:

$$
E_{ij}=
\frac{n_{i\cdot}n_{\cdot j}}{n}.
$$

The chi-square statistic is:

$$
\chi^2=
\sum_{i,j}
\frac{(O_{ij}-E_{ij})^2}{E_{ij}},
$$

where $O_{ij}$ is the observed count.

The degrees of freedom for an $r\times c$ table are:

$$
(r-1)(c-1).
$$

### 16.9 Chi-square results for Titanic

Using the Titanic data:

| Pair | $\chi^2$ | df | p-value | Cramér's V |
|---|---:|---:|---:|---:|
| Pclass × Survived | 102.889 | 2 | $4.55\times10^{-23}$ | 0.340 |
| Sex × Survived | 260.717 | 1 | $1.20\times10^{-58}$ | 0.541 |
| Embarked × Survived | 26.489 | 2 | $1.77\times10^{-6}$ | 0.173 |
| SibSp × Survived | 37.272 | 6 | $1.56\times10^{-6}$ | 0.205 |
| Parch × Survived | 27.926 | 6 | $9.70\times10^{-5}$ | 0.177 |

Cramér's V is:

$$
V=
\sqrt{
\frac{\chi^2}{n\min(r-1,c-1)}
}.
$$

It is a measure of association strength. The Sex–Survived association is the strongest of these pairs; Pclass is next. The p-values are very small, but the data are observational and the sample is not a random sample from a well-defined population. Statistical significance is not causation.

For SibSp and Parch, some expected counts are below 5, so the chi-square approximation is questionable. Combine rare categories or use an exact test if inference matters.

### 16.10 Odds ratio and risk ratio

For a 2×2 table:

|  | Event | No event |
|---|---:|---:|
| Group A | a | b |
| Group B | c | d |

The odds ratio is:

$$
\mathrm{OR}=\frac{a/b}{c/d}=\frac{ad}{bc}.
$$

For female versus male survival:

$$
\mathrm{OR}=
\frac{233\times468}{81\times109}
\approx12.35.
$$

The risk ratio is:

$$
\mathrm{RR}=
\frac{a/(a+b)}{c/(c+d)}
=
\frac{233/314}{109/577}
\approx3.93.
$$

The odds ratio is larger than the risk ratio because survival is not rare. Neither is causal.

### 16.11 Simpson's paradox and confounding

Simpson's paradox occurs when an association in aggregated data reverses or changes when the data are stratified by a third variable. The classic example is a treatment that looks better overall but worse within every subgroup because the groups have different baseline risks.

Titanic is observational, so confounding is everywhere:

- `Embarked` is associated with `Pclass`;
- `Pclass` is associated with `Sex` and `Age`;
- `SibSp` and `Parch` describe family structure;
- `Cabin` missingness is associated with class.

A bivariate plot can show an association that disappears or changes after conditioning. Always ask: what third variable could explain this?

### 16.12 Boxplot quantiles

For sorted values $x_{(1)}\le\cdots\le x_{(n)}$, the median is the middle value or the average of the two middle values. Quartiles split the data into four parts. Different software uses slightly different interpolation rules, so quartiles can differ at the third decimal.

The IQR is:

$$
\mathrm{IQR}=Q_3-Q_1.
$$

The fences are:

$$
Q_1-1.5\,\mathrm{IQR},
\qquad
Q_3+1.5\,\mathrm{IQR}.
$$

The multiplier 1.5 is a convention. Some analyses use 3.0 for extreme outliers. A point beyond the fence is a candidate for inspection, not automatically an error.

### 16.13 KDE bandwidth

For a Gaussian KDE:

$$
\hat{f}_h(x)=
\frac{1}{nh}
\sum_{i=1}^{n}
\frac{1}{\sqrt{2\pi}}
\exp\left(
-\frac{(x-x_i)^2}{2h^2}
\right).
$$

The bandwidth $h$ controls the bias-variance trade-off:

- small $h$: low bias, high variance, spiky;
- large $h$: high bias, low variance, oversmooth.

Seaborn's `bw_adjust` multiplies the automatically selected bandwidth. `cut` controls how far the curve extends beyond the data range. A Gaussian KDE can put density outside the observed range; that is a boundary artifact.

### 16.14 Clustermap distances and linkage

A clustermap computes a distance matrix between rows and between columns, then applies hierarchical clustering.

Common distances:

- Euclidean;
- Manhattan;
- correlation distance.

Common linkage rules:

- single;
- complete;
- average;
- Ward.

Different choices produce different dendrograms. The clustermap is a visualization of one clustering, not a unique truth. With small contingency tables, the result is unstable.

### 16.15 Time series: trend and seasonality

A time series can be decomposed into:

- trend: long-term direction;
- seasonality: repeating pattern;
- cycle: longer irregular fluctuation;
- residual: what remains.

For flights:

- the yearly totals rise from 1,520 to 5,714;
- the monthly pattern peaks in July and August;
- the minimum is 104 in November 1949;
- the maximum is 622 in July 1960.

A lineplot of yearly totals shows the trend but hides seasonality. A month-by-year heatmap shows both. A time-aware model should use a time-based split, not a random split, because random splitting leaks future information into the past.

### 16.16 Pivot-table aggregation

A pivot table is a groupby plus a reshape:

```python
table = df.pivot_table(
    values="Survived",
    index="AgeGroup",
    columns="SibSp",
    aggfunc="mean",
)
```

Internally:

1. group by `AgeGroup` and `SibSp`;
2. apply `mean` to `Survived`;
3. unstack `SibSp` into columns;
4. align the result into a matrix.

The aggregation must match the question. A mean of a 0/1 variable is a rate. A mean of an ordinal code is usually not meaningful. A count is often safer.

---

## 17. Plotting internals and API reference

### 17.1 Matplotlib object model

```text
Figure
  └── Axes
        ├── XAxis
        ├── YAxis
        └── Artists
```

A `Figure` can contain many `Axes`. A `FacetGrid`, `PairGrid`, or `ClusterGrid` manages multiple axes.

### 17.2 Axes-level functions

```python
ax = sns.scatterplot(data=tips, x="total_bill", y="tip")
ax.set_title("Tip versus total bill")
```

Axes-level functions:

- draw on one `Axes`;
- accept `ax=`;
- return an `Axes`;
- are easiest to customize.

### 17.3 Figure-level functions

```python
g = sns.relplot(
    data=tips,
    x="total_bill",
    y="tip",
    hue="sex",
    col="smoker",
)
g.set_axis_labels("Total bill", "Tip")
g.figure.suptitle("Tips by sex and smoker status")
```

Figure-level functions:

- create their own figure;
- manage facets;
- return a grid object;
- use `height`, `aspect`, `facet_kws`;
- are customized through grid methods.

### 17.4 `relplot`

```python
sns.relplot(
    data=tips,
    x="total_bill",
    y="tip",
    kind="scatter",
    hue="sex",
    style="smoker",
    size="size",
    row="sex",
    col=None,
    height=4,
    aspect=1.2,
    alpha=0.7,
)
```

- `kind="scatter"` or `"line"`.
- `row`/`col`: facets.
- `hue`/`style`/`size`: visual channels.
- `height`/`aspect`: facet dimensions.
- Returns `FacetGrid`.

### 17.5 `catplot`

```python
sns.catplot(
    data=titanic,
    x="Pclass",
    y="Age",
    kind="bar",
    hue="Sex",
    height=4,
    aspect=1.2,
)
```

Kinds:

- `"bar"`: mean + interval;
- `"box"`: boxplot;
- `"violin"`: violin;
- `"count"`: countplot;
- `"point"`: point estimate;
- `"strip"`: strip;
- `"swarm"`: swarm.

Returns `FacetGrid`.

### 17.6 `barplot`

```python
sns.barplot(
    data=titanic,
    x="Pclass",
    y="Age",
    hue="Sex",
    estimator="mean",
    errorbar=("ci", 95),
)
```

- `estimator`: `"mean"`, `"median"`, or callable.
- `errorbar`: `("ci", 95)`, `"sd"`, `None`, or a callable.
- `order`: category order.
- `hue_order`: hue order.
- Returns `Axes`.

### 17.7 `boxplot`

```python
sns.boxplot(
    data=titanic,
    x="Pclass",
    y="Age",
    hue="Sex",
    whis=1.5,
    showfliers=True,
)
```

- `whis`: fence multiplier.
- `showfliers=False`: hide points but do not remove data.
- `order`, `hue_order`: category order.
- Returns `Axes`.

### 17.8 `histplot` and `kdeplot`

```python
sns.histplot(
    data=titanic,
    x="Age",
    hue="Survived",
    bins=20,
    kde=True,
    stat="count",
)

sns.kdeplot(
    data=titanic,
    x="Age",
    hue="Survived",
    common_norm=False,
    fill=True,
    bw_adjust=1.0,
)
```

- `bins`: number or edges.
- `stat`: `"count"`, `"density"`, `"probability"`, `"percent"`.
- `common_norm`: normalize groups jointly or separately.
- `bw_adjust`: bandwidth multiplier.
- Returns `Axes`.

### 17.9 `heatmap`

```python
sns.heatmap(
    ct,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=True,
    linewidths=0.5,
)
```

- `annot`: print values.
- `fmt`: format.
- `cmap`: color map.
- `cbar`: colorbar.
- `vmin`, `vmax`: color limits.
- Returns `Axes`.

### 17.10 `clustermap`

```python
sns.clustermap(
    ct,
    annot=True,
    fmt="d",
    cmap="Blues",
    metric="euclidean",
    method="average",
    figsize=(6, 6),
)
```

- `metric`: distance.
- `method`: linkage.
- `figsize`: figure size.
- Returns `ClusterGrid`.

### 17.11 `pairplot`

```python
sns.pairplot(
    data=iris,
    hue="species",
    vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    diag_kind="kde",
    corner=False,
    plot_kws={"alpha": 0.6},
)
```

- `vars`: subset of columns.
- `diag_kind`: `"kde"`, `"hist"`, `None`.
- `corner`: lower triangle only.
- Returns `PairGrid`.

### 17.12 `lineplot`

```python
sns.lineplot(
    data=new,
    x="year",
    y="passengers",
    marker="o",
    errorbar=("ci", 95),
)
```

- `marker`: point marker.
- `errorbar`: uncertainty for aggregated values.
- `hue`: multiple lines.
- `style`: line style.
- Returns `Axes`.

### 17.13 Version differences

Current environment:

- pandas 3.0.5;
- NumPy 2.5.3;
- Matplotlib 3.11.2;
- Seaborn 0.13.2;
- SciPy 1.18.1.

Differences from the official saved outputs:

| Item | Official saved output | Current environment |
|---|---|---|
| `sns.distplot` | works | works but deprecated |
| positional `scatterplot` | works | `TypeError` |
| positional `barplot` | works | `TypeError` |
| positional `boxplot` | works | `TypeError` |
| positional `lineplot` | works | `TypeError` |
| `groupby(...).mean()` on strings | works | `TypeError` |
| `groupby(...).sum()` on category month | works | `TypeError` |
| flights month labels | full names | abbreviations |
| axes representation | `_subplots.AxesSubplot` | current `Axes` |
| pandas text dtype | `object` | `str` |

### 17.14 Execution and display issues

**Bare function reference**

```python
sns.scatterplot
```

Draws nothing. Call the function.

**Missing `plt.show()`**

In a notebook, the last expression may display automatically. In a script, call `plt.show()`.

**Stale outputs**

A notebook can contain outputs from an older run. Rerun the cell in a clean kernel before trusting it.

**Figure-level customization**

`g.set_axis_labels(...)` works on a `FacetGrid`; `ax.set_xlabel(...)` works on an `Axes`. Do not mix them.

**Memory and speed**

Pairplots and clustermaps can be expensive. Subset columns and rows when exploring.

---

## 18. Source-fidelity audit

### 18.1 Official notebook: 33 cells

The official notebook has 25 code cells and 8 markdown cells. It uses:

- `tips`, `flights`, `iris` from Seaborn;
- Titanic from `train.csv`;
- old positional Seaborn calls;
- `distplot`;
- `groupby` without `observed=True`;
- an undefined `df` in the scatterplot cell.

### 18.2 Local notebook: 26 cells

The local notebook has 13 code cells and 13 markdown cells. It uses:

- `tips`, `iris` from Seaborn;
- Titanic from an absolute path;
- modern `relplot`, `catplot`, `boxplot` calls;
- deprecated `distplot`;
- a clustermap typo and duplicate;
- a sparse pivot-table heatmap;
- no flights section.

### 18.3 Dataset and paths

`Day_21/datasets/train.csv`:

- 891 rows × 12 columns;
- 61,194 bytes;
- SHA-256 `7d118fef8b6ccf7f81111877bc388536f7b1e498a655e3d649d19aaa010e9f6f`;
- root `E:\MACHINE_LEARNING\train.csv` is byte-identical.

The official notebook's `pd.read_csv("train.csv")` is relative to the working directory. The dataset is under `datasets/`, so the official notebook needs an adjusted path or a working directory containing `train.csv`.

The local notebook's absolute path works on this machine but is not portable.

### 18.4 Official source-fidelity issues

| Issue | Cell | Consequence |
|---|---:|---|
| `df` undefined | 9 | `NameError` in a clean kernel |
| positional `scatterplot` | 9 | `TypeError` in current Seaborn |
| positional `barplot` | 12 | `TypeError` |
| positional `boxplot` | 14 | `TypeError` |
| `distplot` deprecated | 16 | warning; removal planned |
| `groupby("Embarked").mean()` | 20 | `TypeError` in pandas 3.0 |
| positional `lineplot` | 29 | `TypeError` |
| `flights.groupby("year").sum()` | 28 | `TypeError` in pandas 3.0 |
| relative dataset path | 3 | file not found under `reference_notebooks/` |
| old axes repr | many | display difference only |

### 18.5 Local source-fidelity issues

| Issue | Cell | Consequence |
|---|---:|---|
| absolute Titanic path | 7 | not portable |
| `distplot` deprecated | 12 | warning; removal planned |
| `Sibsp` typo | 17 | `KeyError` |
| duplicate clustermap | 17 | redundant figure |
| sparse pivot heatmap | 22 | mostly empty matrix |
| original "LinePlot" label documented | 21 | current header correctly says heatmap |
| no outputs for most plots | 6, 9, 11, 12, 15 | partial notebook |

### 18.6 README discrepancies

The pre-rewrite README referred to:

- `DAY21_Bivariate-Multivariate-Analysis.ipynb`, which is not present;
- `DAY21_Bivariate-Multivariate-Analysis_simplified.ipynb`, which is not present;
- `reference_notebooks/reference_notebook.ipynb`, but the actual file is `reference_notebooks/day21.ipynb`.

The rewrite corrects the README to the files that actually exist.

### 18.7 What was not changed

- `my_notebook.ipynb` was not modified.
- `reference_notebooks/day21.ipynb` was not modified.
- `datasets/train.csv` was not modified.
- `vendor/` was not modified.

---

## 19. Extended comparisons

### 19.1 Scatterplot and lineplot

| Property | Scatterplot | Lineplot |
|---|---|---|
| x-axis order | not implied | implied |
| Best for | association, clusters | trend, time series |
| Points | individual observations | individual or aggregated |
| Connecting line | no | yes |
| Risk | overplotting | false order, aggregation hiding |

### 19.2 Barplot and boxplot

| Property | Barplot | Boxplot |
|---|---|---|
| Statistic | mean by default | median, quartiles |
| Uncertainty | interval | no interval |
| Outliers | hidden | shown |
| Shape | hidden | partially hidden |
| Best for | group means | group distributions |

### 19.3 Countplot and barplot

| Property | Countplot | Barplot |
|---|---|---|
| y variable | not required | required |
| Statistic | count | aggregated numeric |
| Missing categories | dropped by default | depends |
| Best for | category frequencies | group means/rates |

### 19.4 Heatmap and clustermap

| Property | Heatmap | Clustermap |
|---|---|---|
| Row/column order | input order | clustered order |
| Dendrograms | no | yes |
| Information | matrix values | matrix values + similarity |
| Risk | color scale | unstable small-sample clustering |

### 19.5 Pairplot and correlation matrix

| Property | Pairplot | Correlation matrix |
|---|---|---|
| Shows | pairwise scatter and distributions | pairwise linear correlation |
| Nonlinearity | visible | hidden |
| Groups | hue | not directly |
| Cost | high | low |
| Best for | exploration | quick summary |

### 19.6 Bivariate and multivariate

| Property | Bivariate | Multivariate |
|---|---|---|
| Variables | two | three or more |
| Plot | scatter, bar, box, crosstab | facets, hue/style/size, pairplot, matrix |
| Risk | confounding | overplotting, interpretation burden |
| Best for | focused question | broader screening |

### 19.7 Association and causation

| Property | Association | Causation |
|---|---|---|
| Evidence | observed co-movement | intervention or causal design |
| Confounding | possible | addressed by design |
| Language | "associated with" | "causes" |
| Titanic example | class and survival | not established |

---

## 20. Extended interview bank

### Level 1 - Foundations

1. **What is bivariate analysis?**
   Analysis of two variables together.

2. **What is multivariate analysis?**
   Analysis of three or more variables together.

3. **What is a scatterplot?**
   A plot of one point per observation at `(x, y)`.

4. **What is a crosstab?**
   A joint frequency table for two categorical variables.

5. **What is a heatmap?**
   A matrix visualization that maps values to colors.

6. **What is a clustermap?**
   A heatmap with hierarchical clustering and dendrograms.

7. **What is a pairplot?**
   A grid of pairwise scatterplots and univariate distributions.

8. **What is a lineplot?**
   A plot connecting ordered observations, usually over time.

### Level 2 - Mechanics

9. **Which plot for numeric × numeric?**
   Scatterplot or `relplot`.

10. **Which plot for categorical × numeric?**
    Boxplot, barplot, or violin.

11. **Which plot for categorical × categorical?**
    Crosstab plus heatmap or grouped countplot.

12. **What does `hue` do?**
    Maps a variable to color.

13. **What does `style` do?**
    Maps a variable to marker shape.

14. **What does `size` do?**
    Maps a variable to marker size.

15. **What does `row` do?**
    Creates facet rows.

16. **What does `col` do?**
    Creates facet columns.

17. **What does `normalize="index"` do?**
    Row proportions.

18. **What does `annot=True` do?**
    Prints cell values.

19. **What does `pairplot(vars=...)` do?**
    Limits the pairplot to selected columns.

20. **What does `lineplot` do with repeated x values?**
    Aggregates y and shows a mean with an interval by default.

### Level 3 - Mathematics

21. **What is covariance?**
    The average product of deviations from the means.

22. **What is Pearson correlation?**
    Covariance divided by the product of standard deviations.

23. **What is the range of Pearson correlation?**
    $-1$ to $1$.

24. **What is $R^2$?**
    The squared correlation in a simple linear fit; the proportion of variance linearly explained.

25. **What is a standard error?**
    The standard deviation of a sample statistic, often $s/\sqrt{n}$ for a mean.

26. **What is a confidence interval?**
    A range produced by a procedure with a stated coverage probability.

27. **What is a joint probability?**
    $P(R=i,C=j)=n_{ij}/n$.

28. **What is a conditional probability?**
    $P(C=j\mid R=i)=n_{ij}/n_{i\cdot}$.

29. **What is the chi-square statistic?**
    A sum of squared observed-minus-expected differences divided by expected counts.

30. **What is Cramér's V?**
    A normalized measure of association for contingency tables.

31. **What is an odds ratio?**
    The ratio of odds in two groups.

32. **What is a risk ratio?**
    The ratio of event probabilities in two groups.

33. **What is Simpson's paradox?**
    An association that changes or reverses after stratification.

34. **What is a KDE bandwidth?**
    A smoothing parameter controlling bias and variance.

35. **What is hierarchical clustering?**
    A bottom-up or top-down grouping of rows/columns by distance.

### Level 4 - Debugging

36. **Why does `sns.barplot(x, y)` fail?**
    Current Seaborn requires keyword `x` and `y`.

37. **Why does `groupby("Embarked").mean()` fail?**
    String columns do not support mean in pandas 3.0.

38. **Why does `flights.groupby("year").sum()` fail?**
    Category month does not support sum.

39. **Why does the official scatterplot fail?**
    Undefined `df` and old positional API.

40. **Why does the local clustermap fail?**
    `Sibsp` typo; the column is `SibSp`.

41. **Why is a saved output not proof?**
    It may come from a different kernel or code version.

42. **Why is a count heatmap misleading?**
    Large groups dominate the color scale.

43. **Why is a pairplot slow?**
    It creates roughly $p^2$ panels.

44. **Why is a lineplot wrong for unordered categories?**
    It implies an order that does not exist.

45. **Why is a sparse pivot heatmap weak?**
    Most cells are empty and the aggregation may be meaningless.

### Level 5 - Internals

46. **What is the difference between a Figure and an Axes?**
    A Figure is the canvas; an Axes is one plotting region.

47. **What does `relplot` return?**
    A `FacetGrid`.

48. **What does `catplot` return?**
    A `FacetGrid`.

49. **What does `pairplot` return?**
    A `PairGrid`.

50. **What does `clustermap` return?**
    A `ClusterGrid`.

51. **What does `heatmap` return?**
    An `Axes`.

52. **What does `pd.crosstab` return?**
    A DataFrame.

53. **What does `pivot_table` do internally?**
    Groupby, aggregate, unstack.

54. **What does `groupby` do with missing categories?**
    It excludes them by default; use `dropna=False` or `observed=True` carefully.

55. **What does `hue` do internally?**
    Groups the data and maps groups to colors.

### Level 6 - Design

56. **How do you choose a plot?**
    By variable types, the question, and the number of groups.

57. **How do you avoid overplotting?**
    Alpha, smaller points, jitter, hexbin, density, or faceting.

58. **How do you compare groups fairly?**
    Report counts, rates, and uncertainty; normalize when needed.

59. **How do you handle missing values in plots?**
    Report the denominator; drop, impute, or show missingness explicitly.

60. **How do you avoid leakage in EDA?**
    Do not use test targets to choose features; fit preprocessing on training data only.

61. **How do you validate a feature hypothesis?**
    Use cross-validation and a held-out test set, not a single plot.

62. **How do you decide whether an outlier is an error?**
    Check domain plausibility, provenance, units, and model sensitivity.

63. **How do you present a crosstab?**
    Show counts and row/column percentages; state the denominator.

64. **How do you present a pairplot?**
    Subset variables, color by target, and avoid overloading.

65. **How do you present a time series?**
    Show the trend and seasonality; use a time-aware split.

### Level 7 - Advanced

66. **Why can correlation be zero with a strong relationship?**
    The relationship may be nonlinear or symmetric.

67. **Why is chi-square not a measure of effect size?**
    It depends on sample size; use Cramér's V or odds ratios.

68. **Why can a clustermap be unstable?**
    Small tables and linkage choices can change the dendrogram.

69. **Why is a barplot's bootstrap interval not a Bayesian posterior?**
    It is a frequentist resampling procedure, not a posterior distribution.

70. **Why is a time-based split important for flights?**
    Random splits leak future information into the past.

71. **Why is a pivot table of exact Age sparse?**
    Age is continuous; exact combinations are rare.

72. **Why is averaging Pclass problematic?**
    Pclass is ordinal, not a continuous measurement.

73. **Why is a crosstab heatmap not a correlation heatmap?**
    It shows joint counts or rates, not a correlation matrix.

74. **Why is a pairplot not a feature-selection method?**
    It shows pairwise patterns without controlling for other variables or validating predictive value.

75. **Why is observational EDA not causal inference?**
    Confounding and selection bias can create associations without causation.

---

## 21. Extended examination questions

### 21.1 Theory

1. Explain why bivariate analysis follows univariate analysis.
2. Explain the difference between a count, a rate, and a mean.
3. Explain why a barplot and a boxplot can tell different stories.
4. Explain why a count heatmap can mislead.
5. Explain row normalization versus column normalization.
6. Explain what a clustermap adds to a heatmap.
7. Explain why a pairplot is exploratory.
8. Explain why a lineplot requires an ordered x-axis.
9. Explain why a pivot table can be sparse.
10. Explain why association is not causation.

### 21.2 Numerical

11. Compute the survival rate for Pclass 1: 136 out of 216.
12. Compute the survival rate for Pclass 3: 119 out of 491.
13. Compute the female survival rate: 233 out of 314.
14. Compute the male survival rate: 109 out of 577.
15. Compute the odds ratio for female versus male survival.
16. Compute the risk ratio for female versus male survival.
17. Compute the Cherbourg survival rate: 93 out of 168.
18. Compute the Southampton survival rate: 217 out of 644.
19. Compute the correlation $R^2$ for tips: $0.6757^2$.
20. Compute the flights growth ratio: 5,714 / 1,520.
21. Compute the local pivot table density: 185 / 528.
22. Compute the Pclass 1 female standard error: $13.612/\sqrt{85}$.

### 21.3 Analytical

23. A colleague says first class caused survival. Correct the statement.
24. A colleague reads a count heatmap and says third class had the most survivors. Check the claim.
25. A colleague uses a lineplot for Embarked. What is wrong?
26. A colleague uses a barplot and says it shows counts. What is wrong?
27. A colleague uses a pairplot with 30 variables. What advice do you give?
28. A colleague uses a clustermap and interprets the branch order as causal. Correct it.
29. A colleague averages Pclass and reports "average class 2.3." What is wrong?
30. A colleague uses exact Age in a pivot table and gets a mostly blank heatmap. What should they do?
31. A colleague compares a barplot of means with a boxplot and says they disagree. Explain why both can be correct.
32. A colleague uses the full dataset to choose features and then evaluates on a test set. What is the leakage risk?
33. A colleague says the iris pairplot proves a classifier will be perfect. Correct it.
34. A colleague says the flights lineplot proves monthly passengers rose every month. Correct it.
35. A colleague says a saved notebook output proves the code runs. Correct it.

### 21.4 Answer outlines

11. $136/216=62.96\%$.
12. $119/491=24.24\%$.
13. $233/314=74.20\%$.
14. $109/577=18.89\%$.
15. $(233\times468)/(81\times109)\approx12.35$.
16. $(233/314)/(109/577)\approx3.93$.
17. $93/168=55.36\%$.
18. $217/644=33.70\%$.
19. $0.6757^2\approx0.4566$.
20. $5714/1520\approx3.759$.
21. $185/528\approx0.3504=35.04\%$.
22. $13.612/\sqrt{85}\approx1.476$.

---

## 22. References and glossary

### 22.1 Course references

- CampusX Day 21 official notebook: `vendor/100-days-of-machine-learning/day21-bivariate-analysis/day21.ipynb`.
- Local reference copy: `reference_notebooks/day21.ipynb`.
- Local practical notebook: `my_notebook.ipynb`.
- Dataset: `datasets/train.csv` (Titanic, 891 × 12).

### 22.2 Technical references

- pandas documentation: `crosstab`, `pivot_table`, `groupby`, `cut`.
- Matplotlib documentation: `Figure`, `Axes`, `scatter`, `show`.
- Seaborn documentation: `scatterplot`, `relplot`, `barplot`, `boxplot`, `catplot`, `histplot`, `kdeplot`, `heatmap`, `clustermap`, `pairplot`, `lineplot`.
- SciPy documentation: `chi2_contingency`.
- Titanic dataset documentation.
- Iris and tips datasets from Seaborn.
- Flights dataset from Seaborn.

### 22.3 Glossary

| Term | Meaning |
|---|---|
| association | a statistical relationship between variables |
| barplot | categorical plot of an aggregated numeric value |
| bivariate | two variables together |
| boxplot | five-number summary with outlier candidates |
| categorical | labels or groups |
| chi-square | test statistic comparing observed and expected counts |
| clustermap | heatmap with hierarchical clustering |
| contingency table | counts for combinations of categorical variables |
| correlation | standardized measure of linear association |
| crosstab | joint frequency table |
| faceting | splitting a plot into subplots by a variable |
| heatmap | matrix values mapped to colors |
| hue | color grouping |
| IQR | interquartile range, $Q_3-Q_1$ |
| KDE | kernel density estimate |
| lineplot | plot connecting ordered observations |
| multivariate | three or more variables together |
| odds ratio | ratio of odds in two groups |
| pairplot | grid of pairwise numeric plots |
| pivot table | aggregated matrix from groupby |
| risk ratio | ratio of event probabilities |
| scatterplot | points at `(x, y)` |
| Simpson's paradox | association changes after stratification |
| standard error | standard deviation of a sample statistic |
| tidy data | one row per observation, one column per variable |

### 22.4 Final checklist

- [ ] I can choose a plot from the two variable types.
- [ ] I can explain what `hue`, `style`, `size`, `row`, and `col` do.
- [ ] I can distinguish a count, a rate, and a mean.
- [ ] I can read a crosstab and compute row and column percentages.
- [ ] I can explain why a count heatmap can mislead.
- [ ] I can read a barplot and a boxplot together.
- [ ] I can explain what a clustermap adds and what it does not prove.
- [ ] I can read a pairplot and identify separable and overlapping classes.
- [ ] I can read a lineplot for trend and seasonality.
- [ ] I can explain why the local pivot heatmap is sparse.
- [ ] I can identify the official and local source-fidelity bugs.
- [ ] I can explain why association is not causation.
- [ ] I can explain why preprocessing must be fitted on training data only.
- [ ] I can answer the Level 1–7 interview questions with reasoning.
