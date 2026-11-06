# Day 19 - Understanding Your Data: Descriptive Statistics

> CampusX 100 Days of Machine Learning, Day 19.
> One unified chapter covering the official Titanic notebook and the separate personal notebook.

### How to read this chapter

**Part A** teaches the first-look workflow: shape, sampling, dtypes, missing values, descriptive statistics, duplicates, correlations, and what each result means for the Titanic data. It is sufficient to understand and implement the day without reading Part B.

**Part B** is optional advanced reading: statistical definitions, correlation mathematics, pandas API details, a cell-level source-fidelity audit, extended comparisons, an extended interview bank, and extra examination practice.

The day has no model training and no train/test split. Its quantitative foundations are **counts, percentages, measures of center and spread, quartiles, and correlation coefficients**.

---

# Part A - Main Learning Chapter

## 1. The problem: you have a table, but you do not know it yet

Loading a CSV creates a DataFrame. It does not tell you:

- how many rows and columns you have;
- whether the columns mean what their names suggest;
- which values are missing;
- whether the numeric columns are on sensible scales;
- whether rows are duplicated;
- which columns move together;
- whether the target is balanced;
- which columns are IDs, text, categories, or numbers.

Day 19 is the diagnosis step. Treatment comes later.

```text
raw CSV
  |
  v
load with pandas
  |
  v
seven first-look checks
  |
  v
data-quality work order
  |
  v
cleaning / EDA / modeling
```

The official notebook asks seven questions:

1. How big is the data?
2. How does the data look?
3. What are the column types?
4. Are there missing values?
5. How does the data look mathematically?
6. Are there duplicate values?
7. How are the columns correlated?

The local notebook performs the same checks and adds a correlation heatmap.

### 1.1 Why this step matters

Every later decision depends on this diagnosis:

- **Missing values:** Cabin is mostly missing; Age is partly missing; Embarked has two gaps.
- **Dtypes:** `Pclass` is numeric but ordinal; `PassengerId` is numeric but an ID; `Sex` and `Embarked` are categorical strings.
- **Scale:** Fare ranges from 0 to 512.33, so scaling and outlier treatment will matter.
- **Target balance:** 342 survived and 549 died, so a majority-class baseline is 61.62%.
- **Correlation:** Pclass and Fare have the strongest linear relationships with survival among the numeric columns, but `Sex` is excluded because it is a string.
- **Duplicates:** there are no exact duplicate rows, so counts are not inflated by repeated records.

Skipping this step leads to blind imputation, wrong encodings, misleading correlations, and models trained on columns that should never be features.

### 1.2 Descriptive statistics versus inferential statistics

Day 19 is **descriptive**: it summarizes the observed data.

- Mean age is the mean of the 714 observed ages.
- Survival rate is the proportion of the 891 observed passengers who survived.
- Correlation is computed on the observed pairs.

It is not **inferential**: there are no confidence intervals, p-values, or claims about a larger population. Descriptive statistics describe this dataset; they do not prove facts about all Titanic passengers or future passengers.

---

## 2. Foundations: the DataFrame vocabulary

### 2.1 Rows, columns, and cells

A DataFrame has:

- an **index** (row labels; here 0 to 890);
- **columns** (12 named variables);
- **cells** (individual values).

The Titanic data has:

$$
891 \text{ rows} \times 12 \text{ columns} = 10{,}692 \text{ cells}.
$$

### 2.2 Dtypes

Each column has a dtype:

| Kind | Titanic examples | Meaning |
|---|---|---|
| integer | `PassengerId`, `Survived`, `Pclass`, `SibSp`, `Parch` | whole numbers |
| float | `Age`, `Fare` | continuous numbers, possibly missing |
| string/object | `Name`, `Sex`, `Ticket`, `Cabin`, `Embarked` | text or categories |

Dtype is not the same as statistical role:

- `PassengerId` is numeric but is an identifier, not a feature.
- `Survived` is numeric but is the binary target.
- `Pclass` is numeric but is an ordinal category (1st, 2nd, 3rd class).
- `Sex` is a string but is a categorical feature.
- `Name` and `Ticket` are strings with high cardinality.

### 2.3 Numeric versus categorical

| Type | Examples | Appropriate summaries |
|---|---|---|
| continuous numeric | Age, Fare | mean, median, std, quartiles |
| discrete numeric | SibSp, Parch | counts, mean, median, mode |
| ordinal numeric | Pclass | counts, median, groupby |
| binary target | Survived | proportion, counts |
| nominal categorical | Sex, Embarked | counts, proportions |
| identifier | PassengerId | uniqueness checks |
| high-cardinality text | Name, Ticket, Cabin | uniqueness, feature engineering |

`describe()` and `corr()` treat numeric columns as numbers. That is not always statistically appropriate: Pclass is ordinal, and PassengerId is an ID.

### 2.4 Missing values

A missing value is an absent observation, represented by `NaN` or `None`. It is not zero. It is not an empty string. It is unknown.

Missing values affect:

- counts in `describe()`;
- correlation sample sizes;
- imputation decisions;
- model compatibility.

### 2.5 Duplicates

An exact duplicate row is a row whose values match another row in every column. Near-duplicates (same person with a slightly different spelling) are not detected by `duplicated()`.

### 2.6 Correlation

Correlation measures the strength and direction of a **linear** association between two numeric variables. It is not causation, and it is not a complete description of a relationship.

---

## 3. The seven checks at a glance

| Check | pandas call | Question answered |
|---|---|---|
| Size | `df.shape` | how many rows and columns? |
| Glimpse | `df.sample(5)` | what do individual rows look like? |
| Types | `df.info()` | what are the dtypes and non-null counts? |
| Missing | `df.isnull().sum()` | how many values are absent per column? |
| Math | `df.describe()` | center, spread, and quartiles of numeric columns |
| Duplicates | `df.duplicated().sum()` | how many exact duplicate rows? |
| Correlation | `df.corr(numeric_only=True)` | which numeric columns move together? |

The order matters:

1. size and glimpse give context;
2. types and missing values reveal data quality;
3. describe gives the numeric fingerprint;
4. duplicates check record integrity;
5. correlation suggests relationships worth investigating.

Do not start modeling after step 1. Do not start imputing after step 4 without understanding why values are missing.

---

## 4. Check 1: How big is the data?

```python
df = pd.read_csv("datasets/train.csv")
df.shape
```

Saved output:

```text
(891, 12)
```

The tuple is `(rows, columns)`. There are 891 passengers and 12 columns.

Related calls:

```python
len(df)          # 891
df.columns       # Index of 12 column names
df.index         # RangeIndex(start=0, stop=891, step=1)
df.size          # 10692 = 891 * 12
```

### 4.1 What shape tells you

- The dataset is small: 891 rows.
- There are 12 candidate columns, but not all are features.
- A train/test split will leave relatively few rows per fold.
- Statistical summaries will be sensitive to individual passengers.

### 4.2 What shape does not tell you

- whether the rows are representative;
- whether columns are useful;
- how many values are missing;
- whether rows are duplicated;
- whether the target is balanced.

Shape is the first fact, not the diagnosis.

### 4.3 `shape` is an attribute

```python
df.shape      # (891, 12)
df.shape()    # TypeError: 'tuple' object is not callable
```

`shape` is a property, not a method. `info()` and `describe()` are methods and require parentheses.

---

## 5. Check 2: How does the data look?

### 5.1 `head` versus `sample`

```python
df.head()       # first five rows
df.sample(5)    # five random rows
```

`head()` shows the beginning of the file. If the file is sorted by PassengerId, class, or another variable, the first rows are not a random sample.

`sample(5)` draws random rows and better reveals variety. It is not a substitute for a full summary, but it is a better glimpse than `head()` alone.

### 5.2 The official sample

The official `df.sample(5)` saved five rows, including:

| PassengerId | Survived | Pclass | Name | Sex | Age | SibSp | Parch | Fare | Cabin | Embarked |
|---:|---:|---:|---|---|---:|---:|---:|---:|---|---|
| 499 | 0 | 1 | Allison, Mrs. Hudson J C (Bessie Waldo Daniels) | female | 25.0 | 1 | 2 | 151.5500 | C22 C26 | S |
| 126 | 1 | 3 | Nicola-Yarred, Master. Elias | male | 12.0 | 1 | 0 | 11.2417 | NaN | C |
| 605 | 1 | 1 | Homer, Mr. Harry ("Mr E Haven") | male | 35.0 | 0 | 0 | 26.5500 | NaN | C |
| 752 | 1 | 3 | Moor, Master. Meier | male | 6.0 | 0 | 1 | 12.4750 | E121 | S |
| 173 | 1 | 3 | Johnson, Miss. Eleanor Ileen | female | 1.0 | 1 | 1 | 11.1333 | NaN | S |

This glimpse shows:

- mixed classes and survival outcomes;
- children and adults;
- missing Cabin values;
- different embarkation ports;
- a wide Fare range even in five rows.

### 5.3 The local sample

The local notebook's `df.sample(5)` saved different rows, including Jensen, Chip, Skoog, Beavan, and Sunderland. The rows differ because `sample()` is random and no `random_state` is set.

For reproducible sampling:

```python
df.sample(5, random_state=42)
```

### 5.4 What to look for in a glimpse

- Are values in the expected format?
- Are there obvious missing values?
- Are numbers stored as text?
- Are categories spelled consistently?
- Are there impossible values (negative ages, zero fares, future dates)?
- Do IDs look unique?
- Do text columns contain punctuation, quotes, or unicode?

A five-row sample can miss rare problems. Pair it with `info()`, `describe()`, and `value_counts()`.

### 5.5 Sampling options

```python
df.sample(frac=0.1, random_state=42)  # 10% of rows
df.sample(5, replace=True)             # with replacement
df.sample(5, weights="Fare")           # weighted sampling
```

For a quick look, `df.sample(5, random_state=42)` is enough.

---

## 6. Check 3: What are the column types?

### 6.1 `info()`

```python
df.info()
```

Official saved output:

```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   PassengerId  891 non-null    int64
 1   Survived     891 non-null    int64
 2   Pclass       891 non-null    int64
 3   Name         891 non-null    object
 4   Sex          891 non-null    object
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64
 7   Parch        891 non-null    int64
 8   Ticket       891 non-null    object
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object
 11  Embarked     889 non-null    object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB
```

The local notebook, running pandas 3.0.5, reports `str` instead of `object` for text columns and `83.7 KB` instead of `83.7+ KB`. The data is the same; the dtype representation changed.

### 6.2 Reading the non-null counts

| Column | Non-null | Missing |
|---|---:|---:|
| PassengerId | 891 | 0 |
| Survived | 891 | 0 |
| Pclass | 891 | 0 |
| Name | 891 | 0 |
| Sex | 891 | 0 |
| Age | 714 | 177 |
| SibSp | 891 | 0 |
| Parch | 891 | 0 |
| Ticket | 891 | 0 |
| Fare | 891 | 0 |
| Cabin | 204 | 687 |
| Embarked | 889 | 2 |

`info()` already reveals the missing-value pattern. The next check quantifies it.

### 6.3 `info()` returns `None`

```python
result = df.info()
print(result)   # None
```

`info()` prints to the output stream; it does not return a DataFrame. If you need the data programmatically, use:

```python
df.dtypes
df.isnull().sum()
df.memory_usage(deep=True)
```

### 6.4 Dtype traps

- **Numeric IDs:** `PassengerId` is `int64`, but it is not a meaningful numeric feature.
- **Numeric categories:** `Pclass` is `int64`, but it is ordinal, not continuous.
- **Numeric target:** `Survived` is `int64`, but it is binary.
- **Text that should be categorical:** `Sex` and `Embarked` are strings.
- **Text that should be parsed:** `Cabin` contains deck letters and numbers; `Name` contains titles; `Ticket` contains prefixes.
- **Numbers stored as text:** if a numeric column has even one nonnumeric value, pandas may load it as `object`/`str`.

### 6.5 Selecting dtypes

```python
df.select_dtypes(include="number").columns
df.select_dtypes(exclude="number").columns
```

This is useful before `describe()` or `corr()`: it makes the numeric subset explicit.

---

## 7. Check 4: Are there missing values?

### 7.1 Counts

```python
df.isnull().sum()
```

Saved output:

```text
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
```

`df.isna().sum()` is an alias and gives the same result.

### 7.2 Percentages

```python
(df.isnull().mean() * 100).round(2)
```

| Column | Missing | Total | Percent |
|---|---:|---:|---:|
| Age | 177 | 891 | 19.87% |
| Cabin | 687 | 891 | 77.10% |
| Embarked | 2 | 891 | 0.22% |

Calculations:

$$
\frac{177}{891}\times 100 = 19.8653\%,
$$

$$
\frac{687}{891}\times 100 = 77.1044\%,
$$

$$
\frac{2}{891}\times 100 = 0.2245\%.
$$

### 7.3 Total missing cells

$$
177 + 687 + 2 = 866
$$

missing cells out of

$$
891 \times 12 = 10{,}692.
$$

The overall missing-cell rate is:

$$
\frac{866}{10{,}692}\times 100 = 8.0995\%.
$$

About 8.10% of all cells are missing. That sounds small, but it is concentrated in Cabin (77.10% missing) and Age (19.87% missing).

### 7.4 What the missing pattern suggests

| Column | Missing rate | Likely treatment |
|---|---:|---|
| Cabin | 77.10% | drop, or engineer `HasCabin`/deck |
| Age | 19.87% | impute; consider an indicator |
| Embarked | 0.22% | mode impute or drop the two rows |

These are previews, not final decisions. The treatment depends on the missingness mechanism and on train/test separation.

### 7.5 Missingness mechanisms

- **MCAR (missing completely at random):** missingness is independent of all variables. Complete-case analysis is unbiased, but wasteful.
- **MAR (missing at random):** missingness depends on observed variables but not on the missing value itself. Imputation using observed variables can help.
- **MNAR (missing not at random):** missingness depends on the unobserved value. Example: wealthier passengers may be less likely to report fare. Standard imputation may be biased.

You cannot prove the mechanism from the data alone. Domain knowledge matters.

### 7.6 Do not fill yet

At this stage, measure. Do not call `fillna(0)` because zero is a valid Fare and a plausible age only for infants. Do not drop Cabin rows because that would remove 77% of the dataset. Decide after the split and after understanding the mechanism.

---

## 8. Check 5: How does the data look mathematically?

### 8.1 `describe()`

```python
df.describe()
```

Saved output:

| Statistic | PassengerId | Survived | Pclass | Age | SibSp | Parch | Fare |
|---|---:|---:|---:|---:|---:|---:|---:|
| count | 891.000000 | 891.000000 | 891.000000 | 714.000000 | 891.000000 | 891.000000 | 891.000000 |
| mean | 446.000000 | 0.383838 | 2.308642 | 29.699118 | 0.523008 | 0.381594 | 32.204208 |
| std | 257.353842 | 0.486592 | 0.836071 | 14.526497 | 1.102743 | 0.806057 | 49.693429 |
| min | 1.000000 | 0.000000 | 1.000000 | 0.420000 | 0.000000 | 0.000000 | 0.000000 |
| 25% | 223.500000 | 0.000000 | 2.000000 | 20.125000 | 0.000000 | 0.000000 | 7.910400 |
| 50% | 446.000000 | 0.000000 | 3.000000 | 28.000000 | 0.000000 | 0.000000 | 14.454200 |
| 75% | 668.500000 | 1.000000 | 3.000000 | 38.000000 | 1.000000 | 0.000000 | 31.000000 |
| max | 891.000000 | 1.000000 | 3.000000 | 80.000000 | 8.000000 | 6.000000 | 512.329200 |

### 8.2 Which columns are included?

`describe()` defaults to numeric columns. It includes seven columns:

`PassengerId`, `Survived`, `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`.

It excludes five text columns:

`Name`, `Sex`, `Ticket`, `Cabin`, `Embarked`.

This is why `Sex` does not appear in the correlation matrix either, even though it is strongly associated with survival.

### 8.3 The count row is not always 891

- Most numeric columns have 891 non-null values.
- `Age` has 714 because 177 ages are missing.
- `describe()` computes Age statistics on the 714 observed ages, not on all 891 passengers.

This matters: the mean age 29.70 is the mean of the observed ages. If missingness is related to age or survival, it may not represent the full passenger list.

### 8.4 Mean, median, and skew

| Column | Mean | Median (50%) | Interpretation |
|---|---:|---:|---|
| Age | 29.6991 | 28.0000 | mild right skew |
| Fare | 32.2042 | 14.4542 | strong right skew |
| SibSp | 0.5230 | 0.0000 | zero-inflated |
| Parch | 0.3816 | 0.0000 | zero-inflated |
| Pclass | 2.3086 | 3.0000 | ordinal; more passengers in 3rd class |
| Survived | 0.3838 | 0.0000 | binary; 38.38% survived |
| PassengerId | 446.0000 | 446.0000 | uniform ID, not a feature |

For Fare, the mean is more than twice the median:

$$
\frac{32.2042}{14.4542}\approx 2.23.
$$

A few large fares pull the mean upward. The median is a more robust measure of the typical fare.

For Age, the mean and median are close:

$$
29.6991 - 28.0000 = 1.6991.
$$

The distribution is only mildly right-skewed.

### 8.5 Standard deviation

`describe()` reports the sample standard deviation with `ddof=1`:

$$
s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2}.
$$

For `Survived`, the mean is $p=0.383838$. The population standard deviation of a binary variable is:

$$
\sqrt{p(1-p)}=\sqrt{0.383838\times 0.616162}\approx 0.486319.
$$

The sample standard deviation is slightly larger:

$$
0.486592.
$$

The `describe()` value 0.486592 matches the sample formula.

### 8.6 Quartiles and the IQR

`25%`, `50%`, and `75%` are the quartiles. Pandas uses linear interpolation by default.

For Fare:

$$
Q_1=7.9104,\quad Q_3=31.0000,
$$

$$
\mathrm{IQR}=Q_3-Q_1=23.0896.
$$

A common outlier fence is:

$$
Q_3+1.5\times \mathrm{IQR}
=31.0000+1.5(23.0896)
=65.6344.
$$

There are 116 passengers with Fare above 65.6344, about 13.0% of the dataset. The maximum Fare is 512.3292, far above the fence. Fare has a long right tail.

For Age:

$$
Q_1=20.125,\quad Q_3=38.000,
$$

$$
\mathrm{IQR}=17.875,
$$

$$
Q_3+1.5\times \mathrm{IQR}=64.8125.
$$

There are 11 passengers older than 64.8125. The maximum is 80.

These fences are descriptive flags, not automatic deletion rules. A high fare may be a real first-class ticket, not a data error.

### 8.7 Skewness

As a supplementary check, pandas reports:

```python
df["Fare"].skew()   # 4.7873
df["Age"].skew()    # 0.3891
```

- Fare skewness 4.79: strong right skew.
- Age skewness 0.39: mild right skew.

Skewness is not part of the official seven checks, but it explains why mean and median disagree for Fare.

### 8.8 Zero-inflated counts

`SibSp` and `Parch` have median 0:

- more than half the passengers travelled with zero siblings/spouses or zero parents/children;
- the mean is pulled up by a minority with larger families;
- a bar chart of counts will show a spike at zero.

### 8.9 The target mean is a proportion

`Survived` is coded 0/1. Its mean is:

$$
\frac{342}{891}=0.383838.
$$

So 38.38% survived. The complement is:

$$
1-0.383838=0.616162,
$$

so 61.62% died.

If a model always predicts “died,” its accuracy is 61.62%. That is the majority-class baseline. Accuracy alone can therefore look decent while the model ignores the minority class.

### 8.10 `describe(include="all")`

To include text columns:

```python
df.describe(include="all")
```

This adds `unique`, `top`, and `freq` for categorical/text columns. Examples from the data:

| Column | Unique | Top | Freq |
|---|---:|---|---:|
| Name | 891 | Braund, Mr. Owen Harris | 1 |
| Sex | 2 | male | 577 |
| Ticket | 681 | 347082 | 7 |
| Cabin | 147 | G6 | 4 |
| Embarked | 3 | S | 644 |

`Name` is unique for every passenger; `Ticket` and `Cabin` are shared by some passengers. These are feature-engineering clues, not just trivia.

### 8.11 What `describe()` does not tell you

- the shape of the distribution;
- the presence of multiple modes;
- the relationship between columns;
- whether missingness is informative;
- whether a column is an ID or a category;
- whether the data is representative.

Use plots and groupbys for those questions.

---

## 9. Check 6: Are there duplicate values?

### 9.1 Exact duplicates

```python
df.duplicated().sum()
```

Saved output:

```text
0
```

The local notebook, on pandas 3.0.5, prints:

```text
np.int64(0)
```

The value is the same; the display type differs.

There are no exact duplicate rows. Counts such as “342 survived” are not inflated by repeated records.

### 9.2 How `duplicated()` works

```python
df.duplicated()
```

returns a boolean Series:

- `False` for the first occurrence of a row;
- `True` for later occurrences, by default (`keep="first"`).

```python
df.duplicated(keep=False)   # mark all copies
df.drop_duplicates()        # remove later copies
```

### 9.3 Duplicates on a subset

```python
df.duplicated(subset=["PassengerId"]).sum()
```

This checks whether the ID is unique. For Titanic, `PassengerId` is unique, so the count is 0.

`Ticket` is not unique: 681 unique tickets for 891 passengers. That is not a data-quality failure; it may indicate families or groups sharing a ticket. Do not drop those rows as duplicates.

### 9.4 Exact versus near duplicates

`duplicated()` compares full rows. It will not catch:

- the same person entered with a different spelling;
- the same ticket with a slightly different fare;
- the same company with punctuation differences;
- records that differ in one irrelevant column.

Near-duplicate detection needs domain rules, fuzzy matching, or record linkage.

### 9.5 What to do with duplicates

- If exact duplicates exist, investigate why before dropping.
- If they come from a join, the join may be wrong.
- If they are legitimate repeated events, keep them and aggregate.
- If they are data-entry duplicates, drop or merge with a documented rule.

For Day 19, there is nothing to drop.

---

## 10. Check 7: How are the columns correlated?

### 10.1 The numeric correlation matrix

```python
df.corr(numeric_only=True)
```

The matrix is 7×7 because only seven columns are numeric:

`PassengerId`, `Survived`, `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`.

The diagonal is 1 because every column is perfectly correlated with itself. The matrix is symmetric:

$$
r_{XY}=r_{YX}.
$$

### 10.2 Correlation with the target

```python
df.corr(numeric_only=True)["Survived"]
```

Values:

| Column | Correlation with Survived |
|---|---:|
| Survived | 1.000000 |
| Pclass | -0.338481 |
| Fare | 0.257307 |
| Parch | 0.081629 |
| SibSp | -0.035322 |
| Age | -0.077221 |
| PassengerId | -0.005007 |

The official notebook uses `df.corr()["Survived"]` and saves the same numeric values. In pandas 3.0.5, `df.corr()` on the full frame raises:

```text
ValueError: could not convert string to float: 'Braund, Mr. Owen Harris'
```

because the text columns are included. Use `numeric_only=True`.

### 10.3 Interpreting the values

- **Pclass = -0.3385:** higher class number (3rd class) is associated with lower survival. This is a moderate negative linear association.
- **Fare = 0.2573:** higher fare is associated with higher survival. This is a weak-to-moderate positive linear association.
- **Parch = 0.0816:** very weak positive association.
- **SibSp = -0.0353:** essentially no linear association.
- **Age = -0.0772:** very weak negative association.
- **PassengerId = -0.0050:** essentially zero; an ID should not be interpreted as a feature.

Correlation is not causation. Pclass and Fare are also correlated with each other:

$$
r(\mathrm{Pclass},\mathrm{Fare})=-0.5495.
$$

First-class passengers paid more and survived more. The Fare–survival correlation is partly confounded by class.

### 10.4 Why `Sex` is missing from the matrix

`Sex` is a string column. `corr(numeric_only=True)` excludes it. But `Sex` is strongly associated with survival:

| Sex | Count | Survival rate |
|---|---:|---:|
| female | 314 | 74.20% |
| male | 577 | 18.89% |

This is a supplementary groupby, not an official notebook output. It shows the limitation of a numeric correlation matrix: the strongest categorical association can be invisible.

To quantify a binary categorical association, you could:

- encode `Sex` as 0/1 and compute a point-biserial correlation;
- use a chi-square test of independence;
- use Cramér's V for two categorical variables;
- compare survival rates with `groupby`.

### 10.5 Pclass survival rates

| Pclass | Count | Survival rate |
|---:|---:|---:|
| 1 | 216 | 62.96% |
| 2 | 184 | 47.28% |
| 3 | 491 | 24.24% |

The negative correlation is visible as a monotone decrease in survival from 1st to 3rd class. But the relationship is not perfectly linear; groupby is more informative than a single coefficient.

### 10.6 Embarked survival rates

| Embarked | Count | Survival rate |
|---|---:|---:|
| C | 168 | 55.36% |
| Q | 77 | 38.96% |
| S | 644 | 33.70% |
| NaN | 2 | 100.00% |

The two missing-Embarked passengers both survived. Do not overinterpret a 100% rate based on two rows. Embarked is also confounded with class and fare.

### 10.7 Pearson correlation formula

For paired observations $(x_i,y_i)$:

$$
r=
\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}
 \sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}}.
$$

Properties:

- $r\in[-1,1]$;
- $r=1$: perfect positive linear relationship;
- $r=-1$: perfect negative linear relationship;
- $r=0$: no linear relationship (but a nonlinear relationship may exist);
- $r$ is unitless;
- $r$ is sensitive to outliers.

### 10.8 Pairwise missing values

Pandas correlation uses pairwise complete observations. The Age–Survived correlation uses the 714 passengers with observed Age, not all 891. Different pairs can have different sample sizes.

This is convenient but can make coefficients incomparable if missingness differs a lot across columns.

### 10.9 Correlation is not causation

A correlation can arise from:

- a direct effect;
- reverse causation;
- a common cause (confounding);
- selection;
- coincidence.

Example: Fare and survival are positively correlated, but paying more did not magically save passengers. Fare is a proxy for class, cabin location, and other advantages.

### 10.10 The heatmap

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()
```

The local notebook saves a 640×480 figure with two axes. The heatmap displays the 7×7 numeric correlation matrix.

Reading a heatmap:

- each cell is a correlation;
- the diagonal is 1;
- the matrix is symmetric;
- annotations give exact values;
- color alone can exaggerate small differences.

Better display options:

```python
plt.figure(figsize=(10, 8))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
)
plt.title("Titanic numeric correlations")
plt.show()
```

`vmin=-1`, `vmax=1`, and `center=0` keep the color scale honest.

The local notebook's markdown notes that the kernel crashed after the heatmap render. The saved figure exists; re-running a read-only inspection is safe. A crash during plotting is an environment issue, not a data corruption issue.

### 10.11 Correlation alternatives

- **Spearman rank correlation:** monotonic relationships, robust to some outliers.
- **Kendall tau:** rank concordance, useful for small samples.
- **Point-biserial:** binary vs continuous.
- **Cramér's V:** categorical vs categorical.
- **Mutual information:** nonlinear dependence.

Day 19 uses Pearson because the numeric columns are continuous-ish and the goal is a first look. Later analysis can choose a more appropriate measure.

---

## 11. Putting the results together

### 11.1 The data-quality summary

| Question | Answer |
|---|---|
| Rows | 891 |
| Columns | 12 |
| Numeric columns | 7 |
| Text/categorical columns | 5 |
| Missing cells | 866 (8.10%) |
| Age missing | 177 (19.87%) |
| Cabin missing | 687 (77.10%) |
| Embarked missing | 2 (0.22%) |
| Exact duplicate rows | 0 |
| Survived | 342 (38.38%) |
| Died | 549 (61.62%) |
| Majority baseline | 61.62% |

### 11.2 The feature-quality summary

| Column | Role | Issue |
|---|---|---|
| PassengerId | ID | not a feature |
| Survived | target | binary, imbalanced |
| Pclass | ordinal category | numeric coding may mislead |
| Name | high-cardinality text | extract titles or drop |
| Sex | categorical | strong association; must encode |
| Age | continuous | 177 missing |
| SibSp | count | zero-inflated |
| Parch | count | zero-inflated |
| Ticket | high-cardinality text | extract prefix or drop |
| Fare | continuous | right-skewed, outliers |
| Cabin | high-cardinality text | 687 missing |
| Embarked | categorical | 2 missing |

### 11.3 The cleaning work order

1. Drop `PassengerId` as a feature.
2. Decide how to handle `Cabin` (drop or engineer `HasCabin`/deck).
3. Impute `Age`; consider a missing indicator.
4. Impute `Embarked` (mode) or drop the two rows.
5. Encode `Sex` and `Embarked`.
6. Decide whether `Pclass` is ordinal or one-hot.
7. Consider transforming `Fare` because of skew and outliers.
8. Engineer family size from `SibSp` and `Parch`.
9. Extract titles from `Name` if useful.
10. Fit all imputers/encoders/scalers on training data only.

These are candidates, not commands. The right choices depend on the model and validation results.

### 11.4 The ML workflow position

```text
raw CSV
  |
  v
load
  |
  v
Day 19: inspect
  |
  v
clean / impute / encode / scale
  |
  v
EDA plots (Days 20-22)
  |
  v
train/test split
  |
  v
model training and evaluation
```

Day 19 is before the split. It is acceptable to inspect the full dataset for exploratory understanding. It is not acceptable to fit imputation, scaling, or feature-selection parameters on the test set later.

### 11.5 What not to conclude

- Do not conclude that Age causes survival from $r=-0.0772$.
- Do not conclude that Fare is unimportant because its correlation is only 0.2573.
- Do not conclude that Sex is unimportant because it is absent from the numeric matrix.
- Do not conclude that the data is clean because there are no duplicate rows.
- Do not conclude that 38.38% survival is a population estimate.

---

## 12. Common mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Calling `df.shape()` | `TypeError` | use `df.shape` |
| Assigning `df.info()` | result is `None` | use `df.dtypes`/`isnull()` for data |
| Expecting `describe()` to include strings | text columns missing | use `include="all"` |
| Using `df.corr()` on mixed dtypes | `ValueError` in pandas 3.0 | use `numeric_only=True` |
| Treating `PassengerId` as a feature | meaningless correlation | drop it |
| Treating `Pclass` as continuous | misleading mean | treat as ordinal/categorical |
| Treating `Survived` mean as a rating | wrong interpretation | it is a proportion |
| Filling missing values immediately | biased/impossible values | inspect mechanism first |
| Dropping all rows with missing Cabin | loses 77% of data | drop column or engineer |
| Using `head()` only | misses variety/outliers | use `sample()` and summaries |
| Assuming `sample()` is reproducible | different rows each run | set `random_state` |
| Reading heatmap colors only | exaggerates small values | read annotations; fix scale |
| Confusing correlation with causation | wrong causal claim | use domain knowledge/experiments |
| Treating exact duplicates as all duplicates | misses near-duplicates | define matching rules |

---

## 13. Essential interview and examination preparation

### 13.1 Level 1 - definitions

**Q1. What are descriptive statistics?**

Statistics that summarize the observed data: counts, mean, median, spread, quartiles, proportions.

**Q2. What is a missing value?**

An unknown/absent observation, represented by `NaN` or `None`; not zero.

**Q3. What is a duplicate row?**

A row whose values match another row in every column.

**Q4. What is correlation?**

A measure of the strength and direction of a linear association between two numeric variables.

**Q5. What is the median?**

The 50th percentile; the middle value when the data is sorted.

### 13.2 Level 2 - mechanics

**Q6. What does `df.shape` return?**

A tuple `(rows, columns)`.

**Q7. What does `df.info()` show?**

Column names, non-null counts, dtypes, and memory usage.

**Q8. What does `df.describe()` show?**

Count, mean, std, min, 25%, 50%, 75%, and max for numeric columns.

**Q9. What does `df.isnull().sum()` show?**

The number of missing values per column.

**Q10. What does `df.duplicated().sum()` show?**

The number of exact duplicate rows after the first occurrence.

**Q11. What does `df.corr(numeric_only=True)` show?**

A pairwise Pearson correlation matrix for numeric columns.

**Q12. Why `numeric_only=True`?**

To exclude text/categorical columns and avoid conversion errors.

### 13.3 Level 3 - applied and numerical

**Q13. The data has 891 rows and 12 columns. How many cells?**

$$
891 \times 12 = 10{,}692.
$$

**Q14. Age has 177 missing values out of 891. What percentage is missing?**

$$
\frac{177}{891}\times 100 = 19.8653\%.
$$

**Q15. Cabin has 687 missing values. What percentage is missing?**

$$
\frac{687}{891}\times 100 = 77.1044\%.
$$

**Q16. How many total missing cells are there?**

$$
177+687+2=866.
$$

**Q17. What is the overall missing-cell percentage?**

$$
\frac{866}{10{,}692}\times 100 = 8.0995\%.
$$

**Q18. 342 out of 891 survived. What is the survival rate?**

$$
\frac{342}{891}=0.383838 = 38.38\%.
$$

**Q19. What is the majority-class baseline?**

$$
\frac{549}{891}=0.616162 = 61.62\%.
$$

**Q20. Fare mean is 32.2042 and median is 14.4542. What does that suggest?**

Right skew: large fares pull the mean above the median.

**Q21. What is the IQR for Fare?**

$$
Q_3-Q_1=31.0000-7.9104=23.0896.
$$

**Q22. What is the upper outlier fence for Fare?**

$$
31.0000+1.5(23.0896)=65.6344.
$$

### 13.4 Level 4 - debugging and scenarios

**Q23. `df.shape()` raises `TypeError`. Why?**

`shape` is an attribute, not a method.

**Q24. `df.corr()` raises `ValueError` in pandas 3.0. Why?**

The full frame contains string columns. Use `numeric_only=True`.

**Q25. `df.info()` returns `None`. Is that a bug?**

No. `info()` prints; it does not return a DataFrame.

**Q26. `describe()` shows `count` 714 for Age but 891 for Fare. Why?**

Age has 177 missing values; Fare has none.

**Q27. `sample(5)` shows different rows on each run. Why?**

No `random_state` is set.

**Q28. `duplicated().sum()` is 0, but Ticket has repeated values. Why?**

Duplicate rows require all columns to match. Repeated tickets are not duplicate rows.

**Q29. The correlation matrix excludes Sex. Is Sex useless?**

No. Sex is categorical and strongly associated with survival; it is excluded from the numeric Pearson matrix. Use groupby or encode it.

**Q30. A heatmap cell is light red but the annotation is 0.05. What do you trust?**

The number. Color scales can exaggerate small correlations.

### 13.5 Worked code-reading questions

**Q31. What is the output?**

```python
df["Age"].isnull().sum()
```

177.

**Q32. What is the output?**

```python
df["Survived"].mean()
```

0.3838383838383838, the survival proportion.

**Q33. What is the output?**

```python
df["Pclass"].value_counts().sort_index()
```

1: 216, 2: 184, 3: 491.

**Q34. What is wrong with this code?**

```python
df.corr()["Survived"]
```

It includes string columns and raises in pandas 3.0.5. Use `df.corr(numeric_only=True)["Survived"]`.

**Q35. What is wrong with this conclusion?**

“Age has correlation -0.077, so age does not matter.”

The correlation is weak and linear; the relationship may be nonlinear or confounded. Also, 177 ages are missing.

### 13.6 Exam-style theory with answers

**Question 1.** Why inspect data before modeling?

Because model results depend on dtypes, missingness, scale, target balance, duplicates, and relationships. Inspection prevents silent data errors from becoming silent model errors.

**Question 2.** Why is the mean of a binary variable a proportion?

Because the values are 0 and 1. The sum is the count of 1s, so the mean is count of 1s divided by n.

**Question 3.** Why is the median more robust than the mean for Fare?

The median is less sensitive to the extreme high fares. The mean is pulled upward by the long right tail.

**Question 4.** Why does `describe()` not include Sex?

It defaults to numeric columns, and Sex is text/categorical.

**Question 5.** Why can a numeric correlation matrix miss the strongest feature?

Because categorical features are excluded. Sex is strongly associated with survival but is not numeric.

**Question 6.** Why is `PassengerId` not a feature?

It identifies rows; its numeric values have no causal or predictive meaning. A model using it may memorize rather than generalize.

---

## 14. Revision summary

### 14.1 The seven checks

```text
shape -> sample -> info -> isnull -> describe -> duplicated -> corr
```

### 14.2 The Titanic numbers

| Fact | Value |
|---|---:|
| Shape | (891, 12) |
| Cells | 10,692 |
| Missing cells | 866 (8.10%) |
| Age missing | 177 (19.87%) |
| Cabin missing | 687 (77.10%) |
| Embarked missing | 2 (0.22%) |
| Duplicates | 0 |
| Survived | 342 (38.38%) |
| Died | 549 (61.62%) |
| Pclass correlation | -0.3385 |
| Fare correlation | 0.2573 |
| Age correlation | -0.0772 |
| Sex survival (female/male) | 74.20% / 18.89% |

### 14.3 The three warnings

1. Correlation is not causation.
2. A numeric matrix excludes categorical columns.
3. Missing counts are not missing percentages.

### 14.4 The three decisions this day enables

1. Cabin needs dropping or engineering.
2. Age needs imputation.
3. Sex and Embarked need encoding; Pclass needs an ordinal/categorical decision.

### 14.5 The one-sentence summary

Inspect size, rows, types, missingness, numeric summaries, duplicates, and correlations before deciding how to clean or model the data.

---

# Part B - Advanced Reading and Reference

## 15. Statistical reference

### 15.1 Mean

$$
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i.
$$

Sensitive to outliers. For binary data, it is the proportion of 1s.

### 15.2 Median

The 50th percentile. Sort the values; the median is the middle value (or the average of the two middle values). Robust to extreme values.

### 15.3 Variance and standard deviation

Sample variance:

$$
s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2.
$$

Sample standard deviation:

$$
s=\sqrt{s^2}.
$$

Pandas `std()` defaults to `ddof=1`. Population standard deviation uses `ddof=0`.

### 15.4 Quartiles

- $Q_1$: 25th percentile.
- $Q_2$: 50th percentile (median).
- $Q_3$: 75th percentile.

Pandas uses linear interpolation by default. Different tools may use different quantile rules, so quartiles can differ slightly.

### 15.5 IQR and fences

$$
\mathrm{IQR}=Q_3-Q_1.
$$

Common fences:

$$
\text{lower}=Q_1-1.5\mathrm{IQR},\qquad
\text{upper}=Q_3+1.5\mathrm{IQR}.
$$

A point outside a fence is a flag, not automatically an error.

### 15.6 Skewness

A measure of asymmetry. Positive skew means a longer right tail; negative skew means a longer left tail. Fare's skewness is 4.79, indicating a strong right tail.

### 15.7 Robust statistics

Median and IQR are robust to outliers. Mean and standard deviation are not. For skewed data, report both.

### 15.8 Binary variables

For $p=P(X=1)$:

$$
E[X]=p,\qquad \mathrm{Var}(X)=p(1-p).
$$

The sample standard deviation is slightly larger than $\sqrt{p(1-p)}$ because of the $n-1$ denominator.

### 15.9 Proportions and counts

A count is an integer; a proportion is a count divided by a total. Always report the denominator. “74.2% survived” means 233 of 314 females, not 74.2% of all passengers.

### 15.10 Descriptive versus inferential

Descriptive statistics summarize the sample. Inferential statistics quantify uncertainty about a larger population. Day 19 is descriptive only.

---

## 16. Correlation reference

### 16.1 Covariance

$$
\mathrm{Cov}(X,Y)=
\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y}).
$$

Covariance depends on units; correlation normalizes it.

### 16.2 Pearson correlation

$$
r=\frac{\mathrm{Cov}(X,Y)}{s_X s_Y}.
$$

Properties:

- range $[-1,1]$;
- measures linear association;
- unitless;
- symmetric;
- sensitive to outliers;
- pairwise missing observations.

### 16.3 Spearman correlation

Spearman replaces values with ranks and computes Pearson on the ranks. It measures monotonic association and is more robust to outliers and nonlinear monotone relationships.

### 16.4 Point-biserial correlation

A Pearson correlation between a binary variable and a continuous variable. Encoding Sex as 0/1 and correlating with Survived is a point-biserial correlation.

### 16.5 Cramér's V

A measure of association between two categorical variables, based on the chi-square statistic. Useful for Sex vs Embarked or Pclass vs Embarked.

### 16.6 Pairwise deletion

Pandas correlation uses complete pairs for each coefficient. Different coefficients can use different subsets. This is convenient but can complicate comparisons.

### 16.7 Correlation and causation

Correlation does not establish causation. Confounding, reverse causation, selection, and coincidence can all produce correlation.

### 16.8 Simpson's paradox

An association can reverse or disappear when data is aggregated or disaggregated. Always inspect group-level relationships, not just one overall coefficient.

### 16.9 Correlation and feature selection

A feature can have a low linear correlation with the target but still be useful in a nonlinear model or in interaction with another feature. Do not discard features solely because Pearson $r$ is near zero.

### 16.10 Correlation matrix limitations

- only numeric columns;
- linear only;
- pairwise missing;
- sensitive to outliers;
- ordinal coding can distort;
- IDs produce meaningless coefficients.

---

## 17. pandas API reference

### 17.1 Structure

| Call | Returns |
|---|---|
| `df.shape` | tuple `(rows, cols)` |
| `len(df)` | row count |
| `df.columns` | column index |
| `df.index` | row index |
| `df.size` | number of cells |
| `df.dtypes` | Series of dtypes |
| `df.info()` | prints summary; returns `None` |

### 17.2 Sampling and peeking

| Call | Meaning |
|---|---|
| `df.head(n)` | first n rows |
| `df.tail(n)` | last n rows |
| `df.sample(n)` | random n rows |
| `df.sample(frac=0.1)` | random fraction |
| `df.sample(n, random_state=42)` | reproducible sample |

### 17.3 Missing values

| Call | Meaning |
|---|---|
| `df.isnull()` | boolean frame |
| `df.isna()` | alias |
| `df.isnull().sum()` | count per column |
| `df.isnull().mean()` | fraction per column |
| `df.isnull().sum().sum()` | total missing cells |
| `df.notna()` | negation |
| `df.dropna()` | drop rows/columns with missing |
| `df.fillna(value)` | fill missing |

### 17.4 Summary statistics

| Call | Meaning |
|---|---|
| `df.describe()` | numeric summary |
| `df.describe(include="all")` | include categorical |
| `df.mean(numeric_only=True)` | column means |
| `df.median(numeric_only=True)` | column medians |
| `df.std(numeric_only=True)` | sample standard deviations |
| `df.quantile(0.25)` | quartile |
| `df["Age"].skew()` | skewness |

### 17.5 Duplicates

| Call | Meaning |
|---|---|
| `df.duplicated()` | boolean Series |
| `df.duplicated().sum()` | number of extra copies |
| `df.duplicated(subset=[...])` | duplicates on key columns |
| `df.drop_duplicates()` | remove duplicates |

### 17.6 Correlation

| Call | Meaning |
|---|---|
| `df.corr()` | Pearson matrix |
| `df.corr(numeric_only=True)` | numeric only |
| `df.corr(method="spearman")` | Spearman |
| `df.corr(method="kendall")` | Kendall |
| `df.corr()["Survived"]` | target column |

### 17.7 Categorical summaries

| Call | Meaning |
|---|---|
| `df["Sex"].value_counts()` | frequency table |
| `df["Sex"].value_counts(dropna=False)` | include missing |
| `df["Sex"].nunique()` | distinct count |
| `df["Sex"].unique()` | distinct values |
| `df.groupby("Sex")["Survived"].mean()` | group survival rates |

---

## 18. Source-fidelity audit

### 18.1 Official notebook: 17 cells

| Cell | Content | Saved output |
|---:|---|---|
| 0 | import pandas | - |
| 1 | `read_csv('train.csv')` | - |
| 2 | size heading | - |
| 3 | `df.shape` | `(891, 12)` |
| 4 | glimpse heading | - |
| 5 | `df.sample(5)` | 5 random rows |
| 6 | dtypes heading | - |
| 7 | `df.info()` | 12 columns, non-null counts |
| 8 | missing heading | - |
| 9 | `df.isnull().sum()` | Age 177, Cabin 687, Embarked 2 |
| 10 | math heading | - |
| 11 | `df.describe()` | numeric summary |
| 12 | duplicates heading | - |
| 13 | `df.duplicated().sum()` | `0` |
| 14 | correlation heading | - |
| 15 | `df.corr()['Survived']` | 7 numeric correlations |
| 16 | empty | - |

### 18.2 Local notebook: 23 cells

The local notebook adds markdown explanations and a heatmap. It has 23 cells:

- 2 setup code cells (imports, `read_csv`);
- 6 summary code cells (`shape`, `sample`, `info`, `isnull`, `describe`, `duplicated`);
- 1 heatmap code cell;
- the remaining 14 cells are markdown/explanations.

The heatmap cell has no execution count and saves a Figure output. The local markdown notes that the kernel crashed after the heatmap render.

### 18.3 Path differences

Official:

```python
df = pd.read_csv("train.csv")
```

The official notebook lives in `reference_notebooks/`, where `train.csv` is not present. Running it from `Day_19/` would look for `Day_19/train.csv`, which also does not exist. The dataset is at `Day_19/datasets/train.csv`. The official notebook must be run from a directory containing `train.csv`, or the path must be adjusted.

Local:

```python
df = pd.read_csv(r"E:\MACHINE_LEARNING\train.csv")
```

The root `E:\MACHINE_LEARNING\train.csv` exists and is byte-identical to `Day_19/datasets/train.csv`. The local path works on this machine but is not portable. Prefer:

```python
df = pd.read_csv("datasets/train.csv")
```

### 18.4 Version differences

| Aspect | Official saved output | Local pandas 3.0.5 |
|---|---|---|
| Text dtype | `object` | `str` |
| Memory | `83.7+ KB` | `83.7 KB` |
| Duplicate sum display | `0` | `np.int64(0)` |
| Correlation call | `df.corr()` | `df.corr(numeric_only=True)` |
| Correlation result | works in saved output | `df.corr()` raises on strings |

The official `df.corr()['Survived']` output is valid for the pandas version that produced it. In pandas 3.0.5, the same call raises:

```text
ValueError: could not convert string to float: 'Braund, Mr. Owen Harris'
```

Use `numeric_only=True`.

### 18.5 Random sampling

Neither notebook sets `random_state`. The official sample rows and local sample rows differ. The samples are valid random glimpses, but not reproducible. For reproducible teaching examples, use `random_state=42`.

### 18.6 The heatmap

Local:

```python
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()
```

No figure size, title, `vmin`, `vmax`, or `center` is set. The saved figure is 640×480. The annotations are readable, but the color scale could exaggerate small correlations. The markdown reports a kernel crash after rendering; the saved figure is still present.

### 18.7 README discrepancies

The old README referred to:

- `reference_notebooks/reference_notebook.ipynb`, but the actual file is `reference_notebooks/day19.ipynb`;
- a simplified notebook that is not present;
- a primary learner notebook name that is actually `my_notebook.ipynb`.

The rewrite corrects the README to the files that actually exist.

---

## 19. Extended comparisons

### 19.1 `head` versus `sample`

| Property | `head` | `sample` |
|---|---|---|
| Rows | first n | random n |
| Bias | order-dependent | less order-dependent |
| Reproducibility | deterministic | random unless seeded |
| Use | inspect file start | inspect variety |

### 19.2 `info` versus `describe`

| Property | `info` | `describe` |
|---|---|---|
| Focus | structure and dtypes | numeric statistics |
| Missing counts | yes | count row only |
| Text columns | listed | excluded by default |
| Return | prints; `None` | DataFrame |

### 19.3 `isnull` versus `describe` count

| Property | `isnull().sum()` | `describe()` count |
|---|---|---|
| Scope | all columns | numeric columns |
| Meaning | missing count | non-missing count |
| Use | missingness | statistic denominator |

### 19.4 `duplicated` versus `drop_duplicates`

| Property | `duplicated` | `drop_duplicates` |
|---|---|---|
| Returns | boolean Series | DataFrame |
| Purpose | detect | remove |
| `keep` | first/last/False | first/last/False |
| Subset | optional | optional |

### 19.5 Pearson versus Spearman

| Property | Pearson | Spearman |
|---|---|---|
| Measures | linear | monotonic |
| Input | values | ranks |
| Outliers | sensitive | more robust |
| Nonlinear monotone | weak | stronger |
| Use | continuous linear | ordinal/robust |

### 19.6 Correlation versus groupby

| Property | Correlation | Groupby |
|---|---|---|
| Numeric | required | any dtype |
| Output | coefficient | group statistics |
| Nonlinear | not captured | can reveal |
| Categorical | excluded | supported |
| Causation | no | no |

### 19.7 `describe()` versus `value_counts()`

| Property | `describe()` | `value_counts()` |
|---|---|---|
| Numeric | summary statistics | frequency per value |
| Categorical | only with `include` | natural |
| Use | center/spread | category balance |

---

## 20. Extended interview bank

### Level 1 - Foundations

1. What is descriptive statistics?
2. What is a missing value?
3. What is a duplicate row?
4. What is correlation?
5. What is the median?

### Level 2 - Mechanics

6. What does `shape` return?
7. What does `info()` show?
8. What does `describe()` show?
9. What does `isnull().sum()` show?
10. What does `duplicated().sum()` show?
11. What does `corr(numeric_only=True)` show?
12. What does `sample(5)` do?

### Level 3 - Mathematics

13. 891 rows and 12 columns: how many cells?
14. 177 missing ages out of 891: what percentage?
15. 687 missing cabins out of 891: what percentage?
16. 866 missing cells out of 10,692: what percentage?
17. 342 survivors out of 891: what proportion?
18. What is the majority-class baseline?
19. Fare mean 32.2042, median 14.4542: what does that indicate?
20. What is the IQR of Fare?
21. What is the upper fence of Fare?
22. What does a correlation of -0.338 mean?

### Level 4 - Debugging

23. `df.shape()` raises `TypeError`. Why?
24. `df.corr()` raises `ValueError`. Why?
25. `df.info()` returns `None`. Is that a bug?
26. `describe()` count is 714 for Age. Why?
27. `sample(5)` changes each run. Why?
28. `duplicated().sum()` is 0 but Ticket repeats. Why?
29. The correlation matrix excludes Sex. Why?
30. A heatmap color looks strong but the value is 0.05. What do you trust?

### Level 5 - Internals

31. What is the difference between `object` and `str` dtype?
32. Why does pandas use `ddof=1` for `std()`?
33. How does pandas compute quartiles?
34. How does pandas handle missing values in `corr()`?
35. What does `info()` memory usage include?
36. Why is `PassengerId` correlation meaningless?

### Level 6 - Design

37. Design a reusable first-look function.
38. How would you document a data dictionary?
39. How would you decide between dropping and imputing Cabin?
40. How would you detect near-duplicates?
41. How would you handle class imbalance after seeing the target rate?

### Level 7 - Advanced

42. Compare Pearson, Spearman, point-biserial, and Cramér's V.
43. Explain Simpson's paradox with a tabular example.
44. Explain MCAR, MAR, and MNAR.
45. Why can a low-correlation feature still be useful?
46. How can EDA introduce target leakage?

---

## 21. Extended examination questions

### 21.1 Theory

1. Explain the purpose of the seven first-look checks.
2. Explain why `info()` and `describe()` answer different questions.
3. Explain why missing percentages are more useful than missing counts alone.
4. Explain why the mean of a binary target is a proportion.
5. Explain why the median is robust to outliers.
6. Explain why correlation does not imply causation.
7. Explain why a numeric correlation matrix can miss categorical features.
8. Explain why exact duplicates and near-duplicates need different methods.

### 21.2 Numerical

9. Compute the number of cells in an 891×12 table.
10. Compute the missing percentage for Age.
11. Compute the missing percentage for Cabin.
12. Compute the overall missing-cell percentage.
13. Compute the survival proportion.
14. Compute the majority-class baseline.
15. Compute the IQR of Fare.
16. Compute the upper fence of Fare.
17. Compute the IQR of Age.
18. Compute the upper fence of Age.
19. If 116 of 891 fares exceed the upper fence, what percentage is that?
20. If 11 of 891 ages exceed the upper fence, what percentage is that?

### 21.3 Analytical

21. A colleague says “Cabin is 687, so drop 687 rows.” Correct the statement.
22. A colleague says “Age correlation is -0.077, so age is useless.” Correct the statement.
23. A colleague says “Sex is not in the correlation matrix, so it is not predictive.” Correct the statement.
24. A colleague says “There are no duplicates, so the data is clean.” Correct the statement.
25. A colleague says “Fare causes survival because r=0.257.” Correct the statement.
26. A colleague uses `head()` and concludes the data is sorted by class. What should they also do?

---

## 22. References and glossary

### 22.1 Course references

- CampusX Day 19 official notebook: `vendor/100-days-of-machine-learning/day19-understanding-your-data-descriptive-stats/day19.ipynb`.
- Local reference copy: `reference_notebooks/day19.ipynb`.
- Local practical notebook: `my_notebook.ipynb`.
- Dataset: `datasets/train.csv` (Titanic, 891×12).

### 22.2 Technical references

- pandas `DataFrame` documentation: `shape`, `info`, `describe`, `isnull`, `duplicated`, `corr`, `sample`.
- pandas dtype documentation.
- Pearson correlation and Spearman rank correlation references.
- Missing-data mechanisms: MCAR, MAR, MNAR.
- Titanic dataset documentation.

### 22.3 Glossary

| Term | Meaning |
|---|---|
| cell | one value in a DataFrame |
| correlation | linear association measure in [-1, 1] |
| dtype | data type of a column |
| duplicate | row identical to another across all columns |
| IQR | interquartile range, Q3 − Q1 |
| mean | arithmetic average |
| median | 50th percentile |
| missing value | absent/unknown observation |
| outlier | value far from the bulk of the data |
| proportion | count divided by total |
| quartile | 25th/50th/75th percentile |
| sample | random subset of rows |
| skewness | asymmetry of a distribution |
| standard deviation | square root of variance |
| target | variable to predict |
| variance | average squared deviation from the mean |

### 22.4 Final checklist

- [ ] I can run and interpret `shape`, `sample`, `info`, `isnull`, `describe`, `duplicated`, and `corr`.
- [ ] I can compute missing counts and percentages.
- [ ] I can explain mean versus median and identify skew.
- [ ] I can compute and interpret quartiles and IQR.
- [ ] I can explain why the target mean is a proportion.
- [ ] I can interpret a correlation matrix and its limits.
- [ ] I can explain why Sex is excluded from the numeric matrix.
- [ ] I can list the cleaning decisions suggested by the data.
- [ ] I can explain why preprocessing must be fit on training data only.
- [ ] I can avoid confusing correlation with causation.
