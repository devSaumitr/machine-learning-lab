# Day 24 — Standardization (z-Score Scaling)

> **Course:** CampusX 100 Days of Machine Learning, Day 24 — Feature Scaling: Standardization.  
> **Official notebook:** `vendor/100-days-of-machine-learning/day24-standardization/day24.ipynb` (42 cells: 36 code, 6 markdown).  
> **Personal notebooks:** `my_notebook.ipynb` (106 cells: 35 code, 71 markdown, saved outputs), `DAY24_Standardization-with-Outliers.ipynb` (53 cells: 34 code, 19 markdown, code copy without outputs), `DAY24_Feature-Scaling-stub.ipynb` (5 markdown cells, no code).  
> **Reference copy:** `reference_notebooks/day24.ipynb` (42 cells, same code and outputs as the vendor notebook).  
> **Dataset:** `datasets/Social_Network_Ads.csv` — 400 rows, 5 columns; after `iloc[:, 2:]` the modelling table has `Age`, `EstimatedSalary`, `Purchased`.

This chapter follows the Day 39 two-layer standard. **Part A** is the main learning chapter: it explains the problem, the mathematics, the scikit-learn object, the actual notebook results, the outlier experiment, and the leakage rules. **Part B** is optional advanced reference: formal properties, gradient and distance mathematics, robust alternatives, implementation internals, extended interview questions, and source-fidelity notes.

The most important correction in this chapter is about the saved notebook state. `my_notebook.ipynb` was executed out of order: its saved `scaler.mean_`, `describe`, LogisticRegression, and DecisionTree outputs belong to the **outlier-inclusive** run with 322 training rows and 81 test rows, not to the clean 320/80 split described earlier in the notebook. The chapter separates the saved outputs from reproduced clean-run values so that no result is silently invented or misattributed.

---

# Part A — Main Learning Chapter

## 1. The problem: units are not comparable

### 1.1 The dataset

`Social_Network_Ads.csv` contains 400 rows and five columns:

| Column | Meaning | Type |
|---|---|---|
| `User ID` | identifier | integer |
| `Gender` | `Male` / `Female` | string |
| `Age` | age in years | integer |
| `EstimatedSalary` | salary in dollars | integer |
| `Purchased` | target: 0 or 1 | integer |

The personal notebook keeps only the last three columns:

```python
df = pd.read_csv("datasets/Social_Network_Ads.csv")
df = df.iloc[:, 2:]
```

After this slice:

```text
400 rows × 3 columns
Age, EstimatedSalary, Purchased
```

There are no missing values. The target distribution is:

```text
Purchased = 0: 257 rows
Purchased = 1: 143 rows
```

The two features have very different units and ranges:

| Feature | Minimum | Maximum | Mean | Standard deviation |
|---|---:|---:|---:|---:|
| `Age` | 18 | 60 | 37.655 | 10.4829 |
| `EstimatedSalary` | 15,000 | 150,000 | 69,742.50 | 34,096.96 |

A one-unit change in age means one year. A one-unit change in salary means one dollar. A salary difference of 60,000 and an age difference of 40 are not remotely comparable in raw units. That mismatch is the problem standardization solves.

### 1.2 What a model does with the numbers

A logistic regression computes a linear score such as

$$
\hat{y} = \sigma\left(\beta_0 + \beta_1 \cdot \text{Age} + \beta_2 \cdot \text{EstimatedSalary}\right),
$$

where $\sigma(t)=1/(1+e^{-t})$. The model does not know that age is measured in years and salary in dollars. It only sees numbers.

This creates three practical problems:

1. **Distance-based models** treat a 60,000 salary difference as enormously more important than a 40-year age difference.
2. **Gradient-based optimization** can be dominated by the large-scale feature, making the loss surface badly conditioned.
3. **Regularized models** penalize coefficients, and a coefficient's size depends on the feature's units. A coefficient on salary is naturally tiny because salary is large; a coefficient on age is naturally larger. Penalizing both equally without scaling is unfair.

A model can still fit the raw data. The issue is not that the arithmetic is impossible; the issue is that the geometry and optimization are distorted.

### 1.3 Why the split must come first

Standardization learns two numbers per feature: the mean and the standard deviation. Those numbers are parameters. If they are learned from the full dataset before the train/test split, the test set influences the training pipeline. This is data leakage.

The correct order is:

```text
raw data
   |
   v
train/test split
   |
   v
fit scaler on training data only
   |
   v
transform training data and test data with the same scaler
   |
   v
train and evaluate the model
```

The scaler's `fit` is where learning happens. `transform` is where the learned numbers are applied. The test set may be transformed, but it must never be used to fit the scaler.

### 1.4 What this day is and is not

This day is about **feature scaling by standardization**. It is not about:

- encoding categorical variables (Days 26–27),
- imputing missing values (Days 35–40),
- handling outliers in general (Days 42–44),
- building full preprocessing pipelines (Days 28–29).

The day uses a simple numeric dataset so the effect of scale is visible. Later days combine scaling with encoding, imputation, selection, and pipelines.

---

## 2. What standardization is

### 2.1 Definition

Standardization, also called **z-score scaling**, replaces every value of a feature by the number of standard deviations it lies from that feature's mean:

$$
z_{ij} = \frac{x_{ij}-\mu_j}{\sigma_j},
$$

where:

- $x_{ij}$ is the value of feature $j$ for observation $i$;
- $\mu_j$ is the mean of feature $j$;
- $\sigma_j$ is the standard deviation of feature $j$;
- $z_{ij}$ is the standardized value.

The transformation is applied **per column**, independently. `Age` is standardized using the mean and standard deviation of `Age`; `EstimatedSalary` uses the mean and standard deviation of `EstimatedSalary`.

### 2.2 A small numerical example

Take the ages

$$
[22,\ 35,\ 48].
$$

The mean is

$$
\mu = \frac{22+35+48}{3}=35.
$$

Using the population standard deviation (the convention scikit-learn uses),

$$
\sigma
=
\sqrt{\frac{(22-35)^2+(35-35)^2+(48-35)^2}{3}}
=
\sqrt{\frac{169+0+169}{3}}
=
\sqrt{112.667}
\approx 10.614.
$$

The standardized values are

$$
z_1=\frac{22-35}{10.614}\approx -1.225,
\qquad
z_2=\frac{35-35}{10.614}=0,
\qquad
z_3=\frac{48-35}{10.614}\approx +1.225.
$$

The transformed values have mean 0 and population standard deviation 1. The shape is unchanged: the three values are still equally spaced, only the origin and unit changed.

### 2.3 The two statistics

For a column with $n$ training values:

$$
\mu_j = \frac{1}{n}\sum_{i=1}^{n} x_{ij},
$$

$$
\sigma_j = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_{ij}-\mu_j)^2}.
$$

`StandardScaler` uses the **population** standard deviation, dividing by $n$, not the sample standard deviation dividing by $n-1$. This is why the fitted attribute is called `scale_` rather than `std_`. If you compute the standard deviation with pandas (`ddof=1`), you get a slightly different number.

For a clean 320-row training split of this dataset, the personal notebook's reproduced values are:

```text
scaler.mean_ = [37.871875, 70281.25]
scaler.scale_ = [10.1915631, 34376.713]
```

These are reproduced values, not the saved output of the mixed-state notebook. The saved `my_notebook` output is the outlier-inclusive run discussed in Section 8.

### 2.4 What the transformation preserves

Standardization is an **affine transformation** of each column:

$$
z = \frac{x-\mu}{\sigma} = \frac{1}{\sigma}x - \frac{\mu}{\sigma}.
$$

It preserves:

- the order of values;
- the shape of the distribution;
- the relative spacing within a column (all gaps are divided by the same $\sigma$);
- the Pearson correlation between columns, because a positive affine transformation of one variable does not change correlation.

It changes:

- the origin (the mean becomes 0);
- the unit (the standard deviation becomes 1);
- the numerical scale of distances;
- the size of model coefficients;
- the geometry seen by distance-based and gradient-based models.

It does **not**:

- make a distribution normal;
- remove outliers;
- bound values to $[0,1]$;
- guarantee that a model becomes more accurate.

### 2.5 Unbounded output

Standardized values are not restricted to a fixed interval. A value far above the mean becomes a large positive z-score; a value far below becomes a large negative z-score. In the outlier-inclusive run, the maximum standardized values are approximately

```text
Age:             +5.1
EstimatedSalary: +7.1
```

Those are not errors. They are the correct z-scores of extreme observations relative to the training mean and standard deviation. Standardization **centers** the data; it does not **de-outlier** it.

### 2.6 Standardization is not normalization

The stub notebook says:

> "Standardization also called Normalization."

In this course, that sentence is false or at least dangerously imprecise. Standardization and min–max normalization are different transformations:

| | Standardization (z-score) | Min–max normalization |
|---|---|---|
| Formula | $(x-\mu)/\sigma$ | $(x-\min)/(\max-\min)$ |
| Output range | unbounded | fixed $[0,1]$ for values inside the fitted range |
| Center | mean 0 | minimum 0 |
| Spread | standard deviation 1 | range 1 |
| Sensitive to | mean and standard deviation | minimum and maximum |
| Typical use | distance/gradient models, PCA, regularized linear models | bounded inputs, neural networks, image pixels |
| Outlier behaviour | outliers remain large z-scores | one outlier compresses the rest |

Some textbooks use “normalization” as a generic word for any scaling. In this course, Day 25 uses “normalization” for min–max scaling. Be precise in interviews: say **standardization** when you mean z-scores, and **min–max normalization** when you mean the bounded transformation.

---

## 3. Why standardization matters

### 3.1 Distance-based models

KNN, k-means, and SVM with an RBF kernel use distances or similarity. For two rows $\mathbf{a}$ and $\mathbf{b}$, Euclidean distance is

$$
d(\mathbf{a},\mathbf{b})
=
\sqrt{\sum_{j=1}^{p}(a_j-b_j)^2}.
$$

If one feature is measured in dollars and another in years, the dollar feature dominates the sum. A salary difference of 60,000 contributes $60{,}000^2=3.6\times 10^9$; an age difference of 40 contributes $40^2=1{,}600$. The age feature is effectively invisible.

After standardization, both features are measured in standard-deviation units. A difference of one standard deviation contributes the same squared amount to the distance, regardless of the original unit. That is usually the desired behaviour when no feature has a defensible claim to a larger scale.

There is a nuance: standardization assumes that **equal variance deserves equal weight**. If a feature has large variance because it is genuinely more informative, standardization may down-weight it. Scaling is a modelling choice, not a law of nature.

### 3.2 Gradient-based optimization

For a linear model with mean squared error,

$$
L(\beta)=\frac{1}{n}\|\mathbf{y}-\mathbf{X}\beta\|_2^2,
$$

the gradient is

$$
\nabla_\beta L = -\frac{2}{n}\mathbf{X}^\top(\mathbf{y}-\mathbf{X}\beta),
$$

and the Hessian is proportional to

$$
\mathbf{H}=\frac{2}{n}\mathbf{X}^\top\mathbf{X}.
$$

If one column has values in the hundreds of thousands and another in the tens, the Hessian has wildly different eigenvalues. The loss surface is an elongated valley. Gradient descent then needs a very small learning rate to avoid overshooting the large-scale direction, which makes progress slow in the small-scale direction. Standardization makes the curvature more uniform and often improves conditioning.

For logistic regression, the same idea applies to the Hessian of the negative log-likelihood. The default `LogisticRegression` solver in scikit-learn is `lbfgs`, which is more robust than plain gradient descent, but scale can still affect convergence, regularization, and the fitted coefficients.

### 3.3 Regularized models

Ridge regression minimizes

$$
\|\mathbf{y}-\mathbf{X}\beta\|_2^2+\lambda\|\beta\|_2^2.
$$

Lasso minimizes

$$
\frac{1}{2n}\|\mathbf{y}-\mathbf{X}\beta\|_2^2+\lambda\|\beta\|_1.
$$

The penalty is applied to coefficients. If features have different units, the same coefficient value does not represent the same amount of evidence. A salary coefficient is naturally tiny; an age coefficient is naturally larger. Penalizing them equally without standardization makes the regularization arbitrary. Standardize before penalized linear models if you want the penalty to treat features comparably.

### 3.4 PCA and other variance-based methods

PCA finds directions of maximum variance. If one feature has a much larger numerical range, PCA will align its first component with that feature almost automatically. Standardizing first makes PCA operate on correlation structure rather than raw units. This is usually what analysts mean by PCA on heterogeneous tabular features.

The same caution applies: if the features are already in the same meaningful unit and their variances are meaningful, standardization is a choice, not an obligation.

### 3.5 Neural networks

Neural networks are sensitive to input scale because activation functions, weight initialization, and gradient magnitudes depend on it. Standardizing inputs is a common default. The exact recipe depends on the architecture and the optimizer, but the general principle is the same: keep the optimization problem well-conditioned.

### 3.6 When standardization is usually unnecessary

Tree-based models split on thresholds:

```text
Is Age > 30?
Is EstimatedSalary > 70,000?
```

A monotone transformation such as standardization changes the threshold value but not the ordering of the rows. In exact arithmetic, the same splits are available. Decision trees, random forests, and gradient-boosted trees are therefore usually scale-invariant.

“Usually” matters:

- floating-point rounding can change tie-breaking in impurity calculations;
- regularized tree variants or models that combine trees with linear models may care;
- the implementation may choose different split thresholds when values are transformed;
- interpretation of thresholds changes even when predictions do not.

The notebook's tree experiment is consistent with invariance, but the chapter does not claim that every implementation will produce bit-identical predictions.

### 3.7 The empirical caveat

Standardization is not a guarantee of higher accuracy. It changes the geometry and the optimization problem. Whether that helps depends on the model, the data, the regularization, the solver, and the train/test split.

In the official notebook, the unscaled LogisticRegression scored `0.6583` and the scaled one `0.8667` on a 120-row test set. In the personal notebook's saved outlier-inclusive run, the unscaled model scored `0.8642` and the scaled one `0.8519` on an 81-row test set. In a clean reproduction of the personal split, the unscaled model scored `0.8875` and the scaled one `0.8625`. These are different runs with different splits and different data states. A single accuracy comparison is not a proof that scaling helps or hurts. The mathematical reasons for scaling remain; the empirical result must be measured, not assumed.

---

## 4. The mathematics in detail

### 4.1 Mean and variance after standardization

Let $z_{ij}=(x_{ij}-\mu_j)/\sigma_j$, where $\mu_j$ and $\sigma_j$ are computed on the same $n$ values. Then

$$
\frac{1}{n}\sum_{i=1}^{n} z_{ij}
=
\frac{1}{n\sigma_j}\sum_{i=1}^{n}(x_{ij}-\mu_j)
=
\frac{1}{n\sigma_j}\left(\sum_i x_{ij}-n\mu_j\right)
=0.
$$

The population variance is

$$
\frac{1}{n}\sum_{i=1}^{n} z_{ij}^2
=
\frac{1}{n\sigma_j^2}\sum_i (x_{ij}-\mu_j)^2
=
\frac{\sigma_j^2}{\sigma_j^2}
=1.
$$

So the transformed training column has population mean 0 and population variance 1.

### 4.2 Why pandas `describe()` shows a slightly larger standard deviation

`StandardScaler` divides by the population standard deviation. Pandas `describe()` reports the sample standard deviation with denominator $n-1$. For $n=320$:

$$
\sqrt{\frac{n}{n-1}}
=
\sqrt{\frac{320}{319}}
\approx 1.00156.
$$

That is why a perfectly standardized training column can show `std = 1.0016` in pandas before rounding to `1.0`. The notebook's rounded `describe()` shows `1.0`; the exact value is slightly above 1 because of the sample/population convention.

This is a useful interview detail: “mean 0, std 1” is exact for the population convention used by `StandardScaler`, and approximate in pandas' sample-standard-deviation display.

### 4.3 Inverse transformation

The transformation is invertible when $\sigma_j>0$:

$$
x_{ij} = z_{ij}\sigma_j+\mu_j.
$$

`StandardScaler.inverse_transform` applies this formula. It is useful when a model predicts in standardized space and you need to report original units.

### 4.4 Correlation and covariance

If you standardize every column using its own mean and population standard deviation, the covariance matrix of the transformed data is the correlation matrix of the original data:

$$
\operatorname{Cov}(z_j,z_k)
=
\frac{\operatorname{Cov}(x_j,x_k)}{\sigma_j\sigma_k}
=
\rho_{jk}.
$$

A positive affine transformation of one variable does not change its Pearson correlation with another variable. This is why standardization is often described as “removing units but keeping linear relationships.”

It does not preserve nonlinear relationships in any special way, and it does not make the data Gaussian.

### 4.5 Zero variance

If a feature is constant, $\sigma_j=0$ and the formula divides by zero. Scikit-learn handles this by setting the scale to 1 for zero-variance features, so the transformed column becomes all zeros. A constant feature carries no information and should usually be removed by a variance threshold or domain reasoning.

### 4.6 Outliers move the statistics

The mean and standard deviation are not robust. A single extreme value can move both. In the notebook, appending three rows changes the training means from approximately

```text
clean:   [37.871875, 70281.25]
outlier: [38.2080745, 72152.1739]
```

and the scales from approximately

```text
clean:   [10.1915631, 34376.713]
outlier: [11.1420889, 39318.1617]
```

The outlier run's salary mean is pulled upward by the 250,000 and 350,000 salaries. The scale also grows, so the bulk of the data is compressed into a narrower z-range while the extremes still stand far out.

### 4.7 A quick outlier calculation

Suppose a column has values

$$
[10,\ 11,\ 12,\ 13,\ 1000].
$$

The mean is

$$
\frac{10+11+12+13+1000}{5}=209.2.
$$

The population standard deviation is dominated by the 1000:

$$
\sigma
=
\sqrt{\frac{(10-209.2)^2+\cdots+(1000-209.2)^2}{5}}
\approx 395.1.
$$

The z-score of 1000 is

$$
\frac{1000-209.2}{395.1}\approx 2.00.
$$

The outlier does not look as extreme as it should because it has inflated the standard deviation itself. This is exactly why robust methods use the median and MAD or IQR instead.

### 4.8 Standardization versus other scalers

| Scaler | Formula | Output | Robust to outliers? | Typical use |
|---|---|---|---|---|
| `StandardScaler` | $(x-\mu)/\sigma$ | unbounded, mean 0, std 1 | no | linear models, PCA, distance models |
| `MinMaxScaler` | $(x-\min)/(\max-\min)$ | $[0,1]$ inside the fitted range | no | bounded inputs, neural nets |
| `RobustScaler` | $(x-\operatorname{median})/\operatorname{IQR}$ | unbounded | yes, more robust | outlier-heavy data |
| `MaxAbsScaler` | $x/\max|x|$ | $[-1,1]$, preserves zeros | no | sparse data |
| `PowerTransformer` | Yeo–Johnson or Box–Cox | approximately Gaussian | depends | skewed features |

Standardization is not a substitute for a distribution transform. A skewed salary column remains skewed after standardization; only its center and spread change.

---

## 5. How `StandardScaler` works internally

### 5.1 The object

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

`StandardScaler()` creates an **unfitted transformer**. It has not learned anything yet. The important constructor arguments are:

| Argument | Default | Meaning |
|---|---|---|
| `with_mean` | `True` | subtract the training mean |
| `with_std` | `True` | divide by the training standard deviation |
| `copy` | `True` | copy the input instead of modifying it in place |

If `with_mean=False`, only scaling is performed. If `with_std=False`, only centering is performed. Both `False` would make the transformer an identity transformation, which is rarely useful.

### 5.2 `fit` learns and stores

```python
scaler.fit(X_train)
```

`fit` computes, for each column:

- `mean_`: the training mean;
- `var_`: the training population variance;
- `scale_`: the training population standard deviation, with zero-variance columns replaced by 1;
- `n_features_in_`: the number of input features;
- `feature_names_in_`: the input column names when `X_train` is a DataFrame;
- `n_samples_seen_`: the number of samples seen, useful with `partial_fit`.

`fit` returns `self`, which is why this works:

```python
scaler = scaler.fit(X_train)
```

The trailing underscore convention means “learned from data.” Before `fit`, `scaler.mean_` does not exist.

### 5.3 `transform` applies

```python
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

`transform` does not learn anything. It applies the stored `mean_` and `scale_`:

$$
z = \frac{x-\text{mean\_}}{\text{scale\_}}.
$$

It returns a NumPy array by default, not a DataFrame. The personal notebook restores the column names:

```python
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
```

This is a presentation step. The model does not need the names, but plots and debugging do.

### 5.4 `fit_transform` is a convenience

```python
X_train_scaled = scaler.fit_transform(X_train)
```

This is equivalent to:

```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
```

Use it on training data. Do **not** use it on test data, because it would learn a second set of statistics from the test set:

```python
# Wrong: learns test statistics and applies a different mapping
X_test_scaled = scaler.fit_transform(X_test)

# Correct: uses the training statistics
X_test_scaled = scaler.transform(X_test)
```

### 5.5 Data flow

```text
X_train (raw)
    |
    v
StandardScaler.fit
    |  learns mean_, scale_
    v
StandardScaler.transform
    |
    v
X_train_scaled (z-scores)

X_test (raw)
    |
    v
StandardScaler.transform   <-- same fitted scaler, no new learning
    |
    v
X_test_scaled (z-scores using training statistics)
```

The shape is preserved:

```text
(n, p) raw  ->  (n, p) standardized
```

The dtype may change from integer to floating point. That is expected: z-scores are not integers.

### 5.6 Sparse data and centering

Centering a sparse matrix destroys sparsity: subtracting a non-zero mean turns every stored zero into a non-zero value. For sparse data, use:

```python
StandardScaler(with_mean=False)
```

This scales without centering. Alternatively, use `MaxAbsScaler`, which preserves zeros.

### 5.7 `inverse_transform`

```python
X_original = scaler.inverse_transform(X_scaled)
```

This maps z-scores back to the original units using $x=z\sigma+\mu$. It is useful when a model predicts a standardized target or when you want to report a result in original units.

### 5.8 `partial_fit` and streaming

`StandardScaler` supports `partial_fit` for large or streaming data:

```python
scaler = StandardScaler()
scaler.partial_fit(batch_1)
scaler.partial_fit(batch_2)
```

It updates `mean_`, `var_`, and `n_samples_seen_` incrementally. The result is intended to match a full `fit` up to numerical precision. In a leakage-safe workflow, the batches used for `partial_fit` must still be training data only.

---

## 6. The notebook implementation, step by step

### 6.1 Load and slice

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv("datasets/Social_Network_Ads.csv")
df = df.iloc[:, 2:]
df.sample(5)
```

What happens:

1. `read_csv` loads 400 rows and 5 columns.
2. `iloc[:, 2:]` keeps columns from index 2 onward: `Age`, `EstimatedSalary`, `Purchased`.
3. `sample(5)` shows five random rows; it is a quick sanity check, not a statistical summary.

The personal notebook originally used an absolute Windows path. It has been changed to the relative path `datasets/Social_Network_Ads.csv` so the notebook works when opened from `Day_24/`.

### 6.2 Split first

```python
from sklearn.model_selection import train_test_split as tts

x = df.drop(columns=["Purchased"])
y = df["Purchased"]

X_train, X_test, Y_train, Y_test = tts(
    x,
    y,
    test_size=0.2,
    random_state=42,
)
```

What happens:

- `x` contains `Age` and `EstimatedSalary`.
- `y` contains `Purchased`.
- `test_size=0.2` reserves 20% for testing.
- `random_state=42` makes the shuffle reproducible.
- With the clean 400-row dataset, the split is 320 training rows and 80 test rows.
- With the three outliers appended, the split is 322 training rows and 81 test rows.

The test set is locked at this point. Nothing below may learn from it.

### 6.3 Fit the scaler on training data

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler = scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

What happens:

1. `StandardScaler()` creates an unfitted object.
2. `fit(X_train)` learns the two means and two scales from the training rows.
3. `transform(X_train)` applies the training statistics to the training rows.
4. `transform(X_test)` applies the same statistics to the test rows.

The test rows are never used to compute a mean or a standard deviation.

### 6.4 Restore column names

```python
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
```

`transform` returns a NumPy array. The DataFrame wrapper restores the labels so plots can use `X_train_scaled["Age"]` and `X_train_scaled["EstimatedSalary"]`.

### 6.5 Check the learned statistics

```python
scaler.mean_
```

In the saved `my_notebook` state, the output is:

```text
array([3.82080745e+01, 7.21521739e+04])
```

That is:

```text
Age mean:     38.2080745
Salary mean: 72152.1739
```

This is the **outlier-inclusive** training mean. It is not the clean-run mean. The clean 320-row split, when reproduced, gives:

```text
[37.871875, 70281.25]
```

The difference is caused by the three appended rows. The notebook's saved state was executed after those rows were added, even though the code cell appears before the append cell.

### 6.6 Check the transformed distribution

```python
np.round(X_train_scaled.describe(), 1)
```

The saved output is:

```text
        Age  EstimatedSalary
count  322.0            322.0
mean     0.0              0.0
std      1.0              1.0
min     -3.0             -1.8
25%     -0.6             -0.7
50%     -0.1             -0.0
75%      0.7              0.4
max      5.1              7.1
```

What to notice:

- `count = 322` confirms the outlier-inclusive split.
- `mean = 0.0` and `std = 1.0` are the intended result (rounded).
- `min` and `max` are not bounded; the extremes are the injected outliers.
- The exact pandas standard deviation is about `1.0016` before rounding, because pandas uses the sample standard deviation while `StandardScaler` uses the population standard deviation.

The clean-run reproduced table has a smaller maximum:

```text
max Age:             2.2
max EstimatedSalary: 2.3
```

Those values are from a fresh clean split, not from the saved notebook output.

### 6.7 A data-flow summary

```text
Raw training rows
    |
    +--> Age ------------> mean Age, scale Age ----+
    |                                             |
    +--> EstimatedSalary -> mean Salary, scale Salary --+
                                                      |
                                                      v
                                             z = (x - mean) / scale
                                                      |
                                                      v
                                             standardized training rows
```

For test rows, the same learned means and scales are applied. The test rows do not contribute to the arrows that point into the statistics.

---

## 7. Reading the plots

### 7.1 Scatter before and after

The personal notebook plots:

```python
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 8))

ax1.scatter(X_train["Age"], X_train["EstimatedSalary"])
ax1.set_title("BEFORE SCALLING")

ax2.scatter(X_train_scaled["Age"], X_train_scaled["EstimatedSalary"], color="red")
ax2.set_title("AFTER SCALLING")
```

Before scaling:

- the x-axis is age, roughly 18–60;
- the y-axis is salary, roughly 15,000–150,000;
- the salary axis is orders of magnitude larger, so the cloud looks stretched vertically and the age variation is visually compressed.

After scaling:

- both axes are in standard-deviation units;
- the cloud is roughly centered on (0, 0);
- the two axes are visually comparable.

What the plot proves:

- the numerical scale changed as intended;
- the bulk of the data is centered near zero;
- the shape is still the same cloud, just relabeled.

What the plot does **not** prove:

- that a model will be more accurate;
- that the data are normally distributed;
- that outliers are gone.

### 7.2 KDE plots

The notebook draws kernel density estimates for both features before and after scaling:

```python
sns.kdeplot(X_train["Age"], ax=ax1)
sns.kdeplot(X_train["EstimatedSalary"], ax=ax1)

sns.kdeplot(X_train_scaled["Age"], ax=ax2)
sns.kdeplot(X_train_scaled["EstimatedSalary"], ax=ax2)
```

Before scaling, the two densities live on very different x-ranges. After scaling, both are centered at zero and have comparable spread. The shapes remain: if a feature was skewed before, it is still skewed after. Standardization changes location and scale, not distributional shape.

The notebook has a harmless presentation bug: it calls `ax1.set_title(...)` twice and never titles `ax2` in some cells. The plotted data are still correct; only the label is misleading. This is a reminder to read the code, not just the title.

### 7.3 The outlier plots

After appending the three extreme rows, the raw scatter shows the bulk of the data crushed into a corner by the 250,000 and 350,000 salaries. The standardized scatter still shows the extreme points far from the bulk, with z-scores around +5.1 for age and +7.1 for salary.

The correct conclusion is:

> Standardization centers the data but does not remove outliers. If outliers are a problem, handle them separately with capping, removal, a robust scaler, or a distribution transform.

The incorrect conclusion is:

> Standardization failed because the outliers are still visible.

The outliers are supposed to remain visible. The scaler never promised to remove them.

### 7.4 The blank panel

In one outlier plot, the right-hand `ax2.scatter` line is commented out. The panel is blank because the plotting call is missing, not because the scaled data are empty. This is a notebook bug, not a mathematical result. The real before/after comparison is in the earlier scatter cell and in the later outlier cell where both panels are drawn.

---

## 8. The model experiments and the actual numbers

### 8.1 The official notebook

The vendor notebook uses:

```python
X_train, X_test, y_train, y_test = train_test_split(
    df.drop("Purchased", axis=1),
    df["Purchased"],
    test_size=0.3,
    random_state=0,
)
```

The clean split has 280 training rows and 120 test rows. The saved official outputs are:

| Model | Unscaled accuracy | Scaled accuracy |
|---|---:|---:|
| LogisticRegression | 0.6583333333333333 | 0.8666666666666667 |
| DecisionTreeClassifier | 0.875 | 0.875 |

The official `scaler.mean_` is:

```text
array([3.78642857e+01, 6.98071429e+04])
```

The official standardized `describe()` is:

```text
        Age  EstimatedSalary
count  280.0            280.0
mean     0.0              0.0
std      1.0              1.0
min     -1.9             -1.6
25%     -0.8             -0.8
50%     -0.1              0.0
75%      0.8              0.5
max      2.2              2.3
```

The official notebook's LogisticRegression gap is large: 25 test rows out of 120. The tree is an exact tie in the saved output.

### 8.2 The personal notebook's saved state

The personal notebook uses:

```python
test_size=0.2
random_state=42
```

Its saved outputs are from the **outlier-inclusive** state:

| Model | Unscaled accuracy | Scaled accuracy |
|---|---:|---:|
| LogisticRegression | 0.8641975308641975 | 0.8518518518518519 |
| DecisionTreeClassifier | 0.8888888888888888 | 0.8888888888888888 |

The LogisticRegression scores correspond to 70/81 and 69/81. The tree scores correspond to 72/81 and 72/81. The saved `scaler.mean_` and `describe` also belong to this 322-row training state.

This is why the chapter does not repeat the old claim that the personal notebook's first comparison was “0.8642 vs 0.8519 on a clean split.” Those numbers are real, but they are the outlier-inclusive run. The clean-run comparison must be reproduced separately.

### 8.3 The clean personal split, reproduced

Running the same personal settings on the clean 400-row data gives:

```text
train shape: (320, 2)
test shape:  (80, 2)

LogisticRegression: 0.8875 unscaled, 0.8625 scaled
DecisionTree:       0.8375 unscaled, 0.8375 scaled
```

These are **reproduced** values, not saved notebook outputs. They show that the clean-run LogisticRegression did not improve with scaling on this particular split. The tree tied.

### 8.4 What the numbers mean

The one-row gap in the saved personal run is not evidence:

```text
70/81 = 0.8641975
69/81 = 0.8518519
difference = 1/81 = 0.0123457
```

The standard error of an accuracy near 0.86 with 81 test rows is roughly

$$
\sqrt{\frac{0.86(1-0.86)}{81}}\approx 0.0385.
$$

A one-row difference is well inside sampling noise. Do not conclude that scaling hurt or helped from this run.

The official gap is larger, but it is also version-dependent. Reproducing the official split with a current scikit-learn version gives approximately:

```text
LogisticRegression: 0.875 unscaled, 0.8667 scaled
DecisionTree:       0.875 unscaled, 0.8667 scaled
```

The official `0.6583` is a historical saved output, not a universal law. Different scikit-learn versions, solvers, convergence settings, and regularization defaults can change the result. The safe conclusion is:

> Scaling changes the optimization geometry and can matter a great deal for some models and datasets, but a single accuracy comparison is not proof. Evaluate it with the same folds, the same model settings, and a proper baseline.

### 8.5 Why the tree is usually invariant

A decision tree asks questions of the form:

```text
Age <= 30.5?
EstimatedSalary <= 70,000?
```

Standardization is a monotone transformation. If $a<b$, then $z(a)<z(b)$ because $\sigma>0$. The set of possible splits by order is unchanged. In exact arithmetic, the tree can make the same partitions.

In practice:

- floating-point arithmetic can change impurity calculations at the margin;
- tied thresholds may be broken differently;
- different implementations may choose different split points;
- the printed thresholds change even when predictions do not.

The saved official and personal runs show ties; the current reproduction of the official split shows a one-row difference for the tree. The correct statement is “trees are usually scale-invariant,” not “trees always produce bit-identical predictions.”

### 8.6 Model-sensitivity table

| Model | Needs scaling? | Why |
|---|---|---|
| KNN | Usually yes | Distances are the prediction mechanism. |
| k-means | Usually yes | Clusters are defined by distances. |
| SVM with RBF/polynomial kernel | Usually yes | Kernel distances and margins. |
| Logistic/linear regression with regularization | Usually yes | The penalty applies to coefficients. |
| PCA | Usually yes | Variance is scale-dependent. |
| Neural networks | Usually yes | Optimization and activations are scale-sensitive. |
| Decision trees | Usually no | Splits depend on order, not magnitude. |
| Random forests / gradient boosting | Usually no | Same reason; scaling is mostly harmless. |
| Naive Bayes | Usually no | Likelihoods are estimated per feature. |

“Usually no” does not mean “forbidden.” Scaling a tree model is harmless in most cases, but it adds computation and can make threshold interpretation less direct.

---

## 9. The outlier experiment

### 9.1 The injected rows

The notebook appends:

```python
new_data = pd.DataFrame({
    "Age": [5, 90, 95],
    "EstimatedSalary": [1000, 250000, 350000],
    "Purchased": [0, 1, 1],
})

df = pd.concat([df, new_data], ignore_index=True)
```

The three rows are deliberately extreme:

- age 5 is far below the bulk;
- ages 90 and 95 are far above the bulk;
- salaries 250,000 and 350,000 are far above the original maximum of 150,000;
- salary 1,000 is far below the original minimum of 15,000.

The vendor notebook uses `df.append(...)`, which was removed in pandas 2.0. The personal notebook correctly uses `pd.concat`. This is a practical improvement, not a change in the experiment.

### 9.2 What changes

The clean and outlier training statistics are:

| Statistic | Clean 320 rows | Outlier 322 rows |
|---|---:|---:|
| Age mean | 37.871875 | 38.2080745 |
| Salary mean | 70,281.25 | 72,152.1739 |
| Age scale | 10.1915631 | 11.1420889 |
| Salary scale | 34,376.713 | 39,318.1617 |

The mean and scale both move. The salary mean rises by about 1,871; the salary scale rises by about 4,941. The bulk of the data is therefore standardized against a mean and spread that no longer describe the bulk.

### 9.3 What the standardized extremes look like

The outlier-inclusive `describe()` shows:

```text
Age max:              5.1
EstimatedSalary max:  7.1
```

Those are large z-scores, but they are not infinite. The scaler has done its job: it expressed the extremes relative to the training statistics. It has not removed them.

### 9.4 Why standardization cannot fix outliers

Standardization is a linear transformation:

$$
z = \frac{x-\mu}{\sigma}.
$$

A linear transformation cannot delete an extreme value. It can only move and rescale it. If $x$ is far from the mean, $z$ will be far from zero unless $\sigma$ is made enormous by the outlier itself, which then compresses the rest of the data.

This is the key lesson of the day:

> Standardization is not an outlier treatment. It is a unit change.

### 9.5 What to do instead

Options depend on why the outlier exists:

1. **Data error:** correct or remove the row.
2. **Genuine rare event:** keep it, but use a model that is robust or a transformation that reduces its influence.
3. **Heavy tail:** use a log or power transform (Day 31), or a robust scaler.
4. **Measurement limit:** cap/winsorize at a defensible threshold.
5. **Different population:** model it separately or add an indicator.

Robust scaling uses the median and IQR:

$$
z_{\text{robust}} = \frac{x-\operatorname{median}(x)}{\operatorname{IQR}(x)}.
$$

It is less affected by extremes than the mean and standard deviation. It still does not remove them; it simply uses statistics that are less sensitive to them.

### 9.6 Robust scaler code

```python
from sklearn.preprocessing import RobustScaler

robust = RobustScaler()
X_train_robust = robust.fit_transform(X_train)
X_test_robust = robust.transform(X_test)
```

`RobustScaler` stores `center_` (the median) and `scale_` (the IQR). It is a useful comparison when the data contain genuine extremes. It is not automatically better: if the outliers are errors, fix the data first.

### 9.7 Distribution transforms

If the goal is to reduce skew rather than just change units, use a distribution transform:

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method="yeo-johnson")
X_train_pt = pt.fit_transform(X_train)
X_test_pt = pt.transform(X_test)
```

A power transform changes the shape; standardization does not. Do not confuse the two goals.

---

## 10. The leakage-safe workflow

### 10.1 The basic pattern

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Rules:

1. Split first.
2. Fit the scaler on `X_train` only.
3. Transform `X_train` and `X_test` with the same fitted scaler.
4. Never call `fit` or `fit_transform` on `X_test`.
5. Never compute the mean or standard deviation on the full dataset.

### 10.2 The pipeline pattern

In a real project, scaling is one step among many. Put it inside a pipeline so cross-validation refits it on each training fold:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

`model.fit` fits the scaler on the training data and then fits the classifier on the transformed training data. `model.predict` applies the fitted scaler to the test data and then predicts.

### 10.3 `ColumnTransformer` for mixed types

When numeric and categorical columns coexist:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_features = ["Age", "EstimatedSalary"]
categorical_features = ["Gender"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=1000)),
])
```

`ColumnTransformer` sends each column group to its own pipeline and concatenates the results. The scaler sees only the numeric columns; the encoder sees only the categorical columns.

### 10.4 Cross-validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(scores.mean(), scores.std())
```

Each fold fits its own scaler on that fold's training part. The validation part is transformed with the fold's scaler. This is the only way to get an honest estimate when preprocessing learns parameters.

### 10.5 Common leakage patterns

| Anti-pattern | Why it leaks |
|---|---|
| `scaler.fit(df)` before split | test statistics enter training |
| `scaler.fit_transform(X_test)` | test statistics replace training statistics |
| `df["Age"] = (df["Age"] - df["Age"].mean()) / df["Age"].std()` before split | same problem, manual version |
| Scaling the full dataset and then splitting | test rows influence `mean_` and `scale_` |
| Selecting features on the full dataset | target information leaks through selection |
| Imputing with the full-data mean | test values influence the fill value |

The rule is not “never touch the test set.” The test set must be transformed. The rule is “never learn a parameter from the test set.”

---

## 11. Common mistakes and debugging

### 11.1 Mistake table

| Mistake | Symptom | Fix |
|---|---|---|
| Fitting on the full dataset | Validation score too optimistic; production drop | Split first; fit on train |
| `fit_transform` on test | Inconsistent mapping; leakage | `transform` on test |
| Scaling before imputation | NaN propagates or scaler fails | Impute first, then scale |
| Scaling the target by accident | Predictions in wrong units | Keep target separate; inverse-transform if intentional |
| Expecting normality after scaling | KDE shape unchanged | Use a power transform for skew |
| Expecting outliers to disappear | Large z-scores remain | Treat outliers separately |
| Using min–max on extreme outliers | Bulk compressed into a tiny interval | Use robust scaling or cap |
| Using standardization on a constant column | Zero variance | Drop the column or let sklearn set scale to 1 |
| Forgetting column names after `transform` | Plot/key errors | Rewrap with `pd.DataFrame` |
| Mismatched feature order at prediction | Wrong predictions or shape error | Use `Pipeline`/`ColumnTransformer`; save the fitted object |
| Comparing models on different splits | Misleading accuracy difference | Same folds, same seed, same preprocessing |
| Reading a one-row accuracy gap as proof | Overinterpretation | Use confidence intervals and repeated evaluation |

### 11.2 Debugging checklist

1. Print `X_train.shape` and `X_test.shape`.
2. Print `X_train.dtypes`.
3. Print `X_train.isna().sum()` before and after scaling.
4. Check `scaler.mean_` and `scaler.scale_` after `fit`.
5. Check `np.round(X_train_scaled.describe(), 2)`.
6. Verify `X_test_scaled` uses the same scaler, not a new one.
7. Check that the target is not inside `X_train`.
8. Check for zero-variance columns.
9. Check for `ConvergenceWarning` in logistic regression; scaling may help.
10. Compare raw and scaled models on the same split.
11. Use a pipeline inside cross-validation.
12. Save the fitted scaler with the model, not just the model weights.

### 11.3 A subtle debugging example

If `describe()` shows `std = 1.0016` instead of `1.0`, the scaler is not broken. Pandas uses the sample standard deviation; `StandardScaler` uses the population standard deviation. The exact relationship is

$$
s_{\text{pandas}} = \sigma_{\text{sklearn}}\sqrt{\frac{n}{n-1}}.
$$

For $n=320$, that is $1.00156$. Rounding to one decimal gives `1.0`.

---

## 12. Practical checklist

Before scaling:

- [ ] Split the data.
- [ ] Identify numeric features.
- [ ] Check for missing values and infinities.
- [ ] Check for constant columns.
- [ ] Decide whether the model is scale-sensitive.
- [ ] Decide whether standardization or another scaler is appropriate.
- [ ] Check for outliers and decide how to handle them.

During scaling:

- [ ] `fit` on training data only.
- [ ] `transform` training and test data with the same object.
- [ ] Keep column names for debugging.
- [ ] Check `mean_` and `scale_`.
- [ ] Check the transformed `describe()`.

After scaling:

- [ ] Train the model on the transformed data.
- [ ] Evaluate on the transformed test data.
- [ ] Compare against an unscaled baseline on the same split.
- [ ] Use cross-validation with a pipeline for a reliable estimate.
- [ ] Save the fitted scaler with the model.
- [ ] Document the scaling choice and the training statistics.

---

## 13. Interview essentials

### Level 1 — Definitions

**Q1. What is standardization?**

Standardization is a feature transformation that replaces each value by its z-score:

$$
z=\frac{x-\mu}{\sigma}.
$$

It centers each feature at zero and scales it to unit standard deviation. It is used when features have different units or scales and the model is sensitive to those differences.

**Q2. What is the difference between standardization and normalization?**

In this course, standardization means z-score scaling, $(x-\mu)/\sigma$, with unbounded output and mean 0/std 1. Normalization means min–max scaling, $(x-\min)/(\max-\min)$, with output in $[0,1]$. The words are sometimes used loosely in other contexts, so state the formula you mean.

**Q3. Why do we need feature scaling?**

Many models use distances, dot products, gradients, or coefficient penalties. A feature with a large numerical range can dominate those calculations. Scaling puts features on a comparable footing and can improve optimization and regularization.

### Level 2 — Mechanics

**Q4. What does `StandardScaler.fit` learn?**

It learns `mean_`, `var_`, and `scale_` for each feature, plus `n_features_in_` and, for DataFrame input, `feature_names_in_`. `var_` is the population variance; `scale_` is the population standard deviation, with zero-variance features set to 1.

**Q5. What is the difference between `fit`, `transform`, and `fit_transform`?**

`fit` learns parameters. `transform` applies learned parameters. `fit_transform` calls `fit` and then `transform` on the same data. Use `fit_transform` on training data and `transform` on test data.

**Q6. Why must the scaler be fitted on training data only?**

Because the mean and standard deviation are parameters. If they are learned from the test set or the full dataset, test information enters training. That is data leakage and produces optimistic evaluation.

**Q7. Why does `transform` return a NumPy array?**

Scikit-learn transformers generally return arrays so they can work with many input types. If you need column names for plotting or debugging, wrap the output in a DataFrame using the original columns.

### Level 3 — Mathematics

**Q8. Prove that standardized training data has mean 0.**

$$
\frac{1}{n}\sum_i z_i
=
\frac{1}{n\sigma}\sum_i(x_i-\mu)
=
\frac{1}{n\sigma}\left(\sum_i x_i-n\mu\right)
=0.
$$

**Q9. Why does pandas show a standard deviation slightly above 1 after standardization?**

`StandardScaler` uses the population standard deviation (divide by $n$); pandas `describe()` uses the sample standard deviation (divide by $n-1$). The ratio is $\sqrt{n/(n-1)}$, which is 1.00156 for $n=320$.

**Q10. What happens to correlation under standardization?**

Pearson correlation is unchanged by positive affine transformations of either variable. Standardization changes units but not linear correlation.

**Q11. What happens if a feature has zero variance?**

The formula divides by zero. Scikit-learn sets the scale to 1 for that feature, so the transformed values are all zero. A constant feature is uninformative and should usually be removed.

**Q12. Why are outliers still large after standardization?**

Standardization is linear. It recenters and rescales but does not delete extreme values. If an outlier is far from the mean, it remains far in z-units unless the standard deviation is itself inflated by the outlier, which compresses the rest of the data.

### Level 4 — Diagnosis

**Q13. Your scaled model has lower test accuracy than the unscaled model. What do you check?**

Check whether the difference is larger than sampling noise, whether both models used the same split and settings, whether the scaler was fitted correctly, whether outliers or distribution shift are present, and whether regularization or convergence changed. A single split is not enough to decide.

**Q14. Your training accuracy is high but validation accuracy is low. Could scaling be the cause?**

Scaling alone does not cause overfitting. It can change the optimization path and the effective regularization, but a large train/validation gap usually points to model complexity, too many features, leakage, or a non-representative validation split. Check the pipeline and the data split before blaming the scaler.

**Q15. `LogisticRegression` emits a convergence warning on unscaled data. What do you do?**

Scale the features, increase `max_iter`, or change the solver. Scaling is often the cleanest fix because it improves the conditioning of the optimization problem. Check `n_iter_` and the warning text.

**Q16. Your test set contains a value far outside the training range. Is that an error?**

Not necessarily. The scaler will produce a large z-score. The model may extrapolate or fail honestly. Check whether the value is plausible, whether the training data are representative, and whether a robust scaler or a different model is appropriate.

### Level 5 — Design

**Q17. When would you standardize features?**

Standardize for KNN, k-means, SVM with distance kernels, PCA, regularized linear models, and neural networks when features have different units. Consider it for any model whose optimization or geometry is scale-sensitive.

**Q18. When would you not standardize?**

For tree-based models, standardization is usually unnecessary. It may also be inappropriate when features are already in comparable units and their variances carry meaningful information, or when a bounded representation is required.

**Q19. How do you choose between standardization, min–max, and robust scaling?**

Use standardization for approximately symmetric features and scale-sensitive models; min–max when a bounded range is required and outliers are not extreme; robust scaling when outliers are present but should not dominate; power transforms when skew is the main problem.

**Q20. How do you put scaling into a production pipeline?**

Fit the scaler on training data, save it with the model, and apply the same `transform` to every future input. Never refit on production data. Use a `Pipeline` so preprocessing and model are one object.

### Level 6 — Advanced

**Q21. Why does standardization help gradient descent?**

It makes the Hessian's eigenvalues more comparable, reducing the condition number. The loss surface becomes less elongated, so gradient descent can take larger steps without zigzagging.

**Q22. Why does standardization change regularized linear models?**

The penalty is applied to coefficients. A coefficient's magnitude depends on the feature's units. Standardizing changes the coefficient scale and therefore the effective strength of the penalty.

**Q23. Why are trees usually invariant to monotone transformations?**

A tree splits on order, not magnitude. A monotone transformation preserves order, so the same partitions are available. Floating-point tie-breaking can still cause small implementation-level differences.

**Q24. Does standardization make data Gaussian?**

No. It changes mean and standard deviation, not skew or multimodality. Use a power transform or another distributional method if Gaussianity is the goal.

**Q25. What is the relationship between standardization and Mahalanobis distance?**

Standardizing each feature divides by its marginal standard deviation. Mahalanobis distance goes further and accounts for the full covariance structure. Standardization is a diagonal, marginal rescaling; Mahalanobis uses the inverse covariance matrix.

### Level 7 — Synthesis

**Q26. A model works in cross-validation but fails in production. How could scaling be involved?**

The production feature distribution may differ from training, so the training means and scales may no longer be appropriate. A feature may have shifted, a new category may appear, or the production pipeline may be refitting the scaler. Check schema, distribution shift, and whether the saved scaler is the one used in production.

**Q27. You have a small dataset with 10,000 features. Would you standardize before PCA?**

Usually yes, because raw scales can dominate the principal components. But with 10,000 features and few samples, also consider regularization, feature selection, and the risk of overfitting the covariance estimate. Standardization is necessary but not sufficient.

**Q28. Explain to a stakeholder why scaling is not the same as cleaning.**

Scaling changes the units used to compare features. Cleaning fixes missing, invalid, or inconsistent values. A scaled dataset can still contain errors and outliers; a cleaned dataset may still need scaling. They solve different problems.

---

## 14. Exam practice

### Q1. Compute the mean, population standard deviation, and z-scores for `[22, 35, 48]`.

Mean:

$$
\mu=\frac{22+35+48}{3}=35.
$$

Population variance:

$$
\sigma^2=\frac{(22-35)^2+(35-35)^2+(48-35)^2}{3}
=\frac{169+0+169}{3}=112.667.
$$

Population standard deviation:

$$
\sigma=\sqrt{112.667}\approx 10.614.
$$

z-scores:

$$
z=\left[\frac{-13}{10.614},0,\frac{13}{10.614}\right]
\approx[-1.225,0,1.225].
$$

### Q2. Why does `StandardScaler` use the population standard deviation?

It is the implementation convention: divide by $n$, not $n-1$. The transformed training data then has population variance exactly 1. Pandas `describe()` reports the sample standard deviation, so it shows a value slightly above 1 before rounding.

### Q3. A student fits the scaler on the full dataset and then splits. Explain the leakage.

The scaler's mean and standard deviation are computed using every row, including the rows that later become the test set. The training transformation therefore contains information about the test distribution. The test set is no longer independent, and the evaluation can be optimistic.

### Q4. The official notebook reports unscaled 0.6583 and scaled 0.8667. The personal saved run reports 0.8642 and 0.8519. Are these contradictory?

No. They are different runs:

- official: `test_size=0.3`, `random_state=0`, clean data, 120 test rows;
- personal saved: `test_size=0.2`, `random_state=42`, outlier-inclusive data, 81 test rows;
- personal clean reproduction: 0.8875 vs 0.8625, 80 test rows.

Different splits and data states produce different accuracies. The lesson is to compare models under the same protocol and not to treat one split as universal evidence.

### Q5. Why did the decision tree tie in the saved runs?

A monotone transformation preserves the order of values, so the tree can make the same splits. In exact arithmetic, the predictions are unchanged. Floating-point tie-breaking can occasionally change a split or a prediction, so “usually invariant” is more accurate than “always identical.”

### Q6. After appending three extreme rows, why does the salary mean change so much?

The new salaries include 250,000 and 350,000, far above the original maximum of 150,000. The mean is sensitive to extreme values, so it rises from about 70,281 to about 72,152. The standard deviation also rises, which compresses the bulk of the data into a narrower z-range while the extremes remain far out.

### Q7. When would you use `RobustScaler` instead of `StandardScaler`?

When the data contain genuine outliers and you do not want the mean and standard deviation to be dominated by them. `RobustScaler` uses the median and IQR, which are less sensitive to extremes. It is not a substitute for fixing data errors; if an extreme value is a mistake, correct or remove it.

### Q8. Write a leakage-safe pipeline for this dataset.

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)
score = model.score(X_test, y_test)
```

The pipeline fits the scaler on the training data and applies the same fitted scaler to the test data. If you cross-validate the pipeline, each fold fits its own scaler.

---

## 15. One-page revision

### Definitions

- **Feature scaling:** putting numeric features on comparable scales.
- **Standardization:** $z=(x-\mu)/\sigma$.
- **Min–max normalization:** $x'=(x-\min)/(\max-\min)$.
- **Robust scaling:** $(x-\operatorname{median})/\operatorname{IQR}$.
- **Fit:** learn parameters.
- **Transform:** apply parameters.
- **Leakage:** using information from outside the training data to learn a parameter.

### Core formulas

$$
z=\frac{x-\mu}{\sigma}
$$

$$
\mu=\frac{1}{n}\sum_i x_i
$$

$$
\sigma=\sqrt{\frac{1}{n}\sum_i(x_i-\mu)^2}
$$

$$
x=z\sigma+\mu
$$

$$
s_{\text{pandas}}=\sigma_{\text{sklearn}}\sqrt{\frac{n}{n-1}}
$$

### Workflow

```text
raw data -> split -> fit scaler on train -> transform train/test
         -> model -> evaluate on test
```

### Model sensitivity

| Model | Scale needed? |
|---|---|
| KNN, k-means, SVM | Yes |
| Regularized linear models | Yes |
| PCA | Yes |
| Neural networks | Yes |
| Decision trees / forests | Usually no |
| Naive Bayes | Usually no |

### Leakage rules

1. Split before learning any parameter.
2. Fit the scaler on training data only.
3. Transform test data with the training scaler.
4. Never call `fit_transform` on test data.
5. Put scaling inside a pipeline for cross-validation.
6. Save the fitted scaler with the model.

### Final checklist

- [ ] Data split.
- [ ] Missing values checked.
- [ ] Constant columns checked.
- [ ] Scaler fitted on train.
- [ ] Train and test transformed with the same scaler.
- [ ] `mean_` and `scale_` inspected.
- [ ] `describe()` inspected.
- [ ] Model compared on the same split.
- [ ] Outliers handled separately.
- [ ] Pipeline used for cross-validation.
- [ ] Fitted scaler saved for production.

# Part B — Advanced Reading and Reference

This part preserves the deeper material. It is optional on a first reading of Part A. The goal is not to repeat the basic recipe but to explain why the recipe has the properties it has, where those properties stop holding, and how the scikit-learn implementation behaves at the edges.

## B1. Formal setup and notation

Let $X$ be an $n\times p$ training matrix. Row $i$ is an observation, column $j$ is a feature, and $x_{ij}$ is the value of feature $j$ for observation $i$. Write

$$
\mu_j=\frac{1}{n}\sum_{i=1}^{n}x_{ij},
\qquad
\sigma_j^2=\frac{1}{n}\sum_{i=1}^{n}(x_{ij}-\mu_j)^2,
$$

for the training population mean and population variance of feature $j$. Define

$$
\mu=
\begin{pmatrix}
\mu_1\\
\vdots\\
\mu_p
\end{pmatrix},
\qquad
D=\operatorname{diag}(\sigma_1,\ldots,\sigma_p).
$$

The standardized training matrix can be written compactly as

$$
Z=(X-\mathbf{1}\mu^\top)D^{-1},
$$

where $\mathbf{1}$ is an $n\times 1$ vector of ones. Elementwise,

$$
z_{ij}=\frac{x_{ij}-\mu_j}{\sigma_j}.
$$

This is a **column-wise affine map**. The map is applied independently to each feature; it is not a rotation, a projection, or a distributional transform.

If $\sigma_j=0$, the affine map is not defined because $D^{-1}$ does not exist. Scikit-learn replaces that zero scale by 1, so a constant training column becomes a column of zeros. The constant feature carries no information and should normally be removed.

The inverse map, when all scales are positive, is

$$
X=Z D+\mathbf{1}\mu^\top,
\qquad
x_{ij}=z_{ij}\sigma_j+\mu_j.
$$

The fitted scaler therefore stores two vectors of length $p$: a location vector and a scale vector. Those vectors are the entire learned state needed for transformation. Everything else in the object is metadata or bookkeeping.

### B1.1 Population versus sample convention

The distinction between dividing by $n$ and dividing by $n-1$ is not cosmetic. `StandardScaler` uses the population convention. If $s_j$ is the sample standard deviation reported by pandas,

$$
s_j=\sqrt{\frac{1}{n-1}\sum_i(x_{ij}-\bar{x}_j)^2},
$$

and $\sigma_j$ is the scaler's population scale, then

$$
s_j=\sigma_j\sqrt{\frac{n}{n-1}}.
$$

For $n=320$, the factor is $1.00156$; for $n=80$, it is $1.00631$. This is why a standardized training column can show a pandas `std` slightly above 1 even when the scaler is correct. The transformed column has population variance exactly 1 by construction, and sample variance $n/(n-1)$.

### B1.2 What is learned and what is not

`fit` learns $\mu_j$ and $\sigma_j$ from the rows it receives. It does not learn a distribution, a correlation matrix, a covariance matrix, or a probability model. `transform` applies the stored pair to new rows. The test rows are not used to update the pair.

This is the formal reason leakage is possible: the pair $(\mu_j,\sigma_j)$ is an estimated parameter. If the estimate is computed from data that includes the test rows, the test rows have influenced the training pipeline. The model is then evaluated on data that helped define its preprocessing.

## B2. Geometry, conditioning, and optimization

### B2.1 Standardization as diagonal preconditioning

Suppose the raw feature matrix has columns with very different scales. The normal-equation matrix for ordinary least squares is $X^\top X$. After centering and scaling,

$$
Z=(X-\mathbf{1}\mu^\top)D^{-1},
$$

the normal-equation matrix becomes

$$
Z^\top Z
=
D^{-1}(X-\mathbf{1}\mu^\top)^\top(X-\mathbf{1}\mu^\top)D^{-1}.
$$

The inner matrix is the centered sum-of-squares matrix of the raw features. Dividing by $n$ gives the covariance matrix $\Sigma_X$; therefore

$$
\frac{1}{n}Z^\top Z
=
D^{-1}\Sigma_X D^{-1}
=
R,
$$

where $R$ is the correlation matrix of the training features. Standardization is a **diagonal preconditioner**: it rescales each coordinate so that its marginal variance is 1. It does not remove correlation between features.

For two uncorrelated features with standard deviations $s_1$ and $s_2$, the raw Hessian is proportional to

$$
\begin{pmatrix}
s_1^2 & 0\\
0 & s_2^2
\end{pmatrix},
$$

whose condition number is $(s_1/s_2)^2$ if $s_1\ge s_2$. If $s_1=1000$ and $s_2=1$, the condition number is $10^6$. After standardization, the matrix is the $2\times 2$ identity and the condition number is 1. This is the cleanest mathematical statement of why scaling helps gradient-based optimization: it removes the trivial scale disparity before the optimizer sees the problem.

The condition number is not the only thing that matters. If the features are strongly correlated, the correlation matrix itself can be ill-conditioned. Standardization cannot fix that; PCA, ridge regularization, or feature selection may be needed.

### B2.2 Gradient descent and the shape of the loss surface

For a quadratic loss with Hessian $H$, gradient descent converges slowly when $H$ has both very large and very small eigenvalues. The level sets are elongated ellipses; a step large enough to make progress along the flat direction can overshoot along the steep direction. The condition number $\kappa(H)=\lambda_{\max}/\lambda_{\min}$ controls this mismatch.

Standardization makes the diagonal entries of the Hessian comparable, but it does not make the Hessian spherical when features are correlated. It improves the **scale-induced** part of the conditioning problem, not every possible conditioning problem.

For logistic regression, the Hessian of the negative log-likelihood is

$$
H(\beta)=X^\top W X,
$$

where $W$ is diagonal with entries $p_i(1-p_i)$ for the current fitted probabilities. The same reasoning applies: wildly different column scales make $H$ badly conditioned. The default scikit-learn solver, `lbfgs`, is more robust than plain gradient descent, but scale can still affect convergence, the number of iterations, and the solution when regularization is active.

### B2.3 Coefficient mapping between raw and standardized models

Consider a linear predictor with an intercept:

$$
\eta=b+\sum_{j=1}^{p}w_jx_j.
$$

Substitute $x_j=\sigma_jz_j+\mu_j$:

$$
\eta
=
b+\sum_j w_j(\sigma_jz_j+\mu_j)
=
\left(b+\sum_j w_j\mu_j\right)
+
\sum_j (w_j\sigma_j)z_j.
$$

Therefore the standardized model has coefficients

$$
w_j^{z}=w_j^{x}\sigma_j,
\qquad
b^{z}=b^{x}+\sum_j w_j^{x}\mu_j,
$$

and the inverse mapping is

$$
w_j^{x}=\frac{w_j^{z}}{\sigma_j},
\qquad
b^{x}=b^{z}-\sum_j \frac{w_j^{z}\mu_j}{\sigma_j}.
$$

The fitted **predictions** of an unregularized linear model with an intercept are unchanged by this reparameterization, because both expressions are the same linear function written in different coordinates. The coefficients change because their units change: a coefficient on salary in dollars is not numerically comparable to a coefficient on salary in standard deviations.

This mapping is the correct way to compare coefficients across scaled and unscaled models. Comparing raw coefficient magnitudes directly across features is usually meaningless unless the features share units and comparable scales.

### B2.4 Why regularization breaks the invariance

Ridge regression minimizes

$$
\sum_i\left(y_i-b-\sum_jw_jx_{ij}\right)^2+\lambda\sum_jw_j^2.
$$

Under the reparameterization above, the residual term is unchanged, but the penalty becomes

$$
\lambda\sum_j\left(\frac{w_j^{z}}{\sigma_j}\right)^2
$$

if the penalty is written in terms of the raw coefficients, or

$$
\lambda\sum_j(w_j^{z})^2
$$

if the penalty is written in terms of the standardized coefficients. These are not the same penalty unless all $\sigma_j$ are equal. Standardizing changes which coefficient vectors are cheap and which are expensive, so the fitted coefficients and predictions can change even when the model class is the same.

The same logic applies to Lasso: the $\ell_1$ penalty is not invariant to rescaling coordinates, and the sparsity pattern can change. This is why “standardize before penalized linear models” is a modelling recommendation, not a cosmetic step.

Scikit-learn's `LogisticRegression` is L2-regularized by default (`penalty='l2'`, `C=1.0`), so its fitted probabilities are not guaranteed to be invariant under feature scaling. The official notebook's large unscaled-versus-scaled gap is therefore not surprising in principle. The exact historical number still depends on the solver, version, convergence, and data split; it should not be treated as a universal constant.

### B2.5 Distances and Mahalanobis geometry

For two rows $a$ and $b$, the raw squared Euclidean distance is

$$
\|a-b\|_2^2=\sum_j(a_j-b_j)^2.
$$

In standardized coordinates,

$$
a_j-b_j=\sigma_j(z_{a,j}-z_{b,j}),
$$

so

$$
\|a-b\|_2^2=\sum_j\sigma_j^2(z_{a,j}-z_{b,j})^2.
$$

Raw Euclidean distance is therefore a **weighted** distance in standardized coordinates, with weight $\sigma_j^2$ on coordinate $j$. A feature with a large standard deviation receives a large weight whether or not it is more informative. Standardization sets all those weights to 1.

Mahalanobis distance goes further:

$$
d_M(a,b)=\sqrt{(a-b)^\top\Sigma^{-1}(a-b)}.
$$

It accounts for both the marginal scales and the correlations. Standardization is the diagonal special case that ignores correlations. If $\Sigma$ is diagonal, Mahalanobis distance reduces to Euclidean distance after dividing each coordinate by its standard deviation.

This distinction matters in interviews: standardization does not “whiten” the data. Whitening requires a full linear transformation, usually based on the covariance matrix or PCA. Standardization only rescales axes.

### B2.6 PCA: covariance versus correlation

PCA on raw features finds directions that maximize raw variance. If salary is measured in dollars and age in years, the first component will usually be dominated by salary simply because its numerical variance is larger. Standardizing first makes PCA operate on the correlation matrix rather than the covariance matrix. The resulting components are often more interpretable when the features have different units.

Neither choice is universally correct. If the features are already in a common unit and their variances are meaningful, covariance PCA may be the intended analysis. Standardization is a decision about which variation should count as comparable.

### B2.7 Trees and monotone invariance

A decision tree splits on inequalities such as $x_j\le t$. If $\sigma_j>0$, standardization is strictly increasing:

$$
x_j\le t
\iff
\frac{x_j-\mu_j}{\sigma_j}\le\frac{t-\mu_j}{\sigma_j}.
$$

The set of possible partitions by order is unchanged, so in exact arithmetic the tree can produce the same partitions and predictions. The printed threshold changes, but the ordering of rows does not.

This is an invariance of the **model class**, not a promise of bit-identical output from every implementation. Floating-point rounding, tied impurity values, and implementation-specific tie-breaking can change a split at the margin. The correct interview statement is “trees are usually invariant to monotone feature transformations,” not “trees always give identical predictions.”

The same reasoning explains why unregularized OLS with an intercept is invariant in exact arithmetic, while ridge, Lasso, and default L2 logistic regression are not: the former depends only on the linear span, while the latter penalize coordinate-specific coefficients.

## B3. Statistical properties and interpretation

### B3.1 Affine equivariance

Suppose a feature is transformed as $y=ax+b$ with $a>0$. Its mean becomes $a\mu+b$ and its standard deviation becomes $a\sigma$. Therefore

$$
\frac{y-(a\mu+b)}{a\sigma}
=
\frac{a(x-\mu)}{a\sigma}
=
\frac{x-\mu}{\sigma}.
$$

Standardization is **affine equivariant**: changing the original unit by a positive affine transformation does not change the standardized values. This is exactly why the choice of dollars versus thousands of dollars does not matter after standardization. A negative $a$ flips the sign of the z-scores but preserves their magnitudes; standard deviations are positive, so the usual scaler uses $a>0$.

### B3.2 What is preserved and what is not

A positive affine transformation preserves:

- the order of values;
- all quantiles, up to the same affine map;
- skewness and kurtosis;
- the Pearson correlation with other features;
- the shape of the histogram or density, up to the axis labels.

It changes:

- the mean and standard deviation;
- the numerical values of distances;
- the coefficients of models that use the feature;
- the effective strength of coefficient penalties;
- the numerical conditioning of optimization.

It does not:

- make a skewed distribution symmetric;
- make a multimodal distribution unimodal;
- remove outliers;
- create a Gaussian distribution;
- guarantee better predictive accuracy.

The statement “standardization makes the data normal” is a category error. “Normal” in “normal distribution” is not the same word as “normalize” in “normalize the scale.” A power transform or quantile transform changes shape; standardization changes location and scale.

### B3.3 Estimated statistics are random variables

The scaler's `mean_` and `scale_` are estimates from a finite training sample. If the training set were resampled, they would change. The transformed test values therefore use estimated, not true, population parameters. This is normal and unavoidable; the important discipline is that the estimates come from training data only.

For a feature with population standard deviation $\sigma$, the standard error of the sample mean is approximately $\sigma/\sqrt{n}$ under independent sampling. The standard error of the sample variance is larger and depends on the fourth moment. With small training sets, the learned scale can be noticeably unstable, especially when the feature is heavy-tailed.

This is one reason to prefer robust statistics when the tail is genuinely heavy: the mean and standard deviation can be dominated by a few observations, and their sampling variability can be large.

### B3.4 Distribution shift after deployment

A fitted scaler encodes the training distribution. If production inputs shift, a value that was ordinary in training can become an extreme z-score in production, or vice versa. The scaler is not “wrong”; it is applying the old reference distribution.

Monitoring should therefore include the input distribution, not only the output accuracy. Useful checks include:

- mean and standard deviation of each feature over time;
- quantiles and missingness rates;
- the fraction of values beyond the training range;
- population stability indices or domain-specific drift metrics;
- schema and feature-order checks.

Refitting the scaler on production data without retraining the model is usually a bad idea: it changes the meaning of the model's coefficients while leaving the model fixed. A controlled retraining pipeline should refit preprocessing and model together on a representative training window, then validate before deployment.

### B3.5 Leakage beyond the scaler

Scaling is only one parameter-learning step that can leak. The same rule applies to:

- imputation statistics;
- category vocabularies and target encodings;
- feature selection;
- PCA or other decomposition;
- hyperparameter tuning;
- outlier thresholds;
- calibration parameters.

If a step learns anything from data, it belongs inside the training fold. A `Pipeline` is the practical way to enforce this during cross-validation.

## B4. Robustness and alternative transformations

### B4.1 Influence and breakdown

The sample mean has influence function

$$
\operatorname{IF}(x;\mu)=x-\mu.
$$

A single observation can move the mean by an arbitrarily large amount, so the breakdown point of the mean is 0. The sample variance has influence function proportional to

$$
(x-\mu)^2-\sigma^2,
$$

which is also unbounded. One extreme value can inflate the variance and therefore shrink everyone else's z-scores.

The median and the interquartile range have a breakdown point of about 50%: roughly half the data must be corrupted before the estimate can be driven arbitrarily far. That does not make them immune to contamination, but it makes them much less sensitive than the mean and standard deviation.

### B4.2 A small influence calculation

Take the clean values

$$
[1,\ 2,\ 3,\ 4].
$$

The mean is $2.5$, the population variance is

$$
\frac{(-1.5)^2+(-0.5)^2+(0.5)^2+(1.5)^2}{4}=1.25,
$$

and the population standard deviation is $\sqrt{1.25}\approx1.1180$. The z-score of 4 is

$$
\frac{4-2.5}{1.1180}\approx1.3416.
$$

Now append 100. The new mean is

$$
\frac{1+2+3+4+100}{5}=22.
$$

The new population variance is

$$
\frac{(-21)^2+(-20)^2+(-19)^2+(-18)^2+78^2}{5}
=
\frac{7610}{5}=1522,
$$

so the new standard deviation is $\sqrt{1522}\approx39.0128$. The z-score of 100 is only

$$
\frac{100-22}{39.0128}\approx1.999,
$$

and the z-score of the formerly high value 4 becomes

$$
\frac{4-22}{39.0128}\approx-0.461.
$$

The outlier has compressed the bulk toward the negative side and made itself look less extreme than its raw distance suggests. This is the mathematical reason a z-score threshold such as $|z|>3$ can fail on contaminated data: the standard deviation is estimated from the same contaminated sample.

### B4.3 `RobustScaler`

`RobustScaler` uses

$$
z_j^{\text{robust}}
=
\frac{x_j-\operatorname{median}(x_j)}
{\operatorname{IQR}(x_j)},
$$

where the default interquartile range is $Q_{75}-Q_{25}$. Its fitted attributes are `center_` and `scale_`. The default `quantile_range=(25.0, 75.0)` can be changed; a narrower range is more local but less stable, while a wider range is smoother but less robust.

`unit_variance=False` is the default. If `unit_variance=True`, the scale is divided by the normal-consistency factor

$$
\Phi^{-1}(0.75)-\Phi^{-1}(0.25)\approx1.34898,
$$

so that a normally distributed feature has a robust scale close to 1. This is a convention, not a requirement.

`RobustScaler` still does not remove outliers. It only prevents them from dominating the center and scale. If the extreme values are data errors, fix the data; if they are genuine rare events, decide explicitly whether to keep, cap, model separately, or use a robust method.

### B4.4 Median absolute deviation

The median absolute deviation is

$$
\operatorname{MAD}=\operatorname{median}\left(|x_i-\operatorname{median}(x)|\right).
$$

For approximately normal data, multiplying by 1.4826 gives a consistent estimate of the standard deviation. MAD has a breakdown point of about 50% and is often used in robust z-score rules. It is not what `RobustScaler` uses by default, but it is a useful alternative when the IQR is too coarse.

### B4.5 Min–max scaling and its boundary behaviour

Min–max scaling is

$$
x'_j=\frac{x_j-\min(x_j)}{\max(x_j)-\min(x_j)}.
$$

For training values inside the fitted range, the output lies in $[0,1]$. For a test value outside the training range, the output can lie outside $[0,1]$. For example, if the training range is $[0,100]$ and a test value is 150, the transformed value is 1.5, not 1.0. Scikit-learn's `MinMaxScaler(clip=False)` is the default and does not clip; `clip=True` caps the result at the feature range.

This is an important practical difference from the simplified statement “min–max always gives $[0,1]$.” The guarantee is relative to the fitted range, not to every future input.

Min–max scaling is also highly sensitive to a single extreme value: one large maximum compresses the rest of the data into a narrow interval. Standardization is less dramatic but still affected through the mean and standard deviation.

### B4.6 `MaxAbsScaler`

`MaxAbsScaler` divides each feature by its maximum absolute value:

$$
x'_j=\frac{x_j}{\max_i|x_{ij}|}.
$$

It maps the training data into $[-1,1]$ and preserves zeros. Preserving zeros is valuable for sparse data, where centering would destroy sparsity. It is not robust to a single large magnitude, and it does not center the data.

### B4.7 Power and quantile transforms

`PowerTransformer` applies a parametric monotone transformation (Yeo–Johnson or Box–Cox) to reduce skew and approximate a Gaussian shape. Box–Cox requires strictly positive values; Yeo–Johnson handles zero and negative values. The transformation is learned from the training data and must be applied to test data with the same fitted object.

`QuantileTransformer` maps each feature's empirical distribution to a uniform or normal distribution. It is nonlinear and can be very effective for heavy tails, but it changes distances and can behave unpredictably outside the training range. It also has hyperparameters such as `n_quantiles` and `subsample`.

Neither is a replacement for standardization when the goal is only to put features on comparable scales. They solve a different problem: changing the shape of the distribution.

### B4.8 `Normalizer` is not column standardization

`sklearn.preprocessing.Normalizer` normalizes each **row** to unit norm:

$$
x_i'=\frac{x_i}{\|x_i\|_p}.
$$

It is common in text classification and some similarity pipelines. It does not center or scale columns, and it does not solve the “salary is in dollars, age is in years” problem. Do not confuse it with min–max normalization or with z-score standardization.

## B5. Scikit-learn implementation details

### B5.1 Fitted attributes

After `fit`, `StandardScaler` exposes:

| Attribute | Meaning |
|---|---|
| `mean_` | training mean per feature |
| `var_` | training population variance per feature |
| `scale_` | training population standard deviation, with zero-variance features set to 1 |
| `n_features_in_` | number of features seen during `fit` |
| `feature_names_in_` | feature names when the input is a DataFrame with string column names |
| `n_samples_seen_` | sample count used by `partial_fit` |

There is no `std_` attribute. The name `scale_` reflects the fact that the stored value is the divisor used by `transform`, and it can be 1 for a constant feature even though the true standard deviation is 0.

Before `fit`, accessing `mean_` raises `AttributeError`. The trailing underscore is a scikit-learn convention for “learned from data.”

### B5.2 How `fit` computes the statistics

`StandardScaler` computes the mean and variance in a numerically stable incremental form. The implementation uses an update based on the previous mean, previous variance, and sample count, rather than a naive sum of squares that can lose precision when values are large. This is the same family of ideas used by Welford's online algorithm.

The variance is the population variance. The scale is its square root, with zeros replaced by 1. The replacement is not a mathematical claim that a constant feature has standard deviation 1; it is an implementation choice that prevents division by zero and makes the transformed constant column all zeros.

### B5.3 `fit_transform` and copying

`fit_transform(X)` is equivalent to `fit(X).transform(X)`. It is convenient on training data and may avoid an extra copy internally. It is not a different statistical operation.

The `copy` constructor parameter controls whether the transformer tries to avoid copying during transformation. `copy=False` may modify the input in place for some array types, so do not use it casually on data you still need in raw form. The default `copy=True` is safer.

### B5.4 Sparse matrices

Centering a sparse matrix destroys sparsity. If a sparse row has stored zeros, subtracting a non-zero mean turns every implicit zero into a non-zero value. For this reason, `StandardScaler(with_mean=True)` raises an error for sparse input. Use

```python
StandardScaler(with_mean=False)
```

to scale without centering, or use `MaxAbsScaler`, which preserves zeros by construction. A sparse pipeline should be designed deliberately; converting a large sparse matrix to dense just to center it can exhaust memory.

### B5.5 `partial_fit` and streaming

`partial_fit` updates the running mean and variance from one batch at a time:

```python
scaler = StandardScaler()
scaler.partial_fit(batch_1)
scaler.partial_fit(batch_2)
```

The result is intended to match a full `fit` up to numerical precision. The batches used for `partial_fit` must still be training data. Calling `partial_fit` on a stream that includes future test data is leakage, even if the model itself is trained later.

`partial_fit` is useful when the training set does not fit in memory. It is not a license to refit on production data without a validation strategy.

### B5.6 Feature names and output type

`transform` returns a NumPy array by default. Recent scikit-learn versions support

```python
scaler.set_output(transform="pandas")
```

which makes the transformer return a DataFrame when the input is a DataFrame. This is convenient, but it is version-dependent; do not assume it exists in older environments. Wrapping the array manually is portable:

```python
X_train_scaled = pd.DataFrame(
    scaler.transform(X_train),
    columns=X_train.columns,
    index=X_train.index,
)
```

The index is worth preserving in pandas workflows. Dropping it can silently misalign rows in later joins or plots.

### B5.7 Pipelines and cross-validation

A pipeline makes the preprocessing and model one estimator:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])
```

When `model.fit(X_train, y_train)` is called, the pipeline fits the scaler on `X_train`, transforms it, and fits the classifier on the transformed data. When `model.predict(X_test)` is called, the pipeline applies the fitted scaler to `X_test` and predicts.

During `cross_val_score(model, X, y, cv=5)`, each fold clones the entire pipeline, fits its own scaler on that fold's training part, and evaluates on the validation part. This is the correct way to avoid preprocessing leakage. Fitting the scaler once before cross-validation and then passing the transformed data to `cross_val_score` is not equivalent.

### B5.8 Persistence and numerical precision

Save the fitted pipeline, not just the classifier. A model trained on standardized features is unusable without the exact scaler that produced those features. `joblib.dump` or `pickle` can persist the pipeline, but the environment's library versions should be recorded.

Floating-point precision matters when features have very large offsets and small variations. If values are around $10^{12}$ with differences around $10^{-3}$, float64 may already struggle; centering first can improve accuracy. Standardization itself subtracts the mean, but the subtraction happens in the same floating-point representation, so it cannot recover information that was already lost when the data were stored.

## B6. Model-by-model advanced notes

| Model | Effect of standardization | Important nuance |
|---|---|---|
| OLS with intercept | Predictions invariant in exact arithmetic | Coefficients transform; numerical conditioning improves |
| OLS without intercept | Not invariant | The intercept is constrained to zero, so rescaling changes the fit |
| Ridge / Lasso | Changes fit | Penalty is coordinate-dependent |
| Logistic regression, unregularized | Predictions invariant in exact arithmetic | Coefficients transform; solver still benefits |
| Logistic regression, default L2 | Changes fit | Scikit-learn default is regularized |
| Linear SVM | Changes margin and fit | Margin depends on feature scale |
| RBF/polynomial SVM | Usually important | Kernel distances depend on scale |
| KNN | Usually important | Neighbor identity changes |
| k-means | Usually important | Cluster geometry changes |
| PCA | Important for heterogeneous units | Standardization gives correlation PCA |
| Gaussian Naive Bayes | Usually invariant in exact arithmetic | Numerical and estimation differences remain |
| Multinomial Naive Bayes | Not a simple scaling case | It expects non-negative counts |
| Decision trees / forests / boosting | Usually unnecessary | Monotone invariance; thresholds relabel |
| Neural networks | Usually important | Activations, initialization, and gradients are scale-sensitive |

“Usually invariant” is a statement about the ideal model, not a guarantee for a particular implementation, regularizer, or numerical precision. “Usually unnecessary” does not mean “harmful”; scaling a tree is often harmless but adds computation and changes threshold interpretation.

### B6.1 Binary and count features

Standardizing a binary indicator is mathematically allowed but changes its interpretation. A binary feature with mean $p$ has standard deviation $\sqrt{p(1-p)}$; its standardized values are two points. The coefficient is rescaled accordingly. For linear models with regularization, this can change the effective penalty; for trees, it changes nothing essential.

Count features are often skewed and have a meaningful zero. Standardization preserves the zero only if the mean is zero, which is usually not true. If zero has a special meaning (no events, no purchases), consider a count-aware transform such as $\log(1+x)$ before scaling, and think about whether centering destroys an interpretable baseline.

### B6.2 Target scaling

Scaling the target is a separate decision. For regression, standardizing $y$ changes the scale of residuals and coefficients; predictions can be inverse-transformed back to original units. It may help optimization in some models, but it does not change the information content. For classification, the target is a label, not a feature; do not standardize it.

If you standardize the target, keep the target scaler and apply `inverse_transform` before reporting metrics in original units. Otherwise an RMSE of 0.5 may mean “half a standard deviation,” not “half a dollar.”

## B7. Validation, leakage, and production

### B7.1 The formal leakage rule

A preprocessing parameter may be estimated only from the data that are allowed to influence the model's training. In a train/test protocol, that means the training partition. In $k$-fold cross-validation, it means the training part of each fold. In nested cross-validation, it means the inner training part used for model selection.

The rule is not “never transform the test set.” The test set must be transformed to be scored. The rule is “never estimate a parameter from the test set.”

### B7.2 Cross-validation with preprocessing

Correct:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
```

where `model` is the full pipeline.

Incorrect:

```python
X_scaled = StandardScaler().fit_transform(X)
scores = cross_val_score(classifier, X_scaled, y, cv=5)
```

The second version estimates the scaler from all rows before the folds are formed, so each validation fold has influenced the preprocessing. The resulting score is usually optimistic, sometimes dramatically so for small datasets.

### B7.3 Nested cross-validation

If you tune hyperparameters as well as evaluate performance, use nested cross-validation: the inner loop selects hyperparameters, the outer loop estimates generalization. The scaler is fitted inside both loops, always on the relevant training part. This is more expensive but gives a less biased estimate when the dataset is small and many configurations are tried.

### B7.4 Production checklist

1. Save the fitted scaler or, preferably, the whole pipeline.
2. Record the library versions used to fit it.
3. Validate the input schema: column names, order, dtypes, units.
4. Apply the same transform to every future batch; never refit on production data.
5. Monitor input distributions and the fraction of out-of-range values.
6. Retrain preprocessing and model together when the distribution changes.
7. Keep a raw-data path for debugging; do not overwrite the source data with scaled values.
8. Document the units and the reference statistics.

### B7.5 Reproducibility and historical results

Accuracy numbers from a notebook are tied to the data, split, preprocessing, model defaults, and library versions that produced them. The official Day 24 notebook's unscaled LogisticRegression score of `0.6583` is a saved historical result. A current scikit-learn run on the same split can produce a different score because solver defaults, regularization behavior, and numerical details can change across versions.

The safe way to report such a result is:

> “The saved official notebook reports 0.6583 unscaled and 0.8667 scaled on a 120-row test set with `test_size=0.3`, `random_state=0`. Reproducing the split with the current environment gives different values, so the exact gap is environment-dependent.”

That is more honest than presenting the historical number as a universal property of standardization.

## B8. Expanded experiment and source tables

### B8.1 Official notebook

Settings: clean 400-row dataset, `test_size=0.3`, `random_state=0`, 280 training rows, 120 test rows.

| Quantity | Value |
|---|---:|
| `scaler.mean_` Age | 37.8642857 |
| `scaler.mean_` Salary | 69,807.1429 |
| Raw Age mean / std | 37.9 / 10.2 |
| Raw Salary mean / std | 69,807.1 / 34,641.2 |
| Standardized Age min / max | -1.9 / 2.2 |
| Standardized Salary min / max | -1.6 / 2.3 |
| Standardized Age 25% / 50% / 75% | -0.8 / -0.1 / 0.8 |
| Standardized Salary 25% / 50% / 75% | -0.8 / 0.0 / 0.5 |

Saved model results:

| Model | Unscaled | Scaled | Test rows correct |
|---|---:|---:|---:|
| LogisticRegression | 0.6583333 | 0.8666667 | 79/120 vs 104/120 |
| DecisionTreeClassifier | 0.875 | 0.875 | 105/120 vs 105/120 |

The official notebook's 25% and 75% values above are taken from the saved output. Some earlier drafts of these notes used `0.6/0.6` for the 75% row; that was a transcription error, corrected here against the reference notebook.

### B8.2 Personal notebook, saved outlier-inclusive state

Settings: 403 rows after appending three extremes, `test_size=0.2`, `random_state=42`, 322 training rows, 81 test rows.

| Quantity | Value |
|---|---:|
| `scaler.mean_` Age | 38.2080745 |
| `scaler.mean_` Salary | 72,152.1739 |
| `scaler.scale_` Age | 11.1420889 |
| `scaler.scale_` Salary | 39,318.1617 |
| Standardized Age min / max | -3.0 / 5.1 |
| Standardized Salary min / max | -1.8 / 7.1 |
| Standardized Age 25% / 50% / 75% | -0.6 / -0.1 / 0.7 |
| Standardized Salary 25% / 50% / 75% | -0.7 / -0.0 / 0.4 |

Saved model results:

| Model | Unscaled | Scaled | Test rows correct |
|---|---:|---:|---:|
| LogisticRegression | 0.8641975 | 0.8518519 | 70/81 vs 69/81 |
| DecisionTreeClassifier | 0.8888889 | 0.8888889 | 72/81 vs 72/81 |

These are the outputs stored in `my_notebook.ipynb`. They belong to the outlier-inclusive execution state, not to the clean 320/80 split described earlier in the notebook.

### B8.3 Personal notebook, clean reproduction

Settings: clean 400 rows, `test_size=0.2`, `random_state=42`, 320 training rows, 80 test rows.

| Quantity | Value |
|---|---:|
| `scaler.mean_` Age | 37.871875 |
| `scaler.mean_` Salary | 70,281.25 |
| `scaler.scale_` Age | 10.1915631 |
| `scaler.scale_` Salary | 34,376.713 |
| Standardized Age max | 2.2 |
| Standardized Salary max | 2.3 |

Reproduced model results:

| Model | Unscaled | Scaled | Test rows correct |
|---|---:|---:|---:|
| LogisticRegression | 0.8875 | 0.8625 | 71/80 vs 69/80 |
| DecisionTreeClassifier | 0.8375 | 0.8375 | 67/80 vs 67/80 |

These are reproduced values, not saved notebook outputs. They are included to show how much the data state changes the story. On this clean split, scaling did not improve either model; on the saved outlier-inclusive split, the tree tied and the logistic regression lost one row.

### B8.4 Current reproduction of the official split

A current scikit-learn environment reproduces the official 280/120 split but not the exact historical scores:

| Model | Unscaled | Scaled | Test rows correct |
|---|---:|---:|---:|
| LogisticRegression | 0.875 | 0.8667 | 105/120 vs 104/120 |
| DecisionTreeClassifier | 0.875 | 0.8667 | 105/120 vs 104/120 |

This is a useful reminder that the official saved outputs are evidence about a particular historical run, not a mathematical law. The robust conclusions are:

- scaling can change optimization and regularization;
- tree predictions are usually stable under monotone rescaling;
- a single split is a noisy estimate;
- the exact accuracy gap is not a universal constant.

### B8.5 What the tables do not show

The notebooks do not save confusion matrices, precision, recall, ROC curves, or confidence intervals. Therefore this chapter does not claim that scaling improves a particular error type. Accuracy alone is a coarse summary, especially with the 257/143 class imbalance. A proper comparison would use the same folds, multiple metrics, repeated evaluation, and a pipeline.

## B9. Extended interview bank

The questions below are follow-ups to Part A, not duplicates. They are aimed at deeper reasoning and practical judgement.

### Level 1 — Definitions and framing

**Q1. What does it mean to say standardization is an affine transformation?**

It is a linear rescaling plus a shift: $z=(x-\mu)/\sigma=(1/\sigma)x-\mu/\sigma$. It preserves order and shape but changes origin and unit.

**Q2. What is affine equivariance in this context?**

If a feature is changed by a positive affine map $y=ax+b$, its standardized values are unchanged: $(y-(a\mu+b))/(a\sigma)=(x-\mu)/\sigma$. The choice of unit does not affect the z-scores.

**Q3. What is a preconditioner, and why is standardization one?**

A preconditioner transforms an optimization problem to improve its conditioning. Standardization is a diagonal preconditioner: it rescales each coordinate by its standard deviation but does not rotate or decorrelate the axes.

**Q4. Is standardization a distributional transformation?**

No. It changes location and scale, not skew, multimodality, or tail shape. A power or quantile transform is distributional; standardization is affine.

### Level 2 — Mechanics and API

**Q5. Why is the attribute called `scale_` rather than `std_`?**

Because it is the divisor used by `transform`. For a constant feature, the true standard deviation is 0 but the stored scale is 1, so `scale_` is not always equal to the standard deviation. The name also generalizes to other scalers.

**Q6. What does `StandardScaler(with_mean=False)` do, and when is it useful?**

It divides by the training scale but does not subtract the mean. It is useful for sparse data, where centering would turn implicit zeros into non-zeros and destroy sparsity. It is not equivalent to full standardization.

**Q7. What does `partial_fit` update?**

It updates the running mean, variance, and sample count from each batch. The result approximates a full `fit` up to numerical precision. The batches must still be training data; streaming does not remove the leakage rule.

**Q8. What does `set_output(transform="pandas")` change?**

It changes the output container to a pandas DataFrame when the input is a DataFrame, preserving column names. It does not change the numerical transformation. The option is version-dependent, so portable code often wraps the array manually.

**Q9. Why does `fit_transform` on the test set leak?**

Because `fit_transform` calls `fit` on the test set before transforming it. That estimates a new mean and scale from the test rows, so the test distribution influences the mapping.

### Level 3 — Mathematics

**Q10. Derive the coefficient mapping between raw and standardized linear models.**

With $x_j=\sigma_jz_j+\mu_j$,

$$
b+\sum_jw_jx_j
=
\left(b+\sum_jw_j\mu_j\right)
+
\sum_j(w_j\sigma_j)z_j.
$$

So $w_j^z=w_j^x\sigma_j$ and $b^z=b^x+\sum_jw_j^x\mu_j$. The inverse is $w_j^x=w_j^z/\sigma_j$ and $b^x=b^z-\sum_jw_j^z\mu_j/\sigma_j$.

**Q11. Why does ridge regularization change under standardization?**

The penalty is applied to coefficients, and coefficients scale with the feature units. In raw coordinates, the penalty is $\lambda\sum(w_j^x)^2$; in standardized coordinates, the same model has coefficients $w_j^z=w_j^x\sigma_j$, so the penalty becomes $\lambda\sum(w_j^z/\sigma_j)^2$ if expressed in raw units. Unless all scales are equal, the effective penalty changes.

**Q12. Show that standardization turns the covariance matrix into a correlation matrix.**

For centered data, $\operatorname{Cov}(z_j,z_k)=\operatorname{Cov}(x_j,x_k)/(\sigma_j\sigma_k)=\rho_{jk}$. Therefore the covariance matrix of $Z$ is the correlation matrix of $X$.

**Q13. Why does standardization not change skewness?**

Skewness is based on standardized third moments. A positive affine transformation multiplies deviations and the standard deviation by the same factor, so the ratio is unchanged. The same argument applies to kurtosis.

**Q14. What is the condition number, and how does standardization affect it?**

The condition number is $\lambda_{\max}/\lambda_{\min}$ of the Hessian or normal-equation matrix. It measures how elongated the quadratic loss is. Standardization removes the trivial scale disparity between columns; it does not remove correlation-induced ill-conditioning.

**Q15. Why is raw Euclidean distance a weighted distance in standardized coordinates?**

Because $a_j-b_j=\sigma_j(z_{a,j}-z_{b,j})$, so the squared distance is $\sum_j\sigma_j^2(z_{a,j}-z_{b,j})^2$. The raw distance implicitly weights each standardized coordinate by its variance.

### Level 4 — Diagnosis

**Q16. After scaling, an unregularized OLS model's predictions changed. What should you check?**

Check whether the model has an intercept, whether the same split and rows were used, whether the pipeline was refit on the test data, whether regularization is actually active, and whether numerical precision or feature order changed. With an intercept and exact arithmetic, OLS predictions should be invariant.

**Q17. Logistic regression coefficients changed a lot after scaling. Is that a bug?**

Usually not. Coefficients are expressed in feature units, so rescaling changes their numerical values. What matters is whether predictions and penalized objective are comparable. With scikit-learn's default L2 penalty, scaling can also change the fitted model, so compare predictions on the same validation protocol rather than raw coefficients.

**Q18. A standardized training column shows `std = 1.0016` in pandas. Is the scaler wrong?**

No. `StandardScaler` uses population variance (divide by $n$); pandas uses sample variance (divide by $n-1$). The ratio is $\sqrt{n/(n-1)}$, which is 1.00156 for $n=320$. Rounding hides the difference.

**Q19. Production scores dropped after the preprocessing job was updated. What are the likely causes?**

The scaler may have been refit on a different distribution, the feature order or schema may have changed, the model may have been trained with a different scaler, or the input distribution may have shifted. Compare the saved `mean_`/`scale_`, feature names, and input quantiles before blaming the model.

**Q20. A z-score threshold of 3 flags too few outliers. Why?**

The mean and standard deviation are themselves inflated by the outliers. An extreme value increases $\sigma$, which shrinks its own z-score and compresses the rest of the data. Use robust statistics, domain thresholds, or a model-based outlier method.

### Level 5 — Design

**Q21. When would you deliberately not standardize a linear model?**

When the features are already in comparable, meaningful units and the coefficient scale is part of the intended interpretation; when the model is unregularized and numerically stable; when centering would destroy sparse structure; or when a bounded representation is required. “Always standardize” is a default, not a law.

**Q22. How do you choose between `StandardScaler`, `RobustScaler`, `MinMaxScaler`, and `PowerTransformer`?**

Use `StandardScaler` for approximately symmetric features and scale-sensitive models; `RobustScaler` when genuine outliers should not dominate the center and spread; `MinMaxScaler` when a bounded range is required and future values are expected inside the training range; `PowerTransformer` when skew is the main problem. These choices can be combined, but each must be fitted inside the training pipeline.

**Q23. How would you handle a feature with a heavy tail before scaling?**

First decide whether the tail is real or an error. If real, consider a log or power transform, a robust scaler, or a model that handles the tail. Do not silently cap values without documenting the threshold. Evaluate the choice on validation data, not on the test set.

**Q24. How do you scale features in an online learning system?**

Use `partial_fit` on the training stream, or maintain explicit running statistics. Freeze the scaler when evaluating a model, and retrain preprocessing and model together when the distribution changes. Do not update the scaler on the evaluation stream.

**Q25. Would you standardize before a Gaussian Naive Bayes classifier?**

Usually not necessary. Gaussian Naive Bayes estimates a mean and variance per feature, and the likelihood is invariant to a positive rescaling in exact arithmetic. Numerical and estimation differences can remain, but standardization is not the first priority. Multinomial Naive Bayes is different because it expects non-negative counts.

### Level 6 — Advanced

**Q26. Prove that unregularized OLS with an intercept is invariant to an invertible affine transformation of the features.**

An invertible affine map $x\mapsto Ax+c$ changes the design matrix but not the column space of the augmented matrix $[1,X]$. OLS fits the projection of $y$ onto that column space. Since the column space is unchanged, fitted values and residuals are unchanged; only the coefficients are reparameterized. Standardization is the special case $A=D^{-1}$, $c=-D^{-1}\mu$.

**Q27. Why is default L2 logistic regression not invariant under the same transformation?**

The L2 penalty is imposed on the coefficients in the chosen coordinates. An invertible affine map changes the coefficients, so the same penalty corresponds to a different constraint in the original space. The fitted probabilities can therefore change. The unregularized maximum-likelihood solution is invariant in exact arithmetic.

**Q28. Can standardization hurt KNN?**

Yes. Standardization assumes that equal variance deserves equal weight. If one feature has large variance because it is genuinely more informative, standardization can down-weight it relative to a noisy feature with small variance. Feature weighting, domain knowledge, or learned metrics may be needed.

**Q29. What is the relationship between standardization and whitening?**

Standardization divides each feature by its marginal standard deviation. Whitening applies a full linear transformation that makes the covariance matrix the identity, removing correlations as well. Standardization is diagonal; whitening is not.

**Q30. Why can standardizing a binary feature be misleading?**

It changes the coefficient's unit and, under regularization, its effective penalty. The two standardized values depend on the base rate through $\sqrt{p(1-p)}$. The transformation is mathematically valid but can obscure the interpretation of a 0/1 indicator. For trees, it changes nothing essential.

**Q31. Why does standardization not fix a multimodal distribution?**

It is affine. An affine map preserves the number and relative positions of modes; it only changes the axis labels. A mixture model, a quantile transform, or a different representation is needed if multimodality matters.

**Q32. What happens if the training set has a constant feature?**

The population standard deviation is zero. Scikit-learn sets `scale_` to 1, so the transformed column is all zeros. The feature is uninformative and should usually be removed. If it is retained, its coefficient is unidentifiable or irrelevant depending on the model.

### Level 7 — Synthesis

**Q33. You have one million rows and ten thousand sparse features. How do you scale?**

Avoid dense centering. Use `StandardScaler(with_mean=False)` or `MaxAbsScaler` to preserve sparsity, and keep the transformation inside a sparse-aware pipeline. Consider `partial_fit` if the data do not fit in memory. Evaluate whether scaling is needed at all for the chosen model.

**Q34. Cross-validation shows no difference between scaled and unscaled models, but the holdout shows a large difference. What do you conclude?**

Not that scaling is universally helpful. The holdout may be a lucky or unlucky split, the CV folds may not represent the holdout distribution, or the models may differ in regularization or convergence. Use repeated CV, inspect per-fold variance, and avoid selecting the preprocessing based on the holdout.

**Q35. A stakeholder says “scaling did not improve accuracy, so it is useless.” How do you respond?**

Accuracy on one split is a noisy, model-specific outcome. Scaling changes geometry, conditioning, and regularization; whether that improves accuracy depends on the model and data. It may improve convergence, stability, coefficient comparability, or distance behaviour even when a single accuracy number is unchanged. Evaluate with a proper protocol and the metrics that matter.

**Q36. How would you detect that a deployed scaler is stale?**

Monitor feature means, standard deviations, quantiles, missingness, and the fraction of values outside the training range. Compare them with the fitted `mean_` and `scale_`. A drift in inputs does not always require immediate retraining, but it should trigger investigation and a validation run.

**Q37. Why is “standardization makes data Gaussian” wrong, and what would you use instead?**

Standardization is affine; it preserves skew and multimodality. To change the distributional shape, use a power transform (Yeo–Johnson or Box–Cox) or a quantile transform, and validate the result. Even then, “more Gaussian” is a modelling choice, not a guarantee of better predictions.

## B10. Extended examination questions

### Q1. Derive the population mean and variance of a standardized column.

With $z_i=(x_i-\mu)/\sigma$ and $\mu,\sigma$ computed on the same $n$ values,

$$
\frac{1}{n}\sum_i z_i
=
\frac{1}{n\sigma}\sum_i(x_i-\mu)
=
\frac{1}{n\sigma}\left(\sum_i x_i-n\mu\right)=0.
$$

Also,

$$
\frac{1}{n}\sum_i z_i^2
=
\frac{1}{n\sigma^2}\sum_i(x_i-\mu)^2
=
\frac{\sigma^2}{\sigma^2}=1.
$$

The population variance is 1. The sample variance is $n/(n-1)$.

### Q2. A raw linear model has intercept 1, age coefficient 0.05, and salary coefficient 0.00002. The training means are 38 and 70,000; the scales are 10 and 34,000. Find the standardized coefficients and intercept.

Using $w_j^z=w_j^x\sigma_j$:

$$
w_{\text{age}}^z=0.05\cdot10=0.5,
\qquad
w_{\text{salary}}^z=0.00002\cdot34{,}000=0.68.
$$

The intercept is

$$
b^z=1+0.05(38)+0.00002(70{,}000)
=1+1.9+1.4=4.3.
$$

Check: at the training means, all z-scores are 0, so the standardized prediction is 4.3. The raw prediction at the means is also $1+0.05(38)+0.00002(70{,}000)=4.3$. The inverse mapping returns the original coefficients.

### Q3. Why does the same ridge penalty produce a different effective constraint after standardization?

The raw penalty is $\lambda\sum_j(w_j^x)^2$. Since $w_j^z=w_j^x\sigma_j$, the raw penalty can be written as

$$
\lambda\sum_j\left(\frac{w_j^z}{\sigma_j}\right)^2.
$$

If the model is fitted in standardized coordinates with penalty $\lambda\sum_j(w_j^z)^2$, features with small $\sigma_j$ are penalized more heavily in raw units, and features with large $\sigma_j$ less heavily. The two penalties coincide only when all scales are equal. This is why the fitted coefficients and predictions can change.

### Q4. Starting from $[1,2,3,4]$, append 100. Compute the mean, population standard deviation, and the z-scores of 4 before and after the append.

Before:

$$
\mu=2.5,\qquad
\sigma^2=1.25,\qquad
\sigma\approx1.1180,
\qquad
z_4=\frac{4-2.5}{1.1180}\approx1.3416.
$$

After:

$$
\mu=22,\qquad
\sigma^2=1522,\qquad
\sigma\approx39.0128,
$$

$$
z_{100}=\frac{100-22}{39.0128}\approx1.999,
\qquad
z_4=\frac{4-22}{39.0128}\approx-0.461.
$$

The outlier compresses the bulk and reduces its own apparent extremeness. This is why z-score outlier rules can fail when the statistics are contaminated.

### Q5. Two uncorrelated features have standard deviations 1000 and 1. What is the condition number of the raw normal-equation matrix, and what happens after standardization?

The raw matrix is diagonal with entries proportional to $1000^2$ and $1^2$, so the condition number is $10^6$. After standardization, the matrix is the identity, so the condition number is 1. If the features are correlated, standardization leaves the correlation matrix, whose condition number may still be greater than 1.

### Q6. A `MinMaxScaler` is fitted on training values in $[0,100]$. A test value is 150. What is the transformed value with `clip=False` and with `clip=True`?

With `clip=False`:

$$
\frac{150-0}{100-0}=1.5.
$$

With `clip=True`, the result is capped at 1.0. This shows that min–max output is bounded only when the input lies inside the fitted range or clipping is enabled.

### Q7. Why does `StandardScaler(with_mean=True)` fail on a sparse matrix?

Subtracting a non-zero mean turns every implicit zero into a non-zero value, destroying sparsity and potentially exhausting memory. The implementation rejects this combination. Use `with_mean=False` or a zero-preserving scaler such as `MaxAbsScaler`.

### Q8. Explain why a pipeline is required for honest cross-validation when scaling is used.

The scaler learns `mean_` and `scale_` from data. If it is fitted before the cross-validation folds are formed, every validation fold has contributed to those statistics. A pipeline defers fitting until each fold's training part, so the validation part is transformed with statistics learned only from that fold's training data. This preserves the independence of the validation estimate.

### Q9. Why can standardizing a feature with a heavy tail make the bulk look “too normal” or “too compressed”?

The standard deviation is inflated by the tail, so the bulk's z-scores are divided by a larger number than the bulk's own spread would suggest. The bulk appears compressed toward zero, while the extreme values still stand out. Standardization has not removed the tail; it has changed the unit using a tail-sensitive scale.

### Q10. When is a z-score of 5 not evidence of an error?

When the distribution is genuinely heavy-tailed, when the training scale is underestimated, or when the value is a valid rare event. A z-score is a unit-free description relative to the fitted mean and scale; it is not a probability and not a rule for deleting data. Investigate the data-generating process.

## B11. Coverage and source fidelity

### B11.1 Official notebook map

The vendor notebook has 42 cells. Its main sequence is:

| Cells | Content |
|---|---|
| 0–3 | imports, load `Social_Network_Ads.csv`, drop the first two columns, sample rows |
| 4–5 | train/test split, `test_size=0.3`, `random_state=0`; shapes 280/120 |
| 6–8 | `StandardScaler` fit on training data, transform both splits, show `mean_` |
| 9–13 | display raw and scaled training data, `describe()` |
| 14–19 | scatter and KDE plots before/after scaling |
| 20–31 | LogisticRegression and DecisionTreeClassifier, unscaled vs scaled, accuracy |
| 32 | full-data `describe()` |
| 33–41 | append three extreme rows, re-split (282/121), re-scale, plot |

The official notebook has no saved model results after the outlier re-split. It also uses `df.append`, which was removed in pandas 2.0; the personal notebook uses `pd.concat`.

### B11.2 Personal notebook map

`my_notebook.ipynb` has 106 cells: 35 code and 71 markdown. It preserves the course sequence and adds explanation, but it also contains a few harmless artifacts:

- a duplicated `LogisticRegression` initialization cell;
- KDE cells that set the left title twice and leave the right panel untitled;
- one outlier scatter cell where the right `ax2.scatter` line is commented out, producing a blank panel;
- saved outputs from an outlier-inclusive execution state, even though the clean-run code appears earlier;
- an original absolute Windows dataset path, now changed to the relative `datasets/Social_Network_Ads.csv`;
- a practice reference changed from `topic_notes.md` to `notes.md`.

The implementation itself is preserved. The corrections are path, link, and explanation fixes, not a rewrite of the working code.

`DAY24_Standardization-with-Outliers.ipynb` has 53 cells: 34 code and 19 markdown. It is a code copy of the outlier experiment without saved outputs. It is kept separate, as required by the never-merge rule. Because it has no outputs, this chapter does not attribute any model score to it.

`DAY24_Feature-Scaling-stub.ipynb` has five markdown cells and no code. It contains the sentence “Standardization also called Normalization,” which is imprecise in this course; Day 25 uses “normalization” for min–max scaling. The notes correct that terminology.

### B11.3 Reference notebook

`reference_notebooks/day24.ipynb` has 42 cells and matches the vendor notebook's code and saved outputs. It is the local reference copy and should not be edited. The official results quoted in this chapter come from that file.

### B11.4 Dataset and result provenance

The dataset is the CampusX course copy of `Social_Network_Ads.csv`: 400 rows, five columns, no missing values, target counts 257 zeros and 143 ones. After `iloc[:, 2:]`, the modelling table has `Age`, `EstimatedSalary`, and `Purchased`.

The following values are **saved notebook outputs**:

- official `scaler.mean_ = [37.8642857, 69807.1429]`;
- official describe table and 280/120 shapes;
- official LogisticRegression `0.6583333` / `0.8666667`;
- official DecisionTree `0.875` / `0.875`;
- personal saved `scaler.mean_ = [38.2080745, 72152.1739]`;
- personal saved describe table and 322/81 shapes;
- personal saved LogisticRegression `0.8641975` / `0.8518519`;
- personal saved DecisionTree `0.8888889` / `0.8888889`.

The following values are **reproduced** for this chapter and are labeled as such:

- clean personal means `[37.871875, 70281.25]`;
- clean personal scales `[10.1915631, 34376.713]`;
- clean personal max z-scores `2.2` / `2.3`;
- clean personal LogisticRegression `0.8875` / `0.8625`;
- clean personal DecisionTree `0.8375` / `0.8375`;
- current official-split LogisticRegression `0.875` / `0.8667`;
- current official-split DecisionTree `0.875` / `0.8667`.

No confusion matrices or additional metrics were saved, so none are invented here.

### B11.5 Version caveat

The official saved LogisticRegression gap is large, but reproducing the same split with a current scikit-learn version gives a different result. This chapter therefore treats the official number as historical evidence about a particular environment, not as a universal property of scaling. The stable lesson is mathematical and procedural: scaling changes conditioning and regularization; it does not guarantee accuracy; and comparisons must use the same split, pipeline, and evaluation protocol.

## B12. References

- CampusX, *100 Days of Machine Learning*, Day 24: standardization, `vendor/100-days-of-machine-learning/day24-standardization/`.
- scikit-learn, *Preprocessing data* and `StandardScaler`, `RobustScaler`, `MinMaxScaler`, `MaxAbsScaler`, `PowerTransformer`, `QuantileTransformer`, and `Normalizer` API documentation.
- pandas, `DataFrame.describe` documentation for the sample-standard-deviation convention.
- Hastie, Tibshirani, and Friedman, *The Elements of Statistical Learning*, for conditioning, ridge regression, and the bias–variance view of regularization.
- Huber and Ronchetti, *Robust Statistics*, for influence functions and breakdown points.
- Course dataset: `Social_Network_Ads.csv`, as distributed with the CampusX repository.
