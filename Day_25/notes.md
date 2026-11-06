# Day 25 — Normalization and Min-Max Scaling

> **Course topic:** CampusX Day 25, *Feature Scaling — Normalization / MinMaxScaler*.
> **Personal notebooks:** `my_notebook.ipynb` (61 cells) and `DAY25_Normalization-stub.ipynb` (4 cells).
> **Dataset:** `datasets/wine_data.csv`.
> **Reference:** `reference_notebooks/day25.ipynb` (do not edit).
>
> This chapter follows the Day 39 two-layer structure. **Part A** is the main learning chapter and is sufficient on its own. **Part B** preserves advanced details, variants, and reference material.

---

## Part A — Main Learning Chapter

### A1. Learning objectives

By the end of this chapter you should be able to:

1. Explain why feature scaling is needed and when it is unnecessary.
2. Derive and apply the min-max formula, including a custom output range.
3. Explain exactly what `MinMaxScaler` learns during `fit`.
4. Use `fit_transform` on training data and `transform` on test data without leakage.
5. Interpret the scaler's learned attributes and the notebook's plots.
6. Compare min-max scaling with standardization, max-absolute scaling, robust scaling, and row-wise normalization.
7. Answer interview and exam questions about scale, leakage, outliers, and model families.

---

### A2. The problem: one feature can dominate another

Suppose a dataset has two features:

| Row | Age | Salary |
|---|---:|---:|
| 0 | 20 | 20,000 |
| 1 | 25 | 40,000 |
| 2 | 30 | 60,000 |
| 3 | 35 | 80,000 |
| 4 | 40 | 100,000 |

A distance-based model such as K-nearest neighbours computes something like

$$
d(\mathbf{x}_i,\mathbf{x}_j)
=
\sqrt{(Age_i-Age_j)^2+(Salary_i-Salary_j)^2}.
$$

The salary difference is measured in tens of thousands, while the age difference is measured in single years. A salary gap of 20,000 contributes $20{,}000^2$ to the squared distance; an age gap of 5 contributes only $25$. The model is not necessarily "wrong" — it is faithfully using the units we gave it — but the units now decide which feature matters.

The same problem appears in:

- K-means, which assigns points by distance to cluster centres.
- Support vector machines, especially with RBF kernels.
- Principal component analysis, where variance depends on units.
- Gradient descent and neural networks, where badly scaled features create long, narrow loss surfaces and slow convergence.
- Regularized linear models, because a penalty on coefficients is not comparable across features measured on different scales.

Scaling changes the numerical representation, not the information in the feature. It does not create new information, remove outliers, or make a feature normally distributed.

#### When scaling is usually unnecessary

Tree-based models such as decision trees, random forests, and gradient-boosted trees choose splits by threshold comparisons. A split such as `Salary <= 50000` is equivalent to `Salary/1000 <= 50`. Monotone rescaling therefore does not change the possible splits. Scaling can still matter for numerical stability or for a pipeline shared with other models, but it is not usually required for tree accuracy.

---

### A3. Foundations: what feature scaling changes

Feature scaling applies a transformation to each feature, usually column by column. Two common families are:

- **Range scaling:** map values into a fixed interval, such as $[0,1]$.
- **Location-scale scaling:** subtract a centre and divide by a spread, producing values without a fixed interval.

Min-max scaling belongs to the first family. Standardization belongs to the second.

A scaling transformation should normally be **monotone**: if $x_i < x_j$, the transformed values preserve that order. Min-max scaling is monotone when the training range is positive. This is why it preserves rank order and why it does not change the ordering of values within a feature.

---

### A4. The min-max formula

For one feature with training minimum $x_{\min}$ and training maximum $x_{\max}$, the default min-max transformation is

$$
x' = \frac{x-x_{\min}}{x_{\max}-x_{\min}}.
$$

The symbols mean:

- $x$: an original value.
- $x_{\min}$: the smallest value observed in the training data for that feature.
- $x_{\max}$: the largest value observed in the training data for that feature.
- $x'$: the transformed value.

The formula is an affine transformation. It first shifts the data so the minimum becomes zero, then divides by the range so the maximum becomes one:

$$
x_{\min}\mapsto 0,
\qquad
x_{\max}\mapsto 1.
$$

Values between the training minimum and maximum map between zero and one. Values outside the training range map outside $[0,1]$.

#### General output range

If we want the output interval $[a,b]$ instead of $[0,1]$, the formula is

$$
x' = a + (x-x_{\min})\frac{b-a}{x_{\max}-x_{\min}}.
$$

For $a=0,b=1$ this reduces to the default formula. For $a=-1,b=1$ the training minimum maps to $-1$ and the training maximum maps to $1$.

---

### A5. A small numerical example

Take the feature

$$
[10,20,30,40,50].
$$

Here $x_{\min}=10$ and $x_{\max}=50$, so the range is $40$.

For $x=30$:

$$
x'=\frac{30-10}{50-10}=\frac{20}{40}=0.5.
$$

For every value:

| Original | Calculation | Scaled |
|---:|---|---:|
| 10 | $(10-10)/40$ | 0.00 |
| 20 | $(20-10)/40$ | 0.25 |
| 30 | $(30-10)/40$ | 0.50 |
| 40 | $(40-10)/40$ | 0.75 |
| 50 | $(50-10)/40$ | 1.00 |

The notebook reproduces this exactly:

```python
values = [10, 20, 30, 40, 50]
minimum = min(values)
maximum = max(values)
normalized = [(x - minimum) / (maximum - minimum) for x in values]
print(normalized)
```

Saved output:

```text
[0.0, 0.25, 0.5, 0.75, 1.0]
```

#### Why the operation is per column

A DataFrame may contain `Age` and `Salary`. If we used one global minimum and maximum for the whole table, the salary range would determine the scale of age as well, and the two columns would no longer be independently comparable. Ordinary feature scaling learns one minimum and one maximum **per feature**.

```python
df_min = df.min()
df_max = df.max()
normalized_df = (df - df_min) / (df_max - df_min)
```

Pandas aligns the one-dimensional `df_min` and `df_max` Series with the DataFrame columns, so each column is scaled by its own bounds.

---

### A6. `MinMaxScaler`: creating, fitting, transforming

The scikit-learn API separates three ideas:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()              # create an unfitted object
scaler.fit(X_train)                  # learn the training bounds
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

A compact equivalent for the training data is:

```python
X_train_scaled = scaler.fit_transform(X_train)
```

`fit_transform` is not a different algorithm. It calls `fit` and then `transform` on the same input.

#### What each call means

| Call | What happens | What is returned |
|---|---|---|
| `MinMaxScaler()` | Creates an unfitted transformer. No data is read. | A scaler object |
| `fit(X_train)` | Learns the minimum and maximum of each feature. | The same fitted scaler |
| `transform(X)` | Applies the learned formula to `X`. | A transformed array |
| `fit_transform(X_train)` | Learns from `X_train` and transforms that same data. | A transformed array |
| `inverse_transform(X_scaled)` | Reverses the learned affine map. | Values on the original scale |

The notebook's synthetic DataFrame demonstrates the result:

```text
Original data:
   Age  Salary
0   20   20000
1   25   40000
2   30   60000
3   35   80000
4   40  100000

Normalized data:
    Age  Salary
0  0.00    0.00
1  0.25    0.25
2  0.50    0.50
3  0.75    0.75
4  1.00    1.00
```

Both columns happen to have equally spaced values, so their normalized columns look identical. That is a property of this toy example, not a general rule.

#### What the scaler stores

After fitting, `MinMaxScaler` stores attributes ending in an underscore:

```python
print(scaler.data_min_)
print(scaler.data_max_)
print(scaler.scale_)
```

Saved output for the synthetic data:

```text
Minimum learned per feature: [   20. 20000.]
Maximum learned per feature: [4.e+01 1.e+05]
Scale factor per feature: [5.00e-02 1.25e-05]
```

The scale factor is the reciprocal of the range for the default output range:

$$
\text{scale}_j=\frac{1}{x_{\max,j}-x_{\min,j}}.
$$

For `Age`, $1/(40-20)=0.05$. For `Salary`, $1/(100000-20000)=1.25\times 10^{-5}$.

Internally, scikit-learn stores a per-feature scale and offset. A compact way to describe the default transformation is

$$
X' = X\cdot\text{scale} + \text{min},
$$

where `min_` is derived from `data_min_` and the requested feature range. The formula is equivalent to the fraction above, but the stored affine form makes `transform` fast and makes `inverse_transform` straightforward.

---

### A7. The most important workflow rule: split before scaling

The correct sequence is:

1. Separate features $X$ and target $y$.
2. Split into training and test sets.
3. Fit the scaler on `X_train` only.
4. Transform `X_train` and `X_test` with that same fitted scaler.
5. Train on scaled training data and evaluate on scaled test data.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### Why fitting on the full dataset is leakage

Suppose the test set contains the largest salary in the entire dataset. If we fit the scaler before splitting, the training values are divided by a range that includes that test salary. The training representation therefore depends on information from the test set. This is **data leakage**: the held-out data has influenced the training pipeline.

The test set is supposed to simulate unseen data. Its statistics must not influence any learned preprocessing parameter.

#### The notebook's leakage demonstration

The notebook first creates a 10-row synthetic dataset and splits it 70/30 with `random_state=42`. The scaler is fitted on the 7 training rows. Saved training shape: `(7, 2)`. Saved test shape: `(3, 2)`.

Training-scaled rows include:

```text
        Age    Salary
6  0.652174  0.529412
5  0.521739  0.435294
1  0.000000  0.000000
8  1.000000  1.000000
```

Test-scaled rows include:

```text
        Age    Salary
0 -0.130435 -0.082353
3  0.260870  0.200000
9  1.217391  1.470588
```

The test values leave $[0,1]$ because the test rows contain values below the training minimum and above the training maximum. This is expected and is one of the most important observations in the notebook.

The learned training bounds are:

```text
Training minima: [2.2e+01 2.5e+04]
Training maxima: [4.5e+01 1.1e+05]
```

The training-scaled minimum and maximum are exactly zero and one for both features:

```text
Scaled training minimums: {'Age': 0.0, 'Salary': 0.0}
Scaled training maximums: {'Age': 1.0, 'Salary': 1.0}
```

#### What would go wrong with `fit_transform(X_test)`

If we called `scaler.fit_transform(X_test)`, the scaler would compute new bounds from the three test rows. The training and test data would then be on different scales, and the model's learned coefficients would no longer correspond to the test representation. The correct call is always `scaler.transform(X_test)`.

---

### A8. Why test values can be outside the range

A common misconception is that min-max scaling guarantees every transformed value lies in $[0,1]$. It guarantees that only for the data used to fit the scaler. For a new value $x>x_{\max}$:

$$
x'=\frac{x-x_{\min}}{x_{\max}-x_{\min}}>1.
$$

For $x<x_{\min}$, the transformed value is negative.

This is not a bug. It is information: the test point lies outside the range seen during training. In some applications you may choose to clip values, but clipping is a modelling decision that discards that information. Scikit-learn's `MinMaxScaler(clip=True)` can clip to the requested range, but the default is `clip=False`.

The notebook's unseen-value experiment makes this concrete:

```python
scaler = MinMaxScaler()
scaler.fit(pd.DataFrame({"value": [10, 20, 30]}))
print(scaler.transform([[40]])[0, 0])
```

Saved output:

```text
1.5
```

The training maximum is 30, so 40 maps to $(40-10)/(30-10)=1.5$.

---

### A9. Custom output range

The `feature_range` argument changes the output interval:

```python
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(original)
```

For the salary values `[20000, 40000, 60000, 80000]`, the saved output is:

```text
     Salary
0 -1.000000
1 -0.333333
2  0.333333
3  1.000000
```

The transformation is still affine; only the endpoints change. With a custom range, the scale factor becomes

$$
\text{scale}_j=\frac{b-a}{x_{\max,j}-x_{\min,j}}.
$$

For $[a,b]=[-1,1]$ and a salary range of 60,000, the scale is $2/60000=1/30000$. A salary of 40,000 maps to $-1+(40000-20000)/30000=-1+0.6667=-0.3333$.

The output range does not change the shape of the distribution. It only changes the numerical interval.

---

### A10. The wine dataset: what the notebook actually does

The notebook loads the first three columns of `wine_data.csv`:

```python
df = pd.read_csv(
    r"E:\MACHINE_LEARNING\DATA_SETS\wine_data.csv",
    header=None,
    usecols=[0, 1, 2]
)
df.columns = ["Class label", "Alcohol", "Malic Acid"]
```

The saved table has 178 rows and three columns. `Class label` is the target; `Alcohol` and `Malic Acid` are the two features.

The absolute Windows path is a portability problem. The local dataset is available at `datasets/wine_data.csv`, so a portable version should use a relative path or a path resolved from the notebook location:

```python
from pathlib import Path

data_path = Path("datasets") / "wine_data.csv"
df = pd.read_csv(data_path, header=None, usecols=[0, 1, 2])
```

The notebook then:

1. Splits the data with `test_size=0.2, random_state=23`.
2. Fits `MinMaxScaler` on `x_train`.
3. Transforms `x_train` and `x_test`.
4. Converts the arrays back to DataFrames with the original column names.
5. Describes the scaled training data.
6. Plots the training data before and after scaling.

Saved training shape:

```text
(142, 2)
```

Saved scaled-training summary:

```text
        Alcohol  Malic Acid
count    142.00      142.00
mean       0.52        0.32
std        0.22        0.22
min        0.00        0.00
25%        0.35        0.17
50%        0.53        0.22
75%        0.69        0.47
max        1.00        1.00
```

The minimum and maximum are exactly 0 and 1, as expected. The means and standard deviations are not fixed at any particular values; they depend on the shape of the original distribution. Min-max scaling does not produce mean zero or standard deviation one.

The notebook contains duplicate split-and-scale blocks and repeated plotting cells. These are preserved as part of the personal notebook. The repeated outputs are consistent, but they are not separate experiments.

---

### A11. Reading the graphs

The notebook creates several plots:

1. A KDE plot of `Alcohol`.
2. A scatter plot of `Alcohol` versus `Malic Acid`, coloured by class.
3. Side-by-side scatter plots before and after scaling.
4. Side-by-side KDE plots before and after scaling.

#### KDE before and after

A kernel density estimate approximates the distribution's shape. Min-max scaling is an affine transformation, so the KDE after scaling should have the same shape, just on a different horizontal axis. The peak does not become "more normal"; the skew does not disappear. The plot confirms that scaling changed the axis, not the distribution's shape.

What the plot **cannot** tell us:

- Whether a model will perform better.
- Whether the feature is now normally distributed.
- Whether outliers have been removed.

#### Scatter before and after

In the raw scatter, `Alcohol` and `Malic Acid` may occupy different ranges. After scaling, both axes run from 0 to 1. The relative geometry of the points is preserved up to the per-axis stretch and shift. If the raw plot looks like a narrow vertical band because one axis has a much larger numerical range, the scaled plot makes the two axes visually comparable.

The colours represent the class labels. Scaling does not change class membership, so the colour pattern should remain the same. What changes is the coordinate system.

The notebook's saved figure outputs only record the figure size, not the pixel content. The interpretation above follows from the code and from the affine property of the transform; it is not a claim about a hidden visual result.

---

### A12. Putting scaling inside a pipeline

A pipeline keeps preprocessing and model training together:

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = Pipeline([
    ("scaler", MinMaxScaler()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))
```

Saved output:

```text
Accuracy: 0.6666666666666666
```

This is a toy correctness check on three test rows, not a tuned model. Its value is that the scaler is fitted inside the pipeline on the training data. During cross-validation, each fold would fit its own scaler on that fold's training portion, which prevents leakage across folds.

The pipeline also solves the train/serve consistency problem. At prediction time, `model.predict(new_rows)` applies the same fitted scaler and then the model. You do not have to remember the scaling step manually.

#### Why a pipeline is more than convenience

Without a pipeline, it is easy to:

- Fit the scaler on the full dataset before cross-validation.
- Forget to scale a new row at inference time.
- Use a different column order in training and inference.
- Refit the scaler on test data.

A pipeline makes these mistakes less likely because the preprocessing is part of the estimator object.

---

### A13. What min-max scaling does not do

Min-max scaling:

- Does not remove outliers.
- Does not make data normally distributed.
- Does not make features independent.
- Does not guarantee test values lie in $[0,1]$.
- Does not preserve the original units.
- Does not necessarily improve every model.

It changes the numerical range. That is all.

#### Outliers

Because the minimum and maximum define the transformation, a single extreme value can compress the rest of the feature into a narrow band. For example, if most values lie between 0 and 10 but one value is 10,000, then almost all scaled values lie near 0. This is why robust scaling exists.

---

### A14. Min-max versus standardization

Standardization uses

$$
z=\frac{x-\mu}{\sigma},
$$

where $\mu$ is the training mean and $\sigma$ is the training standard deviation. The result has mean approximately zero and standard deviation approximately one on the fitted data, but it is not bounded.

| Property | Min-max scaling | Standardization |
|---|---|---|
| Formula | $(x-x_{\min})/(x_{\max}-x_{\min})$ | $(x-\mu)/\sigma$ |
| Output range | Fixed interval for fitted data | Unbounded |
| Centre | Minimum maps to lower bound | Mean maps to zero |
| Spread | Range maps to interval width | Standard deviation maps to one |
| Sensitive to | Minimum and maximum | Mean and standard deviation |
| Typical use | Bounded inputs, pixels, coordinates | Gaussian-like features, linear models, PCA |
| Test values | Can leave the interval | Can be any value |

Neither method is universally better. The choice depends on the model, the feature distribution, and whether a bounded representation is useful.

The notebook compares the two on a simple four-row DataFrame:

```text
Min-Max scaling:
   Feature_A  Feature_B
0   0.000000   0.000000
1   0.333333   0.333333
2   0.666667   0.666667
3   1.000000   1.000000

Standardization:
   Feature_A  Feature_B
0  -1.341641  -1.341641
1  -0.447214  -0.447214
2   0.447214   0.447214
3   1.341641   1.341641
```

Both columns happen to be perfectly correlated in this toy example, so the two methods produce proportional columns. Real data rarely behaves this neatly.

---

### A15. Min-max versus row-wise normalization

Scikit-learn's `Normalizer` is a different operation. It scales each **row** so that its vector norm is one:

$$
\mathbf{x}_i'=\frac{\mathbf{x}_i}{\lVert\mathbf{x}_i\rVert}.
$$

This is useful for text features, embeddings, and situations where the direction of a sample matters more than its magnitude. It is not the same as scaling each column to $[0,1]$.

The notebook's example makes the distinction visible:

```text
Row-wise vector normalization:
   Feature_A  Feature_B
0   0.099504   0.995037
1   0.099504   0.995037
2   0.099504   0.995037
3   0.099504   0.995037
```

Every row has the same ratio because every row is a positive multiple of the same vector. The operation normalized each row, not each column.

---

### A16. Common mistakes and debugging

#### Mistake 1: fitting before splitting

```python
# Wrong: test statistics enter the scaler
scaler.fit(X)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Fix: split first, then `scaler.fit(X_train)`.

#### Mistake 2: fitting separately on train and test

```python
# Wrong: two different scalers, two different coordinate systems
X_train_scaled = MinMaxScaler().fit_transform(X_train)
X_test_scaled = MinMaxScaler().fit_transform(X_test)
```

Fix: one fitted scaler, then `transform` both sets.

#### Mistake 3: assuming test values must be in range

A test value outside the training range produces a value outside $[0,1]$. Do not silently clip it unless clipping is part of your modelling decision.

#### Mistake 4: using one global min and max

Ordinary feature scaling is per column. A global min/max mixes unrelated units.

#### Mistake 5: scaling the target without a reason

Scaling $X$ is standard. Scaling $y$ changes the target's units and can change the loss and the interpretation of predictions. Do it only when the workflow requires it, and remember to inverse-transform predictions.

#### Mistake 6: forgetting scaling at inference time

If the model was trained on scaled features, new rows must be transformed with the same fitted scaler. A pipeline prevents this mistake.

#### Mistake 7: confusing min-max with normalization in the vector sense

`MinMaxScaler` is column-wise range scaling. `Normalizer` is row-wise norm scaling.

#### Mistake 8: ignoring constant features

If a feature is constant, its range is zero. The naive formula divides by zero. Scikit-learn handles this case internally, but a constant feature carries no information and should usually be removed.

#### Mistake 9: using an absolute path

The notebook loads the CSV from an absolute Windows path. This breaks when the project moves or runs on another machine. Use a path relative to the notebook or project root.

---

### A17. Where this sits in the ML workflow

Min-max scaling belongs after the train/test split and before model training:

```text
Raw data
   |
   v
Clean and inspect
   |
   v
Separate X and y
   |
   v
Train/test split
   |
   v
Fit MinMaxScaler on X_train
   |
   v
Transform X_train and X_test
   |
   v
Train model
   |
   v
Evaluate on test data
```

Inside cross-validation, the scaler must be fitted inside each fold. A pipeline handles this automatically.

Scaling is often composed with other preprocessing:

```text
ColumnTransformer
   ├── numeric columns -> SimpleImputer -> MinMaxScaler
   ├── ordinal columns -> OrdinalEncoder
   └── nominal columns -> OneHotEncoder
```

Day 28 introduces `ColumnTransformer`, and Day 29 combines it with `Pipeline`.

---

### A18. Interview preparation

#### Level 1 — Recall

**Q1. What is min-max normalization?**

It is a feature-scaling method that maps each feature to a fixed interval, usually $[0,1]$, using the training minimum and maximum:

$$
x'=\frac{x-x_{\min}}{x_{\max}-x_{\min}}.
$$

**Q2. What do `fit` and `transform` do?**

`fit` learns the per-feature minimum and maximum. `transform` applies the learned affine transformation. `fit_transform` does both on the same data.

**Q3. Why is `transform` used on the test set?**

Because the test set must reuse the training bounds. Fitting on test data would produce a different coordinate system and leak test information.

#### Level 2 — Explanation

**Q4. Why can test values be outside $[0,1]$?**

The interval guarantee applies to the data used for fitting. A test value above the training maximum maps above 1; a value below the training minimum maps below 0. This is expected.

**Q5. Does min-max scaling remove outliers?**

No. It rescales them. In fact, an extreme value can become the new maximum and compress the remaining values toward zero.

**Q6. Is min-max scaling the same as standardization?**

No. Min-max scaling uses the minimum and maximum and produces a bounded interval for the fitted data. Standardization uses the mean and standard deviation and produces unbounded z-scores.

#### Level 3 — Application

**Q7. Which models often benefit from scaling?**

Distance-based models (KNN, K-means), margin-based models (SVM), PCA, gradient-based models, and neural networks. Tree-based models usually do not require it.

**Q8. When would you choose robust scaling over min-max scaling?**

When outliers are present and the minimum/maximum would be dominated by extreme values. Robust scaling uses the median and interquartile range, which are less sensitive to extremes.

**Q9. When would you choose max-absolute scaling?**

For sparse data, because max-absolute scaling does not centre the data and therefore preserves zero entries. It divides by the maximum absolute value, mapping values into $[-1,1]$.

#### Level 4 — Analysis

**Q10. Why is fitting a scaler before cross-validation a problem?**

Because the scaler learns from all rows, including the rows that will become validation folds. The validation score is then optimistically biased. Preprocessing must be fitted inside each fold, usually through a pipeline.

**Q11. What happens to a constant feature?**

Its range is zero. The naive formula is undefined. Scikit-learn handles the degenerate case internally, but the feature is uninformative and should usually be dropped.

**Q12. Does min-max scaling preserve the shape of the distribution?**

Yes, up to an affine change of axis. It does not change skewness, multimodality, or the order of values. It only changes the scale.

#### Level 5 — Design

**Q13. How would you scale a dataset with both numeric and categorical columns?**

Use a `ColumnTransformer`: numeric columns can go through an imputer and a scaler; ordinal columns through an ordinal encoder; nominal columns through a one-hot encoder. Keep the whole preprocessing block inside a pipeline.

**Q14. How would you handle new categories at inference time?**

For one-hot encoding, use `handle_unknown="ignore"` so unseen categories become all-zero rows rather than causing an error. For ordinal encoding, define an explicit category order and decide how unknown values should be handled.

**Q15. What would you log or monitor in production?**

The training bounds, the scaler version, the feature order, the proportion of out-of-range values, and the distribution of scaled inputs. A large shift in out-of-range values may indicate data drift.

#### Level 6 — Evaluation

**Q16. How would you test whether scaling improves a model?**

Compare models with and without scaling using the same cross-validation folds and the same preprocessing protocol. Do not compare a scaled model evaluated with leakage against an unscaled model evaluated honestly.

**Q17. Why might scaling fail to improve a tree model?**

Because trees split on order and thresholds. A monotone rescaling does not change which splits are possible, so the fitted tree can be equivalent.

**Q18. What is the cost of min-max scaling?**

It requires storing two numbers per feature and applying an affine transform at inference time. The computational cost is small. The main risk is sensitivity to outliers and leakage if fitted incorrectly.

#### Level 7 — Advanced

**Q19. Can min-max scaling be inverted?**

Yes. For the default range,

$$
x = x'(x_{\max}-x_{\min}) + x_{\min}.
$$

Scikit-learn provides `inverse_transform`. This is useful when a model predicts a scaled target, but it must use the same fitted scaler.

**Q20. What is the difference between min-max scaling and normalization in deep learning?**

"Normalization" is used loosely. Min-max scaling is a fixed range transform. Batch normalization standardizes activations using batch statistics during training. Layer normalization standardizes across features for each sample. They operate on different axes and have different training behaviour.

**Q21. How does scaling interact with regularization?**

For models with a coefficient penalty, scaling changes the numerical size of coefficients. A penalty applied to raw features can unfairly shrink the coefficient of a feature with a large unit. Standardizing features puts the penalty on a comparable scale. Min-max scaling can also help, but standardization is more common for regularized linear models.

---

### A19. Exam-style questions with worked answers

#### Question 1

A feature has training values $[4,8,12,16]$. A test value is $20$. Using default min-max scaling, compute the transformed test value and explain whether it is inside $[0,1]$.

**Answer**

The training minimum is 4 and the maximum is 16, so the range is 12.

$$
x'=\frac{20-4}{16-4}=\frac{16}{12}=1.3333.
$$

The value is outside $[0,1]$ because 20 exceeds the training maximum. This is expected behaviour.

#### Question 2

A scaler is fitted on a training set with `Age` range $[20,60]$ and `Salary` range $[30{,}000,90{,}000]$. Using `feature_range=(-1,1)`, compute the transformed value for `Age=40`.

**Answer**

For `Age`, $a=-1$, $b=1$, $x_{\min}=20$, $x_{\max}=60$:

$$
x'=-1+(40-20)\frac{1-(-1)}{60-20}
=-1+20\cdot\frac{2}{40}
=-1+1=0.
$$

The midpoint of the training range maps to the midpoint of the output range, which is 0.

#### Question 3

Explain why `MinMaxScaler().fit_transform(X_test)` is incorrect in a standard supervised workflow.

**Answer**

It learns new minima and maxima from the test set. The test set is supposed to be unseen, so its statistics must not influence preprocessing. The training and test data would also be scaled with different parameters, making the model's learned coefficients inconsistent with the test representation.

#### Question 4

A constant feature has value 7 in every training row. What does min-max scaling do to it, and why is the feature usually removed?

**Answer**

The range is zero, so the fraction is undefined. Implementations handle this by using a safe scale and mapping the constant to the lower bound of the requested range. The feature still carries no information because every row has the same value; it cannot help a model distinguish rows.

#### Question 5

Compare min-max scaling and standardization for a feature with a strong right skew and several extreme values.

**Answer**

Min-max scaling maps the minimum to 0 and the maximum to 1. Extreme values can compress the majority of the data into a narrow interval. Standardization centres by the mean and divides by the standard deviation; it is also affected by extremes, but it does not force a bounded interval. For strongly skewed data with outliers, robust scaling or a shape-changing transform such as a log transform may be more appropriate. The right choice depends on the model and the goal.

---

### A20. Revision summary

- Min-max scaling maps each feature to a fixed interval using the training minimum and maximum.
- The default formula is $(x-x_{\min})/(x_{\max}-x_{\min})$.
- The general range is $[a,b]$: $a+(x-x_{\min})(b-a)/(x_{\max}-x_{\min})$.
- `fit` learns the bounds; `transform` applies them; `fit_transform` does both on training data.
- Split before scaling. Never fit on test data.
- Test values can lie outside the interval.
- Min-max scaling is sensitive to outliers.
- It preserves the shape of the distribution up to an affine change of axis.
- It is not the same as standardization or row-wise `Normalizer`.
- Tree-based models usually do not need it.
- Put the scaler inside a pipeline to keep cross-validation honest.

---

## Part B — Advanced Reading and Reference

### B1. The general affine form

The min-max transformation can be written as

$$
x' = s x + o,
$$

where

$$
s=\frac{b-a}{x_{\max}-x_{\min}},
\qquad
o=a-sx_{\min}.
$$

This is an affine map: a linear scaling followed by a translation. The map is invertible when $s\neq 0$, which holds when $x_{\max}>x_{\min}$.

Scikit-learn stores `scale_` and `min_` in this form. For the default range, `scale_` is the reciprocal of the training range and `min_` is the negative scaled minimum.

#### Numerical precision

Floating-point arithmetic means a transformed training maximum may be `0.9999999999999999` rather than exactly 1. This is normal. Do not write tests that require exact equality without a tolerance.

---

### B2. Constant features and degenerate ranges

If $x_{\max}=x_{\min}$, the mathematical fraction is undefined. Scikit-learn's internal zero-range handling chooses a safe scale and maps the constant feature to the lower bound of the requested range.

A constant feature is usually removed because it has zero variance and cannot separate observations. Some models tolerate it, but it adds a column with no information.

---

### B3. `MaxAbsScaler`

`MaxAbsScaler` divides each feature by its maximum absolute value:

$$
x'=\frac{x}{\max_j |x_j|}.
$$

Properties:

- It maps the fitted data into $[-1,1]$.
- It does not centre the data.
- It preserves zero entries, so it is suitable for sparse data.
- It is sensitive to a single large absolute value.

It is not the same as min-max scaling. A feature with values $[-10,2]$ maps to $[-1,0.2]$; min-max scaling maps it to $[0,1]$.

```python
from sklearn.preprocessing import MaxAbsScaler

maxabs = MaxAbsScaler()
X_train_scaled = maxabs.fit_transform(X_train)
X_test_scaled = maxabs.transform(X_test)
```

---

### B4. `RobustScaler`

`RobustScaler` uses the median and interquartile range:

$$
x'=\frac{x-\operatorname{median}(x)}{\operatorname{IQR}(x)},
$$

where

$$
\operatorname{IQR}=Q_3-Q_1.
$$

The median and IQR are less influenced by extreme values than the mean and standard deviation. The transformed data is not bounded.

```python
from sklearn.preprocessing import RobustScaler

robust = RobustScaler()
X_train_scaled = robust.fit_transform(X_train)
X_test_scaled = robust.transform(X_test)
```

`RobustScaler` also supports `with_centering` and `with_scaling`, and `unit_variance=True` can scale the IQR to approximately one under a normal distribution.

Use it when outliers are present and you want a scaling that is not dominated by them. It does not remove the outliers; it changes how the rest of the data is represented.

---

### B5. Mean normalization

Mean normalization is sometimes defined as

$$
x'=\frac{x-\operatorname{mean}(x)}{x_{\max}-x_{\min}}.
$$

It centres the feature at its mean and divides by the range. It is less common in scikit-learn because there is no dedicated `MeanNormalizer` class. Standardization is usually preferred when centring is the goal.

Do not confuse mean normalization with standardization: standardization divides by the standard deviation, not the range.

---

### B6. Row-wise `Normalizer`

`Normalizer` operates on each sample:

$$
\mathbf{x}_i'=\frac{\mathbf{x}_i}{\lVert\mathbf{x}_i\rVert_p}.
$$

The `norm` argument can be `'l1'`, `'l2'`, or `'max'`. The operation is useful when the relative proportions of a sample matter more than its absolute magnitude, such as in some text-processing workflows.

It is not a column-wise feature scaler. If you apply `Normalizer` to a tabular dataset, each row becomes a unit vector; the columns no longer have independent ranges.

```python
from sklearn.preprocessing import Normalizer

normalizer = Normalizer(norm="l2")
X_train_normalized = normalizer.fit_transform(X_train)
X_test_normalized = normalizer.transform(X_test)
```

`Normalizer` is stateless in practice; fitting does not learn data-dependent parameters, but the API still uses `fit`/`transform` for consistency.

---

### B7. Scaling and model geometry

#### Distance-based models

For Euclidean distance,

$$
d^2=\sum_j (x_{ij}-x_{kj})^2.
$$

If feature $j$ has a large numerical range, it contributes disproportionately. Scaling changes the relative contribution of each feature. This can improve or harm performance depending on whether the original units were meaningful.

#### Linear models with regularization

For ridge regression, the objective is

$$
\min_{\boldsymbol\beta}
\lVert \mathbf{y}-\mathbf{X}\boldsymbol\beta\rVert_2^2
+\lambda\lVert\boldsymbol\beta\rVert_2^2.
$$

A coefficient's numerical size depends on the scale of its feature. If one feature is measured in millions and another in fractions, the same coefficient value corresponds to very different effects. Standardizing features makes the penalty more comparable. Min-max scaling can also make the penalty comparable, but its behaviour depends on the range and is less standard for this purpose.

#### PCA

PCA maximizes variance. A feature with a large numerical range can dominate the first principal component even if its variation is not scientifically important. Standardization is the usual preprocessing choice for PCA. Min-max scaling is possible but changes the variance structure differently.

#### Neural networks

Gradient descent updates depend on the loss surface. Badly scaled inputs create ill-conditioned curvature; small learning rates are needed for stability, and convergence is slow. Normalizing inputs to a comparable range usually helps. Batch normalization and layer normalization address related problems inside the network, but input scaling is still a standard first step.

---

### B8. Feature names and output types

`MinMaxScaler` accepts a DataFrame or an array. If it receives a DataFrame, it learns one bound per column and returns a NumPy array by default. The column names are not preserved.

The notebook wraps the result back into a DataFrame:

```python
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns,
    index=X_train.index,
)
```

This is useful for plotting and for keeping track of columns. In a pipeline, downstream steps usually receive arrays, so column names may be lost anyway. `ColumnTransformer` can expose feature names through `get_feature_names_out()` in recent scikit-learn versions, but you should not assume names survive every transformation.

Always keep the training column order and use the same order at inference time. A scaler fitted on columns `[Age, Salary]` expects those columns in that order.

---

### B9. Inverse transformation

For the default range,

$$
x = x'(x_{\max}-x_{\min}) + x_{\min}.
$$

For a general range, invert the affine map:

$$
x = \frac{x'-o}{s}.
$$

Scikit-learn exposes `inverse_transform`. This is essential when:

- You scaled the target and need predictions on the original scale.
- You want to inspect scaled values in original units.
- You are debugging a pipeline.

Do not refit a new scaler to invert; use the same fitted object.

---

### B10. Scaling and data leakage in cross-validation

A common incorrect pattern is:

```python
# Wrong: the scaler sees all rows before cross-validation
X_scaled = MinMaxScaler().fit_transform(X)
scores = cross_val_score(model, X_scaled, y, cv=5)
```

The scaler has already seen the validation folds. The correct pattern is:

```python
# Right: the pipeline fits the scaler inside each fold
pipe = Pipeline([
    ("scaler", MinMaxScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])
scores = cross_val_score(pipe, X, y, cv=5)
```

The difference can be small for a stateless transform such as `log1p`, but it is essential for stateful preprocessing such as scaling, imputation, and encoding.

---

### B11. Advanced interview questions

**Q1. Why does `MinMaxScaler` use `data_min_` and `data_max_` rather than recomputing the bounds on every call?**

Because the test data must be transformed with the training bounds. Recomputing on each call would make the transformation data-dependent and would leak test statistics.

**Q2. What is the difference between `scale_` and the range?**

For the default output range, `scale_` is the reciprocal of the range. For a custom range, `scale_` includes the output width. `data_range_` stores the original range.

**Q3. Can min-max scaling make a feature more normal?**

No. It is an affine transformation, so it preserves skewness and the shape of the distribution. A log or power transform changes the shape.

**Q4. Why is min-max scaling often used for image pixels?**

Pixel values are naturally bounded, often in $[0,255]$. Dividing by 255 maps them to $[0,1]$, which is a convenient input range for neural networks. The bounds are known and stable, so the outlier sensitivity is less of a concern.

**Q5. What happens if the training set has a different range from the production data?**

Production values outside the training range map outside the output interval. This may be acceptable, or it may indicate data drift. Monitor the proportion of out-of-range values and consider retraining or clipping if the shift is severe.

**Q6. How does min-max scaling affect the interpretation of coefficients?**

It changes the units of the coefficients. A coefficient now describes the change in the prediction for a one-unit change in the scaled feature, not the original feature. To interpret in original units, inverse-transform the feature or convert the coefficient.

**Q7. Is min-max scaling a linear transformation?**

It is an affine transformation: a linear scaling plus a translation. It is linear in the sense that it preserves straight lines and relative distances along a feature, but it is not a linear map through the origin unless the minimum is zero.

**Q8. What is the role of `clip` in `MinMaxScaler`?**

`clip=True` clips transformed values to the requested feature range. It is useful when downstream models cannot tolerate out-of-range inputs, but it discards information about how far outside the range a value lies. The default is `clip=False`.

---

### B12. Notebook coverage matrix

| Notebook area | Cells | What it teaches | Status |
|---|---:|---|---|
| Hand formula and list example | 4–6 | Formula and edge case | Saved output |
| Manual DataFrame scaling | 7–10 | Per-column scaling | Saved output |
| `MinMaxScaler` fit/transform | 11–15 | API and learned attributes | Saved output |
| Train/test leakage demonstration | 16–19 | Split-before-scale, out-of-range test values | Saved output |
| Shape preservation | 20–21 | Scaling does not change shape | Saved output |
| Comparison with StandardScaler and Normalizer | 22–23 | Different operations | Saved output |
| Common mistakes | 24 | Debugging checklist | Markdown |
| Pipeline | 25–27 | Leak-free workflow | Saved output |
| Experiments | 28–29 | Units, custom range, unseen values | Saved output |
| Interview and practice | 30–32 | Revision | Markdown |
| Cheat sheet | 33 | Summary | Markdown |
| Trailing notes | 34–35, 55–58 | Mean normalization, MaxAbs, Robust, comparison | Markdown |
| Wine data EDA and plots | 36–54 | Real-data scaling and visualization | Saved outputs |
| Experiment banner | 59 | Range exercise | Markdown |
| Key takeaways | 60 | Revision | Markdown |

The stub notebook contains four markdown cells and no executable code. It is preserved as a historical artifact, not as a primary learning source.

---

### B13. Source fidelity and caveats

- The saved outputs in `my_notebook.ipynb` come from the user's environment. They are reported here as saved results, not as independently reproduced values.
- The notebook loads `wine_data.csv` from an absolute path outside the day folder. The dataset is present locally at `datasets/wine_data.csv`; the relative-path version is a portability correction, not a saved notebook result.
- The notebook contains duplicate code cells and repeated plots. They are preserved; the chapter does not treat duplicates as separate experiments.
- The toy pipeline accuracy `0.6666666666666666` is based on three test rows and is not evidence of model quality.
- The notebook's trailing markdown mentions mean normalization, max-absolute scaling, and robust scaling but does not implement them. Part B supplies the missing formulas and API references.
- No model tuning, cross-validation, or statistical significance test was performed in this notebook.

---

### B14. References

- scikit-learn documentation: `MinMaxScaler`, `StandardScaler`, `MaxAbsScaler`, `RobustScaler`, `Normalizer`.
- CampusX Day 25: Feature Scaling — Normalization / MinMaxScaler.
- Day 24 notes: standardization and z-scores.
- Day 28 notes: `ColumnTransformer`.
- Day 29 notes: pipelines, cross-validation, and model export.
- Day 30 notes: `FunctionTransformer` and log transforms.

---

### B15. Final connection

Min-max scaling is a small affine operation with a large workflow lesson: **learn preprocessing parameters from training data only, then reuse them unchanged**. The formula is simple; the discipline is what makes it useful in real ML systems.

The next day moves from numeric scaling to categorical encoding. The same separation of `fit` and `transform` will reappear, and the same leakage rule will apply.
