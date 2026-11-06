# Day 30 — FunctionTransformer and the Log Transform

> **Course topic:** CampusX Day 30, *Function Transformer | Log Transform | Reciprocal | Square Root*.
> **Personal notebooks:**
> `my_notebook.ipynb` (77 cells, documented walkthrough, no stored outputs) and
> `DAY30_Function-Transformer-preview-OUT-OF-SCOPE.ipynb` (29 cells, executed preview with saved outputs).
> **Dataset:** `datasets/train.csv` (Titanic, 891 rows).
> **Reference:** `reference_notebooks/day30.ipynb` (do not edit).
>
> **Part A** is the main learning chapter and is sufficient on its own. **Part B** preserves internals, mathematics, coverage, and advanced reference material.

---

## Part A — Main Learning Chapter

### A1. Learning objectives

By the end of this chapter you should be able to:

1. Explain why a skewed feature can hurt a model.
2. Read a histogram, a Q-Q plot, and a skewness value together.
3. Explain what `FunctionTransformer` wraps and when the wrapped function executes.
4. Distinguish `np.log1p` (a function object) from `np.log1p(X)` (an executed result).
5. Apply `log1p` to a right-skewed feature and compare distributions.
6. Route a transform to one column with `ColumnTransformer`.
7. Explain why `fit_transform` on test is wrong, and why it is harmless for a stateless transform.
8. Compare holdout accuracy with cross-validation correctly.
9. Diagnose the notebook's preserved bugs: test-side refit, whole-data transform before CV, duplicate classifier in CV, deprecated `distplot`, and the malformed helper cell.
10. Explain why `np.sin` is a bad transform for `Fare`.

---

### A2. The problem: skew and scale-sensitive models

Titanic `Fare` is not symmetric. Most passengers paid small fares, and a few paid very large fares. A typical slice looks like:

```text
Fare: 0, 7, 8, 8, 10, 15, 20, 25, 30, 500, 800
```

The mass is on the left; the rare large values form a long right tail. This is **right skew**.

Skew matters differently for different models:

- **Logistic regression** uses a linear combination of features. A feature with a long tail can dominate the gradient and produce a boundary that is sensitive to a few large values.
- **Distance-based models** (KNN, clustering) measure distances; one huge fare can dominate the distance.
- **Decision trees** split on order, so they are less sensitive to scale. But a long tail still creates many possible split points and can make the tree spend capacity on rare values.

The goal of a log transform is to **compress the tail without deleting rows**. It changes the shape of the feature distribution, not the number of examples.

A second issue is scale. A fare of 500 is hundreds of times larger than a fare of 5. A linear model may treat that magnitude difference as importance. Compressing the tail reduces this effect.

---

### A3. Three ways to check whether a feature looks normal

The notebook uses three complementary checks.

#### 1. Histogram

A histogram shows the shape of the distribution:

- A symmetric bell suggests approximate normality.
- A long right tail suggests right skew.
- A long left tail suggests left skew.
- Multiple peaks suggest a mixture or grouping.

The histogram is a visual summary, not a formal test. The bin width can change the impression, so do not rely on one plot.

#### 2. Q-Q plot

A quantile-quantile plot compares the sample quantiles with the quantiles of a theoretical distribution, here the normal distribution:

```python
stats.probplot(X_train["Fare"], dist="norm", plot=plt)
```

The x-axis shows theoretical normal quantiles; the y-axis shows the sorted sample values. If the points lie close to the 45-degree reference line, the sample looks approximately normal.

Systematic bending away from the line indicates a difference in shape:

- The upper tail bending upward suggests a heavier right tail than normal.
- The lower tail bending downward suggests a heavier left tail.
- An S-shape suggests a different skew or a bounded distribution.

A Q-Q plot is more informative than a histogram for tail behavior, but it is still visual evidence, not proof.

#### 3. Skewness

Skewness is a numeric summary of asymmetry. A common definition is:

$$
\text{skew} = \frac{\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^3}{\left(\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2\right)^{3/2}}.
$$

- Near 0: approximately symmetric.
- Positive: right-skewed (long right tail).
- Negative: left-skewed (long left tail).

The magnitude is not standardized across all contexts; use it alongside the plots.

A reproduced calculation on the local training split gives:

| Feature | Raw skew | After `log1p` |
|---|---:|---:|
| `Age` | `0.4363` | `-2.1544` |
| `Fare` | `5.4727` | `0.4401` |

These are supplementary reproduced values, not saved notebook outputs. They show the key lesson:

- `Fare` is strongly right-skewed and the log brings it close to symmetric.
- `Age` is only mildly right-skewed, and the log **overshoots**, making it strongly left-skewed.

So a log transform is not automatically an improvement. It depends on the feature.

---

### A4. The log transform

The notebook uses:

$$
\operatorname{log1p}(x) = \log(1+x).
$$

Why `1+x` instead of `x`?

- `log(0)` is negative infinity, which would poison the column.
- `log1p(0) = 0`, so zeros remain finite.
- For large $x$, `log1p(x)` behaves almost like `log(x)`.

Numerical example:

| $x$ | $\log(1+x)$ |
|---:|---:|
| 0 | 0 |
| 10 | 2.3979 |
| 100 | 4.6151 |
| 1000 | 6.9088 |

The raw ratio between 1000 and 10 is 100. The log ratio is:

$$
\frac{6.9088}{2.3979} \approx 2.88.
$$

The large value is compressed far more than the small value. This is what "tail-taming" means.

#### Domain

`log1p(x)` is mathematically defined for $x > -1$. For this workflow, the feature is non-negative (`Fare >= 0`), so the transform is safe. For general data:

- `np.log(x)` requires $x > 0$.
- `np.log1p(x)` requires $x > -1$.
- Box-Cox requires $x > 0$ strictly.
- Yeo-Johnson accepts any real value.

Do not apply a log transform to arbitrary negative values without understanding the resulting domain.

#### Inverse

The inverse of `log1p` is:

$$
\operatorname{expm1}(y) = e^y - 1.
$$

This is useful if you transform a target and need to return to the original scale. The notebook transforms features, not the target, so the inverse is not used.

---

### A5. What `FunctionTransformer` is

`FunctionTransformer` adapts an ordinary function into a scikit-learn transformer:

```python
from sklearn.preprocessing import FunctionTransformer

trf = FunctionTransformer(func=np.log1p)
```

It gives the function a `fit`/`transform` interface so it can sit inside a `ColumnTransformer` or `Pipeline`.

#### What happens at construction

Nothing is computed. The constructor stores a reference to the function:

```python
trf = FunctionTransformer(func=np.log1p)
```

At this point:

- No data has been passed.
- No values have been transformed.
- `func` holds the function object `np.log1p`.

#### The critical distinction

```python
FunctionTransformer(func=np.log1p)     # stores the function
FunctionTransformer(func=np.log1p(X))  # calls the function now, stores the result
```

- `np.log1p` is a **function object**. It can be called later on any input.
- `np.log1p(X)` is an **array**. It has already been computed on one specific `X`.

`FunctionTransformer` needs the recipe, not the cooked dish. It calls the stored function later inside `transform`.

#### Is it stateless?

For `np.log1p`, yes. The function has no parameters learned from data:

- No mean.
- No standard deviation.
- No categories.
- No quantiles.

`fit` does essentially nothing except validate and return `self`. `transform` applies `log1p` element-wise.

This matters for the notebook's test-side refit bug: refitting a stateless transformer learns nothing, so the numerical result is the same as a correct `transform`. But the habit is still wrong, and the same line would leak with a stateful transformer.

#### Important parameters

| Parameter | Meaning |
|---|---|
| `func` | Function applied by `transform` |
| `inverse_func` | Optional inverse function for `inverse_transform` |
| `validate` | If `True`, checks the input is 2-D and finite |
| `accept_sparse` | Whether sparse input is accepted |
| `check_inverse` | Whether to check that `inverse_func` reverses `func` |
| `feature_names_out` | Controls output feature names |

The notebook uses only `func`.

---

### A6. The data pipeline in the notebook

The notebook uses a three-column slice of the Titanic data:

```python
df = pd.read_csv(r"E:\MACHINE_LEARNING\DATA_SETS\train.csv")
df = df[["Survived", "Age", "Fare"]]
df["Age"] = df["Age"].fillna(df["Age"].mean())

X = df.iloc[:, 1:3]
y = df.iloc[:, 0]
```

Shapes:

```text
df: (891, 3)
X:  (891, 2)   # Age, Fare
y:  (891,)
```

#### The imputation leak

`df["Age"].fillna(df["Age"].mean())` computes the mean over **all 891 rows**, including the rows that will become the test set. This is a simplification and a mild leakage pattern.

The correct workflow is:

1. Split first.
2. Compute the training mean.
3. Fill training and test values with the training mean, or put the imputer inside a pipeline.

The notebook does it before the split. The effect is small for this dataset, but the habit is wrong. The notes preserve the notebook's behavior and explain the correct pattern.

#### The split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=34
)
```

Shapes:

```text
X_train: (712, 2)
X_test:  (179, 2)
```

The fixed `random_state=34` makes the split reproducible.

---

### A7. Baseline: raw features

The notebook creates two fresh estimators:

```python
clf = LogisticRegression()
clf2 = DecisionTreeClassifier()
```

It fits both on the raw training features and evaluates on the test set.

Saved results from the executed preview:

| Model | Raw holdout accuracy |
|---|---:|
| Logistic regression | `0.7206703910614525` |
| Decision tree | `0.6815642458100558` |

These are the baselines that later transforms are compared against.

#### A note on decision-tree randomness

`DecisionTreeClassifier()` has `random_state=None` by default. The tree can randomize feature order when splits tie, so its accuracy can change between runs. A reproduced run on the same data gives `0.6871508379888268` for the raw tree, not the saved `0.6815642458100558`.

The logistic regression results are deterministic for this setup and reproduce exactly. When comparing saved and reproduced tree scores, allow for this randomness. For a stable experiment, set `random_state` in the tree.

---

### A8. The full log transform

The notebook wraps `log1p`:

```python
trf = FunctionTransformer(func=np.log1p)
```

It then transforms the training and test sets:

```python
X_train_trans = trf.fit_transform(X_train)
X_test_trans = trf.fit_transform(X_test)   # preserved bug
```

The first line is correct: fit (nothing to learn) and transform the training data.

The second line refits the transformer on the test data. The correct line is:

```python
X_test_trans = trf.transform(X_test)
```

For `log1p`, refitting learns nothing, so the transformed values and scores are numerically unchanged. The line is still wrong because:

- It teaches the wrong habit.
- It would leak test statistics for a stateful transformer.
- It would produce different behavior if `FunctionTransformer` were replaced by an imputer, scaler, or power transformer.

The notebook includes a corrected cell that uses `transform`, but the original bug is preserved.

#### What the transform does to the data

`log1p` is applied element-wise to both columns:

```text
X_train (712, 2) --log1p--> X_train_trans (712, 2)
X_test  (179, 2) --log1p--> X_test_trans  (179, 2)
```

The shape does not change. The values do:

- Large fares shrink strongly.
- Small fares change little.
- Zero fares remain zero.
- `Age` values also change, even though `Age` is only mildly skewed.

---

### A9. Results after the full log transform

Saved results from the executed preview:

| Model | Raw | Full `log1p` |
|---|---:|---:|
| Logistic regression | `0.7206703910614525` | `0.7374301675977654` |
| Decision tree | `0.6815642458100558` | `0.7094972067039106` |

Both models improve on the holdout split.

- Logistic regression gains about 0.0168, roughly 3 additional correct predictions out of 179.
- Decision tree gains about 0.0279, roughly 5 additional correct predictions.

These are small differences on a single 179-row test set. They are encouraging, but they are not proof that the transform will always help. The CV results below give a more stable comparison.

#### Why did the tree improve?

Trees split on order. A monotone transform like `log1p` preserves the order of values, so every possible split threshold has an equivalent threshold on the log scale. In principle, a tree can achieve the same partitions before and after a monotone transform.

In practice, the tree's greedy split search and its regularization (minimum samples per leaf, maximum depth, etc.) interact with the spacing of values. Compressing the tail can change which splits are considered useful and can reduce the dominance of rare large fares. The improvement is possible, but it is not guaranteed by the math of monotonicity.

#### Why did logistic regression improve?

Logistic regression uses a linear combination. A long-tailed feature can produce large gradients for a few rows and distort the fitted coefficients. Compressing the tail makes the feature's scale more balanced and can produce a cleaner boundary.

The improvement is plausible, but the notebook does not isolate which column caused it. The transform changes both `Age` and `Fare`.

---

### A10. Cross-validation after the full transform

The notebook transforms the entire feature matrix before cross-validation:

```python
X_transf = trf.fit_transform(X)

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

print(np.mean(cross_val_score(clf, X_transf, y, scoring="accuracy", cv=15)))
print(np.mean(cross_val_score(clf2, X_transf, y, scoring="accuracy", cv=15)))
```

Saved results:

| Model | 15-fold CV mean |
|---|---:|
| Logistic regression | `0.6770056497175143` |
| Decision tree | `0.6621092278719396` |

The CV mean is lower than the holdout score. This is not a contradiction:

- The holdout score uses one split of 179 rows.
- The CV mean averages 15 folds on all 891 rows.
- Each fold has a different training/validation composition.
- Fifteen folds of about 59 rows each produce a noisy mean.

The CV mean is generally a better estimate of generalization than a single holdout split, but it has its own variance. Report the mean and the spread, not just the mean.

#### The leakage pattern

`trf.fit_transform(X)` runs before `cross_val_score`. For `log1p`, this is harmless because the transform is stateless. But the pattern is wrong:

- A stateful transformer would learn from all rows, including the validation folds.
- The CV score would be optimistic.
- The correct shape is a pipeline passed to `cross_val_score`, so the transform is fitted inside each fold.

The notebook includes a corrected pipeline cell, but it is not executed.

---

### A11. The recommended pipeline shape

The honest cross-validation pattern is:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("prep", ColumnTransformer([
        ("log", FunctionTransformer(np.log1p), ["Fare"]),
    ], remainder="passthrough")),
    ("clf", LogisticRegression()),
])

np.mean(cross_val_score(pipe, X, y, scoring="accuracy", cv=15))
```

Inside each fold:

1. The `ColumnTransformer` is fitted on the fold's training rows.
2. `Fare` is logged; `Age` passes through.
3. Logistic regression is fitted on the transformed training rows.
4. The validation rows are transformed with the same fitted transformer and scored.

For a stateless `log1p`, the result is numerically the same as the pre-transformed CV. A reproduced run gives `0.6679472693032015`, matching the selective CV logistic-regression result. For a stateful transform, the pipeline result would differ and would be the honest one.

---

### A12. Selective routing: log only `Fare`

The notebook then routes the transform to one column:

```python
trf2 = ColumnTransformer(
    [("log", FunctionTransformer(np.log1p), ["Fare"])],
    remainder="passthrough",
)
```

The tuple is `(name, transformer, columns)`:

- `name`: `"log"`.
- `transformer`: `FunctionTransformer(np.log1p)`.
- `columns`: `["Fare"]`.

`remainder="passthrough"` keeps `Age` unchanged. Without it, the default `remainder="drop"` would remove `Age`.

#### Output order

`ColumnTransformer` concatenates transformed blocks first, then the remainder:

```text
input:  Age, Fare
          |
          +-- Fare -> log1p -> transformed Fare (column 0)
          +-- Age  -> passthrough ----------> raw Age (column 1)
          |
          v
output: [log(Fare), Age]
```

The output is a NumPy array with no column names. Column 0 is logged `Fare`; column 1 is raw `Age`. This ordering is why the helper later uses `X_trans[:, 0]` to refer to transformed `Fare`.

The fit/transform discipline is correct here:

```python
X_train_trans2 = trf2.fit_transform(X_train)
X_test_trans2 = trf2.transform(X_test)
```

Fit on training data; transform test data.

#### Results

Saved holdout results:

| Model | Raw | Full log | Selective log (`Fare` only) |
|---|---:|---:|---:|
| Logistic regression | `0.7206703910614525` | `0.7374301675977654` | `0.7150837988826816` |
| Decision tree | `0.6815642458100558` | `0.7094972067039106` | `0.7039106145251397` |

Selective routing gives a different result from the full log:

- Logistic regression is below its raw baseline when only `Fare` is logged and `Age` stays raw.
- The tree still improves over its raw baseline.

Why? The full log changes both columns. `Age` is only mildly skewed, but the log makes it strongly left-skewed (reproduced skew `-2.1544`). The selective transform leaves `Age` raw, so the linear model sees a different feature geometry. The notebook does not isolate these effects, so do not overinterpret the exact ordering. The holdout differences are only a few predictions.

The important lesson is that **per-column routing gives per-column consequences**. Transforming a column is a modeling decision, not a universal improvement.

---

### A13. The duplicate-classifier bug in selective CV

The notebook then cross-validates the selectively transformed data:

```python
X_transf = trf2.fit_transform(X)

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

print("LR", np.mean(cross_val_score(clf, X_transf, y, scoring="accuracy", cv=15)))
print("DT", np.mean(cross_val_score(clf, X_transf, y, scoring="accuracy", cv=15)))
```

Saved output:

```text
LR 0.6679472693032015
DT 0.6679472693032015
```

Both lines are identical to 16 decimal places. That is the fingerprint of a copy-paste bug: the second line evaluates `clf` again, not `clf2`.

The corrected line is:

```python
print("DT", np.mean(cross_val_score(clf2, X_transf, y, scoring="accuracy", cv=15)))
```

The corrected decision-tree value is not saved. A reproduced run gives `0.6599246704331451`, but the tree's default randomness means this value is not stable across runs. The logistic-regression value is deterministic and reproduces exactly.

The lesson: **identical scores from different model families are almost always a code bug**, not a miracle of agreement. Check the estimator argument.

---

### A14. Seeing the effect: Q-Q before and after

The notebook plots Q-Q comparisons for `Age` and `Fare` before and after the full log transform.

For `Age`:

```python
plt.subplot(121)
stats.probplot(X_train["Age"], dist="norm", plot=plt)
plt.title("AGE Before LOG")

plt.subplot(122)
stats.probplot(X_train_trans["Age"], dist="norm", plot=plt)
plt.title("AGE After LOG")
```

For `Fare`:

```python
plt.subplot(121)
stats.probplot(X_train["Fare"], dist="norm", plot=plt)
plt.title("Fare Before LOG")

plt.subplot(122)
stats.probplot(X_train_trans["Fare"], dist="norm", plot=plt)
plt.title("Fare After LOG")
```

#### How to read the plots

- The left panel is the raw feature.
- The right panel is the logged feature.
- The diagonal is the normal reference.
- Closer to the diagonal means closer to normal.

Expected pattern:

- Raw `Fare` bends far from the diagonal at the top because of the expensive tail.
- Logged `Fare` should hug the line more closely.
- `Age` was never strongly skewed; the change is smaller, and the log may overshoot into left skew.

The notebook repeats the `Age` and `Fare` Q-Q plots in later cells. The repetition is redundant but harmless; it confirms the pattern is stable.

The plots are visual evidence, not proof. The reproduced skew values support the visual reading: `Fare` improves from `5.4727` to `0.4401`; `Age` moves from `0.4363` to `-2.1544`.

---

### A15. The reusable helper and the sine experiment

The notebook introduces a helper that takes any function and applies it to `Fare`:

```python
def apply_transform(transform):
    X = df[["Age", "Fare"]]
    y = df["Survived"]

    trf = ColumnTransformer(
        [("log", FunctionTransformer(transform), ["Fare"])],
        remainder="passthrough",
    )

    X_trans = trf.fit_transform(X)
    clf = LogisticRegression()

    print("Accuracy", np.mean(cross_val_score(
        clf, X_trans, y, scoring="accuracy", cv=15
    )))

    plt.figure(figsize=(14, 4))
    plt.subplot(121)
    stats.probplot(X["Fare"], dist="norm", plot=plt)
    plt.title("Fare Before Transform")
    plt.subplot(122)
    stats.probplot(X_trans[:, 0], dist="norm", plot=plt)
    plt.title("Fare After Transform")
    plt.show()
```

The helper demonstrates two important ideas:

1. **Functions are objects.** `transform` is a parameter; `apply_transform(np.log1p)` and `apply_transform(np.sin)` pass different function objects.
2. **The transform is a modeling choice.** The same machinery can test good and bad ideas.

#### The helper's leakage pattern

The helper fits the transformer on the full `X` and then cross-validates on the transformed data:

```python
X_trans = trf.fit_transform(X)
cross_val_score(clf, X_trans, y, ...)
```

For `log1p`, this is harmless because the transform is stateless. For a stateful transform, it would leak. The honest shape is a pipeline passed to `cross_val_score`, as in A11.

#### The sine experiment

```python
apply_transform(np.sin)
```

Saved output from the executed preview:

```text
Accuracy 0.6184557438794727
```

This is far below the logistic-regression baseline of about `0.72`.

Why is sine bad?

- Sine is periodic. Distinct fares can map to the same value.
- The mapping is not monotonic over the fare range.
- The order information that a linear model or tree could use is destroyed.
- The transform does not compress a tail in a meaningful way; it wraps values.

A negative result is still evidence. It shows that the transform must match the structure of the data. Log compresses a right tail; sine folds values together.

#### A source-fidelity warning about the helper

`my_notebook.ipynb` contains a malformed helper cell. Cell 70 begins with the token `ml` followed by an indented block:

```text
ml
    print("Accuracy", np.mean(cross_val_score(
        clf, X_trans, y, scoring='accuracy', cv=15
    )))
    ...
```

This cell does not define `apply_transform`. It would raise a syntax/indentation error if run. Cell 74 then calls `apply_transform(np.sin)`, which would raise `NameError` because the function was never defined.

The working helper is defined in `DAY30_Function-Transformer-preview-OUT-OF-SCOPE.ipynb`, and the saved `0.6184557438794727` comes from that preview notebook. The malformed cell in `my_notebook.ipynb` is preserved as a notebook bug; the notes explain the intended code without silently editing the notebook.

---

### A16. The deprecation warning

The notebook uses:

```python
sns.distplot(X_train["Age"])
sns.distplot(X_train["Fare"])
```

The executed preview saves a warning:

```text
UserWarning: `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
```

`distplot` is deprecated. Use:

- `sns.histplot(...)` for an axes-level histogram.
- `sns.displot(...)` for a figure-level distribution plot.

The notebook includes a corrected loop:

```python
for col in ["Age", "Fare"]:
    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    sns.histplot(X_train[col], kde=True)
    plt.title(col + " distribution")
    plt.subplot(122)
    stats.probplot(X_train[col], dist="norm", plot=plt)
    plt.show()
```

This cell is marked "NOT EXECUTED". It is the modern replacement; the original `distplot` calls are preserved for course fidelity.

---

### A17. Holdout versus cross-validation

The notebook reports two kinds of numbers:

| Protocol | Data | What it measures |
|---|---|---|
| Holdout | One 80/20 split, 179 test rows | Performance on one unseen sample |
| 15-fold CV | 15 rotations over 891 rows | Average performance across folds |

They are not directly comparable:

- The holdout score is one realization. It can be lucky or unlucky.
- The CV mean averages many realizations but still has variance.
- The CV folds are smaller, so each model is trained on less data.
- The holdout test set is never used for fitting; the CV folds reuse data for training and validation in rotation.

A good workflow is:

1. Use CV to compare transforms and models.
2. Pick the best configuration.
3. Report a final holdout score, or nested CV, for an unbiased estimate.

Do not tune on the holdout set and then report the same holdout score as unbiased.

---

### A18. Common mistakes and debugging

#### Mistake 1: `fit_transform` on test

```python
# Wrong
X_test_trans = trf.fit_transform(X_test)

# Correct
X_test_trans = trf.transform(X_test)
```

For a stateless transform, the numbers may not change. The habit is still wrong and becomes a leak for stateful transforms.

#### Mistake 2: transforming all data before CV

```python
# Wrong shape
X_transf = trf.fit_transform(X)
cross_val_score(clf, X_transf, y, cv=15)
```

Use a pipeline inside `cross_val_score`:

```python
cross_val_score(pipe, X, y, cv=15)
```

#### Mistake 3: evaluating the wrong estimator

Identical CV scores from two different models are a code smell. Check the estimator passed to `cross_val_score`.

#### Mistake 4: using `distplot`

It is deprecated and removed in seaborn 0.14+. Use `histplot` or `displot`.

#### Mistake 5: log on zeros or negatives

`np.log(0)` is negative infinity. Use `log1p` for non-negative data with zeros. Do not apply a log to arbitrary negative values.

#### Mistake 6: assuming log always normalizes

`Age` becomes more skewed after `log1p` in the reproduced calculation. A transform can overshoot. Check the histogram, Q-Q plot, and skew before and after.

#### Mistake 7: reading transformed columns by old names

After `ColumnTransformer`, the output is a NumPy array. Column names are gone. In the selective transform, column 0 is logged `Fare` and column 1 is raw `Age`. Use positions or reconstruct names with `get_feature_names_out()`.

#### Mistake 8: ignoring tree randomness

`DecisionTreeClassifier()` has `random_state=None`. Reproduced scores can differ from saved scores. Set `random_state` for stable comparisons.

#### Mistake 9: treating a holdout gain as proof

A gain of 0.02 on 179 rows is a few predictions. Confirm with CV and, ideally, repeated splits.

#### Mistake 10: defining a helper in a malformed cell

The `my_notebook` helper cell is broken. If a function is undefined, check the cell source before assuming a logic error. The preview notebook contains the working definition.

---

### A19. Where this sits in the ML workflow

The transform belongs inside the preprocessing chain:

```text
Raw data
   |
   v
Train/test split
   |
   v
ColumnTransformer
   |-- numeric imputation
   |-- log transform for skewed columns
   |-- encoding for categorical columns
   |-- passthrough for unchanged columns
   |
   v
Pipeline with model
   |
   v
Cross-validation / holdout evaluation
```

It connects to the earlier days:

- Day 19–20: distribution checks and skew intuition.
- Day 24–25: scaling changes location and spread but not shape.
- Day 26–27: encoding changes categorical representation.
- Day 28: `ColumnTransformer` routes per-column transforms.
- Day 29: `Pipeline` keeps preprocessing inside CV.
- Day 30: `FunctionTransformer` wraps a fixed function; `log1p` is the first distribution transform.
- Day 31: `PowerTransformer` learns the transformation parameter instead of fixing it.

---

### A20. Interview preparation

#### Level 1 — Recall

**Q1. What does `FunctionTransformer` do?**

It wraps an ordinary function into a scikit-learn transformer with `fit` and `transform`, so the function can be used inside `ColumnTransformer` and `Pipeline`.

**Q2. What is `log1p(x)`?**

`log(1+x)`. It is safe at zero because `log1p(0)=0`.

**Q3. What does a Q-Q plot compare?**

Sample quantiles against theoretical quantiles of a reference distribution, usually the normal distribution.

**Q4. What does a positive skew value mean?**

A long right tail.

#### Level 2 — Explanation

**Q5. Why use `log1p` instead of `log` for `Fare`?**

`Fare` contains zeros. `log(0)` is negative infinity; `log1p(0)=0` keeps the values finite.

**Q6. Why does `FunctionTransformer(func=np.log1p)` not compute anything at construction?**

The constructor stores the function object. The function is called later inside `transform`, when data is available.

**Q7. Why is `fit_transform(X_test)` wrong?**

It refits the transformer on test data. For a stateful transformer, that leaks test statistics. The correct call is `transform(X_test)` using the training-fitted object.

**Q8. Why are the scores still valid after the test-side refit in this notebook?**

`log1p` is stateless. Refitting learns nothing, so the transformed values are the same as a correct `transform`. The habit is still wrong.

#### Level 3 — Application

**Q9. How would you log only `Fare` and keep `Age` unchanged?**

```python
ColumnTransformer(
    [("log", FunctionTransformer(np.log1p), ["Fare"])],
    remainder="passthrough",
)
```

**Q10. What is the output column order?**

Transformed `Fare` first, passthrough `Age` second. The output is a NumPy array with no names.

**Q11. How would you cross-validate the transform honestly?**

```python
pipe = Pipeline([
    ("prep", ColumnTransformer([
        ("log", FunctionTransformer(np.log1p), ["Fare"]),
    ], remainder="passthrough")),
    ("clf", LogisticRegression()),
])
cross_val_score(pipe, X, y, cv=15)
```

#### Level 4 — Analysis

**Q12. Why does the full log improve logistic regression while selective logging does not?**

The full log changes both `Age` and `Fare`; the selective log changes only `Fare`. `Age` is only mildly skewed, and logging it makes it strongly left-skewed. The exact contribution of each column is not isolated in the notebook. The holdout differences are only a few predictions, so do not overinterpret them.

**Q13. Why does `Age` become more skewed after `log1p`?**

`Age` is only mildly right-skewed. The log compresses the upper values strongly and expands the lower values, pushing the distribution into a left skew. A transform can overshoot.

**Q14. Why does the decision tree improve after a monotone transform?**

A monotone transform preserves order, so in principle the tree can find equivalent splits. In practice, the greedy split search, regularization, and value spacing interact with the transform, so the fitted tree and its accuracy can change. The improvement is possible but not guaranteed.

#### Level 5 — Design

**Q15. How would you choose between `log1p`, square root, and a power transform?**

Check the distribution shape and domain, try candidates, compare with cross-validation on the same folds, and keep the transform only if it improves a meaningful metric. Do not choose by habit.

**Q16. How would you handle zeros and negatives?**

- Zeros: use `log1p`.
- Strictly positive: `log` or Box-Cox.
- Any sign: Yeo-Johnson.
- Negative values with a meaningful shift: consider a shifted log, but the shift changes the interpretation.

**Q17. How would you make the helper fold-honest?**

Replace the pre-transformed `X_trans` with a pipeline and pass raw `X` to `cross_val_score`.

#### Level 6 — Evaluation

**Q18. Why is the CV mean lower than the holdout accuracy?**

Different protocols and different data. The holdout is one split; the CV mean averages 15 folds of different composition. The CV mean is not required to match the holdout.

**Q19. What does an identical CV score for two models indicate?**

A likely code bug, such as evaluating the same estimator twice. Different model families rarely agree to 16 decimals by chance.

**Q20. How would you report the uncertainty of a CV estimate?**

Report the mean and standard deviation (or standard error) of the fold scores, not just the mean. Consider repeated cross-validation for a more stable estimate.

#### Level 7 — Advanced

**Q21. What is the relationship between `log1p` and the Box-Cox transform?**

Box-Cox is a family of power transforms indexed by $\lambda$. For $\lambda=0$, Box-Cox is the log transform. `log1p` is a shifted log used when zeros are present. Box-Cox requires strictly positive data.

**Q22. Why is `FunctionTransformer` stateless while `PowerTransformer` is stateful?**

`FunctionTransformer` applies a fixed function. `PowerTransformer` estimates $\lambda$ from the training data, so it stores a learned parameter and must be fitted only on training data.

**Q23. What are the risks of applying a transform before cross-validation?**

If the transform is stateful, it learns from validation folds and leaks information. Even for a stateless transform, the pattern is bad because replacing it with a stateful transform later introduces leakage silently.

**Q24. How would you invert a log-transformed prediction?**

Use `np.expm1(y_pred)` for `log1p`, or `np.exp(y_pred)` for natural log. Check the domain and whether the model was trained on the transformed target.

---

### A21. Exam-style questions with worked answers

#### Question 1

A feature has values `0, 10, 100, 1000`. Compute `log1p` for each and compare the ratio of the largest to the smallest nonzero value before and after.

**Answer**

$$
\log(1+0)=0,\quad \log(11)=2.3979,\quad \log(101)=4.6151,\quad \log(1001)=6.9088.
$$

Before, the ratio $1000/10=100$. After, the ratio $6.9088/2.3979\approx 2.88$. The transform compresses the large values much more than the small values.

#### Question 2

Why is `FunctionTransformer(func=np.log1p(X))` wrong?

**Answer**

`np.log1p(X)` executes the function immediately and returns an array. `FunctionTransformer` expects a callable, not a precomputed result. The correct form is `FunctionTransformer(func=np.log1p)`, which stores the function object and applies it later inside `transform`.

#### Question 3

A transformer is stateless. Is `fit_transform(X_test)` harmless?

**Answer**

Numerically, it may be harmless because fitting learns nothing. Procedurally, it is wrong: it breaks the train/test discipline, and it becomes a leak if the transformer is replaced by a stateful one. Always use `transform` on test data.

#### Question 4

The selective `ColumnTransformer` logs `Fare` and passes `Age` through. What are the output columns and their order?

**Answer**

Column 0 is `log1p(Fare)`; column 1 is raw `Age`. The transformed block comes first, then the passthrough block. The output is a NumPy array with no column names.

#### Question 5

Two models produce identical 15-fold CV means. What is the likely cause?

**Answer**

A code bug: the same estimator is being evaluated twice. The notebook's second line passes `clf` instead of `clf2`. Different model families do not agree to 16 decimals by chance.

#### Question 6

Why does `np.sin` perform poorly on `Fare`?

**Answer**

Sine is periodic and non-monotonic. It maps distinct fares to the same value, destroying the order and magnitude information. The resulting feature is not a useful compression of the fare distribution.

#### Question 7

Why is the holdout accuracy higher than the CV mean?

**Answer**

They use different protocols. The holdout is one 80/20 split of 179 test rows; the CV mean averages 15 folds over the 891-row dataset. Different training sizes and validation compositions produce different scores. The CV mean is not required to equal the holdout.

#### Question 8

How would you make the cross-validation fold-honest?

**Answer**

Put the preprocessing and the model in a `Pipeline`, and pass the pipeline and raw `X` to `cross_val_score`. The pipeline fits the preprocessing inside each fold, so validation rows never influence the learned parameters.

---

### A22. Revision summary

- Skew is asymmetry; positive skew means a long right tail.
- Histogram, Q-Q plot, and skewness are complementary checks.
- `log1p(x)=log(1+x)` compresses large values and is safe at zero.
- `FunctionTransformer(func)` stores a function; it does not compute at construction.
- `np.log1p` is a function object; `np.log1p(X)` is an executed array.
- `ColumnTransformer` routes a transform to selected columns; output order is transformed block first, passthrough last.
- Fit on train, transform test. `fit_transform(X_test)` is a preserved bug.
- `log1p` is stateless, so the bug is numerically harmless here; it would leak for a stateful transformer.
- Full-log holdout: LR `0.7374`, DT `0.7095`; raw: LR `0.7207`, DT `0.6816`.
- Full-log CV15: LR `0.6770`, DT `0.6621`.
- Selective-log holdout: LR `0.7151`, DT `0.7039`.
- Selective CV: both printed `0.6679` because of the `clf`-twice bug; corrected DT is not saved.
- `np.sin` scores `0.6185`; it destroys the monotonic fare signal.
- The `my_notebook` helper cell is malformed; the working helper and saved outputs are in the preview notebook.
- `distplot` is deprecated; use `histplot` or `displot`.
- `DecisionTreeClassifier` is stochastic by default; set `random_state` for stable comparisons.

---

## Part B — Advanced Reading and Reference

### B1. `FunctionTransformer` internals

`FunctionTransformer` is a lightweight adapter. Its core behavior is:

```python
class FunctionTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.func(X)
```

The real implementation adds validation, inverse transforms, sparse handling, and feature-name logic, but the idea is this simple.

- `fit` is a no-op for stateless functions.
- `transform` calls `func` on the entire input.
- `inverse_transform` calls `inverse_func` if provided.
- `validate=True` checks that the input is 2-D and finite before applying the function.

`FunctionTransformer` does not vectorize the function itself. If `func` is a Python function that operates on scalars, it may be slow. NumPy universal functions such as `np.log1p` operate element-wise on arrays efficiently.

#### `feature_names_out`

By default, `FunctionTransformer` can produce output feature names in a one-to-one fashion. This is useful when the input has named columns and the transform preserves the number of columns.

#### `check_inverse`

If `inverse_func` is provided, `check_inverse=True` can verify that applying `func` and then `inverse_func` returns the original values within a tolerance. This is useful for debugging custom transforms.

---

### B2. `ColumnTransformer` output ordering and indices

For a `ColumnTransformer` with transformers $T_1, \dots, T_m$ and a remainder $R$:

$$
\text{output} = [T_1(X_{C_1}), T_2(X_{C_2}), \dots, T_m(X_{C_m}), R(X_{C_{\text{rem}}})].
$$

The order is the declaration order, not the original DataFrame order.

For the selective transform:

```python
ColumnTransformer(
    [("log", FunctionTransformer(np.log1p), ["Fare"])],
    remainder="passthrough",
)
```

The output is:

```text
[log1p(Fare), Age]
```

The original order was `Age, Fare`. The transform changes the order.

This is why downstream code must use positions or `get_feature_names_out()`:

```python
trf2.get_feature_names_out()
# ['log__Fare', 'remainder__Age']
```

If you reconstruct a DataFrame, use these names rather than assuming the original order.

---

### B3. Mathematics of skewness and the log transform

#### Skewness

The third standardized moment is:

$$
\gamma_1 = \frac{\mu_3}{\sigma^3},
$$

where $\mu_3$ is the third central moment and $\sigma$ is the standard deviation. Sample estimators differ in bias correction; pandas uses an unbiased estimator by default. The sign is what matters for interpretation.

#### Log compression

For positive $x$:

$$
\frac{d}{dx}\log(1+x) = \frac{1}{1+x}.
$$

The derivative is large near zero and small for large $x$. This means the transform stretches small values and compresses large values. That is exactly the behavior needed for a right tail.

#### Why `log1p` is a shifted log

$$
\log(1+x) = \log(x) + \log\left(1+\frac{1}{x}\right).
$$

For large $x$, the second term is small, so `log1p(x)` is close to `log(x)`. For $x=0$, it is exactly 0.

#### The inverse

$$
y = \log(1+x) \iff x = e^y - 1.
$$

`np.expm1(y)` computes $e^y-1$ accurately for small $y$.

---

### B4. Box-Cox and Yeo-Johnson (outlook)

The log transform is one member of a family of power transforms. Box-Cox is defined as:

$$
x^{(\lambda)} =
\begin{cases}
\dfrac{x^\lambda - 1}{\lambda}, & \lambda \neq 0, \\
\log(x), & \lambda = 0.
\end{cases}
$$

Box-Cox requires $x > 0$. It estimates $\lambda$ from the training data by maximum likelihood, so it is stateful.

Yeo-Johnson extends the family to any real value:

$$
x^{(\lambda)} =
\begin{cases}
\dfrac{(x+1)^\lambda - 1}{\lambda}, & x \ge 0,\ \lambda \neq 0, \\
\log(x+1), & x \ge 0,\ \lambda = 0, \\
-\dfrac{(1-x)^{2-\lambda} - 1}{2-\lambda}, & x < 0,\ \lambda \neq 2, \\
-\log(1-x), & x < 0,\ \lambda = 2.
\end{cases}
$$

`PowerTransformer` in scikit-learn implements both. Day 31 studies these in detail.

The key difference from `FunctionTransformer`:

- `FunctionTransformer` applies a fixed function and learns nothing.
- `PowerTransformer` learns $\lambda$ from data, so it must be fitted only on training data and kept inside a pipeline.

---

### B5. Cross-validation mechanics

`cross_val_score(estimator, X, y, cv=15, scoring="accuracy")`:

1. Splits the data into 15 folds.
2. For each fold, trains on the other 14 folds.
3. Scores the held-out fold.
4. Returns an array of 15 scores.

The mean is:

$$
\bar{a} = \frac{1}{15}\sum_{k=1}^{15} a_k.
$$

The standard deviation of the fold scores estimates the variability across folds. A small mean with a large standard deviation is less trustworthy than a similar mean with a small standard deviation.

For classification, use stratified folds when class proportions matter. `cross_val_score` with a classifier uses `StratifiedKFold` by default for binary/multiclass targets.

#### Why 15 folds?

More folds mean each training set is larger (closer to the full data), but the validation sets are smaller and the scores are more variable. The computational cost grows linearly with the number of folds. Five or ten folds are common; the notebook uses 15, which is slower but gives more folds to average.

---

### B6. Holdout, CV, and nested CV

| Protocol | What is fitted | What is scored | Bias |
|---|---|---|---|
| Holdout | Once on training split | One test split | Can be optimistic or pessimistic due to split luck |
| CV | Once per fold | Mean over folds | Better estimate, still variable |
| Nested CV | Inner CV for tuning, outer CV for evaluation | Outer fold scores | Less biased for the tuning procedure |

If you use CV to choose a transform and then report the same CV score as the final estimate, the estimate is optimistic because the choice used the validation folds. Nested CV separates selection from evaluation.

For this notebook, the transform choice is exploratory; the reported scores are teaching evidence, not a final model-selection claim.

---

### B7. Decision-tree randomness

`DecisionTreeClassifier` uses randomization when multiple splits are equally good. With `random_state=None`, the chosen split can differ between runs, changing the tree and its accuracy.

For reproducible comparisons:

```python
DecisionTreeClassifier(random_state=42)
```

The notebook does not set `random_state`. The saved tree scores are one realization. The reproduced scores differ slightly. This is expected, not a bug.

Logistic regression with the default solver is deterministic for this data, so its scores reproduce exactly.

---

### B8. `distplot`, `histplot`, and `displot`

Seaborn's `distplot` is deprecated and removed in version 0.14. The modern API separates axes-level and figure-level functions:

- `histplot`: axes-level histogram, can overlay a KDE.
- `displot`: figure-level distribution plot, can create facets.

The notebook's corrected loop uses `histplot`. The original `distplot` calls are preserved for course fidelity and produce a warning in the executed preview.

---

### B9. Notebook coverage matrix

| Notebook area | Cells | What it teaches | Saved output |
|---|---:|---|---|
| Setup and imports | 0–6 | Libraries and purpose | None |
| Load, slice, impute, split | 8–12 | Data shape, split, leakage caveat | None in `my_notebook` |
| Normality checks | 14–22 | Histogram, Q-Q, `distplot` warning, corrected loop | Preview outputs |
| Baselines | 23–28 | LR and DT raw scores | Preview: `0.7207`, `0.6816` |
| `FunctionTransformer` | 29–36 | Wrapping, statelessness, test-side refit bug | Preview |
| Full log results | 37–42 | LR `0.7374`, DT `0.7095` | Preview |
| Full-log CV | 43–47 | CV15 `0.6770`, `0.6621`; pipeline correction | Preview |
| Q-Q before/after | 48–53 | Age and Fare visual comparison | Preview figures |
| Selective routing | 54–57 | `ColumnTransformer` on `Fare`; LR `0.7151`, DT `0.7039` | Preview |
| Selective CV bug | 58–62 | `clf` twice; identical `0.6679` | Preview |
| Repeated Q-Q | 63–68 | Duplicate visual checks | Preview |
| Helper and sine | 69–75 | Function objects; `np.sin` `0.6185`; malformed cell | Preview helper output |
| Takeaways | 76 | Revision | None |

`my_notebook.ipynb` has no stored outputs. The executed preview contains the saved outputs quoted above.

---

### B10. Source fidelity and caveats

- `my_notebook.ipynb` has 77 cells and no stored outputs; its comments quote expected results.
- `DAY30_Function-Transformer-preview-OUT-OF-SCOPE.ipynb` has 29 cells and executed outputs. It is the source of the saved scores.
- The dataset is `datasets/train.csv` (891 rows, 12 columns); the notebook uses `Survived`, `Age`, and `Fare`.
- `Age` is imputed with the mean over all 891 rows before the split. This is a preserved simplification and a mild leakage pattern.
- The split is `test_size=0.2`, `random_state=34`: `(712, 2)` train, `(179, 2)` test.
- Saved raw holdout: LR `0.7206703910614525`, DT `0.6815642458100558`.
- Saved full-log holdout: LR `0.7374301675977654`, DT `0.7094972067039106`.
- Saved full-log CV15: LR `0.6770056497175143`, DT `0.6621092278719396`.
- Saved selective holdout: LR `0.7150837988826816`, DT `0.7039106145251397`.
- Saved selective CV prints both `0.6679472693032015` because of the `clf`-twice bug.
- Saved sine CV: `0.6184557438794727`.
- The notebook uses `trf.fit_transform(X_test)`; this is a preserved bug. It is harmless for stateless `log1p`.
- The notebook transforms all of `X` before CV; this is a preserved leakage pattern. It is harmless for stateless `log1p`.
- The notebook uses `distplot`, which is deprecated; the executed preview shows the warning.
- `my_notebook` cell 70 is malformed and does not define `apply_transform`; cell 74 would raise `NameError`. The working helper is in the preview.
- Reproduced supplementary values: raw skew Age `0.4363`, Fare `5.4727`; log skew Age `-2.1544`, Fare `0.4401`; corrected selective CV DT `0.6599246704331451`; pipeline CV LR `0.6679472693032015`.
- Reproduced tree scores differ from saved tree scores because `DecisionTreeClassifier` has `random_state=None`.
- No graph data is stored in `my_notebook`; the preview contains the Q-Q figures.
- No source notebook is modified; bugs remain visible for learning.

---

### B11. References

- scikit-learn documentation: `FunctionTransformer`, `ColumnTransformer`, `Pipeline`, `LogisticRegression`, `DecisionTreeClassifier`, `cross_val_score`.
- SciPy documentation: `scipy.stats.probplot`, skewness.
- pandas documentation: `Series.skew`, `fillna`, `iloc`.
- Seaborn documentation: `histplot`, `displot`, `distplot` deprecation.
- CampusX Day 30: Function Transformer, Log Transform, Reciprocal, Square Root.
- Day 19–20 notes: distribution checks and skew.
- Day 24–25 notes: scaling.
- Day 28 notes: `ColumnTransformer`.
- Day 29 notes: pipelines and cross-validation.
- Day 31 notes: `PowerTransformer`, Box-Cox, Yeo-Johnson.

---

### B12. Final connection

`FunctionTransformer` is the simplest bridge between an ordinary mathematical function and a scikit-learn workflow. It lets you test a transformation without writing a custom estimator. The log transform is the first distribution-shaping transform in the course; it compresses right tails and can improve scale-sensitive models, but it is not universally beneficial.

The notebook's real lessons are:

- A transform is a modeling decision, not a ritual.
- Check the distribution before and after.
- Keep the transform inside the preprocessing chain.
- Fit on training data, transform test data.
- Compare with cross-validation, not one lucky split.
- A pipeline makes the workflow honest; a malformed cell or a copy-paste bug can still mislead you.

The next day turns the fixed log into a learnable power transform: instead of choosing the function by hand, `PowerTransformer` estimates the exponent from the training data.
