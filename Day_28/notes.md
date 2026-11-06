# Day 28 — ColumnTransformer

> **Course topic:** CampusX Day 28, *Column Transformer in Machine Learning*.
> **Personal notebook:** `my_notebook.ipynb` (25 cells).
> **Dataset:** `datasets/covid_toy.csv` (100 rows, 6 columns).
> **Reference:** `reference_notebooks/day28.ipynb` (do not edit).
>
> **Part A** is the main learning chapter and is sufficient on its own. **Part B** preserves advanced details, internals, and reference material.

---

## Part A — Main Learning Chapter

### A1. Learning objectives

By the end of this chapter you should be able to:

1. Explain why different columns need different preprocessing.
2. Build a manual preprocessing workflow and concatenate its blocks correctly.
3. Write a `ColumnTransformer` using `(name, transformer, columns)` triples.
4. Explain `fit`, `transform`, and `fit_transform` for a composite transformer.
5. Predict the output shape from the column selectors and transformer outputs.
6. Explain `remainder="passthrough"` and the default `remainder="drop"`.
7. Diagnose the Day 28 notebook's column-selection bug, test-side refitting bug, and non-reproducible split.
8. Connect `ColumnTransformer` to the pipeline in Day 29.

---

### A2. The problem: one table, several kinds of columns

The `covid_toy` dataset has six columns:

| Column | Meaning | Type | Missing values |
|---|---|---|---|
| `age` | age in years | numeric | 0 |
| `gender` | Male/Female | nominal | 0 |
| `fever` | body temperature | numeric | 10 |
| `cough` | Mild/Strong | ordinal | 0 |
| `city` | Kolkata/Delhi/Mumbai/Bangalore | nominal | 0 |
| `has_covid` | target | binary | 0 |

The feature matrix has five columns, and each one needs different treatment:

- `fever` has missing values, so it needs imputation.
- `cough` has two ordered levels, so it can be ordinal-encoded.
- `gender` and `city` are nominal, so they should be one-hot encoded.
- `age` is numeric and has no missing values, so it can pass through unchanged (or be scaled).

A manual workflow does each step separately:

```python
X_train_fever = si.fit_transform(X_train[["fever"]])
X_test_fever = si.transform(X_test[["fever"]])

X_train_cough = oe.fit_transform(X_train[["cough"]])
X_test_cough = oe.transform(X_test[["cough"]])

X_train_gender_city = ohe.fit_transform(X_train[["gender", "city"]])
X_test_gender_city = ohe.transform(X_test[["gender", "city"]])

X_train_age = X_train.drop(columns=["gender", "fever", "cough", "city"]).values
X_test_age = X_test.drop(columns=["gender", "fever", "cough", "city"]).values

X_train_transformed = np.concatenate(
    (X_train_age, X_train_gender_city, X_train_cough, X_train_fever),
    axis=1,
)
```

This works, but it has several failure modes:

- The concatenation order is easy to get wrong.
- The train and test blocks must be assembled in the same order.
- A new column can be forgotten.
- The preprocessing is not reusable as a single object.
- The same logic must be repeated at inference time.

`ColumnTransformer` turns the plan into one object.

---

### A3. The `ColumnTransformer` syntax

A `ColumnTransformer` receives a list of triples:

```python
ColumnTransformer(
    transformers=[
        (name, transformer, columns),
        ...
    ],
    remainder="drop",
)
```

Each triple has three parts:

| Part | Meaning | Example |
|---|---|---|
| `name` | A unique string used to identify the step | `"impute_fever"` |
| `transformer` | An object with `fit` and `transform` | `SimpleImputer()` |
| `columns` | Column names or integer positions | `["fever"]` |

The output is a NumPy array (or a sparse matrix) formed by applying each transformer to its selected columns and concatenating the results in the order the triples are declared.

```text
Raw DataFrame
   |
   +-- transformer 1 on columns A --> block 1
   +-- transformer 2 on columns B --> block 2
   +-- transformer 3 on columns C --> block 3
   +-- remainder ------------------> block 4
   |
   v
concatenate(block 1, block 2, block 3, block 4)
   |
   v
one output array
```

#### The transformer must be a transformer

A `ColumnTransformer` is for preprocessing steps that implement `fit`/`transform`. A model implements `fit`/`predict`, not `fit`/`transform`. Putting a model inside a `ColumnTransformer` is the wrong tool; the model belongs in a `Pipeline` after the preprocessing.

---

### A4. The Day 28 plan

The notebook's manual path uses:

1. `SimpleImputer()` on `fever`.
2. `OrdinalEncoder(categories=[["Mild", "Strong"]])` on `cough`.
3. `OneHotEncoder(drop="first", sparse_output=False, dtype=np.int32)` on `gender` and `city`.
4. `age` passed through.
5. `np.concatenate` in the order `(age, gender_city, cough, fever)`.

The automated path declares:

```python
transformer = ColumnTransformer(
    transformers=[
        ("tnf1", SimpleImputer(), ["fever"]),
        ("tnf2", OrdinalEncoder(categories=[["Mild", "Strong"]]), ["cough"]),
        ("tnf3", OneHotEncoder(sparse_output=False, drop="first"), ["age", "gender"]),
    ],
    remainder="passthrough",
)
```

Notice the difference: the automated path one-hot encodes **`age` and `gender`**, while the manual path one-hot encodes **`gender` and `city`**. This is a real column-selection bug in the notebook, not a deliberate alternative feature set. The manual path's `city` is missing from the automated path's explicit transformers, so `city` is passed through by `remainder="passthrough"` as a raw string. The automated path also treats the numeric `age` as a categorical variable, creating one column per distinct age.

This bug has a large effect on the output shape, as the next section shows.

---

### A5. Output width: predicting the shape

For the manual path:

- `age`: 1 column.
- `gender` (2 levels, drop first): 1 column.
- `city` (4 levels, drop first): 3 columns.
- `gender` + `city` together: 4 columns.
- `cough` (2 levels, ordinal): 1 column.
- `fever` (imputed): 1 column.

Total:

$$
1+4+1+1=7.
$$

The saved manual output is:

```text
(80, 7)
```

For the automated path with the notebook's column selection:

- `fever` imputed: 1 column.
- `cough` ordinal: 1 column.
- `age` one-hot encoded: if the training split has $u$ distinct ages, `drop="first"` gives $u-1$ columns.
- `gender` one-hot encoded: 2 levels, drop first -> 1 column.
- `city` passthrough: 1 raw string column.
- `remainder="passthrough"` also passes any other unlisted columns, but there are none besides `city`.

Total:

$$
1+1+(u-1)+1+1 = u+3.
$$

The notebook's saved test-side call produces:

```text
(20, 20)
```

That is consistent with a test split containing 17 distinct ages:

$$
17+3=20.
$$

A reproduced run on the local dataset (without the notebook's missing `random_state`) gives 49 distinct training ages and a training output of:

```text
(80, 52)
```

This reproduced value is not a saved notebook output; it is a supplementary calculation showing how the width explodes when `age` is treated as categorical. The exact number depends on the split because `train_test_split` has no `random_state` in the notebook.

The lesson: **read the column selectors before interpreting a shape.** A width of 52 is not a property of `ColumnTransformer`; it is a consequence of one-hot encoding a high-cardinality numeric column.

---

### A6. `remainder`: passthrough versus drop

The `remainder` argument controls what happens to columns that are not explicitly selected by any transformer.

| Value | Behaviour |
|---|---|
| `"drop"` | Drop unlisted columns. This is the default. |
| `"passthrough"` | Keep unlisted columns unchanged and append them after the transformed blocks. |
| An estimator | Apply that estimator to the unlisted columns. |

The default is `"drop"`, which can silently remove important columns. In the notebook, `remainder="passthrough"` keeps `city` and any other unlisted column. Because `city` is a string, the output array contains strings and is not model-ready.

A safer explicit plan is:

```python
ColumnTransformer(
    transformers=[
        ("impute_fever", SimpleImputer(strategy="mean"), ["fever"]),
        ("ord_cough", OrdinalEncoder(categories=[["Mild", "Strong"]]), ["cough"]),
        ("ohe_gender_city", OneHotEncoder(
            drop="first", sparse_output=False, handle_unknown="ignore"
        ), ["gender", "city"]),
    ],
    remainder="passthrough",   # keeps age
)
```

This yields the same 7 columns as the manual path: age 1 + gender/city 4 + cough 1 + fever 1.

---

### A7. `fit`, `transform`, and `fit_transform`

A `ColumnTransformer` is a composite estimator. Its `fit` method fits each sub-transformer on its selected columns:

```python
ct.fit(X_train)
```

Internally, for each triple:

1. Select the columns.
2. Call `fit` on the transformer with those columns.
3. Store the fitted transformer in `transformers_`.

`transform` applies each fitted transformer and concatenates the outputs:

```python
X_train_transformed = ct.transform(X_train)
X_test_transformed = ct.transform(X_test)
```

`fit_transform` is the shortcut for fitting and transforming the same data:

```python
X_train_transformed = ct.fit_transform(X_train)
```

#### The test-side rule

The test set must be transformed with the training-fitted sub-transformers:

```python
X_test_transformed = ct.transform(X_test)
```

The notebook calls:

```python
transformer.fit_transform(X_train).shape
transformer.fit_transform(X_test).shape
```

The second call refits every sub-transformer on the test set. This is a bug:

- The imputer learns the test mean.
- The ordinal encoder checks the test categories.
- The one-hot encoder learns the test categories.
- The output columns may differ from the training output.
- Test information leaks into the preprocessing.

The correct test call is `transformer.transform(X_test)`.

For this notebook, the effect is especially visible because `age` is one-hot encoded. Fitting on the test set learns only the test ages, so the test output has a different width from the training output. A model trained on the training block could not consume the test block.

---

### A8. What the saved outputs show

The notebook's saved outputs are sparse because only the tail expressions ran. The important saved results are:

| Operation | Saved output |
|---|---|
| `df.head()` | five rows with `age`, `gender`, `fever`, `cough`, `city`, `has_covid` |
| `df.isnull().sum()` | `fever` has 10 missing values |
| `X_train_fever` | imputed fever array; some values are `100.875` |
| `X_train_gender_city` | `(80, 4)` integer array |
| `X_train_transformed.shape` | `(80, 7)` |
| `transformer.fit_transform(X_test).shape` | `(20, 20)` |

The saved fever mean is `100.875`, which comes from the notebook's split. A reproduced run with a different split gives a different mean, such as `100.67123288`. This is expected because the split has no `random_state`. The saved value is not wrong; it is simply tied to one random split.

The saved `X_train_gender_city` has four columns. For `gender` (2 levels) and `city` (4 levels) with `drop="first"`, the count is:

$$
(2-1)+(4-1)=1+3=4.
$$

This matches.

The saved manual shape `(80, 7)` matches the manual plan. The saved automated test shape `(20, 20)` is the result of the buggy test-side refit and the `age` one-hot selection.

---

### A9. The data flow of the corrected plan

A corrected `ColumnTransformer` for this dataset is:

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

ct = ColumnTransformer(
    transformers=[
        ("impute_fever", SimpleImputer(strategy="mean"), ["fever"]),
        ("ord_cough", OrdinalEncoder(categories=[["Mild", "Strong"]]), ["cough"]),
        ("ohe_gender_city", OneHotEncoder(
            drop="first",
            sparse_output=False,
            handle_unknown="ignore",
            dtype=np.int32,
        ), ["gender", "city"]),
    ],
    remainder="passthrough",
)
```

The data flow is:

```text
X_train (80 x 5)
   |
   +-- fever ----------> SimpleImputer ----------> 1 column
   +-- cough ----------> OrdinalEncoder ---------> 1 column
   +-- gender, city ---> OneHotEncoder ----------> 4 columns
   +-- age (remainder) -> passthrough -----------> 1 column
   |
   v
concatenate in declaration order
   |
   v
(80, 7)
```

Note the output order: the transformed blocks come first, in declaration order, and the passthrough block comes last. The output is not in the original column order. The age column is last, even though it was first in the DataFrame.

If you need named columns, use:

```python
ct.get_feature_names_out()
```

The names reflect the transformation order, not the original order.

---

### A10. Column selection: names versus indices

`ColumnTransformer` accepts column selectors as:

- A list of column names: `["fever"]`.
- A list of integer positions: `[2]`.
- A boolean mask.
- A callable that receives the DataFrame and returns a selection.
- A `slice` object.

Names are safer when the input is a DataFrame with meaningful column names. Indices are necessary when the input is a NumPy array, because arrays have no column names.

After a transformer runs, its output is usually a NumPy array. If you stack another `ColumnTransformer` after it, the later transformer sees positions, not names. This is why Day 29 uses indices such as `[2]` and `[6]` in its pipeline.

The notebook's manual path uses names because it operates on DataFrames. The automated path also uses names. Day 29 switches to indices after the first transformation.

---

### A11. Why column order matters

A model learns a coefficient or split for each column position. If the training matrix has columns in one order and the test matrix has the same columns in a different order, the model applies the wrong parameters to the wrong features.

`ColumnTransformer` guarantees a consistent order for a fixed declaration:

1. Block 1 output.
2. Block 2 output.
3. ...
4. Remainder output.

As long as you use the same fitted `ColumnTransformer` for train and test, the order is consistent. The danger is manual concatenation, where you can accidentally swap blocks between train and test.

The notebook's manual path uses the order `(age, gender_city, cough, fever)`. If the test concatenation used a different order, the model would be trained on one layout and evaluated on another. The notebook uses the same order for both, which is correct.

---

### A12. Why `ColumnTransformer` is not a model

`ColumnTransformer` is a preprocessing object. It has no `predict` method. It transforms features; it does not learn a target relationship.

A complete model needs a pipeline:

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("preprocess", ct),
    ("classifier", LogisticRegression(max_iter=1000)),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

The pipeline calls `ct.fit_transform` on training data, then fits the classifier. At prediction time, it calls `ct.transform` on new data, then `classifier.predict`. This is the correct composition.

---

### A13. Common mistakes and debugging

#### Mistake 1: forgetting `remainder`

The default `remainder="drop"` silently removes columns that are not explicitly selected. If `age` is not in any transformer and `remainder` is left at the default, `age` disappears.

Fix: select every column explicitly, or set `remainder="passthrough"` deliberately.

#### Mistake 2: duplicate transformer names

Names must be unique. Duplicate names cause an error or make `named_transformers_` ambiguous.

#### Mistake 3: fitting on test data

```python
# Wrong
ct.fit_transform(X_test)
```

Fix: `ct.transform(X_test)`.

#### Mistake 4: treating a numeric column as categorical

The notebook one-hot encodes `age`, creating dozens of columns. This is usually wrong for a continuous numeric feature. Use a scaler or pass it through instead.

#### Mistake 5: leaving a raw string column in the output

`remainder="passthrough"` keeps `city` as a string if it is not explicitly encoded. The output array then has a mixed dtype and cannot be used by most models. Encode every categorical column or drop it.

#### Mistake 6: assuming the output keeps column names

The output is a NumPy array. Column names are lost unless you use `get_feature_names_out` and reconstruct a DataFrame. Downstream steps that need names will fail.

#### Mistake 7: using a non-reproducible split

The notebook's `train_test_split` has no `random_state`. The split, the imputer mean, and the output shapes can change between runs. Use a fixed `random_state` for reproducible study and comparison.

#### Mistake 8: comparing manual and automated outputs without checking the columns

The manual path uses `gender` and `city`; the automated path uses `age` and `gender`. The outputs are not equivalent. Always compare the selected columns, the transformer parameters, and the output names before comparing shapes.

---

### A14. Where this sits in the ML workflow

`ColumnTransformer` belongs after the train/test split and before the model:

```text
Raw data
   |
   v
Identify column types
   |
   v
Train/test split
   |
   v
ColumnTransformer.fit_transform(X_train)
   |
   v
ColumnTransformer.transform(X_test)
   |
   v
Model training and evaluation
```

Inside cross-validation, the whole `ColumnTransformer` must be fitted inside each fold. A `Pipeline` ensures this:

```python
pipe = Pipeline([
    ("preprocess", ct),
    ("model", LogisticRegression(max_iter=1000)),
])
cross_val_score(pipe, X, y, cv=5)
```

---

### A15. Interview preparation

#### Level 1 — Recall

**Q1. What does `ColumnTransformer` do?**

It applies different transformers to different column groups and concatenates their outputs into one array.

**Q2. What are the three parts of a transformer triple?**

`(name, transformer, columns)`: a unique name, a transformer object with `fit`/`transform`, and the columns it should process.

**Q3. What is the default value of `remainder`?**

`"drop"`. Unlisted columns are removed.

#### Level 2 — Explanation

**Q4. Why use `ColumnTransformer` instead of manual concatenation?**

It declares the preprocessing plan once, keeps train and test consistent, avoids manual block tracking, and can be embedded in a pipeline.

**Q5. Why is `fit_transform` used on training data but `transform` on test data?**

`fit_transform` learns the preprocessing parameters from training data and applies them. `transform` reuses those parameters on test data. Fitting on test would leak information and produce a different mapping.

**Q6. What happens to unlisted columns with `remainder="passthrough"`?**

They are appended unchanged after the transformed blocks. With `"drop"`, they are removed.

#### Level 3 — Application

**Q7. How would you impute `fever`, ordinal-encode `cough`, one-hot encode `gender` and `city`, and keep `age`?**

```python
ColumnTransformer([
    ("impute_fever", SimpleImputer(strategy="mean"), ["fever"]),
    ("ord_cough", OrdinalEncoder(categories=[["Mild", "Strong"]]), ["cough"]),
    ("ohe_gender_city", OneHotEncoder(drop="first", sparse_output=False), ["gender", "city"]),
], remainder="passthrough")
```

**Q8. How many columns does that produce for `covid_toy`?**

`fever` 1 + `cough` 1 + `gender`/`city` 4 + `age` 1 = 7.

**Q9. How would you use the transformer inside cross-validation?**

Wrap it in a `Pipeline` with the model and pass the pipeline to `cross_val_score`. The transformer is then fitted on each fold's training portion.

#### Level 4 — Analysis

**Q10. Why does one-hot encoding `age` create a very wide output?**

`age` has many distinct values. One-hot encoding creates one column per distinct value (minus one after dropping). For 49 distinct training ages, that is 48 columns just for `age`.

**Q11. Why is a raw string column in the output a problem?**

Most scikit-learn estimators require numeric input. A mixed-type array containing strings will fail validation. The string column must be encoded or dropped.

**Q12. Why is the output column order not the original DataFrame order?**

`ColumnTransformer` concatenates transformed blocks in declaration order, then appends the remainder. The output order is the plan's order, not the input order.

#### Level 5 — Design

**Q13. How would you handle a dataset with numeric, ordinal, nominal, and missing-value columns?**

Use one `ColumnTransformer` with separate branches: numeric imputation + scaling, ordinal imputation + ordinal encoding, nominal imputation + one-hot encoding, and an explicit remainder policy. Put it inside a pipeline.

**Q14. How would you preserve feature names for debugging?**

Use `get_feature_names_out()` and wrap the output in a DataFrame with those names. Keep a mapping from original columns to output blocks.

**Q15. How would you handle unseen categories in a production pipeline?**

Use `handle_unknown="ignore"` for one-hot encoding, define an explicit unknown category for ordinal encoding, and monitor the frequency of unseen categories.

#### Level 6 — Evaluation

**Q16. How would you verify that the manual and automated paths are equivalent?**

Compare the selected columns, transformer parameters, output shapes, output names, and a sample of values. Do not compare only shapes; different column selections can produce different shapes.

**Q17. Why is a non-reproducible split a problem for evaluating preprocessing?**

Different splits produce different imputer means, different category sets, and different output shapes. Comparisons between runs become unreliable. Fix `random_state` for reproducible experiments.

**Q18. What would you monitor in production?**

The distribution of input columns, missing-value rates, unseen categories, output width, and the proportion of values outside the training range.

#### Level 7 — Advanced

**Q19. How does `ColumnTransformer` handle sparse output?**

It can return a sparse matrix when the combined output is sparse enough, controlled by `sparse_threshold`. This saves memory for high-cardinality one-hot features. You can set `sparse_threshold=0` to always return sparse or `1.0` to always return dense.

**Q20. What is `verbose_feature_names_out`?**

It controls whether `get_feature_names_out` prefixes feature names with the transformer name. Prefixes help avoid collisions when different transformers produce the same column name.

**Q21. Can a `ColumnTransformer` contain another `ColumnTransformer`?**

Yes. A transformer in a triple can itself be a `ColumnTransformer`. This is useful for nested preprocessing plans, but it makes the output order and naming harder to track.

---

### A16. Exam-style questions with worked answers

#### Question 1

A dataset has columns `age` (numeric), `city` (4 levels), and `fever` (numeric, missing). A `ColumnTransformer` imputes `fever`, one-hot encodes `city` with `drop="first"`, and passes `age` through. How many output columns are there?

**Answer**

`fever` 1 + `city` $(4-1)=3$ + `age` 1 = 5 columns.

#### Question 2

The Day 28 notebook's automated path one-hot encodes `age` and `gender`, ordinal-encodes `cough`, imputes `fever`, and passes `city` through. If the training split has 49 distinct ages, how many output columns does it produce?

**Answer**

`fever` 1 + `cough` 1 + `age` $(49-1)=48$ + `gender` $(2-1)=1$ + `city` 1 = 52 columns.

#### Question 3

Explain why `transformer.fit_transform(X_test)` is wrong and what it does to the output.

**Answer**

It refits every sub-transformer on the test set. The imputer learns the test mean, the encoders learn the test categories, and the output width can differ from the training output. Test information leaks into preprocessing, and a model trained on the training block may not be able to consume the test block. The correct call is `transformer.transform(X_test)`.

#### Question 4

Why does the manual path produce `(80, 7)` while the automated path produces a different width?

**Answer**

The manual path one-hot encodes `gender` and `city`, keeps `age` as one numeric column, and produces 7 columns. The automated path one-hot encodes `age` and `gender`, leaves `city` as a passthrough string, and produces a width that depends on the number of distinct ages. The column selections are different, so the outputs are not equivalent.

#### Question 5

What is the effect of `remainder="drop"` on a column that is not selected by any transformer?

**Answer**

The column is removed from the output. This is the default, and it can silently discard useful features. Use `remainder="passthrough"` or select every column explicitly.

---

### A17. Revision summary

- `ColumnTransformer` applies different transformers to different columns.
- The syntax is a list of `(name, transformer, columns)` triples.
- `remainder="drop"` is the default; `"passthrough"` keeps unlisted columns.
- Fit on training data, transform test data.
- The output is a NumPy array in declaration order; names are lost.
- Predict the output width from the selected columns and each transformer's output.
- One-hot encoding a numeric column can explode the width.
- A raw string passthrough makes the output non-numeric.
- Wrap the transformer in a pipeline for cross-validation.
- The Day 28 notebook preserves column-selection and test-refit bugs.

---

## Part B — Advanced Reading and Reference

### B1. Internal working

A fitted `ColumnTransformer` stores:

- `transformers`: the original triples.
- `transformers_`: the fitted transformers, including the remainder transformer.
- `named_transformers_`: a lookup by name.
- `n_features_in_`: the number of input features.
- `feature_names_in_`: the input column names when available.
- `output_indices_`: a mapping from input columns to output columns.

During `fit`, each transformer is fitted on its selected columns. During `transform`, each transformer is applied and the results are concatenated. The remainder is handled by a `FunctionTransformer` that either drops or passes through the unselected columns.

The concatenation uses `scipy.sparse.hstack` when the outputs are sparse and the density is below `sparse_threshold`; otherwise it uses a dense array.

---

### B2. Column selectors

A selector can be:

- A string: `"fever"`.
- A list of strings: `["gender", "city"]`.
- A list of integers: `[2, 6]`.
- A slice: `slice(0, 10)`.
- A boolean mask: `[True, False, True, ...]`.
- A callable: `lambda df: df.columns.str.startswith("num_")`.

Selectors are resolved during `fit`. The same selector is used during `transform`. If the input columns change between fit and transform, the selector may fail or select the wrong columns.

Names are preferred for DataFrames because they are robust to column reordering. Indices are necessary for arrays, but they are fragile if the column order changes.

---

### B3. Output ordering

The output order is:

1. The output of the first transformer.
2. The output of the second transformer.
3. ...
4. The output of the remainder.

Within each transformer, the output order is determined by that transformer. For `OneHotEncoder`, the columns are ordered by feature and then by category.

This ordering is stable for a fixed fitted transformer. It is not necessarily the same as the original DataFrame column order.

---

### B4. `get_feature_names_out`

```python
ct.get_feature_names_out()
```

The method returns one name per output column. By default, names are prefixed with the transformer name to avoid collisions:

```text
impute_fever__fever
ord_cough__cough
ohe_gender_city__gender_Male
ohe_gender_city__city_Delhi
...
remainder__age
```

`verbose_feature_names_out=False` removes the prefixes, but duplicate names can occur. Prefixes are useful for debugging and feature importance.

Not every transformer implements `get_feature_names_out`. If one does not, the method may raise an error. In that case, build the names manually from the transformer's attributes.

---

### B5. `sparse_threshold`

`ColumnTransformer` can return a sparse matrix when the combined output is sparse. The `sparse_threshold` parameter controls the decision:

- `sparse_threshold=0.3` (default): return sparse if the density is below 0.3.
- `sparse_threshold=0`: always return sparse.
- `sparse_threshold=1.0`: always return dense.

Sparse output is useful for high-cardinality one-hot features. Dense output is easier to inspect and is required by some estimators.

---

### B6. `remainder` as an estimator

`remainder` can be a transformer, not just `"drop"` or `"passthrough"`. For example:

```python
ColumnTransformer(
    transformers=[...],
    remainder=StandardScaler(),
)
```

This applies `StandardScaler` to all unselected columns. It is useful when the remainder is a homogeneous numeric block.

The remainder transformer is fitted on the unselected columns during `fit` and applied during `transform`.

---

### B7. Nested `ColumnTransformer`

A `ColumnTransformer` can contain another `ColumnTransformer`:

```python
numeric = ColumnTransformer([
    ("impute", SimpleImputer(), ["age", "fever"]),
    ("scale", StandardScaler(), ["age", "fever"]),
])

full = ColumnTransformer([
    ("numeric", numeric, ["age", "fever"]),
    ("categorical", OneHotEncoder(), ["gender", "city"]),
], remainder="drop")
```

The inner transformer receives its selected columns and returns a block. The outer transformer concatenates it with the other blocks. This is useful for organizing complex plans, but it makes feature names and output order harder to track.

---

### B8. `ColumnTransformer` versus `Pipeline`

| Object | Purpose | Methods |
|---|---|---|
| `ColumnTransformer` | Apply different transformers to different columns | `fit`, `transform`, `fit_transform` |
| `Pipeline` | Chain steps sequentially | `fit`, `transform`, `predict`, `score` |

A `Pipeline` can contain a `ColumnTransformer` as a step. A `ColumnTransformer` cannot contain a model because a model does not implement `transform` in the required sense.

The typical structure is:

```python
Pipeline([
    ("preprocess", ColumnTransformer([...])),
    ("model", LogisticRegression()),
])
```

---

### B9. Leakage in cross-validation

The wrong pattern:

```python
X_transformed = ct.fit_transform(X)
scores = cross_val_score(model, X_transformed, y, cv=5)
```

The transformer has seen all rows, including the validation folds. The correct pattern:

```python
pipe = Pipeline([
    ("preprocess", ct),
    ("model", model),
])
scores = cross_val_score(pipe, X, y, cv=5)
```

The pipeline fits a fresh `ColumnTransformer` inside each fold. This is the only honest way to evaluate a preprocessing pipeline.

---

### B10. Advanced interview questions

**Q1. How does `ColumnTransformer` handle a transformer that returns a sparse matrix and another that returns a dense array?**

It combines them according to `sparse_threshold`. If the overall density is below the threshold, it returns a sparse matrix; otherwise it converts to dense.

**Q2. What happens if two transformers select the same column?**

The column is processed twice and appears twice in the output. `ColumnTransformer` does not deduplicate selections. This can be useful for creating multiple representations, but it can also cause unintended duplication.

**Q3. What happens if a transformer name is duplicated?**

Scikit-learn raises an error because names must be unique. The name is used as a key in `named_transformers_`.

**Q4. Can `ColumnTransformer` select columns by dtype?**

Yes, with a callable selector or `make_column_selector(dtype_include=...)`. This is useful for automatically selecting numeric or categorical columns.

**Q5. How would you debug a shape mismatch after `ColumnTransformer`?**

Check the selected columns, each transformer's output shape, the remainder policy, and the fitted `categories_` or `n_features_in_`. Use `get_feature_names_out` to see the output layout.

**Q6. Why might `ColumnTransformer` produce different output widths for train and test?**

If it is refitted on test, or if a transformer's output depends on the data (for example, one-hot encoding a high-cardinality column), the width can differ. With a fixed fitted transformer and `handle_unknown="ignore"`, the width is stable.

**Q7. What is the difference between `remainder="passthrough"` and adding the remainder columns as a separate transformer?**

`"passthrough"` appends the unselected columns unchanged. A separate transformer applies a transformation to them. The output order is the same: explicit blocks first, remainder last.

**Q8. How does `ColumnTransformer` interact with pandas output?**

Scikit-learn can be configured to output pandas DataFrames with `set_output(transform="pandas")`. This preserves column names when the transformers support it. It is useful for inspection but can add overhead.

---

### B11. Notebook coverage matrix

| Notebook area | Cells | What it teaches | Status |
|---|---:|---|---|
| Imports and theme | 2 | Setup | No output |
| Load data | 3 | Dataset shape and columns | Saved head |
| Missing values | 4 | Fever has 10 missing | Saved counts |
| Train/test split | 5 | 80/20 split, no `random_state` | No output |
| Manual imputation | 7–8 | SimpleImputer on fever | Saved array |
| Manual ordinal encoding | 9–10 | Cough encoding | No output |
| Manual one-hot encoding | 11–14 | Gender and city | Saved array |
| Manual age passthrough | 16 | Select age | No output |
| Manual concatenation | 17 | Shape (80, 7) | Saved shape |
| Automated path | 19–21 | ColumnTransformer triples | No output |
| Automated fit/transform | 22 | Test-side refit | Saved (20, 20) |
| Experiment | 23 | Fix leakage and compare | Markdown |
| Key takeaways | 24 | Revision and bugs | Markdown |

No model is trained. The notebook ends after preprocessing.

---

### B12. Source fidelity and caveats

- Saved outputs come from the user's environment. They are reported as saved results, not independently reproduced.
- The dataset is `covid_toy.csv` with 100 rows and 6 columns; `fever` has 10 missing values.
- The saved fever imputation value `100.875` is from the notebook's random split. A reproduced run without `random_state` gives a different value.
- The saved manual output shape is `(80, 7)`, which matches the manual column plan.
- The saved automated test shape is `(20, 20)`, which comes from `fit_transform(X_test)` and the `age`/`gender` one-hot selection.
- A reproduced training shape for the buggy automated plan is `(80, 52)` with 49 distinct training ages. This is a supplementary calculation, not a saved notebook output.
- The notebook has no `random_state` in `train_test_split`, so shapes and imputation values can change between runs.
- The notebook one-hot encodes `age`, a numeric column, and passes `city` as a raw string. These are preserved bugs.
- No model, accuracy, or cross-validation result is present.

---

### B13. References

- scikit-learn documentation: `ColumnTransformer`, `make_column_selector`, `Pipeline`, `SimpleImputer`, `OrdinalEncoder`, `OneHotEncoder`.
- CampusX Day 28: Column Transformer in Machine Learning.
- Day 26 notes: ordinal and label encoding.
- Day 27 notes: one-hot encoding.
- Day 29 notes: pipelines and cross-validation.

---

### B14. Final connection

`ColumnTransformer` is the object that turns a table of heterogeneous columns into one consistent numeric matrix. Its power comes from declaring the plan once; its danger comes from silent column drops, wrong selectors, and test-side refitting. The next day wraps this object in a `Pipeline`, adds scaling and feature selection, and shows how to validate and export the whole chain.
