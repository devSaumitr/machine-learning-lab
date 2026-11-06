# Day 27 — One-Hot Encoding

> **Course topic:** CampusX Day 27, *One Hot Encoding — Handling Categorical Data*.
> **Personal notebook:** `my_notebook.ipynb` (30 cells).
> **Dataset:** `datasets/cars.csv` (8,128 rows).
> **Reference:** `reference_notebooks/day27.ipynb` (do not edit).
>
> **Part A** is the main learning chapter and is sufficient on its own. **Part B** preserves advanced details, edge cases, and reference material.

---

## Part A — Main Learning Chapter

### A1. Learning objectives

By the end of this chapter you should be able to:

1. Explain why unordered categorical variables must not be integer-encoded as if they had a rank.
2. Build one-hot columns with `pd.get_dummies` and `OneHotEncoder`.
3. Explain the dummy-variable trap and why $k-1$ columns are often used.
4. Fit an encoder on training data and transform test data without leakage.
5. Handle high-cardinality features by grouping rare categories.
6. Read output shapes and verify that every concatenation step is correct.
7. Identify the preserved bugs in the Day 27 notebook and explain the fixes.

---

### A2. The problem: a number can invent an order

Suppose a car dataset has a `fuel` column:

```text
Petrol, Diesel, CNG, LPG
```

If we map the levels to integers:

```text
CNG    -> 0
Diesel -> 1
LPG    -> 2
Petrol -> 3
```

a linear model sees a single feature whose values increase from CNG to Petrol. It may learn that Petrol is "three units more" than CNG, or that the effect of moving from CNG to Diesel is the same as moving from LPG to Petrol. None of those statements is meaningful. Fuel types have no natural order.

A distance-based model has the same problem. The distance between CNG and Petrol becomes larger than the distance between CNG and Diesel simply because of the arbitrary code assignment.

The same issue applies to:

- `owner`: First Owner, Second Owner, Third Owner, Fourth & Above Owner, Test Drive Car.
- `brand`: Maruti, Hyundai, Honda, ...
- `city`: Delhi, Mumbai, Kolkata, ...
- `colour`: red, green, blue.

These are **nominal** variables: categories differ in kind, not in amount.

One-hot encoding avoids inventing an order by giving each level its own binary column.

---

### A3. The one-hot representation

For a feature with levels $\{c_1,\dots,c_k\}$, one-hot encoding creates $k$ binary indicators:

$$
x_j' = \mathbf{1}[x = c_j],
\qquad j=1,\dots,k.
$$

For `fuel` with four levels, a row with `fuel = Diesel` becomes:

| fuel_CNG | fuel_Diesel | fuel_LPG | fuel_Petrol |
|---:|---:|---:|---:|
| 0 | 1 | 0 | 0 |

A row with `fuel = Petrol` becomes:

| fuel_CNG | fuel_Diesel | fuel_LPG | fuel_Petrol |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 1 |

Exactly one column is 1 for each row. The representation makes no claim about order or distance between fuel types.

#### The cost

One-hot encoding increases the number of columns. A feature with 1,000 levels becomes 1,000 columns (or 999 after dropping one). For high-cardinality features, this can dominate memory and make the design matrix sparse and wide.

---

### A4. The dummy-variable trap

Suppose a model has an intercept and all $k$ one-hot columns for a feature. Then for every row:

$$
\sum_{j=1}^{k} x_j' = 1.
$$

The sum of the one-hot columns equals the intercept column. Therefore the columns are linearly dependent: one column is a perfect linear combination of the others. This is **perfect multicollinearity**, often called the **dummy-variable trap**.

For ordinary least squares, the design matrix becomes rank-deficient, so the coefficients are not uniquely identifiable. The model can still make predictions, but the individual coefficients are unstable and their interpretation is ambiguous.

The standard fix is to drop one level, keeping $k-1$ columns. The dropped level becomes the **reference** or **baseline** category. For `fuel` with CNG as the baseline:

| fuel_Diesel | fuel_LPG | fuel_Petrol |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |

An all-zero row now represents CNG. The coefficient of `fuel_Diesel` is interpreted relative to CNG, holding other features fixed.

#### When $k-1$ is not required

Trees and tree ensembles do not use linear coefficients in the same way. Full one-hot columns are usually harmless for trees, though they can make split search slower. Regularized linear models can handle collinearity numerically, but the coefficients are still not uniquely interpretable. For clean statistical interpretation, $k-1$ is the safer default.

#### `drop='if_binary'`

Scikit-learn also supports `drop='if_binary'`, which drops one column only for binary features and keeps all columns for features with more than two levels. This is useful when you want a single coefficient for a binary indicator but full flexibility for multi-level features.

---

### A5. The dataset and the four notebook blocks

The notebook loads `cars.csv`:

```python
cars = pd.read_csv(r"E:\MACHINE_LEARNING\DATA_SETS\cars.csv")
df = cars
```

Saved shape:

```text
(8128, 5)
```

Columns:

```text
brand, km_driven, fuel, owner, selling_price
```

The first row is:

```text
Maruti, 145500, Diesel, First Owner, 450000
```

The notebook has four main blocks:

1. Inspect `owner` frequencies.
2. One-hot encode `fuel` and `owner` with pandas, with and without `drop_first`.
3. One-hot encode with scikit-learn and concatenate with numeric columns.
4. Group rare `brand` levels and one-hot encode the grouped feature.

---

### A6. Owner frequencies

```python
df["owner"].value_counts()
```

Saved output:

```text
First Owner             5289
Second Owner            2105
Third Owner              555
Fourth & Above Owner     174
Test Drive Car             5
```

There are five owner levels. With all levels, one-hot encoding would create five columns; with `drop_first`, four.

The `Test Drive Car` level has only five rows. It is rare, but it is still a legitimate category. Whether to keep it depends on the problem: a rare level can be informative, but it can also cause unstable estimates.

---

### A7. Pandas `get_dummies`

The first pandas call is:

```python
pd.get_dummies(df, columns=["fuel", "owner"])
```

`columns` tells pandas which columns to expand. Other columns pass through unchanged.

Saved output shape:

```text
(8128, 12)
```

The original table had five columns. Expanding `fuel` (4 levels) and `owner` (5 levels) creates nine new columns and removes the two original columns:

$$
5 - 2 + 9 = 12.
$$

The new columns are:

```text
fuel_CNG, fuel_Diesel, fuel_LPG, fuel_Petrol,
owner_First Owner, owner_Fourth & Above Owner,
owner_Second Owner, owner_Test Drive Car, owner_Third Owner
```

Pandas uses the original column name and level as the new column name, separated by an underscore. The output dtype is boolean in modern pandas.

#### `drop_first=True`

```python
pd.get_dummies(df, columns=["fuel", "owner"], drop_first=True)
```

Saved shape:

```text
(8128, 10)
```

The dropped columns are `fuel_CNG` and `owner_First Owner`. The remaining dummies are:

```text
fuel_Diesel, fuel_LPG, fuel_Petrol,
owner_Fourth & Above Owner, owner_Second Owner,
owner_Test Drive Car, owner_Third Owner
```

That is $3+4=7$ dummy columns, so:

$$
5 - 2 + 7 = 10.
$$

The baseline is the combination `fuel=CNG`, `owner=First Owner`. An all-zero row in the dummy block represents that combination.

#### Why pandas is convenient but not pipeline-safe

`pd.get_dummies` is a pure function. It does not remember the training categories. If you call it separately on training and test data:

- The test set may have a category that the training set did not, producing an extra column.
- The test set may lack a category that the training set had, producing a missing column.
- The column order may differ.

For a quick exploratory analysis, this is fine. For a production pipeline, use `OneHotEncoder`, which learns the categories during `fit` and applies the same columns during `transform`.

---

### A8. Scikit-learn `OneHotEncoder`

The notebook creates:

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(
    drop="first",
    sparse_output=False,
    dtype=np.int32,
)
```

The arguments mean:

| Argument | Meaning |
|---|---|
| `drop="first"` | Drop the first level of each feature, keeping $k-1$ columns. |
| `sparse_output=False` | Return a dense NumPy array instead of a sparse matrix. |
| `dtype=np.int32` | Store the indicators as 32-bit integers. |

The default `handle_unknown="error"` raises an error if `transform` sees a category that was not present during `fit`.

#### Fitting and transforming

```python
X_train_new = ohe.fit_transform(X_train[["fuel", "owner"]])
X_test_new = ohe.transform(X_test[["fuel", "owner"]])
```

Saved training shape:

```text
(5689, 7)
```

The seven columns come from:

- `fuel`: 4 levels, drop first -> 3 columns.
- `owner`: 5 levels, drop first -> 4 columns.

$$
3+4=7.
$$

The test block has the same seven columns because `transform` reuses the learned categories. This is the key advantage over `get_dummies`.

#### Learned attributes

After fitting, `OneHotEncoder` stores:

- `categories_`: one array per input feature, containing the levels in sorted order.
- `drop_idx_`: the index of the dropped level for each feature, or `None` if no level was dropped.
- `n_features_in_`: the number of input columns.
- `feature_names_in_`: present when the input is a DataFrame with string column names.

You can inspect the generated feature names with:

```python
ohe.get_feature_names_out(["fuel", "owner"])
```

This returns names such as `fuel_Diesel`, `fuel_LPG`, and so on.

#### `handle_unknown="ignore"`

For production, a common setting is:

```python
OneHotEncoder(
    drop="first",
    sparse_output=False,
    handle_unknown="ignore",
)
```

With `ignore`, an unseen category at transform time becomes an all-zero row for that feature instead of raising an error. This is often necessary when new categories can appear in production, but it means the model cannot distinguish "unknown" from the dropped baseline unless you add an explicit unknown category.

Scikit-learn also supports `min_frequency` and `max_categories` to group infrequent categories automatically, depending on the version. The notebook uses manual grouping for `brand`, which is easier to inspect.

---

### A9. The concatenation step

The notebook keeps `brand` and `km_driven` alongside the encoded `fuel` and `owner` block:

```python
np.hstack((
    X_train[["brand", "km_driven"]].values,
    X_train_new,
)).shape
```

Saved shape:

```text
(5689, 9)
```

The calculation is:

$$
2 \text{ passthrough columns} + 7 \text{ encoded columns} = 9.
$$

However, `brand` is still a string column. `np.hstack` does not encode it; it simply places the strings next to the numeric columns. The resulting array has a mixed dtype (likely object or string), and it is **not** ready for a scikit-learn model. A complete workflow would encode `brand` as well, or drop it, or group it first.

This is an important debugging lesson: a shape can look right while the contents are wrong. Always inspect `dtype` and a few rows, not just `.shape`.

---

### A10. High-cardinality brands and rare-category grouping

The notebook computes brand counts:

```python
counts = df["brand"].value_counts()
```

Saved counts include:

```text
Maruti           2448
Hyundai          1415
Mahindra          772
Tata              734
Toyota            488
Honda             467
Ford              397
Chevrolet         230
Renault           228
Volkswagen        186
BMW               120
Skoda             105
Nissan             81
Jaguar             71
Volvo              67
Datsun             65
Mercedes-Benz      54
Fiat               47
...
Peugeot             1
```

There are 32 brands. One-hot encoding all of them would add 31 columns after dropping one. Many of those columns would be almost always zero, which adds noise and memory cost.

The notebook sets:

```python
threshold = 100
repl = counts[counts <= threshold].index
```

and replaces rare brands:

```python
df["brand"].replace(repl, "unommon")
```

The typo `"unommon"` is preserved in the notebook. The intended label is `"uncommon"`. The typo does not affect the grouping logic, but it produces an awkward column name and is worth fixing in new code.

The saved preview of `repl` shows 15 entries, but the full count table implies more brands at or below 100. The displayed index is truncated or stale. When re-running, inspect `len(repl)` and `df["brand"].nunique()` directly rather than trusting the saved preview.

After grouping, the saved dummy output has 13 columns:

```text
BMW, Chevrolet, Ford, Honda, Hyundai, Mahindra, Maruti,
Renault, Skoda, Tata, Toyota, Volkswagen, unommon
```

That is 12 frequent brands plus the rare-category bucket. The exact set depends on the threshold. The grouping keeps the feature usable without exploding the width.

#### Choosing the threshold

A threshold is a modelling choice. A low threshold keeps more categories and more columns; a high threshold groups more categories and may lose useful distinctions. Common strategies:

- Group categories with fewer than a fixed number of rows.
- Keep the top $m$ categories and group the rest.
- Group categories whose frequency is below a percentage of the dataset.
- Use `OneHotEncoder(min_frequency=...)` where supported.

The threshold should be chosen with validation data, not only by convenience.

---

### A11. Preserved bugs in the notebook

The Day 27 notebook contains several bugs that are intentionally preserved as learning material. Do not copy them into a production workflow.

#### Bug 1: `y` is not the target

```python
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, 0:4],
    df.iloc[:, :-1],
    test_size=0.3,
    random_state=34,
)
```

`X` is the first four columns: `brand`, `km_driven`, `fuel`, `owner`. `y` is also the first four columns because `iloc[:, :-1]` excludes only the last column (`selling_price`). So `y` overlaps `X` and is not the target.

The correct target is:

```python
y = df["selling_price"]
```

or:

```python
y = df.iloc[:, -1]
```

This bug means the notebook cannot train a valid supervised model. No model is trained in the notebook, so no accuracy is reported. If you add a model without fixing the target, the result is meaningless.

#### Bug 2: `brand` remains a string

The `hstack` block keeps `brand` as a raw string. A model cannot fit on that array. The notebook is demonstrating shape arithmetic, not a finished feature matrix.

#### Bug 3: typo `"unommon"`

The rare-category bucket is named `"unommon"` instead of `"uncommon"`. It is harmless but confusing.

#### Bug 4: duplicate import and empty cells

`OneHotEncoder` is imported twice, and some cells contain only intermediate variables or no output. These are cleanup issues, not conceptual errors.

#### Bug 5: absolute dataset path

The CSV is loaded from an absolute Windows path. Use `datasets/cars.csv` for portability.

#### What is correct

- `OneHotEncoder` is fitted on `X_train` and used to `transform` `X_test`.
- The encoder is not refitted on the test set.
- `drop="first"` is used consistently.
- The shape calculations are correct for the encoded block.

The notebook's takeaways cell warns about fitting on test data. The code shown here already uses `transform` for the test block, which is the right pattern.

---

### A12. Complete data flow

```text
cars.csv (8128 x 5)
   |
   v
inspect owner counts
   |
   v
pandas one-hot: 12 columns (full), 10 columns (drop first)
   |
   v
train/test split (70/30, rs=34)
   |
   +---------------------------+
   |                           |
   v                           v
X_train (5689 x 4)          X_test (2439 x 4)
   |                           |
   v                           v
OneHotEncoder.fit_transform  OneHotEncoder.transform
on fuel + owner              on fuel + owner
   |                           |
   v                           v
(5689 x 7)                   (2439 x 7)
   |
   v
hstack with brand + km_driven
   |
   v
(5689 x 9) but brand is still a string
```

The final array is not model-ready because `brand` is unencoded. A corrected workflow would encode `brand` or group it first.

---

### A13. One-hot versus ordinal versus label encoding

| Method | Applied to | Order | Output width | Main risk |
|---|---|---|---|---|
| One-hot | Nominal inputs | None | $k$ or $k-1$ columns | Width explosion |
| Ordinal | Ordered inputs | Declared | 1 column | False spacing / wrong order |
| Label | Target vector | Learned | 1-D vector | Misuse on inputs |

For `fuel`, one-hot is appropriate because fuel types have no order.

For `review` (Poor, Average, Good), ordinal encoding is appropriate if the order is meaningful.

For `purchased` (No, Yes), label encoding is appropriate because it is the target.

---

### A14. One-hot versus target encoding

Target encoding replaces each category with a statistic of the target, such as the mean target value for that category. It can capture more information with fewer columns, but it is prone to leakage: the target statistic for a row may include that row's own target. It requires out-of-fold fitting.

One-hot encoding makes no use of the target. It is safer and easier to explain, but it can be wide. The choice depends on cardinality, data size, and the model.

---

### A15. Common mistakes and debugging

#### Mistake 1: using `get_dummies` separately on train and test

```python
# Wrong: different columns, different order
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
```

Fix: use `OneHotEncoder` fitted on train, or align the columns explicitly after `get_dummies`.

#### Mistake 2: fitting the encoder on the full dataset

```python
# Wrong: learns categories from test data
ohe.fit(df[["fuel", "owner"]])
```

Fix: `ohe.fit(X_train[["fuel", "owner"]])`.

#### Mistake 3: keeping all $k$ dummies with an intercept

This creates perfect multicollinearity. Use `drop="first"` or `drop_first=True` when coefficient interpretation matters.

#### Mistake 4: one-hotting high-cardinality IDs

A column with thousands of unique IDs should not be one-hot encoded. Group rare categories, hash, target-encode, or drop the ID.

#### Mistake 5: forgetting `handle_unknown`

If production data can contain new categories, the default `handle_unknown="error"` will crash. Decide on a policy: ignore, map to an unknown category, or retrain.

#### Mistake 6: checking only the shape

A shape of `(5689, 9)` can hide a string column or a wrong dtype. Inspect `dtype`, column names, and sample rows.

#### Mistake 7: treating the encoded values as quantities

A one-hot 1 means membership, not magnitude. The columns are indicators, not measurements.

#### Mistake 8: using one-hot for ordered features

If the order matters, one-hot discards it. For `education`, one-hot may be more flexible but loses the monotone structure. Choose based on the modelling assumption and validate.

---

### A16. Where this sits in the ML workflow

One-hot encoding belongs after the train/test split and before model training:

```text
Raw data
   |
   v
Identify nominal columns
   |
   v
Train/test split
   |
   v
Fit OneHotEncoder on X_train
   |
   v
Transform X_train and X_test
   |
   v
Train model
   |
   v
Evaluate
```

In a full preprocessing pipeline:

```text
ColumnTransformer
   ├── nominal columns -> SimpleImputer -> OneHotEncoder
   ├── ordinal columns -> OrdinalEncoder
   ├── numeric columns -> SimpleImputer -> Scaler
   └── high-cardinality -> grouping / target encoding
```

Days 28–29 show how to assemble these steps into a `ColumnTransformer` and a `Pipeline`.

---

### A17. Interview preparation

#### Level 1 — Recall

**Q1. What is one-hot encoding?**

It represents each level of a categorical variable as a binary column. A row has a 1 in the column corresponding to its level and 0 elsewhere.

**Q2. When should one-hot encoding be used?**

For nominal input features, where the categories have no meaningful order.

**Q3. What is the dummy-variable trap?**

With all $k$ one-hot columns and an intercept, the columns sum to the intercept column, creating perfect multicollinearity. Dropping one level avoids the trap.

#### Level 2 — Explanation

**Q4. Why is integer encoding bad for nominal features?**

It invents an order and a distance between categories. A model may learn relationships that do not exist.

**Q5. Why keep $k-1$ columns instead of $k$?**

The dropped level becomes the baseline. The remaining coefficients are interpreted relative to it, and the design matrix is no longer perfectly collinear with the intercept.

**Q6. What is the difference between `pd.get_dummies` and `OneHotEncoder`?**

`get_dummies` is a stateless pandas function, convenient for EDA. `OneHotEncoder` is a scikit-learn estimator that learns categories during `fit`, remembers them, and can be used in pipelines.

#### Level 3 — Application

**Q7. A feature has 100 levels. How many columns does one-hot encoding create with and without dropping one?**

Without dropping: 100 columns. With `drop="first"`: 99 columns. If the model has an intercept, 99 is the safer choice for coefficient interpretation.

**Q8. How would you handle a brand column with 500 levels?**

Group rare brands into an "other" bucket, keep the top $m$ brands, or use target encoding/hashing. Do not one-hot all 500 levels unless the dataset is large enough and the model can handle the width.

**Q9. What happens to an unseen category with `handle_unknown="ignore"`?**

The encoded row for that feature is all zeros. The model treats it like the baseline category unless you add an explicit unknown indicator.

#### Level 4 — Analysis

**Q10. Why can one-hot encoding increase overfitting?**

It increases the number of parameters. Rare categories can get coefficients that fit noise. Regularization, grouping, or a simpler encoding can help.

**Q11. Why is one-hot encoding not always the best choice for high-cardinality features?**

The design matrix becomes very wide and sparse. Many columns have very few non-zero entries, which makes estimation unstable and memory usage high.

**Q12. Does one-hot encoding lose information compared with ordinal encoding?**

It loses the order, which is desirable for nominal variables. It does not lose category membership. For ordered variables, losing the order may be a disadvantage.

#### Level 5 — Design

**Q13. How would you encode a dataset with numeric, ordinal, and nominal columns?**

Use a `ColumnTransformer`: numeric columns through an imputer and scaler; ordinal columns through `OrdinalEncoder` with declared order; nominal columns through `OneHotEncoder`. Put the whole transformer inside a pipeline.

**Q14. How would you handle a new category in production?**

Options: `handle_unknown="ignore"`, an explicit "unknown" category in training, a hashing encoder, or retraining with the new category. The choice depends on whether the new category is expected and whether the model can learn from it.

**Q15. How would you choose the rare-category threshold?**

Use validation performance and the frequency distribution. A threshold that is too low keeps noisy rare levels; one that is too high merges useful categories. Treat it as a hyperparameter and evaluate it honestly.

#### Level 6 — Evaluation

**Q16. How would you compare one-hot with target encoding?**

Use the same cross-validation folds and model. For target encoding, fit the encoding inside each fold to avoid leakage. Compare validation performance, stability, and interpretability.

**Q17. Why might dropping a level hurt a tree model?**

Trees can split on each indicator independently. Dropping a level removes the explicit indicator for the baseline, but the baseline is still represented by all zeros. In practice, trees usually handle either representation, though the split search differs.

**Q18. What would you monitor after deployment?**

The frequency of unseen categories, the distribution of encoded columns, and the model's calibration. A new category or a large shift in category frequencies may require retraining.

#### Level 7 — Advanced

**Q19. Can one-hot encoding be used for the target?**

It can, for some neural-network losses, but standard scikit-learn classifiers expect a one-dimensional target of class ids. For those models, use `LabelEncoder` or pass string labels directly if supported.

**Q20. What is the difference between one-hot encoding and hashing?**

Hashing maps categories to a fixed number of columns using a hash function. It bounds the width and handles unseen categories, but it can cause collisions. One-hot encoding is exact but can be very wide.

**Q21. How does one-hot encoding interact with regularization?**

Each level gets its own coefficient. With L2 regularization, the coefficients are shrunk toward zero. With L1, some coefficients can become exactly zero, effectively selecting levels. The baseline choice affects the interpretation of the remaining coefficients.

---

### A18. Exam-style questions with worked answers

#### Question 1

A nominal feature has levels A, B, C, D. How many columns does one-hot encoding create with `drop_first=True`, and what does an all-zero row mean?

**Answer**

Four levels, drop one -> three columns. Suppose A is dropped. The columns are `B`, `C`, `D`. An all-zero row means the category is A, the baseline. The coefficients of B, C, and D are interpreted relative to A.

#### Question 2

A dataset has 8,128 rows and a `brand` column with 32 levels. One-hot encoding all levels would create how many columns after dropping one? Why might that be a problem?

**Answer**

$32-1=31$ columns. Many brands have very few rows, so most of these columns are almost always zero. The matrix becomes wide and sparse, estimation can be unstable, and memory cost increases. Grouping rare brands or using another encoding is often better.

#### Question 3

Explain why `pd.get_dummies` can produce different columns for training and test data.

**Answer**

`get_dummies` learns the categories from the data it is given. If the test set has a category that the training set lacks, it creates an extra column. If the test set lacks a category that the training set has, it creates fewer columns. The column order can also differ. `OneHotEncoder` avoids this by learning the categories once during `fit`.

#### Question 4

A model is trained with all four `fuel` dummies and an intercept. Explain why the coefficients are not uniquely identifiable.

**Answer**

For every row, the four fuel dummies sum to 1, which is the intercept column. Therefore the design matrix has a linear dependency: one column is a linear combination of the others. The normal equations have infinitely many solutions for the coefficients. Predictions may still be unique, but individual coefficients are not.

#### Question 5

The Day 27 notebook sets `y = df.iloc[:, :-1]`. Explain the bug and the fix.

**Answer**

`df.iloc[:, :-1]` selects all columns except the last one, which is `selling_price`. So `y` contains `brand`, `km_driven`, `fuel`, and `owner` — the same columns as `X`, not the target. The fix is `y = df["selling_price"]` or `y = df.iloc[:, -1]`. Without the fix, any supervised model would be trained on the wrong target.

---

### A19. Revision summary

- One-hot encoding gives each nominal level its own binary column.
- It avoids inventing an order.
- With an intercept, drop one level to avoid perfect multicollinearity.
- `pd.get_dummies` is convenient for EDA but stateless.
- `OneHotEncoder` learns categories and is pipeline-safe.
- Fit on training data, transform test data.
- `handle_unknown="ignore"` maps unseen categories to all zeros.
- High-cardinality features need grouping, hashing, target encoding, or another strategy.
- Check dtypes and column contents, not only shapes.
- The Day 27 notebook preserves several bugs; the main one is that `y` is not the target.

---

## Part B — Advanced Reading and Reference

### B1. The linear algebra of the dummy trap

Let $\mathbf{X}$ be the design matrix with an intercept column $\mathbf{1}$ and all $k$ one-hot columns $\mathbf{D}_1,\dots,\mathbf{D}_k$. For every row:

$$
\mathbf{1} = \sum_{j=1}^{k}\mathbf{D}_j.
$$

Therefore the columns are linearly dependent, and $\mathbf{X}$ does not have full column rank. In ordinary least squares, the normal equations

$$
\mathbf{X}^\top\mathbf{X}\boldsymbol\beta=\mathbf{X}^\top\mathbf{y}
$$

have infinitely many solutions when $\mathbf{X}^\top\mathbf{X}$ is singular. Dropping one dummy column restores full rank (assuming no other collinearity).

For ridge regression, the objective

$$
\lVert\mathbf{y}-\mathbf{X}\boldsymbol\beta\rVert_2^2+\lambda\lVert\boldsymbol\beta\rVert_2^2
$$

has a unique solution even when $\mathbf{X}$ is rank-deficient, because the penalty makes the objective strictly convex. The coefficients are still not uniquely interpretable in the original parameterization, and the baseline choice affects them.

---

### B2. Sparse output

By default, `OneHotEncoder(sparse_output=True)` returns a scipy sparse matrix. For a feature with many levels, most entries are zero, so a sparse representation saves memory.

```python
ohe = OneHotEncoder(drop="first", sparse_output=True)
X_train_sparse = ohe.fit_transform(X_train[["fuel", "owner"]])
```

Many scikit-learn estimators accept sparse matrices. If a downstream step requires a dense array, convert explicitly with `.toarray()`, but be aware of the memory cost.

The notebook uses `sparse_output=False` because it wants to inspect and `hstack` the array. For large high-cardinality features, sparse output is often the better default.

---

### B3. Feature names

`get_feature_names_out` returns the names of the generated columns:

```python
ohe.get_feature_names_out(["fuel", "owner"])
```

Example names:

```text
fuel_Diesel
fuel_LPG
fuel_Petrol
owner_Fourth & Above Owner
owner_Second Owner
owner_Test Drive Car
owner_Third Owner
```

These names are useful for debugging, feature importance, and model documentation. If the input is a NumPy array, you must supply the input feature names.

---

### B4. Handling unknown categories

`handle_unknown` options:

- `"error"`: raise an error on unseen categories. This is the default and is useful for catching data problems.
- `"ignore"`: encode unseen categories as all zeros. This is common in production.
- `"infrequent_if_exist"`: map unseen categories to the infrequent category if one exists. This depends on `min_frequency`/`max_categories`.

With `handle_unknown="ignore"` and `drop="first"`, an unseen category is indistinguishable from the dropped baseline in the encoded representation. If that distinction matters, add an explicit `"Unknown"` category during training or use an additional missing-indicator feature.

---

### B5. Infrequent-category grouping in scikit-learn

Recent versions of scikit-learn support automatic grouping:

```python
ohe = OneHotEncoder(
    min_frequency=100,
    max_categories=20,
    handle_unknown="infrequent_if_exist",
    sparse_output=False,
)
```

The exact behaviour depends on the version. `min_frequency` can be an integer count or a proportion. `max_categories` limits the total number of output columns. The grouped category is named `infrequent_sklearn` by default.

The notebook performs manual grouping with pandas, which is transparent and easy to inspect. Automatic grouping is convenient in pipelines but requires checking the version's semantics.

---

### B6. One-hot encoding and model families

#### Linear models

One-hot columns let a linear model estimate a separate coefficient per level. With $k-1$ columns, the dropped level is the reference. This is the standard way to include nominal predictors in a linear model.

#### Trees

Trees can split on individual indicator columns. Full one-hot encoding can make the tree consider many binary splits, which increases training time. Ordinal encoding of a nominal feature is usually wrong because it imposes an order, but trees can sometimes exploit arbitrary codes; this is not a principled solution.

#### Distance-based models

One-hot encoding creates a binary space. The Euclidean distance between two different categories is $\sqrt{2}$ (if both have one 1 and the rest 0), while the distance between the same category is 0. This is a reasonable default, but it treats all category differences as equally large. Embeddings or target encoding can capture richer similarities.

#### Neural networks

One-hot vectors are common inputs for categorical features. Embedding layers learn dense representations and are often more efficient for high cardinality.

---

### B7. One-hot versus frequency encoding

Frequency encoding replaces each category with its count or proportion in the training data. It uses one column and can capture rarity, but it imposes an order based on frequency and may not generalise to unseen categories. It is sometimes used for high-cardinality features, but it can leak if computed on the full dataset.

One-hot encoding is exact and order-free but wide. The choice depends on the cardinality and the model.

---

### B8. One-hot versus target encoding

Target encoding replaces a category with a statistic of the target, such as the mean target for that category. It can be powerful for high-cardinality features, but it is prone to leakage. A safe implementation uses out-of-fold estimates and smoothing:

$$
\text{encoding}_c
=
\frac{n_c\bar{y}_c + m\bar{y}}{n_c+m},
$$

where $n_c$ is the category count, $\bar{y}_c$ is the category mean, $\bar{y}$ is the global mean, and $m$ is a smoothing parameter. The smoothing shrinks rare categories toward the global mean.

One-hot encoding does not use the target, so it is safer but wider.

---

### B9. One-hot versus hashing

Hashing maps categories to a fixed number of columns:

$$
h(c) \bmod B,
$$

where $B$ is the number of buckets. It bounds the width and handles unseen categories, but collisions can merge unrelated categories. It is useful for very high cardinality and streaming data.

One-hot encoding is exact but can be impractically wide.

---

### B10. Advanced interview questions

**Q1. Why does `drop="first"` drop the first category in sorted order rather than an arbitrary category?**

Scikit-learn sorts the categories and drops the first by default. The choice of baseline is arbitrary but affects coefficient interpretation. You can specify which category to drop with `drop=[...]` or by ordering the categories.

**Q2. What is the difference between `drop=None` and `drop="if_binary"`?**

`drop=None` keeps all $k$ columns for every feature. `drop="if_binary"` drops one column only for binary features, keeping all columns for multi-level features. The latter is useful when you want a single binary indicator but full flexibility for multi-level features.

**Q3. How does one-hot encoding affect the intercept?**

With all $k$ dummies, the intercept is collinear with the dummies. With $k-1$, the intercept represents the baseline category's expected value (in a linear model with no other features). The remaining coefficients are differences from the baseline.

**Q4. Why is one-hot encoding not a solution for high-cardinality IDs?**

IDs usually have no generalisable meaning, and one-hot encoding creates a column per ID. The model can memorize IDs and overfit. IDs should usually be dropped or treated with a method designed for high cardinality.

**Q5. Can one-hot encoding handle missing values?**

`OneHotEncoder` can treat missing values as a category if configured, or you can impute before encoding. The right choice depends on whether missingness is informative. Day 37 covers missing categorical data.

**Q6. How would you explain the dummy trap to a non-technical stakeholder?**

If you give the model one yes/no column for every category plus an overall baseline, the columns become redundant: knowing all but one tells you the last. The model cannot tell which category is responsible for the effect. Dropping one category gives the model a reference point.

**Q7. What is the effect of one-hot encoding on feature importance?**

Each level gets its own importance. This can reveal non-monotone effects, but it also splits importance across correlated columns. For a feature with many levels, the individual importances may be unstable.

**Q8. How would you choose between one-hot and ordinal encoding for an ordered feature?**

Consider whether the effect is monotone and whether the spacing between levels is meaningful. If a single coefficient along the order is plausible, ordinal encoding is simpler. If the effect may be non-monotone, one-hot is more flexible. Validate both with cross-validation.

---

### B11. Notebook coverage matrix

| Notebook area | Cells | What it teaches | Status |
|---|---:|---|---|
| Imports and theme | 3–5 | Setup | No output |
| Load cars data | 6–7 | Dataset shape and columns | Saved table |
| Owner counts | 9 | Category frequencies | Saved counts |
| Pandas one-hot | 11–12 | Full dummy expansion | Saved 12 columns |
| Drop first | 13–14 | $k-1$ encoding | Saved 10 columns |
| Train/test split | 16 | Split and preserved `y` bug | Saved sample |
| sklearn encoder | 17–21 | Fit/transform and hstack | Saved shapes |
| Top categories | 22–27 | Rare-brand grouping | Saved counts and dummies |
| Experiment | 28 | Threshold exercise | Markdown |
| Key takeaways | 29 | Revision and bug list | Markdown |

No model is trained. The notebook ends after encoding and grouping.

---

### B12. Source fidelity and caveats

- Saved outputs come from the user's environment. They are reported as saved results, not independently reproduced.
- The notebook loads `cars.csv` from an absolute Windows path. The local copy is at `datasets/cars.csv`; the relative-path version is a portability correction.
- The notebook's `y` is `df.iloc[:, :-1]`, which overlaps `X` and is not the target. This is a preserved bug. The chapter does not claim any model result.
- The `hstack` output keeps `brand` as a string, so the final array is not model-ready.
- The rare-category bucket is spelled `"unommon"` in the notebook; the intended spelling is `"uncommon"`.
- The saved `repl` preview appears truncated relative to the full count table. Re-run the cell to see the complete list.
- The notebook has no graphs, so no graph interpretation is claimed.
- No accuracy, cross-validation, or feature-importance result is reported.

---

### B13. References

- scikit-learn documentation: `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`.
- pandas documentation: `get_dummies`.
- CampusX Day 27: One Hot Encoding — Handling Categorical Data.
- Day 26 notes: ordinal and label encoding.
- Day 28 notes: `ColumnTransformer`.
- Day 29 notes: pipelines and cross-validation.

---

### B14. Final connection

One-hot encoding is the standard answer to a nominal feature, but it is not a free lunch. It avoids false order at the cost of width, and it requires a careful policy for rare and unseen categories. The next day shows how to route different column types — numeric, ordinal, and nominal — through one `ColumnTransformer`, which is the natural home for the encoders you have just learned.
