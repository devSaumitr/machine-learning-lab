# Day 26 — Ordinal Encoding and Label Encoding

> **Course topic:** CampusX Day 26, *Encoding Categorical Data — Ordinal Encoding and Label Encoding*.
> **Personal notebook:** `my_notebook.ipynb` (27 cells).
> **Dataset:** `datasets/customer.csv`.
> **Reference:** `reference_notebooks/day26.ipynb` (do not edit).
>
> **Part A** is the main learning chapter and is sufficient on its own. **Part B** preserves advanced details, edge cases, and reference material.

---

## Part A — Main Learning Chapter

### A1. Learning objectives

By the end of this chapter you should be able to:

1. Distinguish numerical, nominal, ordinal, and target variables.
2. Explain why models need numeric inputs and why the encoding choice matters.
3. Use `OrdinalEncoder` with an explicit category order for ordered input features.
4. Use `LabelEncoder` for a target vector.
5. Explain `fit`, `transform`, `fit_transform`, `categories_`, and `classes_`.
6. Avoid the two classic errors: alphabetical order for ordinal features, and label encoding for nominal inputs.
7. Connect encoding to train/test leakage and to the preprocessing pipelines in Days 28–29.

---

### A2. The problem: models cannot do arithmetic on words

A classifier such as logistic regression computes a weighted sum:

$$
z = w_1x_1 + w_2x_2 + \cdots + w_px_p + b.
$$

If $x_1$ is the string `"Poor"`, the expression is undefined. The model needs numbers.

But not every number is harmless. Suppose we map:

```text
Poor    -> 0
Average -> 1
Good    -> 2
```

This preserves the real order `Poor < Average < Good`. A linear model can use a single coefficient to represent the effect of moving one level up.

Now suppose we map a nominal feature such as `city`:

```text
Delhi    -> 0
Mumbai   -> 1
Kolkata  -> 2
```

The model now sees `Kolkata > Mumbai > Delhi` and may learn a false ordering. If the model uses distances, it will also treat Delhi and Kolkata as "farther apart" than Delhi and Mumbai. The numbers invented a relationship that the data does not contain.

The central rule of this day is:

> **The encoding must match the meaning of the variable.** Ordered inputs may use ordered codes. Nominal inputs need a representation that does not impose an order. Targets need numeric class labels.

---

### A3. Foundations: types of variables

#### Numerical variables

Values are numbers with meaningful arithmetic: age, salary, temperature, price. They may be continuous or discrete.

#### Categorical variables

Values are labels or groups. Categorical variables split into two kinds.

**Nominal:** categories have no natural order.

- Gender: Male, Female, Other.
- City: Delhi, Mumbai, Kolkata.
- Fuel: Petrol, Diesel, CNG.
- Colour: red, green, blue.

**Ordinal:** categories have a meaningful order, but the numerical distances between levels are not necessarily equal.

- Review: Poor, Average, Good.
- Education: School, UG, PG.
- Satisfaction: Low, Medium, High.
- Disease severity: mild, moderate, severe.

The order is real, but "Good minus Average" is not a quantity in the way "30°C minus 20°C" is. Ordinal encoding preserves order, not equal spacing.

#### Target variables

The target $y$ may also be categorical. For binary classification, labels such as `No` and `Yes` must become numeric class ids, usually 0 and 1. For multi-class classification, labels such as `cat`, `dog`, `bird` become class indices.

The target is a one-dimensional vector, not a feature matrix. That is why scikit-learn has a separate `LabelEncoder` for it.

---

### A4. The dataset used in the notebook

The notebook loads `customer.csv` and drops the first two columns:

```python
df = pd.read_csv(r"E:\MACHINE_LEARNING\DATA_SETS\customer.csv")
df = df.iloc[:, 2:]
```

The saved sample row is:

```text
age  gender review education purchased
7   60  Female   Poor    School       Yes
```

After `iloc[:, 2:]`, the working columns are:

| Column | Role | Type |
|---|---|---|
| `review` | input feature | ordinal |
| `education` | input feature | ordinal |
| `purchased` | target | binary nominal |

`age` and `gender` are dropped in this notebook. `gender` is nominal, so it should not be ordinal-encoded; Day 27 covers one-hot encoding for nominal inputs.

The full sliced table has 50 rows. The notebook prints all of them, but the structure is what matters:

```text
   review education purchased
0  Average    School        No
1     Poor        UG        No
2     Good        PG        No
...
49    Good        UG        No
```

---

### A5. Splitting before encoding

The notebook splits the data:

```python
from sklearn.model_selection import train_test_split as tts

X_train, X_test, y_train, y_test = tts(
    df.iloc[:, 0:2],
    df.iloc[:, 2],
    test_size=0.3,
    random_state=34,
)
```

The saved training shape is:

```text
(35, 2)
```

The test set has 15 rows. `random_state=34` makes the split reproducible.

The encoder will be fitted on `X_train` only. The test categories are transformed with the learned mapping. This is the same discipline used for scaling in Days 24–25: **fit on training data, transform everything else**.

---

### A6. `OrdinalEncoder`: declared order

The notebook creates:

```python
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(
    categories=[
        ["Poor", "Average", "Good"],
        ["School", "UG", "PG"],
    ]
)
```

The `categories` argument is a list of lists. Each inner list corresponds to one column, in column order:

- Column 0 (`review`): `Poor -> 0`, `Average -> 1`, `Good -> 2`.
- Column 1 (`education`): `School -> 0`, `UG -> 1`, `PG -> 2`.

The encoder is then fitted and applied:

```python
oe.fit(X_train)
X_train = oe.transform(X_train)
X_test = oe.transform(X_test)
```

The fitted attribute confirms the order:

```python
oe.categories_
```

Saved output:

```text
[array(['Poor', 'Average', 'Good'], dtype=object),
 array(['School', 'UG', 'PG'], dtype=object)]
```

#### Why the order must be declared

If `categories="auto"` is used, scikit-learn learns the categories from the data and sorts strings lexicographically. For `review`, alphabetical order is:

```text
Average, Good, Poor
```

That maps to:

```text
Average -> 0
Good    -> 1
Poor    -> 2
```

The real order is the reverse: `Poor < Average < Good`. A linear model with a positive coefficient would then learn that moving from `Poor` to `Average` *decreases* the encoded value, which contradicts the meaning of the feature. The model can still fit the data, but the coefficient's sign and interpretation are now misleading, and the model may generalise worse because the numerical geometry fights the real order.

The lesson is simple: **for ordinal features, the order is part of the data definition, so you must supply it.**

#### Output type and shape

`OrdinalEncoder` returns a 2-D array with one column per input column. The default output dtype is `float64`, so the values print as `0.`, `1.`, `2.` rather than integers.

The notebook reassigns:

```python
X_train = oe.transform(X_train)
```

This replaces the DataFrame with a NumPy array. Column names and the original index are lost. The notebook preserves this as-is. In new code, wrap the array if you need labels:

```python
import pandas as pd

X_train_encoded = pd.DataFrame(
    oe.transform(X_train),
    columns=X_train.columns,
    index=X_train.index,
)
```

#### What `fit` learns

`OrdinalEncoder.fit` checks the declared categories and stores them in `categories_`. It does not learn a numeric relationship or a distance between categories. The numbers are positional codes assigned by the order you supplied.

`transform` maps each string to its position. An unseen category raises an error by default because `handle_unknown="error"`.

#### Unseen categories

If the test set contains a category that was not in the training data, the default behaviour is an error:

```text
ValueError: Found unknown categories ['...'] in column ...
```

This is deliberate. Silently assigning an arbitrary code could corrupt the meaning of the feature. If you need a fallback, `handle_unknown="use_encoded_value"` with `unknown_value=-1` can be used, but you must decide whether a sentinel is meaningful for your model.

#### Inverse transformation

`inverse_transform` maps codes back to labels:

```python
oe.inverse_transform([[0.0, 1.0]])
```

Result:

```text
[['Poor', 'UG']]
```

This is useful for debugging and for displaying predictions in human-readable form.

---

### A7. `LabelEncoder`: target labels only

The notebook creates a label encoder:

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(y_train)
```

The fitted classes are:

```python
le.classes_
```

Saved output:

```text
array(['No', 'Yes'], dtype=object)
```

The mapping is:

```text
No  -> 0
Yes -> 1
```

The target is transformed:

```python
y_train = le.transform(y_train)
y_test = le.transform(y_test)
```

Saved `y_train`:

```text
array([1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0,
       0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1])
```

This is a one-dimensional integer array of length 35.

#### Why `LabelEncoder` is for the target

`LabelEncoder` accepts a one-dimensional array of labels. It learns the unique classes, sorts them, and assigns integer codes. It is designed for $y$, not for a feature matrix.

Using it on an input feature is a common mistake:

```python
# Wrong for a nominal input
df["city"] = LabelEncoder().fit_transform(df["city"])
```

This imposes an alphabetical order on a nominal variable. A model may then treat the codes as meaningful magnitudes. For nominal inputs, use one-hot encoding (Day 27) or another order-free representation.

For an ordinal input, `OrdinalEncoder` is the right tool because it accepts multiple columns and lets you declare the order. `LabelEncoder` has no `categories` argument; it always learns the classes from the data.

#### Binary and multi-class targets

For binary targets, 0/1 codes are natural. For multi-class targets, `LabelEncoder` assigns 0, 1, 2, ... in sorted order. Classifiers such as logistic regression and decision trees accept these integer class ids. The codes are identifiers, not quantities; the model's loss treats them as categories.

Some classifiers can also accept string labels directly, but encoding them explicitly gives you control over the class order and makes the output easier to inspect.

#### Inverse transformation

```python
le.inverse_transform([0, 1])
```

Result:

```text
['No', 'Yes']
```

Use it when converting predicted class ids back to labels for reporting.

---

### A8. Complete execution flow

The notebook's data flow is:

```text
customer.csv
   |
   v
drop age and gender (iloc[:, 2:])
   |
   v
review, education, purchased
   |
   v
train_test_split(test_size=0.3, random_state=34)
   |
   +---------------------------+
   |                           |
   v                           v
X_train (35, 2)             X_test (15, 2)
   |                           |
   v                           v
OrdinalEncoder.fit(X_train)  OrdinalEncoder.transform(X_test)
   |                           |
   v                           v
encoded train (35, 2)       encoded test (15, 2)
   |
   v
LabelEncoder.fit(y_train)
   |
   v
y_train (35,), y_test (15,)
```

No model is trained in this notebook. The output is a fully encoded feature matrix and target vector, ready for a model in a later step.

#### Saved outputs at a glance

| Operation | Saved result |
|---|---|
| Sample row | `60, Female, Poor, School, Yes` |
| Sliced columns | `review`, `education`, `purchased` |
| `X_train.shape` | `(35, 2)` |
| `oe.categories_` | `Poor/Average/Good`, `School/UG/PG` |
| `le.classes_` | `No`, `Yes` |
| `y_train` | 35 binary integers |
| Model trained? | No |

---

### A9. Why this is not one-hot encoding

One-hot encoding represents $k$ categories with $k$ binary columns (or $k-1$ to avoid the dummy trap). Ordinal encoding represents them with one column of integer codes.

For `review` with three levels:

| Value | Ordinal | One-hot (3 columns) | One-hot (2 columns, drop first) |
|---|---:|---|---|
| Poor | 0 | 1,0,0 | 0,0 |
| Average | 1 | 0,1,0 | 1,0 |
| Good | 2 | 0,0,1 | 0,1 |

Ordinal encoding assumes the levels are ordered and that a single coefficient can capture the effect of moving along that order. One-hot encoding makes no order assumption; each level gets its own coefficient. The trade-off is dimensionality and interpretability.

For `education`, ordinal encoding is reasonable if you believe the effect of schooling is monotone. If the effect may be non-monotone — for example, if `UG` has a different effect from both `School` and `PG` — one-hot encoding is safer.

For `review`, the same logic applies. The course uses ordinal encoding because the levels are explicitly ordered.

---

### A10. Leakage and the fit/transform rule

The notebook fits the encoders on the training data and transforms both sets. This is correct.

A leakage-prone alternative would be:

```python
# Wrong: learns categories from the full dataset
oe.fit(df[["review", "education"]])
X_train = oe.transform(X_train)
X_test = oe.transform(X_test)
```

If the full dataset contains a category that appears only in the test set, the encoder's `categories_` includes it. The training representation then depends on the test set. The effect may be small for a fixed category list, but the habit is wrong and becomes serious when the category list is learned automatically.

The same rule applies to `LabelEncoder`:

```python
# Wrong: target classes learned from train and test together
le.fit(y)
```

For binary targets the class set is usually identical, so the damage is often zero. But in a workflow with rare classes, fitting on the full target can hide a class that is absent from training. Fit on `y_train` only.

#### Cross-validation

In cross-validation, the encoder must be fitted inside each fold. A pipeline handles this:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("encoder", OrdinalEncoder(categories=[
        ["Poor", "Average", "Good"],
        ["School", "UG", "PG"],
    ])),
    ("model", LogisticRegression(max_iter=1000)),
])
```

The pipeline fits a fresh encoder on each fold's training portion.

---

### A11. Common mistakes and debugging

#### Mistake 1: omitting `categories` for ordinal data

```python
# Wrong: alphabetical order
oe = OrdinalEncoder()
```

`Average` becomes 0, `Good` becomes 1, `Poor` becomes 2. The real order is reversed. Always declare the order for ordinal features.

#### Mistake 2: using `LabelEncoder` on input features

`LabelEncoder` is for a one-dimensional target. It has no multi-column category specification. Using it on nominal inputs imposes a false order.

#### Mistake 3: one-hot encoding the target

The target should be a one-dimensional vector of class ids for standard classifiers. One-hot targets are used for some neural-network losses, but that is a different representation and requires a compatible model.

#### Mistake 4: fitting before splitting

Fit encoders on `X_train` and `y_train`, then transform test. Fitting on the full data leaks category information.

#### Mistake 5: losing column names

`oe.transform` returns a NumPy array. Reassigning `X_train = oe.transform(X_train)` discards the DataFrame's column names and index. This is fine for a model that only needs the array, but it makes debugging and plotting harder. Wrap the output in a DataFrame if you need labels.

#### Mistake 6: assuming codes are quantities

`Good = 2` does not mean "twice as good as Average = 1". The codes are ranks, not measurements. A linear model uses them as numbers, which is why the order must be meaningful and the spacing assumption should be considered.

#### Mistake 7: encoding high-cardinality IDs

An ID column with thousands of unique values should not be ordinal-encoded as if the IDs had a meaningful order. It may need to be dropped, hashed, target-encoded, or grouped. Day 27 discusses high-cardinality handling for one-hot encoding.

#### Mistake 8: unseen categories at inference time

The default `handle_unknown="error"` will raise an error if a new category appears. Decide explicitly how to handle unseen categories: collect more data, map to an "unknown" bucket, or use an encoder that supports a sentinel value.

---

### A12. Where this sits in the ML workflow

Encoding belongs after the train/test split and before model training:

```text
Raw data
   |
   v
Identify variable types
   |
   v
Separate X and y
   |
   v
Train/test split
   |
   v
Fit encoders on training data
   |
   v
Transform train and test
   |
   v
Train model
   |
   v
Evaluate
```

In a full preprocessing pipeline, different columns take different paths:

```text
ColumnTransformer
   ├── ordinal columns -> OrdinalEncoder
   ├── nominal columns -> OneHotEncoder
   ├── numeric columns -> SimpleImputer -> Scaler
   └── target -> LabelEncoder (outside X preprocessing)
```

Days 28–29 show how to assemble these steps into a `ColumnTransformer` and a `Pipeline`.

---

### A13. Interview preparation

#### Level 1 — Recall

**Q1. What is ordinal encoding?**

It maps ordered categories to integer codes according to a declared order. For example, `Poor -> 0`, `Average -> 1`, `Good -> 2`.

**Q2. What is label encoding?**

It maps target labels to integer class ids. For binary labels, `No -> 0`, `Yes -> 1`.

**Q3. Which encoder is used for input features?**

`OrdinalEncoder` for ordered inputs. For nominal inputs, one-hot encoding is usually preferred.

#### Level 2 — Explanation

**Q4. Why must the category order be declared for `OrdinalEncoder`?**

Because the default alphabetical order may not match the real order. If the codes are reversed, a model can learn the wrong direction of the effect, and the coefficient interpretation becomes misleading.

**Q5. Why is `LabelEncoder` not suitable for input features?**

It is designed for a one-dimensional target vector. It has no multi-column category specification and it imposes an order on whatever labels it sees. For nominal inputs, that order is false.

**Q6. Does ordinal encoding assume equal distances between levels?**

No. It preserves order but not equal spacing. The codes 0, 1, 2 do not imply that the difference between 0 and 1 equals the difference between 1 and 2. A linear model may still assume a linear effect per code step, which is a modelling assumption.

#### Level 3 — Application

**Q7. How would you encode `education` with levels School, UG, PG?**

Use `OrdinalEncoder(categories=[["School", "UG", "PG"]])` if the effect is believed to be monotone. If the effect may be non-monotone, use one-hot encoding instead.

**Q8. How would you encode a binary target?**

Use `LabelEncoder` on `y_train`, then transform `y_test`. Check `classes_` to confirm the mapping.

**Q9. What happens if the test set contains an unseen category?**

With the default `handle_unknown="error"`, the encoder raises an error. You can use `handle_unknown="use_encoded_value"` with a sentinel such as `-1`, but you must decide whether that sentinel is meaningful.

#### Level 4 — Analysis

**Q10. Why is fitting an encoder on the full dataset a form of leakage?**

The encoder's category list can depend on the test set. The training representation then uses information from the held-out data. The effect is usually small for a fixed category list, but it violates the separation between training and evaluation.

**Q11. What is the difference between ordinal encoding and one-hot encoding in terms of model flexibility?**

Ordinal encoding uses one column and assumes a monotone effect along the order. One-hot encoding uses multiple columns and allows each level to have its own effect. One-hot is more flexible but increases dimensionality and can create collinearity if all levels are kept.

**Q12. Why does `OrdinalEncoder` output floats by default?**

The default `dtype` is `float64`. You can request an integer dtype if the downstream model accepts it. Floats are the common numeric representation for scikit-learn estimators.

#### Level 5 — Design

**Q13. How would you build a preprocessing pipeline for a dataset with numeric, ordinal, and nominal columns?**

Use a `ColumnTransformer` with three branches: numeric columns through an imputer and scaler; ordinal columns through `OrdinalEncoder` with explicit categories; nominal columns through `OneHotEncoder`. Put the whole `ColumnTransformer` inside a `Pipeline` with the model.

**Q14. How would you handle an ordinal feature whose levels are not known in advance?**

Learn the categories from the training data with `categories="auto"` only if the alphabetical order is meaningful. Otherwise, define the order from domain knowledge. If the order can change over time, version the category list with the model.

**Q15. How would you handle a high-cardinality nominal feature?**

Group rare categories into an "other" bucket, use hashing, or use target encoding with proper cross-validation. One-hot encoding every level can create thousands of columns.

#### Level 6 — Evaluation

**Q16. How would you compare ordinal encoding with one-hot encoding for the same feature?**

Use the same cross-validation folds and the same model. Compare validation performance and consider interpretability and dimensionality. Do not choose based on training accuracy alone.

**Q17. Why might ordinal encoding hurt a linear model even when the order is correct?**

Because the model assumes a linear effect per code step. If the true effect is non-linear — for example, `Average` is much better than both `Poor` and `Good` — a single coefficient cannot capture it. One-hot encoding or adding non-linear features may help.

**Q18. What should you monitor after deployment?**

The distribution of categories, the frequency of unseen categories, and the encoded values' distribution. A new category or a shift in category frequencies may require retraining or a revised encoder.

#### Level 7 — Advanced

**Q19. Can `LabelEncoder` be used for multi-class targets?**

Yes. It assigns integer class ids in sorted order. The classifier treats them as class indices, not as quantities. For some loss functions, such as softmax cross-entropy, this is the standard representation.

**Q20. What is the difference between ordinal encoding and target encoding?**

Ordinal encoding maps categories to codes based on a declared order. Target encoding maps categories to a statistic of the target, such as the mean target value. Target encoding can capture more information but is prone to leakage and needs out-of-fold fitting.

**Q21. How does `OrdinalEncoder` handle unknown categories in a pipeline at inference time?**

By default it raises an error. In a pipeline, you can set `handle_unknown="use_encoded_value"` and `unknown_value=-1`. The downstream model must be able to handle the sentinel meaningfully. Alternatively, map unknown categories to an explicit "unknown" training category.

---

### A14. Exam-style questions with worked answers

#### Question 1

A feature has levels `Low`, `Medium`, `High`, `Very High`. You use `OrdinalEncoder()` without `categories`. What mapping does scikit-learn learn, and why is it wrong?

**Answer**

For string categories, the default `categories="auto"` sorts lexicographically. The order becomes:

```text
High -> 0
Low -> 1
Medium -> 2
Very High -> 3
```

This is not the real order `Low < Medium < High < Very High`. The codes now contradict the feature's meaning. A model may learn the wrong direction of the effect. The fix is to pass the ordered list explicitly:

```python
OrdinalEncoder(categories=[["Low", "Medium", "High", "Very High"]])
```

#### Question 2

A dataset has 35 training rows and 15 test rows. `OrdinalEncoder` is fitted on the training set. The test set contains a new category `Excellent`. What happens by default, and what are two possible fixes?

**Answer**

By default, `handle_unknown="error"`, so `transform` raises a `ValueError`. Two fixes:

1. Use `handle_unknown="use_encoded_value", unknown_value=-1` and ensure the model can handle `-1`.
2. Add an explicit `"Unknown"` category to the training data and the category list, so unseen values map to that category.

The choice depends on whether "unknown" is meaningful for the model and whether the new category is expected in production.

#### Question 3

Explain why `LabelEncoder` is appropriate for `purchased` but not for `review`.

**Answer**

`purchased` is the target: a one-dimensional vector of binary labels. `LabelEncoder` maps `No -> 0`, `Yes -> 1`, which is exactly what a classifier needs. `review` is an input feature with an order. `LabelEncoder` would operate on a single column but would learn an alphabetical order and would not accept a multi-column category specification. `OrdinalEncoder` is the right tool because it accepts a declared order.

#### Question 4

A model is trained on ordinal-encoded `review` with codes `Poor=0`, `Average=1`, `Good=2`. The learned coefficient is negative. Does that mean `Good` reviews reduce the predicted probability of purchase? Explain carefully.

**Answer**

Not necessarily. The coefficient applies to the encoded value. A negative coefficient means that, holding other features fixed, a one-code increase is associated with a lower linear predictor. But the model may also include other correlated features, and the coefficient is estimated jointly with them. Moreover, the coding is arbitrary up to the declared order; if the order were reversed, the sign would flip. The coefficient's sign is interpretable only in the context of the chosen coding and the other features. Correlation does not imply causation.

#### Question 5

Why is one-hot encoding often preferred for nominal features, even though it creates more columns?

**Answer**

One-hot encoding makes no order assumption. Each level gets its own coefficient, so the model can learn a different effect for each category. Ordinal encoding imposes a single ordered dimension, which is false for nominal data and may force the model to extrapolate an order that does not exist. The cost is higher dimensionality, potential collinearity, and reduced interpretability.

---

### A15. Revision summary

- Categorical variables can be nominal (no order) or ordinal (ordered).
- Models need numeric inputs, but the encoding must not invent false order.
- `OrdinalEncoder` is for ordered input features; declare `categories` explicitly.
- `LabelEncoder` is for a one-dimensional target vector; check `classes_`.
- Fit encoders on training data only; transform test data.
- Ordinal codes are ranks, not quantities.
- One-hot encoding is the usual choice for nominal inputs.
- Use a pipeline to keep encoding inside cross-validation.
- Unseen categories need an explicit policy.

---

## Part B — Advanced Reading and Reference

### B1. Internal representation

`OrdinalEncoder` stores:

- `categories_`: a list of arrays, one per feature, in column order.
- `n_features_in_`: the number of input columns.
- `feature_names_in_`: present when the input is a DataFrame with string column names.

`transform` looks up each value in the corresponding category array and returns its index. The operation is a dictionary lookup followed by array construction.

`LabelEncoder` stores:

- `classes_`: the sorted unique labels.
- `n_classes_`: the number of classes.

`transform` maps each label to its index in `classes_`.

Neither encoder learns a distance, a probability, or a target statistic. They are deterministic mappings.

---

### B2. Category order and model assumptions

Ordinal encoding is not just a convenience; it encodes a hypothesis. If the codes are `0,1,2`, a linear model represents the effect as

$$
\eta = \beta_0 + \beta_1 \cdot \text{code}.
$$

This implies that moving from 0 to 1 has the same effect as moving from 1 to 2. If the true effect is non-linear, the model is misspecified. One-hot encoding avoids this by giving each level its own coefficient:

$$
\eta = \beta_0 + \beta_1 \cdot \mathbf{1}[\text{level}=1] + \beta_2 \cdot \mathbf{1}[\text{level}=2].
$$

The trade-off is bias versus variance: ordinal encoding is more constrained (lower variance, potentially higher bias); one-hot is more flexible (higher variance, potentially lower bias).

---

### B3. Dummy-variable trap

For a nominal feature with $k$ levels, one-hot encoding with all $k$ columns creates perfect collinearity with an intercept:

$$
\sum_{j=1}^{k} \mathbf{1}[\text{level}=j] = 1.
$$

Dropping one level gives $k-1$ columns and avoids the trap. The dropped level becomes the reference category. This is discussed further in Day 27.

Ordinal encoding uses one column, so it does not create this particular collinearity. It creates a different assumption: a single ordered effect.

---

### B4. Handling unseen categories

`OrdinalEncoder` supports:

```python
OrdinalEncoder(
    categories=[...],
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)
```

The sentinel `-1` is a modelling choice. A linear model may interpret `-1` as an extreme low value, which is usually not what you want. A tree model may isolate it, but the split is arbitrary. A safer approach is to add an explicit `"Unknown"` category during training, so the model learns its effect from data.

For `LabelEncoder`, unseen labels at inference time are not supported directly. In a classification pipeline, you normally encode the target once during training and reuse the class list. If a new class appears in production, the model cannot predict it; you need to retrain with the new class.

---

### B5. Encoding and feature scaling

Ordinal codes are numeric, so a scaler can be applied to them. Is that a good idea?

- For distance-based models, scaling ordinal codes can prevent the encoded feature from dominating.
- For linear models, scaling changes the coefficient's units but not the model's fit (for unregularized models).
- For regularized models, scaling affects the penalty.

However, scaling ordinal codes does not make the spacing between levels equal. It only changes the numerical range. The ordinal assumption remains.

One-hot columns are already on a 0/1 scale; scaling them is usually unnecessary unless the model is sensitive to the number of columns or the regularization scale.

---

### B6. Encoding interactions

An ordinal code can be combined with other features:

```python
df["review_x_education"] = df["review_code"] * df["education_code"]
```

This creates an interaction feature that assumes the effect of one ordered feature depends on another. It is a modelling choice, not a default. For nominal features, interactions are usually created from one-hot columns or learned by the model.

---

### B7. High-cardinality categorical features

Ordinal encoding of a high-cardinality nominal feature is usually wrong because the codes have no meaningful order. One-hot encoding can create too many columns. Alternatives include:

- **Rare-category grouping:** keep frequent categories and map the rest to `"Other"`.
- **Hashing:** map categories to a fixed number of buckets using a hash function.
- **Target encoding:** replace each category with an out-of-fold statistic of the target.
- **Embeddings:** learn a dense representation in a neural network.

Each method has its own leakage and interpretability trade-offs. Day 27 covers rare-category grouping for one-hot encoding.

---

### B8. `OrdinalEncoder` versus `LabelEncoder` at a glance

| Property | `OrdinalEncoder` | `LabelEncoder` |
|---|---|---|
| Input shape | 2-D feature matrix | 1-D target vector |
| Category order | User-declared or learned | Learned, sorted |
| Multi-column | Yes | No |
| Typical use | Ordered input features | Target labels |
| Unknown handling | Configurable | Not designed for inference |
| Inverse transform | Yes | Yes |
| Output | 2-D array | 1-D array |

---

### B9. Advanced interview questions

**Q1. Why does `OrdinalEncoder` accept `categories` as a list of lists rather than a single list?**

Because it can encode multiple columns at once, and each column has its own category order. The outer list aligns with columns; each inner list defines that column's order.

**Q2. What happens if the number of inner lists does not match the number of columns?**

Scikit-learn raises a `ValueError`. The category specification must have one list per feature.

**Q3. Can `OrdinalEncoder` handle missing values?**

By default, missing values are treated as a category and must be included in the category list, or the encoder raises an error. A common approach is to impute before encoding or to include `np.nan` as an explicit category with `encoded_missing_value`. Check the scikit-learn version's supported parameters.

**Q4. Why is label encoding of a multi-class target acceptable?**

Because classifiers treat class ids as indices, not as ordered quantities. The loss function uses the id to select the corresponding class probability. The alphabetical order is arbitrary but does not impose a numerical distance in the loss.

**Q5. What is the difference between ordinal encoding and integer encoding by hand?**

A hand-built dictionary can achieve the same mapping, but `OrdinalEncoder` provides `fit`/`transform`, stores the category order, handles multiple columns, supports inverse transformation, and integrates with pipelines. It also gives consistent error handling for unknown categories.

**Q6. How would you encode a feature with a natural order but unknown spacing, such as education levels?**

Ordinal encoding is a reasonable baseline. If the effect may be non-linear, one-hot encoding or a monotone constraint can be used. The choice should be validated with cross-validation, not assumed.

**Q7. What is the risk of encoding an ID column with `LabelEncoder`?**

The model may learn spurious patterns based on the arbitrary ID order. IDs usually carry no generalisable information and can cause overfitting, especially with high cardinality. Drop them or treat them separately.

**Q8. How does encoding interact with feature importance?**

Ordinal encoding produces one importance per feature, but that importance conflates the order and the effect. One-hot encoding produces one importance per level, which can reveal non-monotone effects. The interpretation depends on the encoding.

---

### B10. Notebook coverage matrix

| Notebook area | Cells | What it teaches | Status |
|---|---:|---|---|
| Imports and theme | 4–5 | Setup | No output |
| Load and inspect data | 6 | Dataset and variable types | Saved sample |
| Slice columns | 8 | Select review, education, purchased | Saved table |
| Train/test split | 9–10 | 70/30 split, shape (35,2) | Saved shape |
| OrdinalEncoder construction | 12–13 | Declared categories | Object repr |
| Fit | 14 | Learn categories | Saved repr |
| Transform train/test | 15–16 | Codes and `categories_` | Saved categories |
| LabelEncoder | 18–23 | Target classes and transform | Saved classes |
| Encoded target | 24 | Binary array | Saved array |
| Experiment | 25 | Drop order and observe | Markdown |
| Key takeaways | 26 | Revision | Markdown |

No model is trained. The notebook ends after encoding.

---

### B11. Source fidelity and caveats

- Saved outputs come from the user's environment. They are reported as saved results, not independently reproduced.
- The notebook loads `customer.csv` from an absolute Windows path. The local copy is at `datasets/customer.csv`; the relative-path version is a portability correction.
- The notebook reassigns `X_train` and `X_test` to NumPy arrays, losing column names. This is preserved.
- The trailing code cells are empty; the notebook does not train a model or evaluate accuracy.
- The notebook's comments mention `gender` as nominal but do not encode it. Day 27 covers one-hot encoding for nominal features.
- No claims are made about model performance because no model was trained.

---

### B12. References

- scikit-learn documentation: `OrdinalEncoder`, `LabelEncoder`, `OneHotEncoder`.
- CampusX Day 26: Encoding Categorical Data — Ordinal Encoding and Label Encoding.
- Day 25 notes: min-max scaling and the fit/transform discipline.
- Day 27 notes: one-hot encoding and the dummy-variable trap.
- Day 28 notes: `ColumnTransformer`.
- Day 29 notes: pipelines and cross-validation.

---

### B13. Final connection

Ordinal encoding and label encoding are simple mappings, but they force you to answer a deeper question: **what does the number mean?** If the number represents rank, ordinal encoding is appropriate. If it represents a class id, label encoding is appropriate. If it represents nothing but a group, one-hot encoding is usually safer.

The next day applies this reasoning to nominal features and shows how one-hot encoding avoids inventing an order.
