# Day 29 — Machine Learning Pipelines A–Z

> **Course topic:** CampusX Day 29, *Machine Learning Pipelines A–Z*.
> **Personal notebooks:** five notebooks, kept separate:
> `DAY29_Workflow-Without-Pipelines.ipynb`,
> `DAY29_Titanic-Using-Pipeline.ipynb`,
> `DAY29_Sklearn-Pipelines-broken.ipynb`,
> `DAY29_Predict-Without-Pipeline.ipynb`,
> `DAY29_Pipeline-Export-Predict.ipynb`.
> **Dataset:** `datasets/train.csv` (Titanic, 891 rows).
> **Artifacts:** `models/{clf,ohe_sex,ohe_embarked}.pkl`, `pipe.pkl`.
> **Reference:** `reference_notebooks/` (four originals, do not edit).
>
> **Part A** is the main learning chapter and is sufficient on its own. **Part B** preserves internals, mathematics, coverage, and advanced reference material.

---

## Part A — Main Learning Chapter

### A1. Learning objectives

By the end of this chapter you should be able to:

1. Perform the complete manual Titanic preprocessing workflow and explain every block.
2. Explain why training-time preprocessing must be repeated identically at inference.
3. Diagnose the two bugs in the broken notebook.
4. Build a `Pipeline` of imputation, encoding, scaling, feature selection, and a classifier.
5. Explain `fit`, `transform`, and `predict` for a pipeline.
6. Trace the actual output of every stage, including the notebook's column-index bug.
7. Interpret holdout accuracy, cross-validation mean, and `GridSearchCV` results correctly.
8. Persist a fitted pipeline and predict from a raw row.
9. Compare manual inference with pipeline inference.
10. Explain why pipelines prevent leakage during cross-validation.

---

### A2. The problem: training and inference must agree

A model does not receive raw columns directly. It receives a numeric matrix produced by preprocessing:

```text
raw row -> impute -> encode -> scale -> select -> model -> prediction
```

During training, you can perform these steps manually. During inference, every step must be repeated with exactly the same learned parameters and the same column order. If one step differs, the model receives a different feature vector and the prediction can be silently wrong.

The manual workflow makes this visible:

- You must remember the imputer means.
- You must remember the encoder category order.
- You must remember which columns were dropped.
- You must concatenate the blocks in the same order.
- You must apply the same dtype conversions.
- You must repeat all of this for one new passenger.

A `Pipeline` stores the fitted preprocessing objects and the model together. One `fit` trains the whole chain; one `predict` replays the chain on raw rows. This is the central idea of Day 29.

---

### A3. The manual workflow

The first notebook, `DAY29_Workflow-Without-Pipelines.ipynb`, performs the workflow by hand.

#### Step 1: load and drop unused columns

```python
df = pd.read_csv(r"E:\MACHINE_LEARNING\DATA_SETS\train.csv")
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
```

The drop is **assigned back** to `df`. This detail matters; the broken notebook forgets it.

After the drop, the DataFrame has 8 columns:

```text
Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
```

The feature matrix has 7 columns:

```text
Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
```

#### Step 2: split

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    df.drop(columns=["Survived"]),
    df["Survived"],
    test_size=0.2,
    random_state=34,
)
```

Saved shapes:

```text
X_train: (712, 7)
X_test:  (179, 7)
```

The fixed `random_state=34` makes the split reproducible. The training set is used to learn imputation and encoding parameters; the test set is transformed with those parameters.

#### Step 3: impute missing values

The saved missing-value counts after the drop are:

| Column | Missing |
|---|---:|
| `Age` | 177 |
| `Embarked` | 2 |
| others | 0 |

The notebook uses:

```python
si_age = SimpleImputer()                       # default strategy="mean"
si_embarked = SimpleImputer(strategy="most_frequent")

X_train_age = si_age.fit_transform(X_train[["Age"]])
X_train_embarked = si_embarked.fit_transform(X_train[["Embarked"]])

X_test_age = si_age.transform(X_test[["Age"]])
X_test_embarked = si_embarked.transform(X_test[["Embarked"]])
```

`fit_transform` learns the training mean/mode and applies it. `transform` reuses the learned values on test data. Fitting on the test set would leak information.

The output blocks are:

```text
X_train_age:      (712, 1)
X_train_embarked: (712, 1)
X_test_age:       (179, 1)
X_test_embarked:  (179, 1)
```

#### Step 4: one-hot encode

```python
ohe_sex = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
ohe_embarked = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

X_train_sex = ohe_sex.fit_transform(X_train[["Sex"]])
X_train_embarked = ohe_embarked.fit_transform(X_train_embarked)

X_test_sex = ohe_sex.transform(X_test[["Sex"]])
X_test_embarked = ohe_embarked.transform(X_test_embarked)
```

`sparse_output=False` returns a dense NumPy array. `handle_unknown="ignore"` maps an unseen category to an all-zero row instead of raising an error.

`Sex` has 2 categories, so the block has 2 columns. `Embarked` has 3 categories (`C`, `Q`, `S`), so the block has 3 columns.

#### Step 5: keep the remaining numeric columns

```python
X_train_rem = X_train.drop(columns=["Sex", "Age", "Embarked"])
X_test_rem = X_test.drop(columns=["Sex", "Age", "Embarked"])
```

The remaining columns are:

```text
Pclass, SibSp, Parch, Fare
```

That is 4 columns.

#### Step 6: concatenate in a fixed order

```python
X_train_trans = np.concatenate(
    (X_train_rem, X_train_age, X_train_sex, X_train_embarked),
    axis=1,
)
X_test_trans = np.concatenate(
    (X_test_rem, X_test_age, X_test_sex, X_test_embarked),
    axis=1,
)
```

The saved test shape is:

```text
(179, 10)
```

The width is:

$$
4 \text{ (remaining)} + 1 \text{ (age)} + 2 \text{ (sex)} + 3 \text{ (embarked)} = 10.
$$

The output order is:

| Position | Block | Meaning |
|---:|---|---|
| 0 | `X_train_rem` | `Pclass` |
| 1 | `X_train_rem` | `SibSp` |
| 2 | `X_train_rem` | `Parch` |
| 3 | `X_train_rem` | `Fare` |
| 4 | `X_train_age` | imputed `Age` |
| 5 | `X_train_sex` | `Sex=female` |
| 6 | `X_train_sex` | `Sex=male` |
| 7 | `X_train_embarked` | `Embarked=C` |
| 8 | `X_train_embarked` | `Embarked=Q` |
| 9 | `X_train_embarked` | `Embarked=S` |

The exact category order is learned from the training data. The table shows the usual order for this dataset.

#### Step 7: train and evaluate

```python
clf = DecisionTreeClassifier()
clf.fit(X_train_trans, Y_train)
y_pred = clf.predict(X_test_trans)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test, y_pred)
```

Saved result:

```text
0.8100558659217877
```

This is 145 correct predictions out of 179:

$$
\frac{145}{179} = 0.8100558659217877.
$$

The decision tree uses the default hyperparameters, including unlimited depth until purity or the minimum split constraints. The notebook does not tune it.

#### Step 8: persist the manual artifacts

```python
os.makedirs("models", exist_ok=True)

with open("models/ohe_embarked.pkl", "wb") as f:
    pickle.dump(ohe_embarked, f)
with open("models/ohe_sex.pkl", "wb") as f:
    pickle.dump(ohe_sex, f)
with open("models/clf.pkl", "wb") as f:
    pickle.dump(clf, f)
```

Only the two encoders and the classifier are saved. The age and embarked imputers are **not** saved. This matters for manual inference: the saved artifacts cannot handle a new row with missing `Age` or `Embarked`.

The manual workflow works, but it requires the user to remember all of these steps. The next sections show what happens when one detail is forgotten.

---

### A4. Why the manual workflow is fragile

The manual workflow has no single object that owns the preprocessing plan. Every inference call must reconstruct the plan from memory:

- Which columns were dropped?
- Which imputer was used for which column?
- What were the learned means and modes?
- What was the category order?
- What was the concatenation order?
- What dtype did the model see?

A small mistake can produce a wrong prediction without an error. For example, if the age block and the fare block are swapped, both are numeric and the model still runs, but the feature meanings are wrong.

The pipeline does not make the modeling problem easier. It makes the **bookkeeping** reliable.

---

### A5. The broken notebook: two preserved bugs

`DAY29_Sklearn-Pipelines-broken.ipynb` is a near-duplicate of the manual workflow with two deliberate mistakes. It is kept as a debugging lesson.

#### Bug 1: the drop is not assigned

```python
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
```

`DataFrame.drop` returns a new DataFrame by default. It does not modify `df` unless you assign the result or use `inplace=True`. Because the result is discarded, `df` still contains `PassengerId`, `Name`, `Ticket`, and `Cabin`.

The split therefore keeps the `Name` strings. The concatenated matrix contains a string column, and the tree cannot consume it. The saved error is:

```text
ValueError: could not convert string 'Sage, Mr. Douglas Bullen'
```

The error appears during `clf.fit(X_train_trans, Y_train)`. The tree's input validation tries to convert the string column to a number and fails.

#### Bug 2: the train and test blocks are swapped

```python
X_test_trans = np.concatenate((X_train_rem, X_train_age, X_train_sex, X_train_embarked), axis=1)
X_train_trans = np.concatenate((X_test_rem, X_test_age, X_test_sex, X_test_embarked), axis=1)
```

`X_test_trans` is built from the **training** blocks, and `X_train_trans` is built from the **test** blocks. The names are swapped.

This means `clf.fit` receives 179 rows but `Y_train` has 712 labels. A reproduced run with bug 1 fixed but bug 2 still present gives:

```text
ValueError: Number of labels=712 does not match number of samples=179
```

The string error in the saved run occurs first and masks the row-count mismatch.

#### The debugging lesson

Fix the drop first:

```python
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
```

Then fix the block pairing:

```python
X_train_trans = np.concatenate(
    (X_train_rem, X_train_age, X_train_sex, X_train_embarked), axis=1
)
X_test_trans = np.concatenate(
    (X_test_rem, X_test_age, X_test_sex, X_test_embarked), axis=1
)
```

Check the shapes after every concatenation. A pipeline makes this pairing automatic because it stores the fitted transformations and applies them in order.

---

### A6. What a `Pipeline` is

A `Pipeline` is an ordered sequence of steps. Every step except the last must implement `fit` and `transform`. The last step is an estimator that implements `fit` and usually `predict`.

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("step_1", transformer_1),
    ("step_2", transformer_2),
    ("model", estimator),
])
```

Each list element is a `(name, object)` tuple:

| Part | Meaning |
|---|---|
| `name` | Unique string used to access the step |
| `object` | An **unfitted** transformer or estimator |

Pass objects, not calls:

```python
# Correct: create an unfitted object
Pipeline([("model", DecisionTreeClassifier())])

# Wrong: this is a fitted model, not a fresh estimator
Pipeline([("model", DecisionTreeClassifier().fit(X, y))])
```

A pipeline is itself an estimator. It has `fit`, `predict`, `score`, and `get_params`.

#### What `fit` does

```python
pipe.fit(X_train, y_train)
```

For every step except the last:

1. Call `fit_transform` on the current data.
2. Pass the output to the next step.

For the last step:

1. Call `fit` on the transformed data and the target.

#### What `predict` does

```python
pipe.predict(X_test)
```

For every step except the last:

1. Call `transform` on the current data.
2. Pass the output to the next step.

For the last step:

1. Call `predict`.

This is why the same preprocessing is applied at training and inference time. The pipeline owns the fitted objects.

#### Accessing steps

```python
pipe.named_steps["trf1"]
pipe.named_steps["trf5"]
```

`pipe.steps` is the ordered list of `(name, object)` pairs.

---

### A7. The five-stage pipeline in the notebook

`DAY29_Titanic-Using-Pipeline.ipynb` builds the pipeline in five stages.

#### Stage 1: imputation

```python
trf1 = ColumnTransformer([
    ("impute_age", SimpleImputer(), [2]),
    ("impute_embarked", SimpleImputer(strategy="most_frequent"), [6]),
], remainder="passthrough")
```

The selectors `[2]` and `[6]` are **integer positions** in the 7-column feature matrix:

```text
0 Pclass, 1 Sex, 2 Age, 3 SibSp, 4 Parch, 5 Fare, 6 Embarked
```

`remainder="passthrough"` keeps the other five columns. The output width is:

$$
1 + 1 + 5 = 7.
$$

The output order is:

| Output index | Source | Content |
|---:|---|---|
| 0 | `impute_age` | imputed `Age` |
| 1 | `impute_embarked` | imputed `Embarked` |
| 2 | remainder | `Pclass` |
| 3 | remainder | `Sex` |
| 4 | remainder | `SibSp` |
| 5 | remainder | `Parch` |
| 6 | remainder | `Fare` |

This order is the key to the next stage.

#### Stage 2: one-hot encoding

```python
trf2 = ColumnTransformer([
    ("ohe_sex_embarked",
     OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
     [1, 6])
], remainder="passthrough")
```

The name says `sex_embarked`, but the selectors `[1, 6]` act on the **output of `trf1`**, not on the original DataFrame.

From the table above:

- Output index 1 is `Embarked`.
- Output index 6 is `Fare`.
- `Sex` is at output index 3.

So `trf2` actually one-hot encodes **`Embarked` and `Fare`**, not `Sex` and `Embarked`. This is a column-index bug in the notebook, and the same pattern appears in the reference notebook.

The width calculation:

- `Embarked` has 3 categories: 3 columns.
- `Fare` has 228 distinct values in the 712-row training split: 228 columns.
- Remainder: 5 columns (`Age`, `Pclass`, `Sex`, `SibSp`, `Parch`).

$$
3 + 228 + 5 = 236.
$$

A reproduced run gives `trf2` output shape `(712, 236)`. This shape is not saved in the notebook; it is a supplementary calculation.

The output order is:

```text
Embarked_C, Embarked_Q, Embarked_S,
Fare_<value_1>, ..., Fare_<value_228>,
Age, Pclass, Sex, SibSp, Parch
```

`Fare` is a continuous numeric column. Treating every distinct fare as a category creates a very wide, extremely sparse representation. Most fare dummies are 1 for a handful of passengers and 0 for everyone else.

#### Stage 3: scaling a slice

```python
trf3 = ColumnTransformer([
    ("scale", MinMaxScaler(), slice(0, 10))
])
```

There is no `remainder` argument, so the default is `remainder="drop"`.

The slice `slice(0, 10)` selects the first 10 columns of the 236-column `trf2` output:

- `Embarked_C`, `Embarked_Q`, `Embarked_S` (3 columns).
- The first 7 fare dummies (7 columns).

The remaining 226 columns are dropped. The output is `(712, 10)`.

This is a second serious issue: `trf3` does not scale all features. It scales only the first 10 and discards the rest, including `Sex`, `Age`, `Pclass`, `SibSp`, and `Parch`.

#### Stage 4: feature selection

```python
trf4 = SelectKBest(score_func=chi2, k=8)
```

`SelectKBest` ranks the 10 scaled columns with the chi-square statistic and keeps the top 8.

The fitted support in the saved pipeline is:

```text
[0, 2, 3, 5, 6, 7, 8, 9]
```

Those positions correspond to:

- `Embarked_C` (index 0)
- `Embarked_S` (index 2)
- six of the first seven fare dummies (indices 3, 5, 6, 7, 8, 9)

`Embarked_Q` (index 1) and one fare dummy (index 4) are not selected.

The model therefore never sees `Sex`, `Age`, `Pclass`, `SibSp`, or `Parch`. It sees only port indicators and a few exact-fare indicators. This is not the feature set the notebook's comments describe.

#### Stage 5: classifier

```python
trf5 = DecisionTreeClassifier()
```

The default tree is fitted on the 8 selected columns.

#### Assembling the pipeline

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("trf1", trf1),
    ("trf2", trf2),
    ("trf3", trf3),
    ("trf4", trf4),
    ("trf5", trf5),
])
```

The complete data flow is:

```text
X_train (712, 7)
   |
   v
trf1: impute Age and Embarked, keep 5 remainder columns
   |
   v
(712, 7)
   |
   v
trf2: one-hot encode Embarked and Fare, keep 5 remainder columns
   |
   v
(712, 236)
   |
   v
trf3: MinMax scale first 10 columns, drop the other 226
   |
   v
(712, 10)
   |
   v
trf4: SelectKBest(chi2, k=8)
   |
   v
(712, 8)
   |
   v
trf5: DecisionTreeClassifier
```

`pipe.fit(X_train, Y_train)` runs this chain and stores every fitted transformer.

---

### A8. The correct minimal index fix

The pipeline's mechanics are correct; its **column selectors** are not. To encode `Sex` and `Embarked` after `trf1`, the selectors must be the output positions of those columns:

```python
trf2 = ColumnTransformer([
    ("ohe_sex_embarked",
     OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
     [1, 3])   # Embarked at 1, Sex at 3
], remainder="passthrough")
```

Then the width is:

- `Embarked`: 3 columns.
- `Sex`: 2 columns.
- Remainder: 5 columns (`Age`, `Pclass`, `SibSp`, `Parch`, `Fare`).

$$
3 + 2 + 5 = 10.
$$

Now `trf3`'s `slice(0, 10)` covers the entire output, so no columns are dropped:

```python
trf3 = ColumnTransformer([
    ("scale", MinMaxScaler(), slice(0, 10))
])
```

`SelectKBest(chi2, k=8)` then selects 8 of the 10 columns. This is the structure the notebook intended.

Scaling one-hot dummy columns is not harmful, but it is often unnecessary. A more conventional plan is to scale only the numeric columns and leave the dummy columns unscaled. For a tree, scaling is not required at all; it matters for distance-based and gradient-based models.

#### A more robust alternative: use names, not positions

Positional indices are fragile because every upstream transformation can change the layout. A safer design is to keep DataFrames with named columns or to use `ColumnTransformer` with explicit named selectors at each stage. Scikit-learn's `set_output(transform="pandas")` can preserve column names for supported transformers:

```python
trf1.set_output(transform="pandas")
trf2.set_output(transform="pandas")
```

This is not used in the notebook, but it is the modern way to reduce index bookkeeping. The Day 28 lesson applies: declare the columns explicitly and verify `get_feature_names_out()`.

---

### A9. Results: manual versus pipeline

The saved holdout results are:

| Workflow | Test accuracy |
|---|---:|
| Manual workflow | `0.8100558659217877` |
| Five-stage pipeline as written | `0.6759776536312849` |

The pipeline is **not** a drop-in improvement here. The pipeline automates the bookkeeping, but it faithfully executes the wrong column selectors. The manual workflow uses `Sex`, `Age`, `Pclass`, `SibSp`, `Parch`, and `Fare`; the pipeline's final 8 features are port and fare indicators only.

The notebook does not run an ablation that isolates the cause of the difference. The lower pipeline score is consistent with the feature-selection bug, but the saved results alone do not prove that the bug is the only cause. The correct conclusion is:

- The manual workflow and the pipeline are not evaluating the same feature set.
- The pipeline's feature set is much narrower and omits strong predictors such as `Sex`.
- The scores should not be read as "pipelines are worse than manual workflows."

This is an important practical lesson: **a pipeline prevents execution drift, but it does not validate your feature engineering.**

---

### A10. Cross-validation with a pipeline

The notebook evaluates the pipeline with:

```python
from sklearn.model_selection import cross_val_score

cross_val_score(pipe, X_train, Y_train, cv=5, scoring="accuracy").mean()
```

Saved result:

```text
0.6263862897665715
```

The five-fold mean is:

$$
\text{CV mean} = \frac{1}{5}\sum_{k=1}^{5} \text{accuracy}_k.
$$

`cross_val_score` clones the pipeline for each fold. Inside each fold:

1. The pipeline is fitted on the fold's training rows.
2. The imputer learns the fold's training mean/mode.
3. The encoders learn the fold's training categories.
4. The scaler learns the fold's training min/max.
5. The selector ranks the fold's training features.
6. The tree is fitted.
7. The pipeline scores the fold's validation rows.

No validation row influences the preprocessing. This is the correct way to evaluate a preprocessing pipeline.

The holdout accuracy `0.6760` and the CV mean `0.6264` are different protocols:

- The holdout score is one split of 179 test rows.
- The CV mean is the average of five validation folds on the 712 training rows.

They are not expected to be equal. The CV mean is usually the more stable estimate, but it has its own variance.

---

### A11. Pipeline exploration

The notebook inspects the fitted imputer:

```python
pipe.named_steps["trf1"].transformers_[1][1].statistics_
```

Saved output:

```text
array(['S'], dtype=object)
```

This is the most frequent `Embarked` value learned from the training data. The path is:

- `named_steps["trf1"]`: the first `ColumnTransformer`.
- `.transformers_[1]`: the second fitted transformer triple, `("impute_embarked", ...)`.
- `[1]`: the transformer object inside that triple.
- `.statistics_`: the learned imputation statistic.

For a `SimpleImputer(strategy="most_frequent")`, `statistics_` contains the mode. For the default mean strategy, it would contain the mean.

This kind of inspection is how you verify that a pipeline learned what you intended.

---

### A12. Grid search

The notebook tunes the tree's maximum depth:

```python
params = {"trf5__max_depth": [1, 2, 3, 4, 5]}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipe, params, cv=5, scoring="accuracy")
grid.fit(X_train, Y_train)
```

The parameter name uses the double-underscore convention:

```text
trf5__max_depth
  |       |
  |       +-- parameter of the step
  +---------- step name in the pipeline
```

Saved result:

```text
grid.best_score_ = 0.6263862897665715
```

This is exactly the same as the untuned CV mean. The saved notebook does not show the individual grid results.

A reproduced run with the same data and pipeline shows that all five depths produce the same mean CV score:

```text
max_depth=1 -> 0.6263862897665715
max_depth=2 -> 0.6263862897665715
max_depth=3 -> 0.6263862897665715
max_depth=4 -> 0.6263862897665715
max_depth=5 -> 0.6263862897665715
```

The search selects `max_depth=1` because it is the first parameter in the grid when scores tie. This is not evidence that depth 1 is uniquely best. The selected feature set is so narrow that the tree's predictions do not change across these depths in cross-validation.

The lesson: always inspect `grid.cv_results_`, not just `best_score_`. A tie can make the "best" parameter arbitrary.

---

### A13. Exporting the pipeline

The notebook saves the whole pipeline:

```python
import pickle
pickle.dump(pipe, open("pipe.pkl", "wb"))
```

`pipe.pkl` contains:

- The fitted imputers.
- The fitted one-hot encoders.
- The fitted scaler.
- The fitted feature selector.
- The fitted decision tree.
- The step names and the pipeline structure.

This is the deployment artifact. The model and its preprocessing cannot drift apart because they are stored in one object.

The export notebook loads it:

```python
import pickle
import numpy as np

pipe = pickle.load(open("pipe.pkl", "rb"))

test_input = np.array(
    [3, "male", 3.5, 0, 2, 18.2458, "S"],
    dtype=object,
).reshape(1, 7)

pipe.predict(test_input)
```

Saved output:

```text
array([0])
```

The raw row has the same 7 columns as the training features, in the same order:

```text
Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
```

The pipeline performs imputation, encoding, scaling, selection, and prediction internally.

The saved output also contains two warnings:

```text
UserWarning: X does not have valid feature names, but SimpleImputer was fitted with feature names
```

The imputers were fitted on DataFrames with named columns, but `predict` receives a NumPy array with no column names. The warning is harmless for this pipeline because the columns are in the correct positional order. To silence it, pass a DataFrame with the training column names.

---

### A14. Manual inference and its failure

`DAY29_Predict-Without-Pipeline.ipynb` performs inference without the pipeline.

#### Step 1: load the three artifacts

```python
ohe_embarked = pickle.load(open(".../models/ohe_embarked.pkl", "rb"))
ohe_sex = pickle.load(open(".../models/ohe_sex.pkl", "rb"))
clf = pickle.load(open(".../models/clf.pkl", "rb"))
```

The notebook uses absolute paths under `BASICS/models/`. The copies in `Day_29/models/` are the local artifacts.

#### Step 2: create one raw row

```python
test_input = np.array(
    [3, "female", 34.5, 0, 2, 15.2458, "C"],
    dtype=object,
).reshape(1, 7)
```

The row means:

| Position | Column | Value |
|---:|---|---|
| 0 | `Pclass` | 3 |
| 1 | `Sex` | female |
| 2 | `Age` | 34.5 |
| 3 | `SibSp` | 0 |
| 4 | `Parch` | 2 |
| 5 | `Fare` | 15.2458 |
| 6 | `Embarked` | C |

#### Step 3: encode by hand

```python
test_input_sex = ohe_sex.transform(test_input[:, 1].reshape(1, 1))
test_input_embarked = ohe_embarked.transform(test_input[:, -1].reshape(1, 1))
test_input_age = test_input[:, 2].reshape(1, 1)
```

#### Step 4: reassemble in the training order

```python
test_input_transformed = np.concatenate(
    (test_input[:, [0, 3, 4, 5]], test_input_age, test_input_sex, test_input_embarked),
    axis=1,
)
```

The order matches the manual training concatenation:

```text
Pclass, SibSp, Parch, Fare, Age, Sex dummies, Embarked dummies
```

#### Step 5: predict

```python
clf.predict(test_input_transformed)
```

Saved result:

```text
The Kernel crashed while executing code in the current cell or a previous cell.
```

The crash is preserved as the lesson. The assembled array is `object` dtype because the raw row is `object` and the age slice remains `object`. Mixing object numeric values with float encoder outputs is fragile.

#### A source-fidelity warning about the saved artifacts

The visible manual notebook creates 10 training columns:

```text
4 remainder + 1 age + 2 sex + 3 embarked = 10
```

But the saved `models/*.pkl` are inconsistent with that run:

- `ohe_sex.drop_idx_` is `[0]`, so it drops the first category and outputs 1 column.
- `ohe_embarked.drop_idx_` is `[0]`, so it outputs 2 columns.
- `clf.n_features_in_` is 8.

Those artifacts were evidently overwritten by a later run with `drop="first"` encoders. With them, the manual inference row assembles 8 columns, matching the saved classifier. A reproduced run with `.astype(float)` on the assembled row returns:

```text
array([0])
```

So the saved kernel crash is not a mathematical impossibility; it is a runtime/dtype fragility in the manual path. This distinction matters:

- The saved result is a kernel crash.
- The reproduced result with an explicit float cast is `[0]`.
- The manual path is fragile because the user must control dtype, column order, and missing-value handling by hand.

The saved manual artifacts also omit the imputers. If a new row has a missing `Age` or `Embarked`, the manual path has no saved object to fill it. The pipeline includes those imputers.

---

### A15. Manual versus pipeline inference

| Aspect | Manual inference | Pipeline inference |
|---|---|---|
| Artifacts | 3 pickles: two encoders and a classifier | 1 pickle: the whole pipeline |
| Preprocessing | Reconstructed by hand | Replayed by the fitted pipeline |
| Column order | User must remember | Stored in the pipeline |
| Dtype handling | User must cast correctly | Pipeline controls the transformations |
| Missing values | No saved imputers in this notebook | Imputers are included |
| Code size | Many cells | One `predict` call |
| Failure mode | Silent wrong order or dtype crash | Usually a clear schema/validation error |

The pipeline is not magic; it can still fail if the raw input schema changes. But it removes the most common source of drift: manually repeating the training transformations.

---

### A16. Why pipelines prevent leakage in cross-validation

The wrong pattern:

```python
X_transformed = preprocess(X)          # sees all rows
scores = cross_val_score(model, X_transformed, y, cv=5)
```

The preprocessing has already seen the validation rows. The imputer mean, encoder categories, and scaler range include information from the validation folds. The CV score is optimistic.

The correct pattern:

```python
pipe = Pipeline([
    ("preprocess", preprocess),
    ("model", model),
])
scores = cross_val_score(pipe, X, y, cv=5)
```

Inside each fold, the pipeline fits a fresh preprocessing object on the fold's training rows only. The validation rows are transformed with that fitted object. No validation information leaks into the preprocessing.

The same principle applies to `GridSearchCV`: pass the pipeline, not pre-transformed data.

---

### A17. Common mistakes and debugging

#### Mistake 1: forgetting to assign `drop`

```python
# Wrong: result discarded
df.drop(columns=["Name"])

# Correct
df = df.drop(columns=["Name"])
```

#### Mistake 2: swapping train and test blocks

Always pair `X_train_*` blocks with `X_train_trans` and `X_test_*` blocks with `X_test_trans`. Check the row counts.

#### Mistake 3: using positional indices after an upstream transformation

After `trf1`, the column order is not the original order. In this notebook, index 1 is `Embarked`, not `Sex`. Always inspect `get_feature_names_out()` or the transformer's output order.

#### Mistake 4: one-hot encoding a continuous column

`Fare` has 228 distinct training values. One-hot encoding it creates 228 columns and a very sparse representation. Use binning, target encoding, or a numeric transformation instead.

#### Mistake 5: dropping columns accidentally with `trf3`

`ColumnTransformer` defaults to `remainder="drop"`. A transformer that selects only `slice(0, 10)` discards the other columns. If you want to keep them, set `remainder="passthrough"` or select them explicitly.

#### Mistake 6: scaling one-hot columns unnecessarily

Scaling dummy columns is usually harmless but rarely useful. For trees, scaling is irrelevant. For linear models, scale numeric columns and leave dummies as 0/1 unless you have a reason.

#### Mistake 7: fitting on test data

Use `fit_transform` on training data and `transform` on test data. The pipeline enforces this automatically when used correctly.

#### Mistake 8: comparing holdout accuracy and CV mean as if they were the same

The holdout score uses one test split. The CV mean uses several validation folds. They answer different questions and have different variance.

#### Mistake 9: treating `best_score_` as proof of tuning gain

Inspect `cv_results_`. If all parameter values tie, the selected "best" parameter is arbitrary.

#### Mistake 10: pickling across incompatible versions

A pickle can fail to load if the scikit-learn version or the code path changes. Pin versions and keep the training code with the artifact.

---

### A18. Where this sits in the ML workflow

The pipeline is the assembly layer for the preprocessing block:

```text
Raw data
   |
   v
Clean and select columns
   |
   v
Train/test split
   |
   v
Pipeline
   |-- impute
   |-- encode
   |-- scale
   |-- select
   |-- model
   |
   v
Cross-validation / GridSearch
   |
   v
Persist the fitted pipeline
   |
   v
Predict raw rows in production
```

It connects the earlier days:

- Day 24–25: scaling becomes a pipeline stage.
- Day 26–27: encoding becomes a pipeline stage.
- Day 28: `ColumnTransformer` becomes the first stage.
- Day 29: the stages are assembled, validated, tuned, and exported.
- Day 30: `FunctionTransformer` and distribution transforms slot into the same structure.

---

### A19. Interview preparation

#### Level 1 — Recall

**Q1. What is a `Pipeline`?**

An ordered sequence of transformers ending in an estimator, stored as one object. `fit` trains every stage; `predict` replays the transformations and then predicts.

**Q2. What must every non-final step implement?**

`fit` and `transform`. The final step must implement `fit` and usually `predict`.

**Q3. What does `pipe.named_steps` give you?**

A dictionary-like access to the steps by name.

**Q4. What is `pipe.pkl`?**

A pickled fitted pipeline containing preprocessing and the model.

#### Level 2 — Explanation

**Q5. Why is a pipeline safer than manual preprocessing?**

It stores the fitted transformations and applies them in the same order at inference, preventing drift in imputation values, category order, column order, and dtype handling.

**Q6. Why does `cross_val_score` need the pipeline rather than pre-transformed data?**

If you pre-transform all data, the preprocessing sees the validation folds. The pipeline fits preprocessing inside each fold, so validation data never influences the learned parameters.

**Q7. What does `trf5__max_depth` mean in a parameter grid?**

The `max_depth` parameter of the step named `trf5`.

#### Level 3 — Application

**Q8. How would you build imputation, encoding, scaling, selection, and a classifier in one pipeline?**

```python
pipe = Pipeline([
    ("impute", ColumnTransformer([
        ("age", SimpleImputer(), [2]),
        ("embarked", SimpleImputer(strategy="most_frequent"), [6]),
    ], remainder="passthrough")),
    ("encode", ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [1, 3]),
    ], remainder="passthrough")),
    ("scale", MinMaxScaler()),
    ("select", SelectKBest(chi2, k=8)),
    ("model", DecisionTreeClassifier()),
])
```

**Q9. How would you tune the tree depth?**

```python
GridSearchCV(pipe, {"model__max_depth": [1, 2, 3, 4, 5]}, cv=5)
```

**Q10. How would you save and load the whole workflow?**

```python
with open("pipe.pkl", "wb") as f:
    pickle.dump(pipe, f)

with open("pipe.pkl", "rb") as f:
    loaded = pickle.load(f)

loaded.predict(raw_row)
```

#### Level 4 — Analysis

**Q11. Why does the notebook's pipeline perform worse than the manual workflow?**

The pipeline's `trf2` selects indices `[1, 6]` after `trf1`, which are `Embarked` and `Fare`, not `Sex` and `Embarked`. It one-hot encodes a high-cardinality continuous column, then `trf3` keeps only the first 10 columns and drops the rest. The final model never sees `Sex`, `Age`, `Pclass`, `SibSp`, or `Parch`. The pipeline is executing a different, much narrower feature set.

**Q12. Why does `trf3` drop columns?**

`ColumnTransformer` defaults to `remainder="drop"`. Since `trf3` selects only `slice(0, 10)`, every column outside that slice is dropped. This is not a pipeline bug; it is the declared behavior.

**Q13. Why does `GridSearchCV.best_score_` equal the untuned CV mean?**

All five `max_depth` values produce the same cross-validated score on the selected features. The search picks the first parameter in the grid when scores tie.

#### Level 5 — Design

**Q14. How would you avoid positional-index bugs in a complex pipeline?**

Use named DataFrame columns at each stage, `set_output(transform="pandas")` where supported, explicit `get_feature_names_out()` checks, and assertions on shapes after each stage.

**Q15. How would you handle missing values at inference time?**

Keep the imputers inside the pipeline. The manual path in this notebook does not save them, so it cannot handle missing values.

**Q16. How would you deploy this model?**

Persist the fitted pipeline, validate the input schema, load the pipeline in the serving process, and call `predict` on raw rows. Pin the scikit-learn version used to create the pickle.

#### Level 6 — Evaluation

**Q17. How would you compare the manual and pipeline workflows fairly?**

Use the same feature set, the same split, the same model, and the same evaluation protocol. Here the feature sets differ, so the scores are not a fair comparison of pipeline versus manual execution.

**Q18. Why is the holdout accuracy higher than the CV mean?**

They use different data and different protocols. The holdout score is one split; the CV mean averages five folds on the training set. A single split can be easier or harder than the average fold.

**Q19. What would you inspect in `grid.cv_results_`?**

Mean and standard deviation of test scores for each parameter, fit time, score time, and the rank of each configuration. This reveals ties and instability.

#### Level 7 — Advanced

**Q20. How does `Pipeline` handle `fit_transform` for intermediate steps?**

It calls `fit_transform` on the first step with the original data, then `fit_transform` on each subsequent intermediate step with the previous output, and finally `fit` on the last estimator.

**Q21. What is the difference between `Pipeline` and `make_pipeline`?**

`Pipeline` requires explicit names. `make_pipeline` generates names automatically from the class names. The mechanics are the same, but parameter-grid names differ.

**Q22. What is the difference between `ColumnTransformer` and `make_column_transformer`?**

`ColumnTransformer` takes `(name, transformer, columns)` triples. `make_column_transformer` takes `(transformer, columns)` pairs and generates names. Both perform per-column transformations; neither is a sequential pipeline.

**Q23. What are the risks of pickling a pipeline?**

Version incompatibility, missing custom classes, security risks from loading untrusted pickles, and silent behavior changes after library upgrades. Prefer versioned environments and integrity checks.

---

### A20. Exam-style questions with worked answers

#### Question 1

A manual workflow concatenates 4 remainder columns, 1 age column, 2 sex dummies, and 3 embarked dummies. What is the output width?

**Answer**

$$
4 + 1 + 2 + 3 = 10.
$$

The saved test shape is `(179, 10)`.

#### Question 2

After `trf1`, the output columns are `Age`, `Embarked`, `Pclass`, `Sex`, `SibSp`, `Parch`, `Fare`. A `ColumnTransformer` then selects `[1, 6]` for one-hot encoding. Which original features are encoded?

**Answer**

Index 1 is `Embarked`; index 6 is `Fare`. The transformer encodes `Embarked` and `Fare`, not `Sex` and `Embarked`. `Sex` is at index 3.

#### Question 3

The training split has 3 embarked categories and 228 distinct fares. `trf2` one-hot encodes both and passes through 5 columns. What is the output width?

**Answer**

$$
3 + 228 + 5 = 236.
$$

#### Question 4

Explain why `trf3 = ColumnTransformer([("scale", MinMaxScaler(), slice(0, 10))])` drops 226 columns.

**Answer**

`ColumnTransformer` defaults to `remainder="drop"`. The transformer selects only the first 10 columns; every column outside the slice is unselected and therefore dropped.

#### Question 5

Why is `pipe.predict(raw_row)` more robust than the manual inference notebook?

**Answer**

The pipeline stores the fitted imputers, encoders, scaler, selector, and model, and applies them in the correct order. The manual path requires the user to reproduce the order, dtypes, and missing-value handling by hand; the saved notebook crashes on a dtype-fragile object array.

#### Question 6

Why does `cross_val_score(pipe, X, y, cv=5)` avoid leakage?

**Answer**

It clones the pipeline for each fold and fits the preprocessing only on that fold's training rows. The validation rows are transformed with the fold-specific fitted objects, so they do not influence the learned parameters.

#### Question 7

The saved `grid.best_score_` equals the untuned CV mean. What should you check before concluding that tuning did not help?

**Answer**

Inspect `grid.cv_results_`. If all parameter values tie, the best score equals the common score by construction, and the selected parameter is arbitrary. The saved notebook does not show the individual results.

---

### A21. Revision summary

- A pipeline chains preprocessing and a model into one object.
- `fit` trains every stage; `predict` replays the transformations and predicts.
- Manual preprocessing works but is fragile: column order, dtypes, and learned parameters must be repeated exactly.
- The broken notebook fails because `drop` is unassigned and train/test blocks are swapped.
- The five-stage pipeline in the notebook has a real index bug: `trf2` encodes `Embarked` and `Fare`, not `Sex` and `Embarked`.
- `trf3` keeps only the first 10 columns and drops the rest; the final model omits `Sex`, `Age`, `Pclass`, `SibSp`, and `Parch`.
- Manual accuracy is `0.8101`; the pipeline accuracy is `0.6760`. They use different feature sets, so the comparison is not a fair test of pipelines.
- CV mean is `0.6264`; GridSearch best score ties at the same value because all depths tie.
- Manual inference is fragile; the saved run crashes, while a reproduced float cast returns `[0]`.
- `pipe.pkl` stores the whole chain and predicts raw rows in one call.
- Use pipelines inside cross-validation to prevent leakage.

---

## Part B — Advanced Reading and Reference

### B1. Pipeline internals

A `Pipeline` stores:

- `steps`: the ordered list of `(name, object)` tuples.
- `named_steps`: a dictionary-like view of the steps.
- `n_features_in_`: the number of input features seen during fit.
- `feature_names_in_`: the input feature names when the first step receives a DataFrame.

During `fit`, the pipeline calls `_fit` on each step. For intermediate steps it uses `fit_transform` when available; otherwise it calls `fit` and then `transform`. The final step is fitted with the transformed data and the target.

During `predict`, the pipeline calls `transform` on every intermediate step and `predict` on the final estimator. During `predict_proba`, it calls `predict_proba` on the final estimator.

The pipeline does not copy the data unnecessarily; each step receives the previous step's output.

#### Caching

`Pipeline(memory=...)` can cache the transformed outputs of intermediate steps. This is useful during long grid searches because repeated parameter combinations can reuse earlier transformations. Caching is disabled by default. The cache key includes the transformer parameters and the input data.

Caching must be used carefully: a stale cache after changing the data or code can produce wrong results.

---

### B2. `make_pipeline`

```python
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(trf1, trf2, trf3, trf4, trf5)
```

`make_pipeline` is a shorthand that generates step names from the class names. For example, the steps might be named `columntransformer-1`, `columntransformer-2`, and `decisiontreeclassifier`.

The mechanics are identical to `Pipeline`. The difference is parameter naming:

```python
# Pipeline with explicit names
{"trf5__max_depth": [1, 2, 3]}

# make_pipeline with generated names
{"decisiontreeclassifier__max_depth": [1, 2, 3]}
```

The notebook comments out `make_pipeline` and keeps the explicit `Pipeline`, which makes the grid parameter names readable.

---

### B3. `ColumnTransformer` versus `make_column_transformer`

These are different objects with similar names:

| Object | Purpose | Input format |
|---|---|---|
| `ColumnTransformer` | Apply different transformers to different columns | `(name, transformer, columns)` triples |
| `make_column_transformer` | Shorthand for `ColumnTransformer` | `(transformer, columns)` pairs |

The notebook's markdown heading says "DIFF BETWEEN COLUMN TRANSFORMER AND MAKE_COLUMN TRANSFORMER" but does not contain an explanation. The key point is:

- `Pipeline` is about **sequence**.
- `ColumnTransformer` is about **parallel per-column branches**.
- `make_pipeline` and `make_column_transformer` are convenience constructors, not different algorithms.

---

### B4. Column-index arithmetic

A `ColumnTransformer` output is the concatenation of:

1. The output of the first transformer.
2. The output of the second transformer.
3. ...
4. The remainder block.

The output positions are not the original DataFrame positions. This is why `trf2`'s `[1, 6]` selects `Embarked` and `Fare` after `trf1`.

A useful debugging pattern is:

```python
trf1.fit(X_train)
print(trf1.get_feature_names_out())
```

The names show the output order explicitly. For the notebook's `trf1`, the names are:

```text
impute_age__Age
impute_embarked__Embarked
remainder__Pclass
remainder__Sex
remainder__SibSp
remainder__Parch
remainder__Fare
```

The prefix before `__` is the transformer name; the suffix is the original column name.

For `trf2`, `get_feature_names_out()` shows the one-hot categories:

```text
ohe_sex_embarked__x1_C
ohe_sex_embarked__x1_Q
ohe_sex_embarked__x1_S
ohe_sex_embarked__x6_0.0
...
```

The `x1` and `x6` prefixes are the positional selectors, confirming that the encoder received positions 1 and 6.

---

### B5. Mathematics of the stages

#### Mean imputation

For a column with observed values $x_1, \dots, x_n$, the mean strategy fills missing entries with:

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i.
$$

The fitted value is stored in `statistics_`.

#### Most-frequent imputation

For a categorical column, the mode is:

$$
\hat{x} = \arg\max_{c} \sum_{i=1}^{n} \mathbf{1}(x_i = c).
$$

For `Embarked`, the training mode is `S`, which matches the saved `statistics_` output.

#### Min-max scaling

For a feature with training minimum $a$ and maximum $b$:

$$
x' = \frac{x - a}{b - a}.
$$

The fitted values are stored in `data_min_`, `data_max_`, and `scale_`. If $a = b$, scikit-learn handles the degenerate range by using a scale of 1.

In the notebook, `MinMaxScaler` is applied only to the first 10 columns of `trf2`, not to all 236.

#### One-hot encoding width

For a categorical feature with $K$ observed categories:

- Without dropping: $K$ columns.
- With `drop="first"`: $K-1$ columns.

For multiple features, sum the widths. For the notebook's `trf2`:

$$
3 + 228 = 231
$$

one-hot columns, plus 5 remainder columns, for 236 total.

#### Chi-square feature selection

For each feature, `chi2` compares observed counts $O$ with expected counts $E$ under the null hypothesis that the feature is independent of the class:

$$
\chi^2 = \sum \frac{(O - E)^2}{E}.
$$

Large values indicate stronger dependence. `SelectKBest` keeps the $k$ features with the largest scores. The test requires non-negative feature values; one-hot dummies and non-negative numeric columns satisfy this, but negative scaled values do not.

In the notebook, the selected features are dominated by fare dummies, which have many rare categories. Chi-square can assign high scores to rare categories, which is one reason the selection is unstable.

#### Accuracy

$$
\text{accuracy} = \frac{\text{number of correct predictions}}{\text{number of predictions}}.
$$

For the manual workflow:

$$
\frac{145}{179} = 0.8100558659217877.
$$

For the pipeline:

$$
\frac{121}{179} = 0.6759776536312849.
$$

#### Cross-validation mean

For $K$ folds with validation accuracies $a_1, \dots, a_K$:

$$
\text{CV mean} = \frac{1}{K}\sum_{k=1}^{K} a_k.
$$

For $K=5$, the saved mean is `0.6263862897665715`.

#### Grid search objective

`GridSearchCV` evaluates each parameter combination with cross-validation and selects:

$$
\hat{\theta} = \arg\max_{\theta} \text{CV mean}(\theta).
$$

When several $\theta$ values have the same CV mean, the first one in the grid is selected. This is why `max_depth=1` can be reported as "best" even when all depths tie.

---

### B6. Cross-validation and nested cross-validation

`cross_val_score(pipe, X, y, cv=5)` performs $K$-fold cross-validation:

1. Split the data into $K$ folds.
2. For each fold, train on $K-1$ folds and validate on the held-out fold.
3. Return the $K$ validation scores.

The pipeline is cloned for each fold. The preprocessing is fitted only on the fold's training portion.

`GridSearchCV` adds a parameter search on top of CV. If you use the same CV folds for both tuning and final evaluation, the final score can be optimistic because the model has seen the validation folds during tuning. Nested cross-validation uses an outer loop for evaluation and an inner loop for tuning:

```text
outer fold:
    inner CV: tune hyperparameters
    fit best model on outer training set
    evaluate on outer validation set
```

Nested CV gives a less biased estimate of the tuning procedure's generalization, at a higher computational cost.

---

### B7. Persistence and deployment

`pickle` serializes Python objects, including fitted scikit-learn estimators. It is convenient but has risks:

- **Version sensitivity:** a pickle created with one scikit-learn version may not load correctly with another.
- **Code dependency:** custom transformers must be importable when loading.
- **Security:** never load untrusted pickles; they can execute arbitrary code.
- **Schema drift:** the pipeline expects the same raw input columns and dtypes it was trained on.

For production, alternatives include `joblib` (often more efficient for NumPy-heavy objects), ONNX, or a model server with explicit schema validation. Regardless of the format, the preprocessing and the model must be versioned together.

The Day 29 pattern is:

```python
with open("pipe.pkl", "wb") as f:
    pickle.dump(pipe, f)

with open("pipe.pkl", "rb") as f:
    pipe = pickle.load(f)

prediction = pipe.predict(raw_row)
```

The raw row must have the same 7 columns in the same order as training.

---

### B8. Alternatives and comparisons

| Approach | Advantages | Disadvantages |
|---|---|---|
| Manual preprocessing | Transparent; easy to inspect each block | Repetition, drift, leakage risk, fragile inference |
| `ColumnTransformer` only | Handles per-column branches; reusable | Still needs a separate model and manual sequencing |
| `Pipeline` | Stores the full chain; safe CV; one-object deployment | Wrong selectors are executed faithfully; less transparent if overused |
| `make_pipeline` | Concise; auto names | Generated names make parameter grids less readable |
| Custom transformer | Encapsulates domain logic | More code; must be picklable and versioned |

A pipeline is not a substitute for validating the feature engineering. The Day 29 notebook demonstrates both truths: the pipeline correctly automates the chain, and the chain itself contains a feature-selection bug.

---

### B9. Notebook coverage matrix

| Notebook | Cells | What it teaches | Saved result |
|---|---:|---|---|
| `DAY29_Workflow-Without-Pipelines.ipynb` | 25 | Manual impute/encode/concat/train/evaluate/persist | Accuracy `0.8100558659217877`; test shape `(179, 10)` |
| `DAY29_Titanic-Using-Pipeline.ipynb` | 46 | Five-stage pipeline, CV, GridSearch, export | Accuracy `0.6759776536312849`; CV `0.6263862897665715`; Grid best `0.6263862897665715`; `statistics_` `['S']` |
| `DAY29_Sklearn-Pipelines-broken.ipynb` | 17 | Unassigned `drop`; swapped train/test blocks | `ValueError: could not convert string 'Sage, Mr. Douglas Bullen'` |
| `DAY29_Predict-Without-Pipeline.ipynb` | 15 | Manual inference from three pickles | Kernel crash on `predict` |
| `DAY29_Pipeline-Export-Predict.ipynb` | 9 | Load `pipe.pkl`; predict a raw row | `array([0])` plus feature-name warnings |

No notebook in this day contains a graph or a confusion matrix. The results are numeric scores and prediction arrays.

---

### B10. Source fidelity and caveats

- Saved outputs are reported as saved. Reproduced values are labeled as supplementary.
- The dataset is `datasets/train.csv` (891 rows, 12 columns); after dropping four columns, the feature matrix has 7 columns.
- The split is `test_size=0.2`, `random_state=34`: `(712, 7)` train, `(179, 7)` test.
- The manual test matrix is `(179, 10)`.
- The manual accuracy is `0.8100558659217877`.
- The pipeline's `trf2` selects `[1, 6]` after `trf1`, which are `Embarked` and `Fare`; this is a preserved bug.
- A reproduced `trf2` output is `(712, 236)`; this shape is not saved in the notebook.
- `trf3` keeps only the first 10 columns and drops the remaining 226; this follows from the default `remainder="drop"`.
- The saved pipeline's `trf4` support is `[0, 2, 3, 5, 6, 7, 8, 9]`; the final model uses 8 features derived from `Embarked` and `Fare`.
- The pipeline accuracy is `0.6759776536312849`; the CV mean is `0.6263862897665715`; the GridSearch best score equals the CV mean.
- A reproduced GridSearch shows all five `max_depth` values tie; this is a supplementary check, not a saved output.
- The saved `models/*.pkl` have `drop_idx_=[0]` and `clf.n_features_in_=8`, while the visible manual notebook code would produce 10 columns. The artifacts appear to have been overwritten by a later drop-first run. This inconsistency is preserved and documented.
- Manual inference assembles an object-dtype row; the saved run crashes. A reproduced `.astype(float)` returns `[0]`. The crash is a runtime/dtype fragility, not a proof that the model is invalid.
- `pipe.pkl` predicts `[0]` for `[3, "male", 3.5, 0, 2, 18.2458, "S"]` with two harmless feature-name warnings.
- No accuracy, CV, or GridSearch result is reported for the manual inference or export notebooks.
- No source notebook is modified; the bugs remain visible for learning.

---

### B11. References

- scikit-learn documentation: `Pipeline`, `make_pipeline`, `ColumnTransformer`, `make_column_transformer`, `SimpleImputer`, `OneHotEncoder`, `MinMaxScaler`, `SelectKBest`, `chi2`, `cross_val_score`, `GridSearchCV`.
- CampusX Day 29: Machine Learning Pipelines A–Z.
- Day 24–25 notes: scaling.
- Day 26–27 notes: encoding.
- Day 28 notes: `ColumnTransformer`.
- Day 30 notes: `FunctionTransformer` and distribution transforms.

---

### B12. Final connection

A pipeline is the object that turns a sequence of preprocessing decisions into a reproducible workflow. Its value is not that it makes the model better; its value is that it makes the workflow **consistent** between training, validation, and inference. The Day 29 notebooks show both sides of that value:

- The manual workflow achieves `0.8101` but is fragile and hard to reproduce.
- The pipeline automates the chain and reports `0.6760`, but it faithfully executes a feature-selection bug.

The next step is to inspect every stage, verify the feature names, and compare the intended plan with the executed plan. A pipeline is only as correct as the columns you give it.
