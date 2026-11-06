# Day_29 — Machine Learning Pipelines A–Z — Topics Guide (Day 29)

> Deep notes for **Day_29**. Structure follows AGENTS.md Part 9. Body content is preserved from day materials (`topics.md`, `topic_notes.md`, prior `notes.md`) — not a hollow outline.

> Companion: `my_notebook.ipynb` · reference: `reference_notebooks/` · data: `datasets/`

---

## Source material (preserved)

### From `topics.md` (course-aligned quick guide)

# Machine Learning Pipelines A–Z — Topics Guide (Day 29)

> Primary (study in this order):
> 1. `DAY29_Workflow-Without-Pipelines.ipynb` — manual, correct  
> 2. `DAY29_Sklearn-Pipelines-broken.ipynb` — bugs preserved on purpose  
> 3. `DAY29_Titanic-Using-Pipeline.ipynb` — pipeline + CV + GridSearch + pickle  
> 4. `DAY29_Predict-Without-Pipeline.ipynb` — manual inference (crashes)  
> 5. `DAY29_Pipeline-Export-Predict.ipynb` — one-line inference  
>
> Secondary: CampusX `day29-sklearn-pipelines` (`reference_notebook.ipynb` + extras).  
> Dataset: `datasets/train.csv` (Titanic).

## 1. What is it?

A **Pipeline** chains transformers and a final estimator into **one object**. `fit` runs every stage in order on training data; `predict` replays transforms then the model. Preprocessing and the model travel together.

## 2. Why do we need it?

Training-time steps (impute → encode → scale → select) must be **byte-identical** at inference. Doing them by hand means repeating column order, dtypes, and parameters — one slip and predictions silently corrupt. A pipeline freezes the chain so train and serve cannot drift apart. It also keeps preprocessing **inside** each CV fold (no leakage).

## 3. How does it work?

```text
raw row
  → stage1.transform
  → stage2.transform
  → ...
  → model.predict
```

- `Pipeline([...])` takes a list of `(name, object)` tuples.  
- Every object except the last must have `fit`/`transform`.  
- The last step is the estimator (`fit`/`predict`).  
- `pipe.fit(X, y)` → each mid-step `fit_transform`, final step `fit`.  
- `pipe.predict(X)` → each mid-step `transform`, final `predict`.

`ColumnTransformer` handles **per-column** math inside one stage; `Pipeline` handles **sequence** of stages.

## 4. Syntax breakdown

### Pipeline tuples

```python
Pipeline([
    ("prep", ColumnTransformer(...)),   # name, transformer object
    ("model", LogisticRegression()),    # name, estimator object
])
```

| Rule | Why |
|------|-----|
| list of 2-tuples | ordered stages; names unique for `pipe.named_steps` |
| pass **objects**, not calls | `LogisticRegression()` is unfitted; `logreg.fit(...)` already fitted would be wrong |
| last step predicts | earlier steps only transform |

**Wrong:**

```python
Pipeline([("model", LogisticRegression().fit(X, y))])  # already fitted — bad
Pipeline([("prep", preprocess_fn(X))])                 # called too early; not an estimator
```

### Manual vs pipeline (same math)

```python
# manual
Xt = scaler.fit_transform(Xtr)
model.fit(Xt, ytr)

# pipeline
pipe.fit(X, y)   # scaler + model inside
```

### cross_val_score with pipeline

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
# each fold: fit pipe on fold-train, score fold-val
```

Pipe must be **unfitted** when passed in — CV clones it per fold.

### GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipe, param_grid={"model__max_depth": [3, 5]}, cv=5)
grid.fit(X_train, y_train)
grid.best_estimator_   # whole pipeline with best params
```

Double underscore `model__max_depth` means: parameter `max_depth` of the step named `model`.

### Persistence

```python
import pickle
pickle.dump(pipe, open("pipe.pkl", "wb"))
pipe2 = pickle.load(open("pipe.pkl", "rb"))
pipe2.predict(raw_row)   # preprocess + model in one call
```

## 5. Practical implementation (correct pipeline path)

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv("datasets/train.csv")
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

X = df.drop(columns=["Survived"])
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=34
)

# Column indices after drop — notebook style (numpy positions)
trf1 = ColumnTransformer([
    ("impute_age", SimpleImputer(strategy="mean"), [2]),
    ("impute_emb", SimpleImputer(strategy="most_frequent"), [6]),
], remainder="passthrough")

trf2 = ColumnTransformer([
    ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), [1, 6]),
], remainder="passthrough")

# ... scale / select as in notebook ...
pipe = Pipeline([
    ("trf1", trf1),
    ("trf2", trf2),
    ("model", DecisionTreeClassifier()),
])

pipe.fit(X_train, y_train)
print(accuracy_score(y_test, pipe.predict(X_test)))
print(cross_val_score(pipe, X, y, cv=5).mean())
```

Exact stage list matches your notebook — keep its column indices and strategies when comparing runs.

## 6. Execution walkthrough (day narrative)

**A. Manual workflow**  
Drop columns → split → impute Age/Embarked → one-hot Sex/Embarked → concatenate in a fixed order → `DecisionTree.fit` → `predict` → **accuracy 0.81005** → `pickle` encoders + model for later.

**B. Broken notebook (do not copy)**  
Same idea but: `df.drop` **not assigned** (strings remain) and `X_test_trans` built from **train** blocks → `ValueError` converting `'Sage, Mr. Douglas Bullen'`. Lesson: assignment matters; train/test transforms must stay paired.

**C. Pipeline notebook**  
Declare `trf1…trf5` → `Pipeline` → `fit`/`predict` → `cross_val_score(cv=5)` → `GridSearchCV` → `pickle.dump(pipe)` (whole chain).

**D. Predict without pipeline**  
Load three pickles → hand-encode one passenger → concatenate → **crash** on `predict` (dtype mix object+float). Fragile: ~11 cells for one row.

**E. Predict with pipeline**  
Load `pipe.pkl` → `np.array([...]).reshape(1,7)` → `pipe.predict` → `array([0])`. Harmless `UserWarning` about feature names if numpy in / names out.

## 7. Important observations

| Topic | Rule |
|-------|------|
| Leakage | Fit preprocessing only on train (or inside CV via Pipeline) |
| Manual vs pipeline | Same math; pipeline is one object + safe CV |
| Holdout vs CV | One split score vs mean of many folds — do not compare raw numbers as equals |
| Column indices | Post-`ColumnTransformer` arrays are positional — track indices |
| Pickle | Version- and code-sensitive; ship code + pickle together |
| Broken files | Educational; keep bugs visible, fix only in corrected copies |
| GridSearch params | Prefix with step name: `model__max_depth` |

## 8. Connection with previous concepts

- **Day 24–25:** scalers become pipeline stages.  
- **Day 26–27:** encoders become pipeline stages.  
- **Day 28:** ColumnTransformer is the usual **first** stage inside the pipeline.  
- **Day 30–31:** `FunctionTransformer` / `PowerTransformer` slot in the same way.  
- **Earlier CV:** `cross_val_score` without a pipeline on pre-transformed `X` leaks — the notebook’s CV cells make that concrete.

## Common mistakes

1. Transforming all of `X` before `cross_val_score` (fold leakage).  
2. `fit_transform` on test in manual code.  
3. Passing an already-fitted estimator into `Pipeline`.  
4. Swapping train/test transformed arrays (broken notebook).  
5. Forgetting `handle_unknown="ignore"` → crash on new categories at inference.  
6. Comparing holdout accuracy to CV mean as if they were the same metric protocol.  
7. Expecting `pipe.predict` to accept wrong column order/dtypes — raw row must match training columns.

## Mini cheat sheet

```text
CONCEPT     → one object = preprocessing + model
SYNTAX      → Pipeline([("name", trf), ("name2", model)])
FIT         → fit trains all stages; predict replays transforms
CV          → pass UNFITTED pipe to cross_val_score
TUNE        → GridSearchCV(pipe, {"step__param": ...})
PERSIST     → pickle the WHOLE pipe
TRAP        → pre-transform outside CV; fit on test; manual order drift
```

### From `topic_notes.md` / deep notes

# Machine Learning Pipelines A–Z (Day 29)

> Playlist: Day 29 Machine Learning Pipelines A–Z.
> Notebooks (``): `DAY29_Workflow-Without-Pipelines` (correct manual, acc 0.81005, SOURCE of `models/*.pkl`), `DAY29_Sklearn-Pipelines-broken` (ValueError on `Sage, Mr. Douglas Bullen`, do-not-copy), `DAY29_Titanic-Using-Pipeline` (correct pipeline + CV/GridSearch, SOURCE of `pipe.pkl`), `DAY29_Predict-Without-Pipeline` (manual inference, crashes on `predict`), `DAY29_Pipeline-Export-Predict` (one-line `pipe.predict → [0]`). Pickles: `pipe.pkl`, `models/{clf,ohe_sex,ohe_embarked}.pkl` (copies alongside; originals in `BASICS/`).

## 1. Learning Objectives

- Do the correct manual flow and feel its fragility (column tracking, repeat-at-inference).
- Build the 5-stage pipeline (impute → one-hot → scale → select → tree), validate (CV), tune (GridSearch), persist (pickle).
- Contrast 11-cell manual inference vs 1-line pipeline inference.
- Diagnose the two preserved failures (unassigned `drop` + swapped transforms; manual-inference crash).

## 2. Why This Topic Matters

Training-time preprocessing must be byte-identical at inference. Manual repetition drifts; pipelines freeze the chain (preprocessing + model) into one versioned object — the difference between a demo and a deployable.

## 3. Prerequisites

- Days 26–28 (encoders, scaling, ColumnTransformer); `train_test_split`; DecisionTree intuition; `accuracy_score`.

## 4. First-Principles Explanation

Manual = you are the pipeline (remember every fill/encode/order/dtype). Pipeline = the object remembers: `fit` trains all stages on train; `predict` replays them on raw rows. `pipe.pkl`/`models/*.pkl` are frozen memory.

## 5. Formal Definition

- **`Pipeline([(name, estimator)])`**: chained steps with `fit/transform/predict`; `fit` calls `fit_transform` down the chain, `predict` calls `transform` then final `predict`. **`make_pipeline`** = same, auto-named.
- **`cross_val_score(cv=5)`**: 5-fold generalization estimate. **`GridSearchCV`**: exhaustive hyperparameter search with CV.
- **Pickle persistence**: `pickle.dump(obj)` / `pickle.load(open(path,"rb"))` — version- and code-sensitive.

## 6. How It Works Internally

- Manual (correct): assigned `drop(PassengerId/Name/Ticket/Cabin)` → split(.2, rs=34) → `SimpleImputer(mean Age/most_frequent Embarked)` → `OneHotEncoder(sparse_output=False, handle_unknown="ignore")` on Sex/Embarked → drop originals → `concatenate` (train order) → `DecisionTree.fit` → `predict` → `accuracy_score 0.81005` (179 test) → `makedirs models` + dumps.
- Broken: same but `drop` unassigned (Name strings survive) + `X_test_trans = concatenate(X_train_...)` swap → `ValueError: could not convert string 'Sage, Mr. Douglas Bullen'` on `fit`. No accuracy/pickle.
- Pipeline: `trf1=ColumnTransformer(impute Age[2]/Embarked[6], passthrough)` (index selectors on numpy) → `trf2=OHE[1,6]` → `trf3=MinMaxScaler slice(0,10)` → `trf4=SelectKBest(chi2, k=8)` → `trf5=DecisionTree` → `Pipeline` → `fit/predict` → `accuracy` → `cross_val_score(cv=5)` → `GridSearchCV` → `dump(pipe.pkl)`.
- Manual inference: load 3 pickles → hand-encode one passenger → `concatenate([0,3,4,5]+age+sex+embarked)` → crash on `predict` (dtype mixing object+float; kernel crash preserved).
- Pipeline inference: load `pipe.pkl` → `np.array([3,"male",3.5,0,2,18.2458,"S"]).reshape(1,7)` → `predict → [0]` (+2 harmless `UserWarning: X has no feature names...` — pass a named DataFrame to silence).

## 7. Step-by-Step Examples

Raw 7-col Titanic row → (manual) 11 cells of impute/encode/concat/predict vs (pipeline) `pipe.predict(row)`. Both notebooks use the same 7-col shape with different values — run both to compare.

## 8. Code Examples

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import pickle

pipe = Pipeline([
    ("impute", ColumnTransformer([
        ("num", SimpleImputer(strategy="mean"), [2]),
        ("cat", SimpleImputer(strategy="most_frequent"), [6]),
    ], remainder="passthrough")),
    ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    # ... scale slice, SelectKBest(chi2, k=8), DecisionTree (see notebook trf3-trf5)
])
pipe.fit(X_train, y_train)
pipe.predict(X_test)
cross_val_score(pipe, X_train, y_train, cv=5)
GridSearchCV(pipe, param_grid, cv=5).fit(X_train, y_train)

with open("pipe.pkl", "wb") as f: pickle.dump(pipe, f)
pipe2 = pickle.load(open("pipe.pkl", "rb"))
pipe2.predict(raw_row)  # raw 7-col row, no manual steps
```

## 9. Output Interpretation

- `accuracy 0.81005` = manual baseline on 179 test rows (compare pipeline accuracy/CV mean beside it).
- CV scores = generalization spread (higher variance → less stable). GridSearch best params = tuned config.
- `predict → [0]` = did-not-survive for that raw input. `UserWarning` about feature names = cosmetic (numpy in vs named fit).
- `ValueError ... 'Sage, Mr. Douglas Bullen'` = string column leaked into the tree (the broken notebook's lesson).

## 10. Important Parameters and Variations

| Piece | Key args in your work | Notes |
|---|---|---|
| `train_test_split` | `test_size=0.2, random_state=34` | reproducibility |
| `SimpleImputer` | `mean` / `most_frequent` | per dtype |
| `OneHotEncoder` | `sparse_output=False, handle_unknown="ignore"` | dense + unseen-safe |
| `MinMaxScaler` | slice(0,10) | scale first 10 post-OHE cols |
| `SelectKBest` | `chi2, k=8` | keep top-8 (chi2 needs non-negative) |
| `cross_val_score` | `cv=5` | estimate, not tune |
| `GridSearchCV` | `param_grid, cv=5` | tune |
| `Pipeline` vs `make_pipeline` | named vs auto names | same mechanics |

## 11. Practical Use Cases

- Ship one `pipe.pkl` (preprocessing + model) instead of three encoder/model files + a README of manual steps. New data = raw rows in, predictions out.

## 12. Common Mistakes and Misconceptions

1. `df.drop(...)` without assignment/`inplace` does nothing (broken notebook bug #1).
2. Swapped train/test transform variables (bug #2).
3. `fit_transform` on test; fitting imputers/encoders before split.
4. Manual inference dtype/order drift (crash notebook) — cast to training order/dtypes or use pipeline.
5. `chi2` on negative features fails; `MinMaxScaler` slice indices shift if upstream columns change.
6. Pickle across sklearn versions/code paths breaks — pin versions; prefer `pipe.pkl` (one object) over three loose files.

## 13. Relationship With Other Topics

Days 19–22 (what to fix) → 24–28 (the pieces) → 29 (the assembly). `Pipeline` is the exam of the whole block: impute (28) + encode (26–27) + scale (24–25) + combine (28) + select + classify, validated and persisted.

## 14. Revision Summary

- Manual (0.81005, 3 pickles, 11-cell inference, crashes) vs Pipeline (CV+GridSearch, 1 pickle, 1-line inference, robust). Freeze the chain; validate with CV; tune with GridSearch; never `fit` on test.

## 15. Practice Questions

1. Why must inference repeat training preprocessing exactly?
2. What two bugs kill the broken notebook, and what error does each cause?
3. What does each of `trf1..trf5` do?
4. Why does `SelectKBest(chi2)` require non-negative features?
5. `cross_val_score` vs `GridSearchCV` — estimate vs tune?
6. Why does manual inference crash while `pipe.predict` works on the same-shaped row?
7. What does the `UserWarning: X has no feature names` mean, and how do you silence it?
8. Why is one `pipe.pkl` safer to deploy than three loose `models/*.pkl` + manual steps?

<details><summary>Answers</summary>

1. Any deviation (order/dtype/fill/category) silently corrupts features → wrong predictions.
2. Unassigned `drop` → strings reach tree → ValueError; swapped transforms → train/test mismatch (masked by bug #1).
3. Impute / one-hot / scale slice / select-k-8 / decision tree.
4. chi2 statistic is defined on non-negative counts/frequencies.
5. CV estimates generalization of a fixed config; GridSearch searches configs using CV.
6. Manual reassembly mixed dtypes/order; pipeline replays the exact fitted transforms.
7. Fitted with DataFrame names, predicted with nameless numpy — cosmetic; pass a DataFrame with the same column names.
8. One versioned object, no manual-step drift; preprocessing travels with the model.
</details>

---

## Section map (Part 9 checklist)

Content above covers the day’s topics. When expanding further, place new material under these headings without deleting preserved sources:

- **1. Introduction** — Course intro / what this day covers (from topics + notebook headings).
- **2. Problem Being Solved** — Why this technique exists; what breaks if ignored.
- **3. Required Foundations** — Prerequisites from earlier days.
- **4. Conceptual Understanding** — Core idea in plain English with technical accuracy.
- **5. Mathematical Foundation** — Formulas, variables, numerical examples (when relevant).
- **6. Internal Working** — fit/transform internals, what is stored, execution flow.
- **7. Python and Scikit-learn Implementation** — Complete meaningful implementation patterns.
- **8. Syntax Breakdown** — Arguments, defaults, object vs call, return values.
- **9. Practical Examples** — Before/after tables, runnable snippets from the day.
- **10. Visual Interpretation** — Graphs: axes, pattern, conclusion, limits.
- **11. Comparison with Related Techniques** — vs alternatives; when to use / avoid.
- **12. Assumptions and Limitations** — Hard limits and edge cases.
- **13. Common Mistakes** — Beginner + workflow mistakes with fixes.
- **14. Real-World Applications** — Project scenarios and decision process.
- **15. Connection to the ML Workflow** — Where it sits in the pipeline; course links.
- **16. Interview Questions and Answers** — Levels 1–6 with reasoned answers.
- **17. Practice Questions** — Conceptual / code / debug / decision; answers separate.
- **18. Topic Summary and Connections** — Cheat sheet + link to next topic.

---

## Interview quick set

Use preserved Q&A in the body above. Add Level 1–6 follow-ups here as you study:

1. What problem does this day solve?
2. How does the method work internally?
3. What is the common leakage / domain mistake?
4. How do you compare alternatives with CV?
5. Where does this sit in a Pipeline?

## Practice

Re-run `my_notebook.ipynb`, predict one output before running, then check. Update mastery checkboxes as you go.

## Mastery checklist

- [ ] I can explain the problem in plain words
- [ ] I can write the core implementation
- [ ] I can explain important syntax / defaults
- [ ] I can name common mistakes and fixes
- [ ] I can connect this day to the ML workflow
- [ ] I can answer interview follow-ups
