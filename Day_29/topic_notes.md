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
