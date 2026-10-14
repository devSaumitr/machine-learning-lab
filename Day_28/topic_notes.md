# ColumnTransformer (Day 28)

> Playlist: Day 28 Column Transformer / How to use ColumnTransformer in Sklearn.
> Notebook: `DAY28_Column-Transformer.ipynb` (covid_toy.csv: age/gender/fever/cough/city→has_covid; WITHOUT-CT (80,7) vs WITH-CT (20,23)).

## 1. Learning Objectives

- Reproduce the manual path (impute → ordinal → one-hot → concatenate) and the `ColumnTransformer` path.
- Write `(name, transformer, columns)` triples with `remainder="passthrough"`.
- Explain why `transform` (not `fit_transform`) belongs on test data.
- Map this notebook to Day-29 `trf1`.

## 2. Why This Topic Matters

Different columns need different math (fever → median, cough → ordinal, gender/city → one-hot, age → pass). Doing it by hand means tracking every block and concatenation order — the exact pain `ColumnTransformer` removes, and the exact object inside Day-29 pipelines.

## 3. Prerequisites

- Days 26–27 encoders; `SimpleImputer`; `train_test_split`; `np.concatenate`/`hstack`.

## 4. First-Principles Explanation

List, per column group, "which transformer touches it". `ColumnTransformer` runs all groups in parallel and glues the outputs into one array. `remainder="passthrough"` keeps unlisted columns (age) as-is.

## 5. Formal Definition

- **`ColumnTransformer(transformers=[(name, transformer, columns)], remainder=...)`**: applies each transformer to its columns, concatenates outputs. `fit` learns all sub-transformers on train; `transform` applies them.

## 6. How It Works Internally

- WITHOUT-CT: `SimpleImputer(fever)` → `OrdinalEncoder(["Mild","Strong"])` on cough → `OneHotEncoder(drop="first")` on gender/city → age passthrough → `np.concatenate` → (80,7). You track order manually.
- WITH-CT: `tnf1=(imputer, [fever])`, `tnf2=(ordinal, [cough])`, `tnf3=(one-hot, [age, gender])`, `remainder="passthrough"` → `fit_transform(train)` / `fit_transform(test)` → (20,23). Preserved quirks: CT one-hots age+gender while manual one-hots gender+city (compare, don't assume identical); test uses `fit_transform` twice (bug — see §12); split has no `random_state` (non-reproducible); only tail rows show execution (71,78,79,81).

## 7. Step-by-Step Examples

covid_toy row → tnf1 fills fever → tnf2 maps Mild/Strong→0/1 → tnf3 expands gender(+age per this notebook) → remainder passes the rest → one output row. Same row through manual path needs three separate transforms + manual `concatenate` in the same order.

## 8. Code Examples

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

ct = ColumnTransformer([
    ("impute_fever", SimpleImputer(strategy="mean"), ["fever"]),
    ("ord_cough", OrdinalEncoder(categories=[["Mild", "Strong"]]), ["cough"]),
    ("ohe_cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), ["gender", "city"]),
], remainder="passthrough")

Xtr = ct.fit_transform(X_train)  # learn on TRAIN
Xte = ct.transform(X_test)       # apply to test
```

## 9. Output Interpretation

- (80,7) manual vs (20,23) CT reflect different column choices/splits in this demo, not a universal ratio — read the transformer's column lists to explain any shape.
- Executed tail indices (71,78,79,81) only; `null` above means cells never ran — re-run top-to-bottom.

## 10. Important Parameters and Variations

| Arg | Meaning |
|---|---|
| `(name, transformer, columns)` | name for debugging, estimator, column selector (names or indices) |
| `remainder="passthrough"/"drop"` | keep or drop unlisted columns |
| `sparse_threshold` | dense vs sparse output |
| `verbose_feature_names_out` | readable output names via `get_feature_names_out()` |

Day-29 uses index-based columns on numpy arrays (`trf1` imputes Age[2]/Embarked[6]) — same object, different selector style.

## 11. Practical Use Cases

- Any mixed-type table: one `ct` replaces pages of manual block-tracking; embed as pipeline step `trf1`.

## 12. Common Mistakes and Misconceptions

1. `fit_transform` on test (twice here) → leakage + mismatched categories. Fix: `transform`.
2. Column mismatch manual-vs-CT (gender+city vs age+gender) — compare before concluding equivalence.
3. No `random_state` in split → irreproducible shapes/order.
4. Forgetting `remainder` drops columns silently (default `"drop"`).
5. Index-vs-name selectors break after `train_test_split` → array conversion (Day-29 uses indices deliberately).

## 13. Relationship With Other Topics

Days 24–27 (the three transformers) → Day 28 (combine) → Day 29 (automate: `ct` becomes pipeline `trf1`, followed by `trf2` OHE / `trf3` scale / `trf4` select / `trf5` model).

## 14. Revision Summary

- One object, per-column transformers, single output array. Fit-train/transform-test; declare columns + remainder; reproducible split.

## 15. Practice Questions

1. Write the `ColumnTransformer` for: impute fever / ordinal cough / one-hot gender+city / passthrough age.
2. What does `remainder="passthrough"` do, and what is the default?
3. Why is `fit_transform(X_test)` wrong here?
4. Manual gives (80,7), CT gives (20,23) — what two things explain the difference?
5. How does this notebook's `ct` relate to Day-29 `trf1`?
6. Split has no `random_state` — what breaks for revision?

<details><summary>Answers</summary>

1. See §8 code block.
2. Keeps unlisted columns; default `"drop"` removes them.
3. Learns test statistics/categories — leakage + train/test mapping mismatch.
4. Different column selections (gender+city vs age+gender) and different effective splits/rows executed.
5. Same pattern: imputation stage inside the larger pipeline; Day-29 uses index selectors.
6. Reproducibility — shapes/rows/order change per run.
</details>
